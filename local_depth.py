"""Depth Anything AC: local batch runner.

This is the depth fork's counterpart to IFAN's `local_infer.py`, and it keeps the same
promises: point it at an image or a folder and it processes everything, writes each
result beside its source in a `depth/` subfolder with a `_depth` suffix, and speaks the
interface contract in `INTERFACE.md` (tool `depthac-depth`, contract 1) so a handler can
drive it without reading the console.

What is different, because depth is not deblurring:

* **CPU only.** The project is a CPU project: no CUDA, no xformers, no GPU wheels.
  At a 518 px working size a 16.9 Mpx frame takes about 0.7 s.
* **The model has a fixed working size, and it is stated rather than hidden.** The short
  side is scaled to 518 px, the map is computed there and interpolated back to the input
  size with bilinear interpolation. The result is the input's frame, pixel for pixel --
  which is what a mask in Affinity needs -- but its detail is 518-based. This is the
  model's nature, not a downscale this runner chose.
* **The output is a 16-bit greyscale TIFF**, not a picture: one channel, Adobe Deflate,
  0 = far and 65535 = near (the app's normalised disparity). A 8-bit colour preview is
  written next to it purely to be looked at.
* **Nothing is refused for size**, because nothing here costs memory in proportion to the
  image: the network always sees the same small frame, and the only full-size buffer is
  the map itself.
"""

import argparse
import json
import logging
import os
import sys
import time
from datetime import datetime

# The vendored DINOv2 asks for xFormers and logs a warning when it is absent. On a CPU
# project it is always absent, and the fallback path is the normal one, so the line would
# only ever be noise. Silence that one logger before the model modules are imported.
logging.getLogger('dinov2').setLevel(logging.ERROR)

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from matplotlib import colormaps
from PIL import Image

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

from depth_anything.dpt import DepthAnything_AC


CONTRACT = 1
TOOL_ID = 'depthac-depth'
TOOL_VERSION = '1.0.0'

IMAGE_EXTS = ('.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff', '.webp')

DEFAULT_DEST = 'both'               # 'output', 'subfolder', 'suffix' or 'both'
DEFAULT_SUBFOLDER = 'depth'         # folder created beside the inputs
DEFAULT_SUFFIX = '_depth'           # appended to each output file name
COLOR_SUFFIX = '_color'             # the preview beside it, always 8-bit PNG
COLOR_EXT = '.png'

DEFAULT_SHORT_SIDE = 518            # the model's training size; not a display choice
PATCH_MULTIPLE = 14                 # the network's patch size
TIFF_DEFLATE = 8
DEFAULT_CKPT = os.path.join(HERE, 'checkpoints', 'depth_anything_AC_vits.pth')

MODEL_CONFIG = {'encoder': 'vits', 'features': 64,
                'out_channels': [48, 96, 192, 384], 'version': 'v2'}

# The names the browser app offers, mapped to matplotlib's colormaps. The app's default
# is Spectral_r, and the preview matches it so the two interfaces show the same picture.
COLORMAPS = {'spectral': 'Spectral_r', 'inferno': 'Inferno', 'gray': None}

MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def find_images(root, recursive):
    """Return image paths under `root` (a file or a directory)."""
    if os.path.isfile(root):
        return [root]

    paths = []
    if recursive:
        for dirpath, _dirnames, filenames in os.walk(root):
            for name in sorted(filenames):
                if name.lower().endswith(IMAGE_EXTS) and name != 'Thumbs.db':
                    paths.append(os.path.join(dirpath, name))
    else:
        for name in sorted(os.listdir(root)):
            path = os.path.join(root, name)
            if os.path.isfile(path) and name.lower().endswith(IMAGE_EXTS):
                paths.append(path)
    return paths


def is_within(path, folder):
    """True when `path` lies inside `folder` (case-insensitively, drive-aware)."""
    path = os.path.abspath(path)
    folder = os.path.abspath(folder)
    try:
        return os.path.commonpath([path, folder]) == folder
    except ValueError:      # different drives on Windows
        return False


def load_model(ckpt):
    """Load the AC network on the CPU, refusing a checkpoint that does not fit it.

    `app.py` loads with `strict=False` and carries on; a checkpoint missing parameters
    would leave part of the network random and produce a plausible-looking map built on
    noise. Here a missing parameter is a failed model, since there is nothing else it can
    be. Extra tensors are fine -- the checkpoint carries one the network does not use.
    """
    if not os.path.isfile(ckpt):
        raise RuntimeError('checkpoint not found: {}\nDownload depth_anything_AC_vits.pth '
                           'from the links in README.md and place it in checkpoints{}.'
                           .format(ckpt, os.sep))

    model = DepthAnything_AC(MODEL_CONFIG)
    state = torch.load(ckpt, map_location='cpu')
    missing, unexpected = model.load_state_dict(state, strict=False)
    if missing:
        raise RuntimeError('{} does not fit this network: {} parameter(s) missing ({}). '
                           'The result would be part random, so the run stops.'
                           .format(os.path.basename(ckpt), len(missing),
                                   ', '.join(missing[:3])))
    model.eval()
    return model, len(state), len(unexpected)


def read_image(path):
    """Read `path` as an RGB array in its own bit depth, without its alpha channel.

    OpenCV reads 8- and 16-bit unsigned images; anything else is refused rather than
    guessed at. The alpha channel comes off silently, as it does in the deblur runner:
    the whole frame is what a depth map is of, and the original still has it.
    """
    raw = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    if raw is None:
        raise RuntimeError('could not read {}'.format(path))

    if raw.dtype == np.uint16:
        bits = 16
    elif raw.dtype == np.uint8:
        bits = 8
    else:
        raise RuntimeError('{} holds {} data; this runner reads 8- and 16-bit unsigned '
                           'images only'.format(os.path.basename(path), raw.dtype))

    if raw.ndim == 3 and raw.shape[2] == 4:
        raw = raw[:, :, :3]
    elif raw.ndim == 3 and raw.shape[2] == 2:
        raw = raw[:, :, 0]

    if raw.ndim == 2:
        raw = cv2.cvtColor(raw, cv2.COLOR_GRAY2RGB)
    else:
        raw = raw[:, :, ::-1]                   # BGR -> RGB

    return raw, bits


def working_size(height, width, short_side):
    """The frame the network actually sees: short side `short_side`, /14 aligned.

    The same rule the app uses. It is rounded *up* to a multiple of 14, so the working
    frame is never smaller than the short side asked for.
    """
    scale = short_side / min(height, width)
    small_h = ((int(height * scale) + PATCH_MULTIPLE - 1) // PATCH_MULTIPLE) * PATCH_MULTIPLE
    small_w = ((int(width * scale) + PATCH_MULTIPLE - 1) // PATCH_MULTIPLE) * PATCH_MULTIPLE
    return small_h, small_w


def infer(model, raw, bits, short_side):
    """Return `(depth, (small_w, small_h))`: float32 in [0, 1] at the input's size.

    The normalisation is per image and happens on the small map, before it is
    interpolated back -- the same order the app uses, so a number here means what it
    meant there. 0 is the far end of the frame and 1 the near end.
    """
    height, width = raw.shape[:2]
    if min(height, width) < PATCH_MULTIPLE:
        raise RuntimeError('{}x{} is too small for a network that works in {} px patches'
                           .format(width, height, PATCH_MULTIPLE))

    small_h, small_w = working_size(height, width, short_side)
    scale = 65535.0 if bits == 16 else 255.0

    # Resizing in the file's own depth keeps a 100 Mpx frame from being expanded to a
    # float copy before anything is computed. A 16-bit source loses 1/65535 per step;
    # an 8-bit source loses half a level. Neither is visible in a map that is quantised
    # to 16 bits at the end anyway.
    small = cv2.resize(raw, (small_w, small_h), interpolation=cv2.INTER_CUBIC)
    image = small.astype(np.float32) / scale
    image = (image - MEAN) / STD
    tensor = torch.from_numpy(image.transpose(2, 0, 1)).unsqueeze(0).float()

    with torch.no_grad():
        prediction = model(tensor)['out']
        low, high = prediction.min(), prediction.max()
        depth = (prediction - low) / (high - low + 1e-6)

    if depth.dim() == 3:
        depth = depth.unsqueeze(1)
    depth = F.interpolate(depth, size=(height, width), mode='bilinear', align_corners=True)
    depth = depth.squeeze().cpu().numpy().astype(np.float32)

    return depth, (small_w, small_h)


def to_16bit(depth):
    """float32 [0, 1] -> the stored map. Rounding, not truncation: 0.5 is 32768."""
    return np.rint(np.clip(depth, 0.0, 1.0) * 65535.0).astype(np.uint16)


def color_lut(name):
    """A 65536-entry RGB lookup table, so the preview costs 3 bytes a pixel.

    Going through the colormap directly would hand back a float64 RGBA array -- half a
    gigabyte at 17 megapixels. Indexing a table is the same picture for the 16-bit map
    that is actually being stored, and it is the delivered value that is coloured.
    """
    levels = np.linspace(0.0, 1.0, 65536)
    if COLORMAPS[name] is None:
        grey = (levels * 255.0).round().astype(np.uint8)
        return np.repeat(grey[:, None], 3, axis=1)
    rgba = colormaps[COLORMAPS[name]](levels)
    return (rgba[:, :3] * 255.0).round().astype(np.uint8)


def write_map(path, map16):
    """Write the 16-bit greyscale map. TIFF gets Adobe Deflate, not OpenCV's LZW.

    LZW *inflates* this kind of data: 11.5 MiB against 7.9 MiB for the same 17 Mpx map,
    because the low byte of a smooth 16-bit gradient is close to random.
    """
    os.makedirs(os.path.dirname(path) or '.', exist_ok=True)
    params = []
    if path.lower().endswith(('.tif', '.tiff')):
        params = [cv2.IMWRITE_TIFF_COMPRESSION, TIFF_DEFLATE]
    if not cv2.imwrite(path, map16, params):
        raise RuntimeError('could not write {}'.format(path))


def write_color(path, map16, lut):
    """Write the 8-bit preview beside the map, from the same 16-bit values."""
    os.makedirs(os.path.dirname(path) or '.', exist_ok=True)
    Image.fromarray(lut[map16], mode='RGB').save(path)


def container_name(path):
    """The contract's word for the container: `tiff`, `png`, `jpeg` or None."""
    ext = os.path.splitext(path)[1].lower()
    return {'.tif': 'tiff', '.tiff': 'tiff', '.png': 'png',
            '.jpg': 'jpeg', '.jpeg': 'jpeg'}.get(ext)


def now_iso():
    """Local time with its UTC offset, the form the contract shows."""
    return datetime.now().astimezone().isoformat(timespec='seconds')


def result_entry(path, status, output=None, color_map=None, error=None,
                 original=None, processed=None, seconds=None):
    """One `results` entry, in the shape INTERFACE.md fixes."""
    return {
        'input': os.path.abspath(path),
        'output': os.path.abspath(output) if output else None,
        'color_map': os.path.abspath(color_map) if color_map else None,
        'status': status,
        'error': error,
        'input_size': [original[0], original[1]] if original else None,
        'output_size': [processed[0], processed[1]] if processed else None,
        'offset': [0, 0] if status == 'ok' else None,
        'bit_depth': 16 if status == 'ok' else None,
        'channels': 1 if status == 'ok' else None,
        'container': container_name(output) if output else None,
        'seconds': seconds,
    }


def write_manifest(path, manifest):
    """Write the run record, replacing any earlier copy in one step.

    Called once as the run starts and again as each file lands, so a run that is
    interrupted still leaves a record of what it managed to do rather than nothing.
    """
    directory = os.path.dirname(os.path.abspath(path))
    if directory:
        os.makedirs(directory, exist_ok=True)
    temporary = path + '.part'
    with open(temporary, 'w', encoding='utf-8') as handle:
        json.dump(manifest, handle, indent=2)
        handle.write('\n')
    os.replace(temporary, path)


def process(model, path, map_path, color_path, short_side, lut):
    """Depth-map one image, writing the map and (if asked) its preview."""
    raw, bits = read_image(path)
    height, width = raw.shape[:2]

    started = time.time()
    depth, small = infer(model, raw, bits, short_side)
    elapsed = time.time() - started

    map16 = to_16bit(depth)
    write_map(map_path, map16)
    if color_path:
        write_color(color_path, map16, lut)

    return elapsed, (width, height), (width, height), small, bits


def main():
    parser = argparse.ArgumentParser(
        description='Depth Anything AC: local batch runner for folders of images. '
                    'CPU only.')
    parser.add_argument('--input', '-i', required=True, help='input image or folder')
    parser.add_argument('--dest', choices=['output', 'subfolder', 'suffix', 'both'],
                        default=None,
                        help='where results go: "output" writes into --output; '
                             '"subfolder" writes into <input>/<--subfolder>; "suffix" '
                             'writes beside each input as <name><--suffix>.tif; "both" '
                             'does the last two (default: %s)' % DEFAULT_DEST)
    parser.add_argument('--subfolder', default=DEFAULT_SUBFOLDER,
                        help='subfolder name used by --dest subfolder and both '
                             '(default: %(default)s)')
    parser.add_argument('--suffix', default=DEFAULT_SUFFIX,
                        help='file-name suffix used by --dest suffix and both '
                             '(default: %(default)s)')
    parser.add_argument('--output', '-o', default=None,
                        help='destination root, used by --dest output '
                             '(default: <project>\\output)')
    parser.add_argument('--ckpt', default=None,
                        help='path to the pretrained checkpoint '
                             '(default: checkpoints\\depth_anything_AC_vits.pth)')
    parser.add_argument('--short-side', type=int, default=DEFAULT_SHORT_SIDE,
                        help='the model\'s working size: the image\'s short side is '
                             'scaled to this before inference and the map is '
                             'interpolated back (default: %(default)s, the training size). '
                             'Changing it departs from what the model was trained on')
    parser.add_argument('--ext', default='tif', choices=['tif', 'png'],
                        help='container for the 16-bit map (default: %(default)s)')
    parser.add_argument('--colormap', default='spectral',
                        choices=list(COLORMAPS),
                        help='colours for the 8-bit preview (default: %(default)s)')
    parser.add_argument('--no-color-map', action='store_true',
                        help='write the 16-bit map only, with no 8-bit preview')
    parser.add_argument('--recursive', '-r', action='store_true',
                        help='recurse into subfolders, keeping the folder structure')
    parser.add_argument('--manifest', default=None, metavar='PATH',
                        help='write a JSON record of the run (interface contract %d) to '
                             'PATH; a second program reads that rather than the console'
                             % CONTRACT)
    parser.add_argument('--version', action='version',
                        version='{} {}'.format(TOOL_ID, TOOL_VERSION),
                        help='print the tool id and version, "%s %s", and exit'
                             % (TOOL_ID, TOOL_VERSION))
    args = parser.parse_args()

    # Naming --output is shorthand for --dest output.
    if args.dest is None:
        args.dest = 'output' if args.output is not None else DEFAULT_DEST
    if args.dest == 'output' and args.output is None:
        args.output = os.path.join(HERE, 'output')

    if args.short_side < PATCH_MULTIPLE:
        sys.exit('--short-side must be at least {} px'.format(PATCH_MULTIPLE))
    if not os.path.exists(args.input):
        sys.exit('Input not found: {}'.format(args.input))
    ckpt = args.ckpt or DEFAULT_CKPT
    if not os.path.isfile(ckpt):
        sys.exit('Checkpoint not found: {}\nDownload depth_anything_AC_vits.pth from the '
                 'links in README.md and place it in checkpoints{}.'.format(ckpt, os.sep))

    try:
        model, tensors, unused = load_model(ckpt)
    except Exception as ex:
        sys.exit(str(ex))

    device_name = 'cpu'
    print('device     : {}'.format(device_name))
    print('model      : {}, {} px short side'.format(MODEL_CONFIG['encoder'],
                                                     args.short_side))
    print('checkpoint : {} ({} tensors, every parameter matched{})'
          .format(os.path.relpath(ckpt, HERE), tensors,
                  '' if not unused else ', {} unused'.format(unused)))
    print('colour map : {}'.format('off' if args.no_color_map else
                                   '{} -> <name>{}{}'.format(args.colormap, COLOR_SUFFIX,
                                                             COLOR_EXT)))
    if args.manifest:
        print('manifest   : {}'.format(os.path.abspath(args.manifest)))

    lut = None if args.no_color_map else color_lut(args.colormap)

    # Results are written relative to the input: into a subfolder of it, beside each
    # image with a suffix, or both.  "output" sends them to a separate root instead.
    root = args.input if os.path.isdir(args.input) else os.path.dirname(os.path.abspath(args.input))
    subfolder = os.path.join(root, args.subfolder)
    in_subfolder = args.dest in ('subfolder', 'both')
    beside = args.dest in ('suffix', 'both')

    manifest = None
    if args.manifest:
        manifest = {
            'contract': CONTRACT,
            'tool': {'id': TOOL_ID, 'version': TOOL_VERSION},
            'started': now_iso(),
            'finished': None,
            'device': device_name,
            'options': {
                'dest': args.dest,
                'suffix': args.suffix,
                'short_side': args.short_side,
                'ext': args.ext,
                'color_map': not args.no_color_map,
                'colormap': None if args.no_color_map else args.colormap,
            },
            'counts': {'found': 0, 'written': 0, 'failed': 0, 'skipped': 0},
            'results': [],
        }

    def finish(written=0, failed=0):
        """Stamp the record as finished and put it on disk, if one was asked for."""
        if not manifest:
            return
        manifest['counts'].update(written=written, failed=failed)
        manifest['finished'] = now_iso()
        write_manifest(args.manifest, manifest)

    images = find_images(args.input, args.recursive)
    # Never pick up our own earlier results, or a second run would map the maps.
    skipped = 0
    skipped_paths = []
    if in_subfolder or beside:
        kept = []
        for path in images:
            stem = os.path.splitext(os.path.basename(path))[0]
            if in_subfolder and is_within(path, subfolder):
                skipped += 1
                skipped_paths.append(path)
            elif beside and args.suffix and (
                    stem.endswith(args.suffix) or
                    stem.endswith(args.suffix + COLOR_SUFFIX)):
                skipped += 1
                skipped_paths.append(path)
            else:
                kept.append(path)
        images = kept

    if manifest:
        manifest['results'].extend(result_entry(path, 'skipped') for path in skipped_paths)
        manifest['counts'].update(found=len(images) + skipped, skipped=skipped)
        write_manifest(args.manifest, manifest)

    if not images:
        if skipped:
            print('Nothing to do: all {} image(s) under {} are already processed'
                  .format(skipped, args.input))
            finish()
            return
        finish()
        sys.exit('No images found under {}'.format(args.input))

    if args.dest == 'output':
        destination = os.path.abspath(args.output)
    elif args.dest == 'subfolder':
        destination = subfolder
    elif args.dest == 'suffix':
        destination = 'beside each input, as <name>{}<ext>'.format(args.suffix)
    else:
        destination = '{} as <name>{}<ext>'.format(subfolder, args.suffix)

    print('images     : {} found under {}'.format(len(images), args.input))
    if skipped:
        print('skipped    : {} already-processed file(s)'.format(skipped))
    print('destination: {}'.format(destination))
    print('')

    total = 0.0
    done = 0
    failed = 0
    for index, path in enumerate(images, 1):
        relative = os.path.relpath(path, root)
        directory, filename = os.path.split(relative)
        stem = os.path.splitext(filename)[0]
        name = stem + (args.suffix if beside else '') + '.' + args.ext
        color_name = stem + (args.suffix if beside else '') + COLOR_SUFFIX + COLOR_EXT

        if args.dest == 'suffix':
            map_path = os.path.join(root, directory, name)
            color_path = os.path.join(root, directory, color_name) if lut is not None else None
        else:
            target = subfolder if in_subfolder else os.path.abspath(args.output)
            map_path = os.path.join(target, directory, name)
            color_path = os.path.join(target, directory, color_name) if lut is not None else None

        try:
            elapsed, original, processed, small, bits = process(
                model, path, map_path, color_path, args.short_side, lut)
        except Exception as ex:
            failed += 1
            print('[{:>3}/{}] FAILED {}: {}'.format(index, len(images), relative, ex))
            if manifest:
                manifest['results'].append(result_entry(path, 'failed', error=str(ex)))
                manifest['counts']['failed'] = failed
                write_manifest(args.manifest, manifest)
            continue

        done += 1
        total += elapsed
        print('[{:>3}/{}] {}  {}x{} -> {}x{}  [model {}x{}]  {}-bit in  {:.2f}s'
              .format(index, len(images), relative, original[0], original[1],
                      processed[0], processed[1], small[0], small[1], bits, elapsed))

        if manifest:
            manifest['results'].append(result_entry(
                path, 'ok', output=map_path, color_map=color_path, original=original,
                processed=processed, seconds=round(elapsed, 2)))
            manifest['counts']['written'] = done
            write_manifest(args.manifest, manifest)

    print('')
    summary = 'Done: {} of {} image(s) in {:.1f}s'.format(done, len(images), total)
    if done:
        summary += ' (average {:.2f}s)'.format(total / done)
    if failed:
        summary += ', {} failed'.format(failed)
    print(summary)

    finish(done, failed)
    # The contract's exit code: 0 only when every requested file was written.
    sys.exit(0 if failed == 0 else 1)


if __name__ == '__main__':
    main()

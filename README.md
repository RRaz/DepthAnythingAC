---
title: DepthAnything AC
emoji: 🌖
colorFrom: blue
colorTo: green
sdk: gradio
sdk_version: 5.35.0
app_file: app.py
pinned: false
license: cc-by-nc-4.0
short_description: This is official demo of Depth Anything At Any Condition
---

Check out the configuration reference at https://huggingface.co/docs/hub/spaces-config-reference

> **Unofficial fork — modified, and non-commercial use only.** A copy of the Depth
> Anything AC demo by ghost233lism, changed to run locally and to write the depth map
> out as 16-bit data. It is **not affiliated with, or endorsed by**, the original
> authors, Meta, or the Depth Anything V2 authors. Licensed **CC-BY-NC-4.0** — see
> [`LICENSE.md`](LICENSE.md) for what that permits, and
> [`THIRD_PARTY_NOTICES.md`](THIRD_PARTY_NOTICES.md) for the terms of each component,
> including the Apache-2.0 code inherited from Depth Anything V2.

## Purpose of this fork

Two ways in, one model:

- **`local_depth.py`** — a batch runner in the form of IFAN's `local_infer.py` (the
  deblur fork). Point it at an image or a folder and every input gets a 16-bit greyscale
  TIFF depth map in a `depth/` subfolder, named after its source, with an 8-bit colour
  preview beside it. It also writes a manifest (interface contract 1) so a program —
  the Affinity handler — can drive it without reading the console.
- **`app.py`** — the original Gradio interface, left as it is, for looking at a frame in
  colour and for video.

Both are **CPU only**. This is a CPU project: there is no CUDA path and no GPU wheel to
install. A 16.9 Mpx frame takes about 0.7 s.

### What the depth map is

A 16-bit greyscale TIFF — one channel, Adobe Deflate, exactly the input's dimensions.
0 is the far end of the frame and 65535 the near end (the app's normalised disparity).
The 16 bits are for the dark end: on a 16.9 Mpx frame the darkest 16% of pixels have
8192 levels to spread over, where 8 bits would leave them 32.

The resolution is the model's, not the file's: the short side is scaled to 518 px, the
map is computed there and interpolated back up. The output frame matches the input pixel
for pixel — which is what a mask needs — but its detail is 518-based. The runner prints
the working size on every line rather than leaving it to be discovered.

### Usage

```bat
rem a folder: results land in <folder>\depth
.venv\Scripts\python.exe local_depth.py -i "D:\photos\bazaar"

rem one file, plus a record for another program to read
.venv\Scripts\python.exe local_depth.py -i "D:\photos\bazaar\IMG_1234.tif" --manifest run.json

rem or drop folders on run_depth.bat, or put them in input\ and double-click it
```

| Option | Default | Meaning |
|---|---|---|
| `--dest` | `both` | `both`, `subfolder`, `suffix` or `output` |
| `--subfolder` | `depth` | subfolder for results |
| `--suffix` | `_depth` | appended to each map's name |
| `--ext` | `tif` | `tif` (16-bit, deflate) or `png` |
| `--colormap` | `spectral` | preview colours: `spectral`, `inferno`, `gray` |
| `--no-color-map` | off | write the map only |
| `--short-side` | `518` | the model's working size; changing it leaves the training scale |
| `--recursive`, `-r` | off | walk subfolders |
| `--manifest PATH` | — | JSON record of the run (contract 1) |
| `--version` | — | print `depthac-depth 1.0.0` and exit |

### What changed, and when

Modified **2026-10-18** (the browser app) and **2026-10-20** (the batch runner) by Ryan
Raz ([RRaz](https://github.com/RRaz)), against the `app.py` as published in the Space.
The changes are confined to the fork's own files: the model code under
`depth_anything/`, the vendored DINOv2 copy under `torchhub/` and the checkpoints are
untouched.

| File | Change |
|---|---|
| `local_depth.py` | **new** — CPU batch runner: 16-bit greyscale TIFF per input in `depth/`, `_depth` suffix, optional 8-bit preview, per-file failures, `--version` and `--manifest` |
| `run_depth.bat` | **new** — drop-zone wrapper around `local_depth.py` |
| `app.py` | **`save_raw_depth_log()` added** — writes the normalised depth map to `logs/` as a 16-bit PNG (`I;16`); `create_colored_depth_map()` calls it |
| `app.py` | the checkpoint path is resolved against the script's own folder, so it runs from anywhere |
| `app.py` | `matplotlib.cm.get_cmap` → `matplotlib.colormaps[...]`, which recent matplotlib removed |
| `app.py` | the Gradio server binds `127.0.0.1` rather than `0.0.0.0`, so a local run is not exposed to the network |
| `app.py` | **all CUDA branches removed** — `model.cuda()` and the per-frame `.cuda()` calls are gone, so the browser app is CPU only like the runner |
| `run_app.bat`, `stop_app.bat` | **new** — local launcher, which clears a stale listener off port 7860 before starting |
| `LICENSE.md`, `THIRD_PARTY_NOTICES.md`, `LICENSES/` | **new** — licence, attribution, and the licence texts themselves |

## Credits

This fork is based on the original DepthAnything-AC project by ghost233lism:
https://huggingface.co/spaces/ghost233lism/DepthAnything-AC

Modifications in this repository were made by RRaz.

## Licence

**Non-commercial use only** — CC-BY-NC-4.0, the same terms upstream declares in the
front matter above. Personal, educational and research use is fine; use in paid work,
a product or a service is not. The AC checkpoint and the DINOv2 backbone are included
byte-for-byte as upstream publishes them, because running offline is the point of the
fork — non-commercial redistribution with attribution is what CC-BY-NC-4.0 allows.

If you need commercial use, that has to come from the original rights holders; this
fork cannot grant it.

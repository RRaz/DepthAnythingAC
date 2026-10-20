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

## Purpose of This Fork

The main purpose of this fork is to save the generated depth map as a 16-bit raw data file on the local machine.

This fork also supports running DepthAnything-AC locally on CPU, making it easier to use outside the hosted Hugging Face Space environment.

### What changed, and when

Modified **2026-10-18** by Ryan Raz ([RRaz](https://github.com/RRaz)), against the
`app.py` as published in the Space. The changes are confined to the fork's own files:
the model code under `depth_anything/`, the vendored DINOv2 copy under `torchhub/` and
the checkpoints are untouched.

| File | Change |
|---|---|
| `app.py` | **`save_raw_depth_log()` added** — writes the normalised depth map to `logs/` as a 16-bit PNG (`I;16`), which is the point of the fork; `create_colored_depth_map()` calls it |
| `app.py` | the checkpoint path is resolved against the script's own folder, so it runs from anywhere |
| `app.py` | `matplotlib.cm.get_cmap` → `matplotlib.colormaps[...]`, which recent matplotlib removed |
| `app.py` | the Gradio server binds `127.0.0.1` rather than `0.0.0.0`, so a local run is not exposed to the network |
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

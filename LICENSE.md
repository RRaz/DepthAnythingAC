# Licence

**This repository is for non-commercial use only.**

Depth Anything AC is distributed under the
[Creative Commons Attribution-NonCommercial 4.0 International licence](LICENSES/CC-BY-NC-4.0.txt)
(CC-BY-NC-4.0), the full text of which is in [`LICENSES/`](LICENSES). That licence lets
you copy, share and adapt the work **for non-commercial purposes**, provided you give
attribution and say what you changed. Commercial use is not permitted — not by this
fork, and not by anything it includes.

Upstream declares the same terms: the Hugging Face Space
[`ghost233lism/DepthAnything-AC`](https://huggingface.co/spaces/ghost233lism/DepthAnything-AC)
carries `license: cc-by-nc-4.0` in its `README.md`, and this fork neither relaxes nor
extends that.

## What that means in practice

- Personal, educational and research use: fine.
- Depth maps used in paid work, a product, or a service: **not permitted**.
- If you need commercial use, that has to come from the original rights holders — the
  Depth Anything AC authors for the model, and Meta for the DINOv2 backbone. This fork
  is not in a position to grant it, and nothing here should be read as doing so.

## Which terms apply to which files

The repository is a mixture, and every part has to satisfy its own terms. Per file and
per component, with copyright holders and sources:
[`THIRD_PARTY_NOTICES.md`](THIRD_PARTY_NOTICES.md).

Two consequences worth stating plainly:

- The code under `depth_anything/` comes from Depth Anything V2, whose code is
  **Apache-2.0** (a permissive licence with its own attribution conditions). It is
  redistributed here inside a CC-BY-NC-4.0 project, so the stricter non-commercial
  term applies to the repository as a whole.
- The two checkpoints in `checkpoints/` are redistributed exactly as the upstream
  Space ships them, for the same reason: the fork's purpose is to run offline.

## Changes in this fork

Modified **2026-10-18** by Ryan Raz ([RRaz](https://github.com/RRaz)), from
`ghost233lism/DepthAnything-AC` as published on Hugging Face (Space revision
`019a038`). See the "Purpose of This Fork" section of [`README.md`](README.md) for
what changed and why. In short: the fork's own additions are `run_app.bat` and
`stop_app.bat` (a local launcher that clears a stale listener off the port first) and
the changes to `app.py` that let the model run locally, on CPU as well as GPU, and
that save the depth map as a 16-bit raw file for use as a layer elsewhere. Those
additions are offered under the same CC-BY-NC-4.0 so the repository is uniform.

This fork is **not affiliated with, or endorsed by**, the Depth Anything AC authors,
Meta, or the Depth Anything V2 authors.

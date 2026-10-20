# Third-party notices

This repository is a copy of someone else's work with local changes on top of it. Both
the licence it inherits (CC-BY-NC-4.0) and the Apache-2.0 code it contains require that
attribution be kept and that changes be stated. This file is that record; keep it in
place if you redistribute any of this.

Component-by-component:

| Component | Where | © | Source | Terms |
|---|---|---|---|---|
| **Depth Anything AC** — demo app, model code and trained weights | `app.py`, `depth_anything/`, `util/`, `toyset/`, `requirements.txt`, `checkpoints/depth_anything_AC_vits.pth` | ghost233lism, 2025 | Hugging Face Space [`ghost233lism/DepthAnything-AC`](https://huggingface.co/spaces/ghost233lism/DepthAnything-AC) | [CC-BY-NC-4.0](LICENSES/CC-BY-NC-4.0.txt) |
| **Depth Anything V2** — the architecture the above builds on | parts of `depth_anything/` (`dpt.py`, `dinov2.py`, `util/`) | the Depth Anything V2 authors, 2024 | [DepthAnything/Depth-Anything-V2](https://github.com/DepthAnything/Depth-Anything-V2) | [Apache-2.0](LICENSES/Apache-2.0.txt) |
| **DINOv2** — vendored copy, unmodified | `torchhub/facebookresearch_dinov2_main/`, `checkpoints/dinov2_vits14/dinov2_vits14_pretrain.pth` | Meta Platforms, Inc. and affiliates | [facebookresearch/dinov2](https://github.com/facebookresearch/dinov2) | [CC-BY-NC-4.0](LICENSES/CC-BY-NC-4.0.txt) — code *and* weights, per its own `LICENSE` |
| **This fork's changes** | `run_app.bat`, `stop_app.bat`, changes to `app.py`, `LICENSE.md`, this file | Ryan Raz ([RRaz](https://github.com/RRaz)), 2026 | — | [CC-BY-NC-4.0](LICENSES/CC-BY-NC-4.0.txt) |

Notes on the table:

- **DINOv2 is not Apache-2.0.** It is often assumed to be, because parts of its
  ecosystem are; the copy vendored here carries its own `LICENSE`, and that file is the
  CC-BY-NC-4.0 text. Its README says the same in words: "DINOv2 code and model weights
  are released under the CC-BY-NC 4.0 license." The LICENSE inside
  `torchhub/facebookresearch_dinov2_main/` is left exactly where it is.
- **Two licences, one direction.** Apache-2.0 is permissive and CC-BY-NC-4.0 is not, so
  a repository containing both is bound by the stricter of the two: non-commercial.
  Nothing here is Apache-2.0-only.
- **The weights are included, deliberately.** The AC checkpoint and the DINOv2 backbone
  are redistributed byte-for-byte as the upstream Space publishes them, because the
  point of this fork is to run the model offline. Non-commercial redistribution with
  attribution is what CC-BY-NC-4.0 permits; selling it is not.

## If you redistribute this

Credit the Depth Anything AC authors (ghost233lism) and Meta for the DINOv2 backbone,
link the licence, say that you changed the files if you did, and keep
`licenses/CC-BY-NC-4.0.txt` and the vendored DINOv2 `LICENSE` with the copy. Do not add
terms that restrict what CC-BY-NC-4.0 allows.

None of this is legal advice. If you intend to use the model commercially, the question
is not what this repository says but what the original rights holders will agree to.

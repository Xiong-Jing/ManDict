# Pinned RT-DETR subset

Repository: https://github.com/lyuwenyu/RT-DETR
Commit: `df2a96c8c3a098567a1149bf0248da6efe6d495d`.

`rtdetr_pytorch/` contains byte-identical upstream source snapshots listed in
`UPSTREAM.json`. `LICENSE` is upstream Apache-2.0. Source attribution and upstream
DETR attribution remain present. `configs/` records the official R18 override and
common R50 configuration used to instantiate the original R18-vd architecture.

`rtdetr_runtime/` copies the selected operators into an isolated package; only
`src.core.register` / distribution helper imports are redirected to a local
single-process registry shim. The optional attention-mask path uses PyTorch
`Tensor.to` in place of the upstream `Tensor.astype` typo. Exact copy hashes and source mapping are recorded
in `RUNTIME_PATCHES.json`. The original global `src` package is never imported,
so optional teacher/OCR/transformer packages cannot be loaded accidentally.

Project adaptations are implemented in `cgis/models.py` / `cgis/losses.py`:

- Preserve aspect ratio and the valid centered-letterbox rectangle. Exclude padded
  centers from encoder top-query selection and mask padded cross-attention
  values. For exceptionally narrow images with fewer than 300 valid encoder
  anchors, repeat valid choices instead of selecting padded or infinite anchors.
- Expose the last decoder layer's hidden ordinary queries via a forward hook;
  remove the denoising prefix before any research extension.
- Preserve the official R18-vd stem/stages, hybrid encoder, three decoder layers,
  deformable-attention and iterative bounding-box calculations. No image
  multi-scale/anisotropic square resize is performed inside the detector.
- Reinitialize four-class score heads and denoising embeddings; require all other
  official COCO checkpoint tensors to load without shape mismatches. Paddle-exported
  checkpoints omit PyTorch BatchNorm counters; initialize only those counters to
  zero and list them explicitly in the transfer report.
- Cast denoising embeddings to the activation dtype before concatenation. This
  preserves bf16-compatible boolean-mask handling in torch 2.1 attention.
- Apply confidence-weighted positives and category-specific ignore masks uniformly
  to baseline/final main, encoder, decoder-auxiliary and denoising losses. Main
  detection uses the original IoU-target varifocal + 5 L1 + 2 GIoU formulation.
- The `combined` experimental arm adds local C2 refinement, detached signed
  utility targets under one coarse assignment, and local/ring context scoring
  with contrastive supervision. These project modules are not upstream components.

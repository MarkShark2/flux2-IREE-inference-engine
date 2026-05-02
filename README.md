# FLUX.2 IREE Inference Engine

This repository is a focused implementation path for running `black-forest-labs/FLUX.2-klein-4B` on Vulkan-first hardware through IREE, with later dynamic LoRA support.

## What Is Done So Far

- Built a small, flat export pipeline under `model-export-pipeline/`.
- Implemented phase 1 quantization: weight-only signed int4 (group size 128) for diffusion and text-encoder weights, with Hugging Face-style output layout.
- Added a quantized model loader that dequantizes at the load boundary, keeping core PyTorch math quantization-agnostic.
- Implemented full phase 2 pure-PyTorch inference (not a smoke probe) and generated validated 1024x1024 images from the quantized model.
- Integrated official FLUX.2 code through the repository submodule at `submodules/flux2`.
- Cleaned project structure and docs to keep machine-local helpers out of the pushed repo.

## Current Direction

The next major step is phase 3: export the working quantized inference path as MLIR/IREE artifacts with fixed initial shapes, then run on the BC-250 Vulkan target.

Near-term priorities:

1. Keep `02_pure_pytorch_inference.py` as the behavior reference.
2. Start `03_export_mlir.py` from the same quantized model and model code path.
3. Validate first end-to-end IREE execution on Vulkan, then iterate on performance and memory.
4. Add dynamic LoRA loading after the base IREE path is stable.
# Model Export Pipeline

This directory is the small working area for preparing FLUX.2 Klein 4B for the IREE inference engine. Keep it flat and practical: top-level numbered Python files for the main steps, a small `src/` package for shared code, `quantized/` for generated model files, and `output/` for images and logs.

No phase 0 is planned. Phase 1 starts with the minimum setup needed to quantize the model:

```text
01_quantize.py
src/flux2_iree/paths.py
```

The path helper resolves the active Hugging Face snapshot from `refs/main` in `/home/mark/IREE-inference-engine/models--black-forest-labs--FLUX.2-klein-4B` and points official FLUX.2 imports at the repository submodule in `../submodules/flux2`.

Generated files belong in ignored folders:

- `quantized/` for int4 model outputs and the Klein VAE copy/conversion.
- `output/` for images, logs, and small run reports.

Do not add `.gitkeep` placeholders or recreate the old nested `artifacts/`, `scripts/`, and `tests/` layout unless a later prompt explicitly asks for that structure.

Phase 2 is the full pure-PyTorch image generation path against the quantized model:

```text
02_pure_pytorch_inference.py
src/flux2_iree/quantized_model.py
src/flux2_iree/phase2_generate.py
```

The loader dequantizes packed int4 weights at the boundary. The generation path then uses ordinary floating-point PyTorch tensor math adapted from the official FLUX.2 repository, which keeps the core inference code aligned with the later IREE/Turbine direction. The phase 2 success artifact is a real 1024x1024 PNG in `output/`.

Phase 3 starts from that same code path and quantized model root. The next script should be `03_export_mlir.py`, exporting the smallest useful MLIR artifact first with fixed batch and resolution shapes, then writing generated compiler/runtime products under `output/` or another ignored path.
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


MODEL_CACHE_NAME = "models--black-forest-labs--FLUX.2-klein-4B"
QUANTIZED_MODEL_NAME = "FLUX.2-klein-4B-int4"


@dataclass(frozen=True)
class PipelinePaths:
    workspace_root: Path
    engine_root: Path
    pipeline_root: Path
    model_cache_root: Path
    lora_root: Path
    venv_python: Path
    quantized_root: Path
    output_root: Path

    @property
    def flux2_submodule_root(self) -> Path:
        return self.engine_root / "submodules" / "flux2"

    @property
    def flux2_src_root(self) -> Path:
        return self.flux2_submodule_root / "src"

    @property
    def quantized_model_root(self) -> Path:
        return self.quantized_root / QUANTIZED_MODEL_NAME

    def resolve_model_snapshot(self) -> Path:
        ref_file = self.model_cache_root / "refs" / "main"
        if not ref_file.is_file():
            raise FileNotFoundError(f"Missing Hugging Face ref file: {ref_file}")
        snapshot_id = ref_file.read_text(encoding="utf-8").strip()
        if not snapshot_id:
            raise RuntimeError(f"Empty Hugging Face ref file: {ref_file}")
        snapshot = self.model_cache_root / "snapshots" / snapshot_id
        if not snapshot.is_dir():
            raise FileNotFoundError(f"Resolved snapshot does not exist: {snapshot}")
        return snapshot


def get_paths() -> PipelinePaths:
    pipeline_root = Path(__file__).resolve().parents[2]
    engine_root = pipeline_root.parent
    workspace_root = engine_root.parent
    return PipelinePaths(
        workspace_root=workspace_root,
        engine_root=engine_root,
        pipeline_root=pipeline_root,
        model_cache_root=workspace_root / MODEL_CACHE_NAME,
        lora_root=workspace_root / "loras",
        venv_python=engine_root / ".venv" / "bin" / "python",
        quantized_root=pipeline_root / "quantized",
        output_root=pipeline_root / "output",
    )


def main() -> None:
    paths = get_paths()
    print(f"workspace_root={paths.workspace_root}")
    print(f"engine_root={paths.engine_root}")
    print(f"pipeline_root={paths.pipeline_root}")
    print(f"model_cache_root={paths.model_cache_root}")
    print(f"model_snapshot={paths.resolve_model_snapshot()}")
    print(f"flux2_submodule_root={paths.flux2_submodule_root}")
    print(f"flux2_src_root={paths.flux2_src_root}")
    print(f"lora_root={paths.lora_root}")
    print(f"venv_python={paths.venv_python}")
    print(f"quantized_model_root={paths.quantized_model_root}")
    print(f"output_root={paths.output_root}")


if __name__ == "__main__":
    main()
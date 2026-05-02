from __future__ import annotations

import argparse
import json
import shutil
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import torch
from safetensors import safe_open
from safetensors.torch import save_file

from flux2_iree.paths import get_paths


GROUP_SIZE = 128
QUANTIZED_SUFFIX = "__packed_int4"
SCALES_SUFFIX = "__scales"


@dataclass
class TensorRecord:
    name: str
    source_file: str
    output_file: str
    decision: str
    dtype: str
    shape: list[int]
    output_dtype: str
    output_shape: list[int]
    scales_name: str | None = None
    scales_shape: list[int] | None = None
    group_size: int | None = None
    pad_count: int = 0
    reason: str | None = None


def pack_signed_int4(values: torch.Tensor) -> torch.Tensor:
    encoded = (values.to(torch.int16) + 8).clamp(0, 15).to(torch.uint8).flatten()
    if encoded.numel() % 2:
        encoded = torch.cat([encoded, torch.zeros(1, dtype=torch.uint8)])
    pairs = encoded.view(-1, 2)
    return (pairs[:, 0] | (pairs[:, 1] << 4)).contiguous()


def should_quantize(name: str, tensor: torch.Tensor, group_size: int) -> tuple[bool, str]:
    if not tensor.is_floating_point():
        return False, "non-floating tensor"
    if tensor.ndim < 2:
        return False, "rank below 2"
    if tensor.numel() < group_size:
        return False, "smaller than one quantization group"
    lowered = name.lower()
    if "embed" in lowered or "embedding" in lowered:
        return False, "embedding tensor kept for stability"
    if "norm" in lowered or "layernorm" in lowered:
        return False, "normalization tensor kept for stability"
    if lowered.endswith(".bias") or lowered == "bias":
        return False, "bias tensor kept for stability"
    if min(tensor.shape) < 8:
        return False, "skinny tensor kept for stability"
    return True, "matrix-like floating weight"


def quantize_tensor(name: str, tensor: torch.Tensor, group_size: int) -> tuple[dict[str, torch.Tensor], TensorRecord]:
    original_shape = list(tensor.shape)
    flat = tensor.detach().to(torch.float32).flatten().cpu()
    pad_count = (-flat.numel()) % group_size
    if pad_count:
        flat = torch.cat([flat, torch.zeros(pad_count, dtype=torch.float32)])

    groups = flat.view(-1, group_size)
    scales = groups.abs().amax(dim=1).div(7.0).clamp_min(1e-8).to(torch.float16).contiguous()
    quantized = torch.round(groups / scales.to(torch.float32).unsqueeze(1)).clamp(-8, 7).to(torch.int8)
    packed = pack_signed_int4(quantized)
    packed_name = f"{name}{QUANTIZED_SUFFIX}"
    scales_name = f"{name}{SCALES_SUFFIX}"
    record = TensorRecord(
        name=name,
        source_file="",
        output_file="",
        decision="quantized_int4",
        dtype=str(tensor.dtype).replace("torch.", ""),
        shape=original_shape,
        output_dtype="uint8",
        output_shape=list(packed.shape),
        scales_name=scales_name,
        scales_shape=list(scales.shape),
        group_size=group_size,
        pad_count=pad_count,
        reason="matrix-like floating weight",
    )
    return {packed_name: packed, scales_name: scales}, record


def keep_tensor(name: str, tensor: torch.Tensor, reason: str) -> tuple[dict[str, torch.Tensor], TensorRecord]:
    kept = tensor.detach().cpu().contiguous().clone()
    record = TensorRecord(
        name=name,
        source_file="",
        output_file="",
        decision="kept",
        dtype=str(tensor.dtype).replace("torch.", ""),
        shape=list(tensor.shape),
        output_dtype=str(kept.dtype).replace("torch.", ""),
        output_shape=list(kept.shape),
        reason=reason,
    )
    return {name: kept}, record


def copy_sidecars(source_root: Path, output_root: Path) -> None:
    for source in source_root.rglob("*"):
        if not source.is_file():
            continue
        relative = source.relative_to(source_root)
        target = output_root / relative
        if source.suffix == ".safetensors" and relative.parts[:1] != ("vae",):
            continue
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)


def quantize_safetensors_file(source_file: Path, output_file: Path, source_root: Path, output_root: Path, group_size: int) -> list[TensorRecord]:
    payload: dict[str, torch.Tensor] = {}
    records: list[TensorRecord] = []
    relative_source = source_file.relative_to(source_root).as_posix()
    relative_output = output_file.relative_to(output_root).as_posix()

    with safe_open(source_file, framework="pt", device="cpu") as handle:
        for name in handle.keys():
            tensor = handle.get_tensor(name)
            should_pack, reason = should_quantize(name, tensor, group_size)
            if should_pack:
                additions, record = quantize_tensor(name, tensor, group_size)
            else:
                additions, record = keep_tensor(name, tensor, reason)
            record.source_file = relative_source
            record.output_file = relative_output
            payload.update(additions)
            records.append(record)
            del tensor

    output_file.parent.mkdir(parents=True, exist_ok=True)
    save_file(
        payload,
        output_file,
        metadata={
            "format": "pt",
            "quantization": "weight_only_signed_int4",
            "group_size": str(group_size),
            "source_file": relative_source,
        },
    )
    return records


def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def build_manifest(source_root: Path, output_root: Path, records: list[TensorRecord], group_size: int) -> dict[str, Any]:
    quantized = sum(1 for record in records if record.decision == "quantized_int4")
    kept = len(records) - quantized
    return {
        "model": "black-forest-labs/FLUX.2-klein-4B",
        "source_snapshot": str(source_root),
        "output_root": str(output_root),
        "format": "huggingface_snapshot_with_custom_int4_safetensors",
        "quantization": {
            "type": "weight_only_signed_int4",
            "group_size": group_size,
            "packed_suffix": QUANTIZED_SUFFIX,
            "scales_suffix": SCALES_SUFFIX,
        },
        "summary": {
            "total_tensors": len(records),
            "quantized_tensors": quantized,
            "kept_tensors": kept,
        },
        "tensors": [asdict(record) for record in records],
    }


def rewrite_text_encoder_index(output_root: Path, records: list[TensorRecord]) -> None:
    weight_map: dict[str, str] = {}
    for record in records:
        output_path = Path(record.output_file)
        if not output_path.parts or output_path.parts[0] != "text_encoder":
            continue
        shard_name = output_path.name
        if record.decision == "quantized_int4":
            weight_map[f"{record.name}{QUANTIZED_SUFFIX}"] = shard_name
            if record.scales_name is not None:
                weight_map[record.scales_name] = shard_name
        else:
            weight_map[record.name] = shard_name

    if not weight_map:
        return

    text_encoder_root = output_root / "text_encoder"
    total_size = sum(path.stat().st_size for path in text_encoder_root.glob("*.safetensors"))
    write_json(
        text_encoder_root / "model.safetensors.index.json",
        {
            "metadata": {
                "total_size": total_size,
                "quantization": "weight_only_signed_int4",
                "packed_suffix": QUANTIZED_SUFFIX,
                "scales_suffix": SCALES_SUFFIX,
            },
            "weight_map": dict(sorted(weight_map.items())),
        },
    )


def safetensors_to_quantize(source_root: Path) -> list[Path]:
    files: list[Path] = []
    for source in sorted(source_root.rglob("*.safetensors")):
        relative = source.relative_to(source_root)
        if relative.parts[:1] == ("vae",):
            continue
        files.append(source)
    return files


def main() -> None:
    parser = argparse.ArgumentParser(description="Quantize FLUX.2 Klein 4B into a simple HF-style int4 layout.")
    parser.add_argument("--overwrite", action="store_true", help="Replace the existing quantized output directory.")
    parser.add_argument("--group-size", type=int, default=GROUP_SIZE, help="Number of scalar weights per int4 scale.")
    parser.add_argument("--rewrite-index-only", action="store_true", help="Rewrite the text encoder safetensors index from an existing manifest.")
    args = parser.parse_args()

    paths = get_paths()
    source_root = paths.resolve_model_snapshot()
    output_root = paths.quantized_model_root

    if args.rewrite_index_only:
        manifest_path = output_root / "quantization_manifest.json"
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        records = [TensorRecord(**record) for record in manifest["tensors"]]
        rewrite_text_encoder_index(output_root, records)
        print(f"rewrote {output_root / 'text_encoder' / 'model.safetensors.index.json'}")
        return

    if output_root.exists():
        if not args.overwrite:
            raise SystemExit(f"Output already exists, pass --overwrite to replace it: {output_root}")
        shutil.rmtree(output_root)

    print(f"source snapshot: {source_root}")
    print(f"output root: {output_root}")
    print(f"group size: {args.group_size}")

    copy_sidecars(source_root, output_root)

    records: list[TensorRecord] = []
    for source_file in safetensors_to_quantize(source_root):
        output_file = output_root / source_file.relative_to(source_root)
        print(f"quantizing {source_file.relative_to(source_root)}")
        records.extend(quantize_safetensors_file(source_file, output_file, source_root, output_root, args.group_size))

    manifest = build_manifest(source_root, output_root, records, args.group_size)
    write_json(output_root / "quantization_config.json", manifest["quantization"])
    write_json(output_root / "quantization_manifest.json", manifest)
    rewrite_text_encoder_index(output_root, records)

    print(f"wrote {output_root}")
    print(f"quantized tensors: {manifest['summary']['quantized_tensors']}")
    print(f"kept tensors: {manifest['summary']['kept_tensors']}")


if __name__ == "__main__":
    main()
from __future__ import annotations

import json
from dataclasses import dataclass
from functools import reduce
from operator import mul
from pathlib import Path
from typing import Any

import torch
from safetensors import safe_open


PACKED_SUFFIX = "__packed_int4"
SCALES_SUFFIX = "__scales"


@dataclass(frozen=True)
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

    @property
    def numel(self) -> int:
        return reduce(mul, self.shape, 1)

    @property
    def packed_name(self) -> str:
        return f"{self.name}{PACKED_SUFFIX}"


def load_manifest(model_root: Path) -> dict[str, Any]:
    manifest_path = model_root / "quantization_manifest.json"
    return json.loads(manifest_path.read_text(encoding="utf-8"))


def tensor_records(manifest: dict[str, Any]) -> list[TensorRecord]:
    return [TensorRecord(**item) for item in manifest["tensors"]]


def records_for_component(records: list[TensorRecord], component: str) -> list[TensorRecord]:
    if component == "root":
        return [record for record in records if "/" not in record.output_file]
    prefix = f"{component}/"
    return [record for record in records if record.output_file.startswith(prefix)]


def choose_matrix_record(records: list[TensorRecord], component: str) -> TensorRecord:
    candidates = [
        record
        for record in records_for_component(records, component)
        if record.decision == "quantized_int4" and len(record.shape) == 2
    ]
    if not candidates:
        raise RuntimeError(f"No quantized matrix tensor found for component: {component}")
    return min(candidates, key=lambda record: record.numel)


def unpack_signed_int4(packed: torch.Tensor, value_count: int) -> torch.Tensor:
    packed_i16 = packed.to(torch.int16).flatten()
    low = (packed_i16 & 0x0F) - 8
    high = ((packed_i16 >> 4) & 0x0F) - 8
    values = torch.empty(packed_i16.numel() * 2, dtype=torch.int8)
    values[0::2] = low.to(torch.int8)
    values[1::2] = high.to(torch.int8)
    return values[:value_count]


def load_tensor(model_root: Path, record: TensorRecord, dtype: torch.dtype = torch.float32) -> torch.Tensor:
    tensor_path = model_root / record.output_file
    with safe_open(tensor_path, framework="pt", device="cpu") as handle:
        if record.decision == "kept":
            return handle.get_tensor(record.name).to(dtype=dtype).contiguous().clone()

        if record.decision != "quantized_int4":
            raise RuntimeError(f"Unsupported tensor decision: {record.decision}")
        if record.group_size is None or record.scales_name is None:
            raise RuntimeError(f"Incomplete quantization metadata for {record.name}")

        packed = handle.get_tensor(record.packed_name).contiguous().clone()
        scales = handle.get_tensor(record.scales_name).to(torch.float32).contiguous().clone()

    padded_count = record.numel + record.pad_count
    quantized = unpack_signed_int4(packed, padded_count).to(torch.float32)
    groups = quantized.view(-1, record.group_size)
    dequantized = groups * scales.view(-1, 1)
    return dequantized.flatten()[: record.numel].view(record.shape).to(dtype=dtype).contiguous()


def tensor_stats(tensor: torch.Tensor) -> dict[str, Any]:
    values = tensor.detach().to(torch.float32).cpu()
    return {
        "shape": list(tensor.shape),
        "dtype": str(tensor.dtype).replace("torch.", ""),
        "min": float(values.min()),
        "max": float(values.max()),
        "mean": float(values.mean()),
        "std": float(values.std(unbiased=False)),
    }
from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import torch
from einops import rearrange
from PIL import Image
from safetensors import safe_open
from transformers import AutoModelForCausalLM, AutoTokenizer

from flux2_iree.paths import get_paths
from flux2_iree.quantized_model import TensorRecord, load_manifest, load_tensor, tensor_records


def ensure_flux2_import_path() -> None:
    flux2_src = get_paths().flux2_src_root
    if not flux2_src.is_dir():
        raise FileNotFoundError(f"Missing FLUX.2 submodule source directory: {flux2_src}")
    if str(flux2_src) not in sys.path:
        sys.path.insert(0, str(flux2_src))


class LocalQwen3Embedder(torch.nn.Module):
    def __init__(self, model_dir: Path, tokenizer_dir: Path, device: str):
        super().__init__()
        self.model = AutoModelForCausalLM.from_pretrained(
            model_dir,
            torch_dtype=torch.bfloat16,
            device_map=device,
            local_files_only=True,
        )
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_dir, local_files_only=True)
        self.max_length = 512

    @torch.no_grad()
    def forward(self, prompts: list[str]) -> torch.Tensor:
        input_ids = []
        attention_masks = []
        for prompt in prompts:
            messages = [{"role": "user", "content": prompt}]
            text = self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=False,
            )
            model_inputs = self.tokenizer(
                text,
                return_tensors="pt",
                padding="max_length",
                truncation=True,
                max_length=self.max_length,
            )
            input_ids.append(model_inputs["input_ids"])
            attention_masks.append(model_inputs["attention_mask"])

        ids = torch.cat(input_ids, dim=0).to(self.model.device)
        masks = torch.cat(attention_masks, dim=0).to(self.model.device)
        output = self.model(input_ids=ids, attention_mask=masks, output_hidden_states=True, use_cache=False)
        selected = torch.stack([output.hidden_states[index] for index in [9, 18, 27]], dim=1)
        return rearrange(selected, "b c l d -> b l (c d)")


def load_quantized_flux_model(device: str):
    ensure_flux2_import_path()
    from flux2.model import Flux2, Klein4BParams

    paths = get_paths()
    model_root = paths.quantized_model_root
    manifest = load_manifest(model_root)
    records = [record for record in tensor_records(manifest) if record.output_file == "flux-2-klein-4b.safetensors"]

    state_dict: dict[str, torch.Tensor] = {}
    for index, record in enumerate(records, start=1):
        print(f"loading quantized flux tensor {index}/{len(records)}: {record.name}", flush=True)
        state_dict[record.name] = load_tensor(model_root, record, dtype=torch.bfloat16)

    with torch.device("meta"):
        model = Flux2(Klein4BParams()).to(torch.bfloat16)
    model.load_state_dict(state_dict, strict=True, assign=True)
    del state_dict
    return model.to(device).eval()


def diffusers_vae_key(target_key: str) -> str:
    if target_key.startswith("encoder.quant_conv."):
        return target_key.removeprefix("encoder.")
    if target_key.startswith("decoder.post_quant_conv."):
        return target_key.removeprefix("decoder.")
    if target_key.startswith("bn."):
        return target_key

    if target_key.startswith("encoder.norm_out."):
        return target_key.replace("encoder.norm_out.", "encoder.conv_norm_out.", 1)
    if target_key.startswith("decoder.norm_out."):
        return target_key.replace("decoder.norm_out.", "decoder.conv_norm_out.", 1)

    if target_key.startswith("decoder.up."):
        parts = target_key.split(".")
        level = int(parts[2])
        parts[2] = str(3 - level)
        target_key = ".".join(parts)

    replacements = [
        ("encoder.down.", "encoder.down_blocks."),
        ("decoder.up.", "decoder.up_blocks."),
        (".block.", ".resnets."),
        (".downsample.conv.", ".downsamplers.0.conv."),
        (".upsample.conv.", ".upsamplers.0.conv."),
        (".nin_shortcut.", ".conv_shortcut."),
        ("encoder.mid.block_1.", "encoder.mid_block.resnets.0."),
        ("encoder.mid.block_2.", "encoder.mid_block.resnets.1."),
        ("decoder.mid.block_1.", "decoder.mid_block.resnets.0."),
        ("decoder.mid.block_2.", "decoder.mid_block.resnets.1."),
        ("encoder.mid.attn_1.norm.", "encoder.mid_block.attentions.0.group_norm."),
        ("decoder.mid.attn_1.norm.", "decoder.mid_block.attentions.0.group_norm."),
        ("encoder.mid.attn_1.proj_out.", "encoder.mid_block.attentions.0.to_out.0."),
        ("decoder.mid.attn_1.proj_out.", "decoder.mid_block.attentions.0.to_out.0."),
        ("encoder.mid.attn_1.q.", "encoder.mid_block.attentions.0.to_q."),
        ("encoder.mid.attn_1.k.", "encoder.mid_block.attentions.0.to_k."),
        ("encoder.mid.attn_1.v.", "encoder.mid_block.attentions.0.to_v."),
        ("decoder.mid.attn_1.q.", "decoder.mid_block.attentions.0.to_q."),
        ("decoder.mid.attn_1.k.", "decoder.mid_block.attentions.0.to_k."),
        ("decoder.mid.attn_1.v.", "decoder.mid_block.attentions.0.to_v."),
    ]
    source_key = target_key
    for old, new in replacements:
        source_key = source_key.replace(old, new)
    return source_key


def load_local_klein_vae(device: str):
    ensure_flux2_import_path()
    from flux2.autoencoder import AutoEncoder, AutoEncoderParams

    paths = get_paths()
    snapshot = paths.resolve_model_snapshot()
    source_file = snapshot / "vae" / "diffusion_pytorch_model.safetensors"
    ae = AutoEncoder(AutoEncoderParams())
    target_state = ae.state_dict()
    converted: dict[str, torch.Tensor] = {}

    with safe_open(source_file, framework="pt", device="cpu") as handle:
        for target_key, target_tensor in target_state.items():
            source_key = diffusers_vae_key(target_key)
            tensor = handle.get_tensor(source_key)
            if target_tensor.ndim == 4 and tensor.ndim == 2:
                tensor = tensor[:, :, None, None]
            converted[target_key] = tensor.to(dtype=target_tensor.dtype).contiguous()

    ae.load_state_dict(converted, strict=True)
    return ae.to(device).eval()


def save_image(tensor: torch.Tensor, path: Path) -> None:
    x = tensor.clamp(-1, 1)
    x = rearrange(x[0], "c h w -> h w c")
    image = Image.fromarray((127.5 * (x + 1.0)).cpu().byte().numpy())
    path.parent.mkdir(parents=True, exist_ok=True)
    image.save(path, quality=95, subsampling=0)


def generate_image(
    prompt: str,
    output: Path,
    width: int = 1024,
    height: int = 1024,
    seed: int = 12345,
    device: str = "cuda",
) -> dict[str, Any]:
    ensure_flux2_import_path()
    from flux2.sampling import batched_prc_img, batched_prc_txt, denoise, get_schedule, scatter_ids

    if device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for the full 1024x1024 phase 2 generation run")

    paths = get_paths()
    snapshot = paths.resolve_model_snapshot()
    torch_device = torch.device(device)

    print("loading local text encoder", flush=True)
    text_encoder = LocalQwen3Embedder(snapshot / "text_encoder", snapshot / "tokenizer", device=device).eval()
    with torch.no_grad():
        ctx = text_encoder([prompt]).to(torch.bfloat16)
        ctx, ctx_ids = batched_prc_txt(ctx)
    text_encoder = text_encoder.cpu()
    del text_encoder
    torch.cuda.empty_cache()

    print("loading quantized diffusion model", flush=True)
    model = load_quantized_flux_model(device=device)

    print("loading local Klein VAE", flush=True)
    ae = load_local_klein_vae(device=device)

    with torch.no_grad():
        shape = (1, 128, height // 16, width // 16)
        generator = torch.Generator(device=device).manual_seed(seed)
        noise = torch.randn(shape, generator=generator, dtype=torch.bfloat16, device=torch_device)
        x, x_ids = batched_prc_img(noise)
        timesteps = get_schedule(4, x.shape[1])
        x = denoise(model, x, x_ids, ctx, ctx_ids, timesteps=timesteps, guidance=1.0)
        x = torch.cat(scatter_ids(x, x_ids)).squeeze(2)
        decoded = ae.decode(x).float()
        save_image(decoded, output)

    report = {
        "prompt": prompt,
        "seed": seed,
        "width": width,
        "height": height,
        "steps": 4,
        "guidance": 1.0,
        "device": device,
        "torch_version": torch.__version__,
        "model_root": str(paths.quantized_model_root),
        "source_snapshot": str(snapshot),
        "output": str(output),
    }
    report_path = output.with_suffix(".json")
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return report
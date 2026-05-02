from __future__ import annotations

import argparse
import json
from pathlib import Path

from flux2_iree.paths import get_paths
from flux2_iree.phase2_generate import generate_image


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate a 1024x1024 image with the quantized Klein model in pure PyTorch.")
    parser.add_argument("--prompt", default="a high quality photograph of a glass teapot on a wooden table, morning window light, detailed reflections")
    parser.add_argument("--output", default=None, help="PNG output path. Defaults to output/phase2_1024.png.")
    parser.add_argument("--width", type=int, default=1024)
    parser.add_argument("--height", type=int, default=1024)
    parser.add_argument("--seed", type=int, default=12345)
    parser.add_argument("--device", choices=["cuda"], default="cuda")
    args = parser.parse_args()

    output = Path(args.output) if args.output else get_paths().output_root / "phase2_1024.png"
    report = generate_image(
        prompt=args.prompt,
        output=output,
        width=args.width,
        height=args.height,
        seed=args.seed,
        device=args.device,
    )
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
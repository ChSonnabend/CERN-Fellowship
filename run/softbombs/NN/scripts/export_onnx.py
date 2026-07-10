#!/usr/bin/env python3
import argparse
from pathlib import Path

from _bootstrap import add_project_src

add_project_src()

from softbombs.config import load_config
from softbombs.onnx_export import export_checkpoint_to_onnx


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--config", required=True)
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--output", default=None)
    parser.add_argument("--opset-version", type=int, default=17)
    parser.add_argument("--skip-verify", action="store_true")
    args = parser.parse_args()

    config = load_config(args.config)
    training_dir = Path(config["training"]["output_dir"])
    checkpoint = Path(args.checkpoint) if args.checkpoint else training_dir / "best_model.pt"
    output = Path(args.output) if args.output else training_dir / f"{checkpoint.stem}.onnx"
    metadata = export_checkpoint_to_onnx(
        checkpoint,
        output,
        config_override=config,
        opset_version=args.opset_version,
        verify=not args.skip_verify,
    )
    print(f"Wrote {metadata['onnx']}")
    print(metadata)


if __name__ == "__main__":
    main()


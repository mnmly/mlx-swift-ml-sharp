#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.10"
# dependencies = [
#   "torch",
#   "numpy",
#   "safetensors",
# ]
# ///
"""Convert ml-sharp PyTorch weights to MLX safetensors format.

Usage:
    uv run scripts/convert.py -i ../../python/ml-sharp/sharp_2572gikvuh.pt
    uv run scripts/convert.py -i path/to/model.pt -o path/to/output.safetensors
"""

import argparse
import numpy as np
from pathlib import Path
from typing import Dict

try:
    import torch
except ImportError:
    raise ImportError("PyTorch is required. Run: uv run scripts/convert.py")

from safetensors.numpy import save_file


def convert_conv2d_weight(weight: np.ndarray) -> np.ndarray:
    """Convert Conv2d weight from OIHW to OHWI format."""
    return weight.transpose(0, 2, 3, 1)


def convert_conv_transpose2d_weight(weight: np.ndarray) -> np.ndarray:
    """Convert ConvTranspose2d weight from IOHW to OHWI format."""
    return weight.transpose(1, 2, 3, 0)


def is_conv2d_weight(key: str, shape: tuple) -> bool:
    if len(shape) != 4:
        return False
    if "deconv" in key:
        return False
    if "upsample" in key and "weight" in key and shape[2] == shape[3] == 2:
        return False
    return True


def is_conv_transpose2d_weight(key: str, shape: tuple) -> bool:
    if len(shape) != 4:
        return False
    if "deconv" in key:
        return True
    if "upsample" in key and shape[2] == shape[3] == 2:
        return True
    if "head.1.weight" in key and shape[2] == shape[3] == 2:
        return True
    return False


def convert_state_dict(state_dict: Dict[str, "torch.Tensor"], verbose: bool = False) -> Dict[str, np.ndarray]:
    mlx_weights = {}
    for key, value in state_dict.items():
        np_value = value.cpu().float().numpy()
        original_shape = np_value.shape

        if is_conv_transpose2d_weight(key, np_value.shape):
            np_value = convert_conv_transpose2d_weight(np_value)
            if verbose:
                print(f"  ConvTranspose2d: {key}: {original_shape} -> {np_value.shape}")
        elif is_conv2d_weight(key, np_value.shape):
            np_value = convert_conv2d_weight(np_value)
            if verbose:
                print(f"  Conv2d: {key}: {original_shape} -> {np_value.shape}")
        elif len(np_value.shape) == 4 and np_value.shape[2] <= 16 and np_value.shape[3] <= 16:
            np_value = convert_conv2d_weight(np_value)
            if verbose:
                print(f"  Conv2d (auto): {key}: {original_shape} -> {np_value.shape}")

        mlx_weights[key] = np_value

    return mlx_weights


def load_pytorch_checkpoint(path: Path) -> Dict[str, "torch.Tensor"]:
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    if isinstance(checkpoint, dict):
        if "state_dict" in checkpoint:
            return checkpoint["state_dict"]
        elif "model" in checkpoint:
            return checkpoint["model"]
        return checkpoint
    return checkpoint


def convert(input_path: Path, output_path: Path, verbose: bool = True) -> None:
    if verbose:
        print(f"Loading: {input_path}")
    state_dict = load_pytorch_checkpoint(input_path)

    if verbose:
        print(f"Converting {len(state_dict)} tensors...")
    mlx_weights = convert_state_dict(state_dict, verbose=verbose)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    save_file(mlx_weights, str(output_path))

    if verbose:
        size_mb = output_path.stat().st_size / 1e6
        print(f"Saved: {output_path} ({size_mb:.1f} MB)")


def main() -> int:
    parser = argparse.ArgumentParser(description="Convert ml-sharp PyTorch weights to MLX safetensors")
    parser.add_argument("-i", "--input", type=Path, required=True, help="Input PyTorch checkpoint (.pt)")
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        default=None,
        help="Output path (default: same name as input with .safetensors extension)",
    )
    parser.add_argument("-q", "--quiet", action="store_true", help="Suppress output")
    args = parser.parse_args()

    if not args.input.exists():
        print(f"Error: {args.input} not found")
        return 1

    output = args.output or args.input.with_suffix(".safetensors")
    convert(args.input, output, verbose=not args.quiet)
    print("Done!")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

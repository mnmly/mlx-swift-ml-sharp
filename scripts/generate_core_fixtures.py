#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.10"
# dependencies = [
#   "torch",
#   "timm",
#   "plyfile",
#   "scipy"
# ]
# ///
"""Generate numerical parity fixtures for ML-Sharp initializer/composer core."""

from __future__ import annotations

import json
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Any

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
PYTHON_SRC = (REPO_ROOT / "../../python/ml-sharp/src").resolve()
if str(PYTHON_SRC) not in sys.path:
    sys.path.insert(0, str(PYTHON_SRC))

from sharp.models.composer import GaussianComposer
from sharp.models.initializer import create_initializer
from sharp.models.params import DeltaFactor, InitializerParams


def encode_tensor(tensor: torch.Tensor | None) -> dict[str, Any] | None:
    if tensor is None:
        return None
    tensor = tensor.detach().cpu().float().contiguous()
    return {
        "shape": list(tensor.shape),
        "values": tensor.reshape(-1).tolist(),
    }


def make_image(height: int, width: int, start: float, stop: float) -> torch.Tensor:
    total = 3 * height * width
    values = torch.linspace(start, stop, total, dtype=torch.float32)
    return values.reshape(1, 3, height, width)


def make_depth(channels: int, height: int, width: int, start: float, stop: float) -> torch.Tensor:
    total = channels * height * width
    values = torch.linspace(start, stop, total, dtype=torch.float32)
    return values.reshape(1, channels, height, width)


def make_delta(num_layers: int, base_height: int, base_width: int, start: float, stop: float) -> torch.Tensor:
    total = 14 * num_layers * base_height * base_width
    values = torch.linspace(start, stop, total, dtype=torch.float32)
    return values.reshape(1, 14, num_layers, base_height, base_width)


def build_case(
    *,
    name: str,
    initializer_params: InitializerParams,
    delta_factor: DeltaFactor,
    image: torch.Tensor,
    depth: torch.Tensor,
    delta: torch.Tensor,
    min_scale: float,
    max_scale: float,
    color_activation_type: str,
    opacity_activation_type: str,
    color_space: str,
    base_scale_on_predicted_mean: bool,
    scale_factor: int = 1,
) -> dict[str, Any]:
    initializer = create_initializer(initializer_params)
    composer = GaussianComposer(
        delta_factor=delta_factor,
        min_scale=min_scale,
        max_scale=max_scale,
        color_activation_type=color_activation_type,
        opacity_activation_type=opacity_activation_type,
        color_space=color_space,
        base_scale_on_predicted_mean=base_scale_on_predicted_mean,
        scale_factor=scale_factor,
    )

    init_output = initializer(image, depth)
    gaussians = composer(
        delta=delta,
        base_values=init_output.gaussian_base_values,
        global_scale=init_output.global_scale,
        flatten_output=True,
    )

    return {
        "name": name,
        "initializer_params": asdict(initializer_params),
        "composer_params": {
            "delta_factor": asdict(delta_factor),
            "min_scale": min_scale,
            "max_scale": max_scale,
            "color_activation_type": color_activation_type,
            "opacity_activation_type": opacity_activation_type,
            "color_space": color_space,
            "base_scale_on_predicted_mean": base_scale_on_predicted_mean,
            "scale_factor": scale_factor,
        },
        "image": encode_tensor(image),
        "depth": encode_tensor(depth),
        "delta": encode_tensor(delta),
        "expected": {
            "feature_input": encode_tensor(init_output.feature_input),
            "global_scale": encode_tensor(init_output.global_scale),
            "base_values": {
                "mean_x_ndc": encode_tensor(init_output.gaussian_base_values.mean_x_ndc),
                "mean_y_ndc": encode_tensor(init_output.gaussian_base_values.mean_y_ndc),
                "mean_inverse_z_ndc": encode_tensor(init_output.gaussian_base_values.mean_inverse_z_ndc),
                "scales": encode_tensor(init_output.gaussian_base_values.scales),
                "quaternions": encode_tensor(init_output.gaussian_base_values.quaternions),
                "colors": encode_tensor(init_output.gaussian_base_values.colors),
                "opacities": encode_tensor(init_output.gaussian_base_values.opacities),
            },
            "gaussians": {
                "mean_vectors": encode_tensor(gaussians.mean_vectors),
                "singular_values": encode_tensor(gaussians.singular_values),
                "quaternions": encode_tensor(gaussians.quaternions),
                "colors": encode_tensor(gaussians.colors),
                "opacities": encode_tensor(gaussians.opacities),
            },
        },
    }


def main() -> int:
    output_path = REPO_ROOT / "Tests/MLXSharpTests/Fixtures/core-parity.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    cases = [
        build_case(
            name="default_two_layer_surface_min",
            initializer_params=InitializerParams(),
            delta_factor=DeltaFactor(),
            image=make_image(8, 8, 0.05, 0.95),
            depth=make_depth(2, 8, 8, 1.25, 6.75),
            delta=make_delta(2, 4, 4, -0.2, 0.2),
            min_scale=0.0,
            max_scale=10.0,
            color_activation_type="sigmoid",
            opacity_activation_type="sigmoid",
            color_space="linearRGB",
            base_scale_on_predicted_mean=True,
        ),
        build_case(
            name="base_depth_and_linear_disparity",
            initializer_params=InitializerParams(
                num_layers=3,
                color_option="first_layer",
                first_layer_depth_option="base_depth",
                rest_layer_depth_option="linear_disparity",
                normalize_depth=False,
                base_depth=12.0,
                scale_factor=1.25,
                disparity_factor=1.5,
            ),
            delta_factor=DeltaFactor(xy=0.01, z=0.02, color=0.4, opacity=0.5, scale=0.75, quaternion=0.1),
            image=make_image(10, 12, 0.1, 0.8),
            depth=make_depth(2, 10, 12, 2.0, 9.0),
            delta=make_delta(3, 5, 6, -0.35, 0.45),
            min_scale=0.25,
            max_scale=6.0,
            color_activation_type="exp",
            opacity_activation_type="softplus",
            color_space="sRGB",
            base_scale_on_predicted_mean=False,
        ),
        build_case(
            name="single_layer_surface_max_gray",
            initializer_params=InitializerParams(
                num_layers=1,
                color_option="none",
                first_layer_depth_option="surface_max",
                rest_layer_depth_option="surface_max",
                normalize_depth=True,
                feature_input_stop_grad=True,
                scale_factor=0.75,
            ),
            delta_factor=DeltaFactor(xy=0.005, z=0.015, color=0.2, opacity=1.25, scale=0.5, quaternion=0.25),
            image=make_image(6, 10, 0.2, 0.7),
            depth=make_depth(1, 6, 10, 1.0, 5.0),
            delta=make_delta(1, 3, 5, -0.1, 0.15),
            min_scale=0.1,
            max_scale=4.0,
            color_activation_type="softplus",
            opacity_activation_type="sigmoid",
            color_space="linearRGB",
            base_scale_on_predicted_mean=True,
        ),
    ]

    payload = {"cases": cases}
    output_path.write_text(json.dumps(payload, indent=2) + "\n")
    print(f"wrote {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

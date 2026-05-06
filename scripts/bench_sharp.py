#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.10"
# dependencies = [
#   "torch",
#   "timm",
#   "plyfile",
#   "numpy",
#   "scipy",
#   "safetensors",
# ]
# ///
"""Benchmark ml-sharp PyTorch predictor — stage-level timing.

Outputs JSON in the same schema as the Swift sharp-bench binary so results
can be compared directly.

Usage:
    uv run scripts/bench_sharp.py --weights ../../python/ml-sharp/sharp_2572gikvuh.pt
    uv run scripts/bench_sharp.py --weights sharp.pt --size 1536x1536 --iterations 10 --warmup 3
    uv run scripts/bench_sharp.py --help

To compare Swift vs Python:
    swift run sharp-bench --weights sharp_2572gikvuh.safetensors > swift.json
    uv run scripts/bench_sharp.py --weights ../../python/ml-sharp/sharp_2572gikvuh.pt > py.json
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path
from statistics import mean, median, stdev
from typing import Any

import torch


# ---------------------------------------------------------------------------
# Stats helpers
# ---------------------------------------------------------------------------

def summarise(samples: list[float]) -> dict[str, float]:
    s = sorted(samples)
    n = len(s)
    mu = mean(s)
    med = median(s)
    return {
        "mean": mu,
        "median": med,
        "min": s[0],
        "max": s[-1],
        "stddev": stdev(s) if n > 1 else 0.0,
    }


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------

def load_model(weights_path: str, device: torch.device, dtype: torch.dtype):
    """Load the ml-sharp predictor from a PyTorch checkpoint (.pt)."""
    sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "python" / "ml-sharp" / "src"))
    from sharp.models import create_predictor, PredictorParams

    params = PredictorParams()
    model = create_predictor(params)

    checkpoint = torch.load(weights_path, map_location="cpu", weights_only=True)
    state = checkpoint.get("model", checkpoint)
    model.load_state_dict(state, strict=True)
    model.to(device=device, dtype=dtype)
    model.eval()
    return model, params


# ---------------------------------------------------------------------------
# Stage-level benchmark
# ---------------------------------------------------------------------------

def run_benchmark(
    model,
    params,
    image_size: tuple[int, int],
    focal_pixels: float,
    iterations: int,
    warmup: int,
    device: torch.device,
    dtype: torch.dtype,
    label: str,
) -> dict[str, Any]:
    H, W = image_size
    fov = focal_pixels if focal_pixels > 0 else W * 0.58
    disp_factor = torch.tensor([fov / W], dtype=dtype, device=device)
    image = torch.ones(1, 3, H, W, dtype=dtype, device=device) * 0.5

    stage_samples: dict[str, list[float]] = {
        "monodepth": [],
        "init_align": [],
        "feature_model": [],
        "prediction_head": [],
        "composer": [],
    }
    total_samples: list[float] = []

    def sync():
        if device.type == "cuda":
            torch.cuda.synchronize()
        elif device.type == "mps":
            torch.mps.synchronize()

    def tick() -> float:
        sync()
        return time.perf_counter()

    def tock(t0: float) -> float:
        sync()
        return (time.perf_counter() - t0) * 1000.0  # ms

    with torch.no_grad():
        for step in range(warmup + iterations):
            t_total = tick()

            # Stage 1 — monodepth
            t = tick()
            mono_out = model.monodepth_model(image)
            dt_mono = tock(t)

            # Stage 2 — disparity scaling + depth alignment + initialiser
            t = tick()
            df = disp_factor[:, None, None, None]
            monodepth = df / mono_out.disparity.clamp(1e-4, 1e4)
            monodepth, _ = model.depth_alignment(monodepth, None,
                                                  mono_out.decoder_features)
            init_out = model.init_model(image, monodepth)
            dt_init = tock(t)

            # Stage 3 — feature model (GaussianDPT)
            t = tick()
            image_features = model.feature_model(
                init_out.feature_input,
                encodings=mono_out.output_features,
            )
            dt_feat = tock(t)

            # Stage 4 — prediction head
            t = tick()
            delta = model.prediction_head(image_features)
            dt_head = tock(t)

            # Stage 5 — composer
            t = tick()
            _ = model.gaussian_composer(
                delta=delta,
                base_values=init_out.gaussian_base_values,
                global_scale=init_out.global_scale,
            )
            dt_comp = tock(t)

            dt_total = tock(t_total)

            if step >= warmup:
                stage_samples["monodepth"].append(dt_mono)
                stage_samples["init_align"].append(dt_init)
                stage_samples["feature_model"].append(dt_feat)
                stage_samples["prediction_head"].append(dt_head)
                stage_samples["composer"].append(dt_comp)
                total_samples.append(dt_total)

    return {
        "label": label,
        "iterations": iterations,
        "warmup": warmup,
        "dtype": str(dtype).replace("torch.", ""),
        "inputShape": [1, 3, H, W],
        "stageStatsMs": {k: summarise(v) for k, v in stage_samples.items()},
        "totalStatsMs": summarise(total_samples),
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--weights", required=True, help="Path to ml-sharp .pt checkpoint")
    p.add_argument("--size", default="1536x1536",
                   help="Input resolution HxW (default 1536x1536)")
    p.add_argument("--iterations", type=int, default=10)
    p.add_argument("--warmup", type=int, default=3)
    p.add_argument("--dtype", choices=["float32", "float16"], default="float32")
    p.add_argument("--device", default="cpu",
                   help="torch device: cpu | cuda | mps")
    p.add_argument("--label", default="python-torch")
    p.add_argument("--focal", type=float, default=0.0,
                   help="Focal length in pixels (0 = imageWidth * 0.58)")
    return p.parse_args()


def main():
    args = parse_args()

    H, W = map(int, args.size.lower().split("x"))
    dtype = torch.float32 if args.dtype == "float32" else torch.float16
    device = torch.device(args.device)

    print(f"Loading model from {args.weights}...", file=sys.stderr)
    model, params = load_model(args.weights, device, dtype)
    print(f"Running benchmark {H}x{W}, {args.iterations} iters, {args.warmup} warmup...",
          file=sys.stderr)

    result = run_benchmark(
        model=model,
        params=params,
        image_size=(H, W),
        focal_pixels=args.focal,
        iterations=args.iterations,
        warmup=args.warmup,
        device=device,
        dtype=dtype,
        label=args.label,
    )

    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()

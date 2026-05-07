"""Standalone Newton-Schulz throughput micro-benchmark (Sprint 1.1 Phase A.5.2).

Runs the 5-step Polar Express iteration on a fixed random 2D matrix, reports
time per call for fp32 and fp16 compute. Provides a hard reference for NS
performance on gfx1151 independent of PyTorch training, so Phase B can
compute the theoretical ceiling of the fp16-NS optimization.

Usage:
    python scripts/bench_newton_schulz.py
    python scripts/bench_newton_schulz.py --iters 2000 --shape 768 2816
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from halo_training.muon import zeropower_via_polar_express


def bench(shape, iters, dtype, device="cuda"):
    m = torch.randn(*shape, device=device, dtype=torch.float32)
    # warmup
    for _ in range(5):
        _ = zeropower_via_polar_express(m, steps=5, dtype=dtype)
    torch.cuda.synchronize()
    t0 = time.time()
    for _ in range(iters):
        _ = zeropower_via_polar_express(m, steps=5, dtype=dtype)
    torch.cuda.synchronize()
    return (time.time() - t0) / iters


def _bench_all_shapes(shapes, iters):
    results = []
    for shape in shapes:
        row = {"shape": list(shape)}
        for dtype_name, dtype in [("fp32", torch.float32), ("fp16", torch.float16)]:
            t_ms = bench(shape, iters, dtype) * 1000
            row[f"{dtype_name}_ms"] = t_ms
        row["fp16_speedup"] = row["fp32_ms"] / row["fp16_ms"]
        results.append(row)
        print(f"  shape={tuple(shape)}  fp32={row['fp32_ms']:.3f}ms  "
              f"fp16={row['fp16_ms']:.3f}ms  speedup={row['fp16_speedup']:.2f}x")
    return results


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--iters", type=int, default=500)
    p.add_argument("--shape", type=int, nargs=2, default=None,
                   help="Single matrix shape to bench. If omitted, a curated "
                        "set of representative OdinFlat 2D param shapes is used.")
    p.add_argument("--out", default="docs/perf/normuon-ns-benchmark.json")
    args = p.parse_args()

    if args.shape is not None:
        shapes = [tuple(args.shape)]
    else:
        # Representative OdinFlat 2D parameter shapes (from build_imu1_optimizer's
        # 2D group). d_model=768, n_layer=14, SwiGLU expands 768 -> 2816 etc.
        # Grouped here so the benchmark reflects the real workload.
        shapes = [
            (768, 768),       # qkv projections, output proj, shortconv W
            (768, 2816),      # SwiGLU gate_proj / up_proj
            (2816, 768),      # SwiGLU down_proj
            (768, 128),       # FactorizedEmbedding projection (small min-dim)
            (128, 768),       # FactorizedLMHead projection (small min-dim)
            (256, 768),       # tok_embeddings.projection
        ]

    print(f"NS micro-benchmark: {args.iters} iters per config")
    print(f"Hardware: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'cpu'}")
    print()

    results = _bench_all_shapes(shapes, args.iters)

    # Also run "realistic total NorMuon step" benchmark: call NS once per param
    # in a flat OdinFlat 2D-group list. Count is ~60 params per Phase A profile.
    # We approximate with a tight mix below; true count read from profile text.
    print()
    print("Estimated total NS cost per optimizer step (sum of per-shape calls):")
    # Heuristic: 14 layers * (qkv, o, shortconv_W, swiglu_gate, swiglu_up, swiglu_down)
    # = 84 2D params of mixed shapes. The dominant cost is SwiGLU's 3 matmuls per layer.
    shape_counts = {
        (768, 768): 14 * 4,      # ~4 square projections per layer
        (768, 2816): 14 * 2,     # gate_proj + up_proj per layer
        (2816, 768): 14 * 1,     # down_proj per layer
        (768, 128): 1,           # embed projection
        (128, 768): 1,           # lm_head projection
        (256, 768): 1,           # tok_embeddings projection
    }
    total_fp32 = 0.0
    total_fp16 = 0.0
    for row in results:
        count = shape_counts.get(tuple(row["shape"]), 0)
        total_fp32 += row["fp32_ms"] * count
        total_fp16 += row["fp16_ms"] * count
    if total_fp32 > 0:
        reduction = (1 - total_fp16 / total_fp32) * 100
        print(f"  Total fp32 NS/step: {total_fp32:.2f} ms")
        print(f"  Total fp16 NS/step: {total_fp16:.2f} ms")
        print(f"  Savings at fp16:    {total_fp32 - total_fp16:.2f} ms  "
              f"({reduction:.1f}% reduction)")
    else:
        reduction = 0.0
        print("  (shape not in default set; per-step estimate skipped)")

    out = {
        "iters": args.iters,
        "device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "per_shape": results,
        "estimated_total_ns_per_step_ms": {
            "fp32": total_fp32,
            "fp16": total_fp16,
            "reduction_pct": reduction,
        },
    }

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nWritten {args.out}")


if __name__ == "__main__":
    main()

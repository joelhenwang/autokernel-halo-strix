"""Phase I ship-gate measurement: Triton fused_swiglu vs autograd-safe HIP
vs eager at production shape (B=16, T=512, H=2048, fp16).

Run on a CUDA machine (B). Ship gate: Triton fwd+bwd >= 1.05x autograd-safe
HIP (via torch.ops.autokernel.silu_gate_mul).

Usage:
    python scripts/phase_i_triton_ship_gate.py

Output: docs/perf/triton-swiglu-ship-gate-bench.md

Plan: Phase I of master remediation plan (2026-05-11 continuation).
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import torch
import torch.nn.functional as F

from scripts.kernel_bench_harness import bench_kernel_fwd_bwd, print_result


def main():
    if not torch.cuda.is_available():
        print("Phase I requires CUDA. Exiting.")
        return 1

    # Production shape: OdinFlat SwiGLU block, batch=16, T=512, H=2048.
    # gate and up tensors each shape [B, T, H] = [16, 512, 2048].
    shape = (16, 512, 2048)
    dtype = torch.float16

    # 1) Eager reference (Inductor-fused when under torch.compile, but here
    # just plain op). Lower bound for "what autokernel has to beat".
    def eager_swiglu(gate, up):
        return F.silu(gate) * up

    # 2) Autograd-safe HIP path via torch.ops.autokernel.silu_gate_mul.
    #    Requires kernels.hip._torch_ops to be imported to register.
    try:
        import kernels.hip._torch_ops  # noqa: F401
        autograd_hip_available = True
    except Exception as exc:
        print(f"HIP autograd op not available: {exc}")
        autograd_hip_available = False

    def autograd_hip_swiglu(gate, up):
        return torch.ops.autokernel.silu_gate_mul(gate.contiguous(), up.contiguous())

    # 3) Triton fused_swiglu via our Phase D.B kernel.
    try:
        from kernels.triton.fused_swiglu import fused_swiglu, _TRITON_AVAILABLE
        triton_available = _TRITON_AVAILABLE
    except Exception as exc:
        print(f"Triton fused_swiglu not available: {exc}")
        triton_available = False
        fused_swiglu = None

    # Run benches.
    results = {}

    print(f"=== Phase I ship-gate bench ===")
    print(f"Shape: {shape}, dtype: {dtype}")
    print()

    if autograd_hip_available:
        print("Benching: eager vs autograd-safe HIP...")
        r = bench_kernel_fwd_bwd(
            name="eager-vs-autograd-hip",
            triton_fn=autograd_hip_swiglu,
            reference_fn=eager_swiglu,
            shape=shape, dtype=dtype, input_count=2,
            warmup=50, iters=200,
        )
        print_result(r)
        print()
        results["eager-vs-autograd-hip"] = r

    if triton_available:
        print("Benching: eager vs Triton...")
        r = bench_kernel_fwd_bwd(
            name="eager-vs-triton",
            triton_fn=fused_swiglu,
            reference_fn=eager_swiglu,
            shape=shape, dtype=dtype, input_count=2,
            warmup=50, iters=200,
        )
        print_result(r)
        print()
        results["eager-vs-triton"] = r

        if autograd_hip_available:
            print("Benching: autograd-safe HIP vs Triton (SHIP GATE)...")
            r = bench_kernel_fwd_bwd(
                name="autograd-hip-vs-triton",
                triton_fn=fused_swiglu,
                reference_fn=autograd_hip_swiglu,
                shape=shape, dtype=dtype, input_count=2,
                warmup=50, iters=200,
            )
            print_result(r)
            print()
            results["autograd-hip-vs-triton"] = r

    # Emit markdown report.
    out = REPO_ROOT / "docs" / "perf" / "triton-swiglu-ship-gate-bench.md"
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as f:
        f.write("# Phase I: Triton fused SwiGLU Ship-Gate Bench (2026-05-11)\n\n")
        f.write(f"Production shape: {shape}, dtype: {dtype}.\n\n")
        f.write("Three bench pairs (each 200 timed iters + 50 warmup):\n\n")
        f.write("| Comparison | fwd speedup | fwd+bwd speedup |\n")
        f.write("|---|---:|---:|\n")
        for name in ["eager-vs-autograd-hip", "eager-vs-triton", "autograd-hip-vs-triton"]:
            if name in results:
                r = results[name]
                if r.get("skipped"):
                    f.write(f"| {name} | skipped | skipped |\n")
                else:
                    f.write(f"| {name} | {r['speedup_fwd']:.3f}x | {r['speedup_fwd_bwd']:.3f}x |\n")
            else:
                f.write(f"| {name} | _unavailable_ | _unavailable_ |\n")

        f.write("\n## Ship-gate decision\n\n")
        if "autograd-hip-vs-triton" in results:
            r = results["autograd-hip-vs-triton"]
            if r.get("skipped"):
                f.write("**UNDETERMINED** — bench was skipped.\n\n")
            else:
                speedup = r["speedup_fwd_bwd"]
                if speedup >= 1.05:
                    f.write(f"**PASS** — Triton fused_swiglu fwd+bwd speedup "
                            f"**{speedup:.3f}x** over autograd-safe HIP. "
                            f">= 1.05x threshold.\n\n"
                            f"**Action**: proceed to Phase H Sprint 3A bisect. "
                            f"If any H probe passes stability, enable "
                            f"`--optimize-kernels` on Sprint 3A with Triton path.\n")
                else:
                    f.write(f"**FAIL** — Triton fused_swiglu fwd+bwd speedup "
                            f"**{speedup:.3f}x** over autograd-safe HIP. "
                            f"Below 1.05x threshold.\n\n"
                            f"**Action**: skip Phase H. Sprint 3A ships without "
                            f"`--optimize-kernels`. Triton path has no clear "
                            f"throughput advantage on OdinFlat's SwiGLU block.\n")
        else:
            f.write("**UNDETERMINED** — autograd-hip comparison unavailable.\n")

        f.write("\n## Per-bench detail\n\n")
        for name, r in results.items():
            f.write(f"### {name}\n\n")
            if r.get("skipped"):
                f.write(f"_skipped: {r.get('reason', 'unknown')}_\n\n")
                continue
            f.write(f"- ref_fwd_us: {r['ref_fwd_us']:.1f}\n")
            f.write(f"- triton_fwd_us: {r['triton_fwd_us']:.1f}\n")
            f.write(f"- speedup_fwd: {r['speedup_fwd']:.3f}x\n")
            f.write(f"- ref_fwd_bwd_us: {r['ref_fwd_bwd_us']:.1f}\n")
            f.write(f"- triton_fwd_bwd_us: {r['triton_fwd_bwd_us']:.1f}\n")
            f.write(f"- speedup_fwd_bwd: {r['speedup_fwd_bwd']:.3f}x\n\n")

    print(f"Wrote {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

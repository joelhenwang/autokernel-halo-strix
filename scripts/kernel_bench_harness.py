"""Isolated throughput benchmark harness for Triton kernels.

Times the forward pass, backward pass, and full fwd+bwd cycle of a kernel
in isolation. Used to decide Phase D ship gates:

    result = bench_kernel_fwd_bwd(
        name="fused_swiglu",
        triton_fn=MyFusedSwiGLU.apply,
        reference_fn=lambda gate, up: F.silu(gate) * up,
        shape=(16, 512, 2048),
        dtype=torch.float16,
        input_count=2,
        warmup=50,
        iters=200,
    )
    # result["speedup_fwd"], result["speedup_fwd_bwd"]

Plan: Phase D.A.4 of master remediation plan.
"""

from __future__ import annotations

import argparse
import time
from typing import Callable, Dict, List, Tuple

import torch


def _cuda_sync():
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def _bench(fn_call: Callable, *, warmup: int, iters: int) -> float:
    """Return median seconds per iter over ``iters`` timed runs."""
    for _ in range(warmup):
        fn_call()
    _cuda_sync()
    start = time.perf_counter()
    for _ in range(iters):
        fn_call()
    _cuda_sync()
    return (time.perf_counter() - start) / iters


def bench_kernel_fwd_bwd(
    name: str,
    triton_fn: Callable,
    reference_fn: Callable,
    shape: Tuple[int, ...],
    dtype: torch.dtype = torch.float16,
    input_count: int = 1,
    warmup: int = 50,
    iters: int = 200,
) -> Dict:
    """Benchmark forward-only and forward+backward times for both
    implementations. Returns a dict with per-phase timings."""
    if not torch.cuda.is_available():
        return {"skipped": True, "reason": "CUDA not available"}
    device = "cuda"

    # Forward-only inputs (no grad)
    def _mk_nograd():
        return [torch.randn(*shape, dtype=dtype, device=device) for _ in range(input_count)]

    inputs_fwd = _mk_nograd()

    def call_ref_fwd():
        reference_fn(*inputs_fwd)

    def call_trit_fwd():
        triton_fn(*inputs_fwd)

    t_ref_fwd = _bench(call_ref_fwd, warmup=warmup, iters=iters)
    t_trit_fwd = _bench(call_trit_fwd, warmup=warmup, iters=iters)

    # Forward+backward inputs (requires_grad=True)
    def _mk_grad():
        return [
            torch.randn(*shape, dtype=dtype, device=device, requires_grad=True)
            for _ in range(input_count)
        ]

    def call_ref_fwd_bwd():
        inputs = _mk_grad()
        out = reference_fn(*inputs)
        grad = torch.randn_like(out)
        out.backward(grad)

    def call_trit_fwd_bwd():
        inputs = _mk_grad()
        out = triton_fn(*inputs)
        grad = torch.randn_like(out)
        out.backward(grad)

    t_ref_fb = _bench(call_ref_fwd_bwd, warmup=warmup, iters=iters)
    t_trit_fb = _bench(call_trit_fwd_bwd, warmup=warmup, iters=iters)

    return {
        "name": name,
        "shape": shape,
        "dtype": str(dtype),
        "input_count": input_count,
        "iters": iters,
        "ref_fwd_us": t_ref_fwd * 1e6,
        "triton_fwd_us": t_trit_fwd * 1e6,
        "speedup_fwd": t_ref_fwd / t_trit_fwd,
        "ref_fwd_bwd_us": t_ref_fb * 1e6,
        "triton_fwd_bwd_us": t_trit_fb * 1e6,
        "speedup_fwd_bwd": t_ref_fb / t_trit_fb,
    }


def print_result(result: Dict):
    """Human-readable one-line summary."""
    if result.get("skipped"):
        print(f"[{result.get('name', '?')}] SKIPPED: {result.get('reason', '')}")
        return
    print(f"[{result['name']}] shape={result['shape']} dtype={result['dtype']}")
    print(f"  fwd:     ref={result['ref_fwd_us']:.1f}us  "
          f"triton={result['triton_fwd_us']:.1f}us  "
          f"speedup={result['speedup_fwd']:.2f}x")
    print(f"  fwd+bwd: ref={result['ref_fwd_bwd_us']:.1f}us  "
          f"triton={result['triton_fwd_bwd_us']:.1f}us  "
          f"speedup={result['speedup_fwd_bwd']:.2f}x")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", required=True)
    ap.add_argument("--reference", required=True, help="module.fn")
    ap.add_argument("--triton", required=True)
    ap.add_argument("--shape", required=True, help="'16,512,2048'")
    ap.add_argument("--dtype", choices=["fp16", "bf16"], default="fp16")
    ap.add_argument("--inputs", type=int, default=1)
    ap.add_argument("--warmup", type=int, default=50)
    ap.add_argument("--iters", type=int, default=200)
    args = ap.parse_args()

    import importlib

    def _load(dotted):
        mod, fn = dotted.rsplit(".", 1)
        return getattr(importlib.import_module(mod), fn)

    dt = {"fp16": torch.float16, "bf16": torch.bfloat16}[args.dtype]
    shape = tuple(int(x) for x in args.shape.split(","))
    result = bench_kernel_fwd_bwd(
        name=args.name,
        reference_fn=_load(args.reference),
        triton_fn=_load(args.triton),
        shape=shape,
        dtype=dt,
        input_count=args.inputs,
        warmup=args.warmup,
        iters=args.iters,
    )
    print_result(result)


if __name__ == "__main__":
    main()

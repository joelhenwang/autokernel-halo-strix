"""Parity test harness for Triton kernels.

Given a reference implementation (usually the eager PyTorch equivalent)
and a Triton-backed replacement, runs forward + backward on a panel of
random inputs, asserts max absolute and relative error are within
per-dtype thresholds.

Usage:

    from scripts.kernel_parity_harness import compare_forward_backward

    def eager_swiglu(gate, up):
        return torch.nn.functional.silu(gate) * up

    def triton_swiglu(gate, up):
        return MyFusedSwiGLU.apply(gate, up)

    result = compare_forward_backward(
        name="fused_swiglu",
        reference_fn=eager_swiglu,
        triton_fn=triton_swiglu,
        shapes=[(4, 8, 128), (16, 32, 256)],
        dtypes=[torch.float16],
        input_count=2,
    )
    assert result["all_pass"], result["failures"]

Plan: Phase D.A.3 of master remediation plan.
"""

from __future__ import annotations

import argparse
from typing import Callable, Dict, List, Tuple

import torch

# Per-dtype tolerance bounds. Tight for fp32, looser for fp16/bf16.
DEFAULT_TOL = {
    torch.float32: {"fwd_abs": 1e-5, "fwd_rel": 1e-5,
                     "bwd_abs": 1e-4, "bwd_rel": 1e-4},
    torch.float16: {"fwd_abs": 2e-3, "fwd_rel": 2e-3,
                     "bwd_abs": 5e-3, "bwd_rel": 5e-3},
    torch.bfloat16: {"fwd_abs": 1e-2, "fwd_rel": 1e-2,
                      "bwd_abs": 3e-2, "bwd_rel": 3e-2},
}


def _random_inputs(shapes, dtype, device, requires_grad=True):
    return [
        torch.randn(*shape, dtype=dtype, device=device, requires_grad=requires_grad)
        for shape in shapes
    ]


def _compare_tensors(a: torch.Tensor, b: torch.Tensor, tol: Dict, label: str):
    """Returns (pass, max_abs_err, max_rel_err, msg)."""
    if a.shape != b.shape:
        return False, float("inf"), float("inf"), (
            f"{label} shape mismatch: {tuple(a.shape)} vs {tuple(b.shape)}"
        )
    a_f, b_f = a.detach().float(), b.detach().float()
    abs_err = (a_f - b_f).abs().max().item()
    rel_err = (abs_err / (a_f.abs().max().item() + 1e-12))
    ok_abs = abs_err < tol.get(f"{label}_abs", 1e-3)
    ok_rel = rel_err < tol.get(f"{label}_rel", 1e-3)
    return (ok_abs or ok_rel), abs_err, rel_err, ""


def compare_forward_backward(
    name: str,
    reference_fn: Callable,
    triton_fn: Callable,
    shapes: List[Tuple[int, ...]],
    dtypes: List[torch.dtype] = (torch.float16,),
    input_count: int = 1,
    tol: Dict = None,
    n_trials: int = 3,
    seed: int = 42,
    verbose: bool = True,
) -> Dict:
    """Run forward + backward parity test. Returns dict with results."""
    if not torch.cuda.is_available():
        return {"skipped": True, "reason": "CUDA not available"}

    device = "cuda"
    failures = []
    trials_run = 0

    tol_cache = tol or {}
    for dtype in dtypes:
        dtype_tol = tol_cache.get(dtype, DEFAULT_TOL.get(dtype, DEFAULT_TOL[torch.float16]))
        for shape in shapes:
            shapes_list = [shape] * input_count
            for trial in range(n_trials):
                torch.manual_seed(seed + trial)
                # Reference inputs (requires_grad for backward test)
                ref_inputs = _random_inputs(shapes_list, dtype, device, requires_grad=True)
                trit_inputs = [t.detach().clone().requires_grad_(True) for t in ref_inputs]

                # Forward
                try:
                    ref_out = reference_fn(*ref_inputs)
                    trit_out = triton_fn(*trit_inputs)
                except Exception as exc:  # noqa: BLE001
                    failures.append({
                        "phase": "forward_exec", "dtype": str(dtype), "shape": shape,
                        "trial": trial, "error": str(exc),
                    })
                    continue
                trials_run += 1

                ok, abs_err, rel_err, msg = _compare_tensors(
                    ref_out, trit_out, dtype_tol, "fwd"
                )
                if not ok:
                    failures.append({
                        "phase": "forward", "dtype": str(dtype), "shape": shape,
                        "trial": trial, "abs_err": abs_err, "rel_err": rel_err,
                        "msg": msg,
                    })
                    continue
                if verbose:
                    print(f"  [{name}] fwd {dtype} {shape} trial={trial}: "
                          f"abs={abs_err:.2e} rel={rel_err:.2e} OK")

                # Backward
                try:
                    grad_out = torch.randn_like(ref_out)
                    ref_out.backward(grad_out)
                    trit_out.backward(grad_out)
                except Exception as exc:  # noqa: BLE001
                    failures.append({
                        "phase": "backward_exec", "dtype": str(dtype), "shape": shape,
                        "trial": trial, "error": str(exc),
                    })
                    continue

                for i, (rinp, tinp) in enumerate(zip(ref_inputs, trit_inputs)):
                    if rinp.grad is None and tinp.grad is None:
                        continue
                    if (rinp.grad is None) != (tinp.grad is None):
                        failures.append({
                            "phase": "backward", "dtype": str(dtype), "shape": shape,
                            "trial": trial,
                            "msg": f"input {i}: grad presence mismatch "
                                   f"(ref={rinp.grad is not None}, trit={tinp.grad is not None})",
                        })
                        continue
                    ok, abs_err, rel_err, msg = _compare_tensors(
                        rinp.grad, tinp.grad, dtype_tol, "bwd"
                    )
                    if not ok:
                        failures.append({
                            "phase": "backward", "dtype": str(dtype), "shape": shape,
                            "trial": trial, "input_idx": i,
                            "abs_err": abs_err, "rel_err": rel_err, "msg": msg,
                        })
                    elif verbose:
                        print(f"  [{name}] bwd[{i}] {dtype} {shape} trial={trial}: "
                              f"abs={abs_err:.2e} rel={rel_err:.2e} OK")

    result = {
        "name": name,
        "trials_run": trials_run,
        "failures": failures,
        "all_pass": len(failures) == 0,
    }
    if verbose:
        status = "PASS" if result["all_pass"] else "FAIL"
        print(f"[{name}] {status} — {trials_run} trials, {len(failures)} failures")
    return result


def main():
    ap = argparse.ArgumentParser(
        description="Standalone parity harness runner. Provide two Python "
                    "callables via --reference and --triton (module.fn notation). "
                    "Inputs generated from --shapes."
    )
    ap.add_argument("--name", required=True)
    ap.add_argument("--reference", required=True)
    ap.add_argument("--triton", required=True)
    ap.add_argument("--shapes", nargs="+", required=True, help="e.g. '4,8,128'")
    ap.add_argument("--dtype", choices=["fp16", "bf16", "fp32"], default="fp16")
    ap.add_argument("--inputs", type=int, default=1)
    ap.add_argument("--trials", type=int, default=3)
    args = ap.parse_args()

    import importlib

    def _load(dotted):
        mod, fn = dotted.rsplit(".", 1)
        return getattr(importlib.import_module(mod), fn)

    ref_fn = _load(args.reference)
    trit_fn = _load(args.triton)
    dt = {"fp16": torch.float16, "bf16": torch.bfloat16, "fp32": torch.float32}[args.dtype]
    shapes = [tuple(int(x) for x in s.split(",")) for s in args.shapes]

    result = compare_forward_backward(
        name=args.name,
        reference_fn=ref_fn,
        triton_fn=trit_fn,
        shapes=shapes,
        dtypes=[dt],
        input_count=args.inputs,
        n_trials=args.trials,
    )
    import sys
    sys.exit(0 if result["all_pass"] else 1)


if __name__ == "__main__":
    main()

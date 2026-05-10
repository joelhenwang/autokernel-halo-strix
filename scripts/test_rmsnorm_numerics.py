"""Phase III numeric harness: compare rmsnorm HIP kernel vs reference.

Reproduces the drift observed in Phase II bisect (rmsnorm HIP produces
36x smaller activations on OdinFlat at step 200 vs baseline).

Tests:
  1. Single-invocation accuracy vs reference (ref: torch RMSNorm)
  2. Chained N-invocation drift (simulate OdinFlat depth)
  3. Weight precision sensitivity (fp32 vs fp16 weight input)
  4. Batch size sensitivity (tiny vs realistic)
  5. Rounding bias (is the error zero-mean or biased?)

Run:
    python scripts/test_rmsnorm_numerics.py
"""

from __future__ import annotations

import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402


def rmsnorm_reference(x: torch.Tensor, weight: torch.Tensor,
                      eps: float = 1e-6) -> torch.Tensor:
    """Reference RMSNorm matching models/_components.py::RMSNorm.

    The REFERENCE implementation as defined in OdinFlat's RMSNorm module.
    Under autocast fp16, pow(2) runs in fp16 but mean() is usually
    upcast to fp32. We replicate that here for fairness — the
    baseline pathway our HIP kernel is compared against in training.
    """
    norm = torch.rsqrt(x.float().pow(2).mean(-1, keepdim=True) + eps)
    return (x.float() * norm * weight.float()).to(x.dtype)


def rmsnorm_hip(x: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    """Thin wrapper that calls our HIP kernel path."""
    from kernels.hip.rmsnorm import kernel_fn
    return kernel_fn(x, weight)


def _err_stats(a: torch.Tensor, b: torch.Tensor) -> dict:
    """Statistics comparing a (test) to b (reference)."""
    a_f = a.float()
    b_f = b.float()
    diff = a_f - b_f
    rel = diff / (b_f.abs() + 1e-8)
    return {
        "max_abs_err": diff.abs().max().item(),
        "mean_abs_err": diff.abs().mean().item(),
        "mean_rel_err": rel.abs().mean().item(),
        "max_rel_err": rel.abs().max().item(),
        "signed_mean_rel_err": rel.mean().item(),   # sign of systematic bias
        "ratio_of_norms": a_f.norm().item() / (b_f.norm().item() + 1e-12),
    }


def test_single_invocation():
    """Sanity: kernel matches reference to within fp16 precision on a single call."""
    torch.manual_seed(0)
    B_T = 8192
    D = 768

    x = torch.randn(B_T, D, device="cuda", dtype=torch.float16)
    w = torch.randn(D, device="cuda", dtype=torch.float16) * 0.3 + 1.0  # ~RMSNorm init

    ref = rmsnorm_reference(x, w)
    hip = rmsnorm_hip(x, w)

    e = _err_stats(hip, ref)
    print(f"  Single invocation:")
    print(f"    max_abs_err:   {e['max_abs_err']:.6f}")
    print(f"    mean_abs_err:  {e['mean_abs_err']:.6f}")
    print(f"    max_rel_err:   {e['max_rel_err']:.6f}")
    print(f"    signed_mean_rel_err: {e['signed_mean_rel_err']:+.6f}")
    print(f"    ratio_of_norms (hip/ref): {e['ratio_of_norms']:.6f}")
    return e


def test_chained_drift(n_chain: int = 14, weight_seed: int = 1):
    """Chain rmsnorm N times with realistic residual-scale intermediates.

    Simulates OdinFlat's 14 layers where RMSNorm is applied in each
    layer's pre_norm and ffn_norm. Between norms we inject a residual-
    like perturbation so x doesn't stay trivially at unit norm.

    Returns the norm ratio (hip/ref) at each step so we can see drift.
    """
    torch.manual_seed(weight_seed)
    B_T = 8192
    D = 768
    eps = 1e-6

    # One weight per "layer" — different gammas simulate 14 different
    # RMSNorm modules in OdinFlat.
    weights = [
        torch.randn(D, device="cuda", dtype=torch.float16) * 0.3 + 1.0
        for _ in range(n_chain)
    ]

    # Start with a realistic activation (magnitude ~1.0 like post-embedding)
    x0 = torch.randn(B_T, D, device="cuda", dtype=torch.float16)
    x_ref = x0.clone()
    x_hip = x0.clone()

    ratios = []
    print(f"  Chained drift ({n_chain} sequential rmsnorm applications):")
    print(f"    step   ratio_hip/ref   norm_hip      norm_ref      rel_err")
    for i in range(n_chain):
        x_ref = rmsnorm_reference(x_ref, weights[i], eps=eps)
        x_hip = rmsnorm_hip(x_hip, weights[i])

        e = _err_stats(x_hip, x_ref)
        ratios.append(e["ratio_of_norms"])

        # Residual-add to keep magnitude non-collapsing. Simulates the
        # residual skip around each RMSNorm in a transformer block.
        residual = torch.randn(B_T, D, device="cuda", dtype=torch.float16) * 0.5
        x_ref = x_ref + residual
        x_hip = x_hip + residual

        print(f"    {i:3d}    {e['ratio_of_norms']:.6f}       "
              f"{x_hip.float().norm().item():.3f}       "
              f"{x_ref.float().norm().item():.3f}       "
              f"{e['mean_rel_err']:.6f}")

    return ratios


def test_chained_drift_no_residual(n_chain: int = 14):
    """Chain WITHOUT residual: shows pure RMSNorm behavior.

    After the first application, rmsnorm's output has per-row norm ≈ sqrt(D)
    (unit RMS). Chained applications of rmsnorm-only on unit-RMS input
    should be IDEMPOTENT (to within the weight gammas).
    """
    torch.manual_seed(2)
    B_T = 8192
    D = 768
    eps = 1e-6

    weights = [
        torch.randn(D, device="cuda", dtype=torch.float16) * 0.3 + 1.0
        for _ in range(n_chain)
    ]

    x0 = torch.randn(B_T, D, device="cuda", dtype=torch.float16)
    x_ref = x0.clone()
    x_hip = x0.clone()

    print(f"  Chained drift WITHOUT residual ({n_chain} rmsnorm-only):")
    print(f"    step   ratio_hip/ref   norm_hip      norm_ref")
    for i in range(n_chain):
        x_ref = rmsnorm_reference(x_ref, weights[i], eps=eps)
        x_hip = rmsnorm_hip(x_hip, weights[i])
        e = _err_stats(x_hip, x_ref)
        print(f"    {i:3d}    {e['ratio_of_norms']:.6f}       "
              f"{x_hip.float().norm().item():.3f}       "
              f"{x_ref.float().norm().item():.3f}")


def test_weight_dtype():
    """Does fp32 weight vs fp16 weight matter?"""
    torch.manual_seed(3)
    B_T = 4096
    D = 768

    x = torch.randn(B_T, D, device="cuda", dtype=torch.float16)
    w_fp32 = torch.randn(D, device="cuda", dtype=torch.float32) * 0.3 + 1.0
    w_fp16 = w_fp32.half()

    hip_from_fp32 = rmsnorm_hip(x, w_fp32)  # kernel casts internally
    hip_from_fp16 = rmsnorm_hip(x, w_fp16)
    ref = rmsnorm_reference(x, w_fp32)

    print(f"  Weight dtype sensitivity:")
    print(f"    HIP(fp32w) vs ref: {_err_stats(hip_from_fp32, ref)['mean_rel_err']:.6f}")
    print(f"    HIP(fp16w) vs ref: {_err_stats(hip_from_fp16, ref)['mean_rel_err']:.6f}")
    print(f"    HIP(fp32w) vs HIP(fp16w): "
          f"{_err_stats(hip_from_fp32, hip_from_fp16)['mean_rel_err']:.6f}")


def test_different_shapes():
    """Verify correctness across D sizes, batch sizes."""
    print(f"  Shape sensitivity:")
    print(f"    shape        ratio_hip/ref   mean_rel_err")
    torch.manual_seed(7)
    for B_T, D in [(64, 128), (64, 768), (1024, 768), (8192, 768), (8192, 1024)]:
        x = torch.randn(B_T, D, device="cuda", dtype=torch.float16)
        w = torch.ones(D, device="cuda", dtype=torch.float16)
        ref = rmsnorm_reference(x, w)
        hip = rmsnorm_hip(x, w)
        e = _err_stats(hip, ref)
        print(f"    ({B_T:>5d}, {D:>5d})   {e['ratio_of_norms']:.6f}       "
              f"{e['mean_rel_err']:.6f}")


def main() -> int:
    if not torch.cuda.is_available():
        print("Need CUDA"); return 1

    print("=" * 70)
    print("Phase III numeric harness: rmsnorm HIP vs reference")
    print("=" * 70)

    print("\n[1] Single invocation")
    test_single_invocation()

    print("\n[2] Chained drift (with residual) — simulates OdinFlat depth")
    test_chained_drift(n_chain=14)

    print("\n[3] Chained drift (rmsnorm-only, no residual)")
    test_chained_drift_no_residual(n_chain=14)

    print("\n[4] Weight dtype sensitivity")
    test_weight_dtype()

    print("\n[5] Shape sensitivity")
    test_different_shapes()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

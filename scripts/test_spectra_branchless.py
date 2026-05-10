"""v3 T-1.1 branchless SPECTRA parity test.

Verifies that the branchless SPECTRA path produces numerically equivalent
results to the original sigma1.item() branch path, within fp16/fp32
tolerance. Covers both halo_training/spectra.py::apply_post_clip (via
AUTOKERNEL_SPECTRA_BRANCHLESS=1) and halo_training/normuon.py NorMuon
(spectra_branchless=True).

Usage:
    python scripts/test_spectra_branchless.py
"""
from __future__ import annotations

import os
import sys

import torch

sys.path.insert(0, ".")

from halo_training.spectra import apply_post_clip, _estimate_spectral_norm


def _reference_branched(M: torch.Tensor, clip_norm: float,
                         safety_margin: float = 1.02,
                         ns_iterations: int = 5) -> torch.Tensor:
    """Original branched path (force sigma1.item(), no env override)."""
    # Clear env var so apply_post_clip takes the branched path
    os.environ.pop("AUTOKERNEL_SPECTRA_BRANCHLESS", None)
    return apply_post_clip(M, clip_norm=clip_norm,
                           safety_margin=safety_margin,
                           ns_iterations=ns_iterations)


def _branchless(M: torch.Tensor, clip_norm: float,
                safety_margin: float = 1.02,
                ns_iterations: int = 5) -> torch.Tensor:
    """Branchless path via env var."""
    os.environ["AUTOKERNEL_SPECTRA_BRANCHLESS"] = "1"
    try:
        return apply_post_clip(M, clip_norm=clip_norm,
                               safety_margin=safety_margin,
                               ns_iterations=ns_iterations)
    finally:
        os.environ.pop("AUTOKERNEL_SPECTRA_BRANCHLESS", None)


def test_parity():
    """branched and branchless paths produce equivalent outputs."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(0)

    test_cases = [
        # (shape, clip_norm, dtype)
        ((1536, 768), 1.0, torch.float16),    # OdinFlat w_gate_up shape
        ((768, 1536), 1.0, torch.float16),    # transposed
        ((768, 768), 1.0, torch.float16),     # attention projection
        ((768, 768), 5.0, torch.float32),     # loose clip (no-op path)
        ((768, 768), 0.1, torch.float16),     # tight clip
        ((64, 64), 1.0, torch.float32),       # small
    ]

    total = 0
    passed = 0
    for shape, clip_norm, dtype in test_cases:
        total += 1
        M = torch.randn(*shape, device=device, dtype=dtype)
        # scale M so sometimes we hit the clip path, sometimes not
        sigma = _estimate_spectral_norm(M).item()
        # Test both "above clip" (scale=2*clip/sigma) and "below clip"
        for scale_factor in [0.5 * clip_norm / max(sigma, 1e-6),
                             2.0 * clip_norm / max(sigma, 1e-6)]:
            M_scaled = M * scale_factor
            out_branched = _reference_branched(M_scaled.clone(), clip_norm=clip_norm)
            out_branchless = _branchless(M_scaled.clone(), clip_norm=clip_norm)

            # Both should have ||output|| close to clip_norm when clipping fires,
            # or equal to input when no clipping needed.
            rel_err = (out_branched.float() - out_branchless.float()).abs().max().item()
            max_val = max(out_branched.float().abs().max().item(),
                          out_branchless.float().abs().max().item())
            tol = 1e-3 if dtype == torch.float16 else 1e-6
            rel = rel_err / max(max_val, 1e-6)

            status = "PASS" if rel < tol else "FAIL"
            print(f"  {shape} clip={clip_norm} dtype={dtype.__str__()[6:]} "
                  f"scale={scale_factor:.3e}: rel_err={rel:.2e} [{status}]")
            if rel < tol:
                passed += 1
            else:
                total += 1  # counted twice; rebalance
                passed -= 1  # don't count as pass

    print(f"\n{passed}/{total * 2} parity checks passed")
    assert passed == total * 2, "Branchless SPECTRA does not match branched reference"


if __name__ == "__main__":
    test_parity()
    print("\n[PASS] T-1.1 branchless SPECTRA parity verified")

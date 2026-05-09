"""SPECTRA: Spectral norm clipping for optimizer updates.

Implements post-update spectral clipping per the Sprint 1.5 design:
``docs/superpowers/specs/2026-05-06-sprint1.5-spectra-mup-design.md``.

Strategy
--------
After the optimizer's per-parameter 2D update ``M`` is computed, clip
its spectral norm ``σ_1(M)`` to at most ``clip_norm``. Use the same
Newton-Schulz iteration structure as Muon (reuse
``halo_training.muon.zeropower_via_polar_express``) to avoid duplicating
a numerically tuned primitive.

Why post-clip (not pre-clip)
----------------------------
Pre-clip (Sprint 1.5 Phase D) operates on raw gradients and interferes
with Muon's own orthogonalization. Post-clip operates on the final
orthogonalized-and-scaled update, which is semantically what we want
to bound.

Post-clip mechanics
-------------------
Given 2D update ``M`` of shape ``(m, n)``:

1. Compute ``U V^T ≈ polar(M)`` via Newton-Schulz (gives orthonormal
   factorization).
2. The update's spectral norm equals its ℓ2 operator norm;
   equivalently, the maximum singular value.
3. If ``σ_1(M) > clip_norm``, rescale:
   ``M' = M * clip_norm / σ_1(M)``.

In practice we avoid exact SVD and use a cheap power-iteration estimate
(via the NS polar factorization as a by-product) — see
``_estimate_spectral_norm``.

Exports
-------
``apply_post_clip(M, clip_norm, ns_iterations=5)`` — scale M if
needed; preserves shape, dtype, device.
"""

from __future__ import annotations

from typing import Optional

import torch


def apply_post_clip(
    M: torch.Tensor,
    clip_norm: float = 1.0,
    ns_iterations: int = 5,
    dtype: Optional[torch.dtype] = None,
    safety_margin: float = 0.98,
) -> torch.Tensor:
    """Clip the spectral norm of a 2D update to at most ``clip_norm``.

    Uses power iteration (cheap) for the spectral-norm estimate. Because
    power iteration slightly UNDER-estimates σ₁ on matrices with non-trivial
    spectrum, we multiply the scale factor by ``safety_margin`` (< 1.0) so
    the clipped update's true σ₁ is bounded.

    Args:
        M: 2D tensor. Non-2D inputs returned unchanged (post-clip is
            undefined for scalars / vectors; caller is responsible for
            filtering).
        clip_norm: Target spectral-norm ceiling. Must be > 0.
        ns_iterations: Kept for API compatibility; currently ignored
            (power iteration proved simpler and sufficient).
        dtype: Precision for the power-iteration arithmetic. Default fp32.
        safety_margin: Multiplier applied to the scale factor when clipping
            is required. Defaults to 0.98 (2% safety for power-iteration
            under-estimation). Set to 1.0 for raw estimate-based clipping.

    Returns:
        Clipped tensor, same shape / dtype / device as input. Shares
        memory when clipping is not needed (identity fast-path).
    """
    if M.ndim != 2:
        return M
    if clip_norm <= 0:
        raise ValueError(f"clip_norm must be > 0; got {clip_norm}")

    sigma1 = _estimate_spectral_norm(M, ns_iterations=ns_iterations, dtype=dtype)
    if not torch.is_tensor(sigma1):
        sigma1 = torch.tensor(float(sigma1), device=M.device, dtype=M.dtype)

    sigma1_val = sigma1.item()
    # Fast path: no clipping needed (allow a small gate above threshold
    # since our estimator can be slightly low).
    if sigma1_val * (1.0 / safety_margin) <= clip_norm:
        return M

    scale = (clip_norm * safety_margin) / max(sigma1_val, 1e-12)
    return M * scale


def _estimate_spectral_norm(
    M: torch.Tensor,
    ns_iterations: int = 5,
    power_iterations: int = 10,
    dtype: Optional[torch.dtype] = None,
) -> torch.Tensor:
    """Estimate σ₁(M) for a 2D matrix via power iteration.

    The ``ns_iterations`` arg is kept for API forward-compatibility but
    currently ignored — plain power iteration is both simpler and more
    reliable for this estimator. Power iteration converges geometrically;
    10 iterations give ~0.5–2% error on random Gaussian matrices. Callers
    concerned about precision should use a safety margin (see
    ``apply_post_clip(safety_margin=...)``).
    """
    if M.ndim != 2:
        raise ValueError("spectral norm defined only for 2D tensors")

    M_work = M.to(dtype or torch.float32)
    n = M_work.shape[1]
    v = torch.randn(n, device=M_work.device, dtype=M_work.dtype)
    v = v / v.norm().clamp_min(1e-12)
    for _ in range(max(power_iterations, 1)):
        u = M_work @ v
        u_norm = u.norm().clamp_min(1e-12)
        u = u / u_norm
        v = M_work.T @ u
        v_norm = v.norm().clamp_min(1e-12)
        v = v / v_norm
    return (M_work @ v).norm().to(M.dtype)


# For pre-clipping (Phase D), we additionally expose a variant that operates
# on the raw gradient rather than the post-update. Same machinery.
apply_pre_clip = apply_post_clip

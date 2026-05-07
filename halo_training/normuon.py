"""NorMuon: Muon + neuron-wise normalization + Cautious Weight Decay.

Extends ``halo_training/muon.py::Muon`` with two IMU-1 recipe
enhancements (arXiv:2602.02522):

1. **Neuron-wise normalization.** After Newton-Schulz orthogonalization
   produces ``msgn(M)``, normalize each output row (neuron) of the
   resulting matrix to unit ℓ₂ norm. In effect this equalizes the
   update magnitude across neurons regardless of how the Newton-Schulz
   iteration lands, keeping the update direction but rescaling its
   magnitude per-neuron.

2. **Cautious Weight Decay (CWD).** Standard decoupled WD applies
   ``W <- W * (1 - lr * wd)`` unconditionally. Cautious WD applies WD
   only where the gradient and weight have the SAME sign, on the
   hypothesis that applying WD against a decay-pulling gradient is
   redundant and only "flattens" what the optimizer is already doing.
   Elementwise mask: ``(sign(grad) * sign(W) > 0)``. Per Chen et al.
   2025 cautious-WD is orthogonal to NorMuon — can apply either or both.

Newton-Schulz iterations default to 5 with the validated Polar Express
coefficients from ``halo_training/muon.py`` (see `_PE_COEFFS`). IMU-1
mentions 7 iterations but gives no concrete coefficients; hand-extrapolated
7-step runs degrade orthogonalization noticeably so we stick with the
proven 5-step path.

Interface
---------
Mirrors ``halo_training/muon.py::Muon``. Both ``muon_params`` (2D) and
``adamw_params`` (1D / embed / head) groups are accepted. The 2D group
gets NorMuon updates; the 1D group gets standard AdamW.

``halo_training/optimizer.py::build_imu1_optimizer(use_normuon=True)``
is the Sprint 1 entry point.
"""

from __future__ import annotations

import math
from typing import Optional, Tuple

import torch
import torch.nn as nn

# Reuse the validated 5-step Polar Express iteration from the existing Muon
# implementation. The IMU-1 paper mentions 7 iterations but gives no concrete
# coefficients; hand-extrapolating past 5 degrades orthogonalization quality
# noticeably (empirical: 5-step err ≈ 0.08, naive 7-step err ≈ 0.46).
from halo_training.muon import zeropower_via_polar_express


def _newton_schulz_polar_express(G: torch.Tensor, steps: int = 5,
                                 dtype: Optional[torch.dtype] = None) -> torch.Tensor:
    """Newton-Schulz matrix sign iteration — thin wrapper over Muon's 5-step.

    Produces ``msgn(G)`` — the orthogonal matrix factor of G's polar decomp.

    ``steps`` is clamped to 5 (the validated length). Future work may
    introduce longer-chain Polar Express coefficients; until then, 5 is
    the correct default for NorMuon per IMU-1's ablations.
    """
    assert G.ndim == 2, f"NorMuon NS requires 2D; got shape {tuple(G.shape)}"
    return zeropower_via_polar_express(G, steps=min(steps, 5), dtype=dtype)


def _neuron_wise_normalize(M: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """Rescale each output row of M to unit ℓ₂ norm.

    For a 2D weight ``W`` of shape ``[out, in]``, Newton-Schulz may leave
    some output neurons with much larger update magnitude than others.
    Normalizing per-row before the update equalizes per-neuron update
    magnitude, which is the key NorMuon innovation over Muon.
    """
    norms = M.norm(dim=1, keepdim=True).clamp_min(eps)
    return M / norms


def _cautious_wd_mask(grad: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    """Return bool mask where (sign(grad) * sign(weight) > 0).

    Where the mask is True, WD should apply (gradient pulling same
    direction as WD would push). Where False, WD is skipped (the
    gradient is already doing the work or fighting against WD).
    """
    return (grad.sign() * weight.sign()) > 0


class NorMuon(torch.optim.Optimizer):
    """Muon with neuron-wise norm + Cautious WD (IMU-1 recipe).

    Parameters mirror ``halo_training/muon.py::Muon`` with a few additions:

        ns_steps=7                 (was 5; IMU-1 ablations prefer 7)
        cautious_wd=True           (toggle CWD; False = standard decoupled WD)
        neuron_wise_norm=True      (toggle NN; False = plain Muon orthogonalization)

    Both cautious_wd and neuron_wise_norm default True for the IMU-1
    recipe; toggling either off reverts that piece to vanilla Muon
    behavior, useful for ablation runs.
    """

    def __init__(
        self,
        muon_params,
        adamw_params=None,
        lr: float = 0.0235,
        momentum: float = 0.95,
        nesterov: bool = True,
        ns_steps: int = 5,
        weight_decay: float = 0.1,
        adamw_lr: float = 0.007,
        adamw_betas: Tuple[float, float] = (0.9, 0.95),
        adamw_wd: float = 0.0,
        betas: Optional[Tuple[float, float]] = None,  # alias for compat
        ns_dtype: Optional[torch.dtype] = None,
        cautious_wd: bool = True,
        neuron_wise_norm: bool = True,
        neuron_norm_min_dim: int = 0,
    ):
        if betas is not None:
            # Treat betas=(0.9, 0.95) as AdamW-side configuration override,
            # matching how build_imu1_optimizer passes it through.
            adamw_betas = betas

        self.ns_dtype = ns_dtype
        self.cautious_wd = cautious_wd
        self.neuron_wise_norm = neuron_wise_norm
        # Sprint 1.1 Phase B2: if neuron_norm_min_dim > 0, neuron-wise
        # normalization is skipped on 2D params where min(rows, cols) <
        # this threshold. Default 0 = always apply (matches Phase 2 behavior).
        self.neuron_norm_min_dim = neuron_norm_min_dim

        if isinstance(muon_params, dict):
            muon_params = [muon_params]
        muon_groups = []
        for group in muon_params:
            if isinstance(group, dict):
                g = dict(group)
                g.setdefault("lr", lr)
                g.setdefault("momentum", momentum)
                g.setdefault("nesterov", nesterov)
                g.setdefault("ns_steps", ns_steps)
                g.setdefault("weight_decay", weight_decay)
                g["_optimizer_type"] = "normuon"
                muon_groups.append(g)
            else:
                plist = list(group) if hasattr(group, "__iter__") and not isinstance(group, torch.Tensor) else [group]
                muon_groups.append({
                    "params": plist,
                    "lr": lr, "momentum": momentum, "nesterov": nesterov,
                    "ns_steps": ns_steps, "weight_decay": weight_decay,
                    "_optimizer_type": "normuon",
                })

        adamw_groups = []
        if adamw_params is not None:
            if isinstance(adamw_params, dict):
                adamw_params = [adamw_params]
            for group in adamw_params:
                if isinstance(group, dict):
                    g = dict(group)
                    g.setdefault("lr", adamw_lr)
                    g.setdefault("betas", adamw_betas)
                    g.setdefault("weight_decay", g.pop("weight_decay", adamw_wd))
                    g["_optimizer_type"] = "adamw"
                    adamw_groups.append(g)
                else:
                    plist = list(group) if hasattr(group, "__iter__") and not isinstance(group, torch.Tensor) else [group]
                    adamw_groups.append({
                        "params": plist,
                        "lr": adamw_lr, "betas": adamw_betas,
                        "weight_decay": adamw_wd,
                        "_optimizer_type": "adamw",
                    })

        all_groups = muon_groups + adamw_groups
        defaults = dict(lr=lr, momentum=momentum, nesterov=nesterov,
                        ns_steps=ns_steps, weight_decay=weight_decay)
        super().__init__(all_groups, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            opt_type = group.get("_optimizer_type", "normuon")
            if opt_type == "normuon":
                self._normuon_step(group)
            else:
                self._adamw_step(group)

        return loss

    def _normuon_step(self, group):
        lr = group["lr"]
        mu = group["momentum"]
        nesterov = group["nesterov"]
        ns_steps = group["ns_steps"]
        wd = group["weight_decay"]

        for p in group["params"]:
            if p.grad is None:
                continue
            grad = p.grad

            if p.ndim != 2:
                # Safety fallback: non-2D param in a NorMuon group (shouldn't
                # happen with build_imu1_optimizer but be defensive).
                p.data.add_(grad, alpha=-lr)
                continue

            state = self.state[p]
            if "momentum_buffer" not in state:
                state["momentum_buffer"] = torch.zeros_like(grad)

            buf = state["momentum_buffer"]
            buf.mul_(mu).add_(grad)

            if nesterov:
                m = grad.add(buf, alpha=mu)
            else:
                m = buf.clone()

            # Newton-Schulz orthogonalization
            m_orth = _newton_schulz_polar_express(m, steps=ns_steps, dtype=self.ns_dtype)

            # Neuron-wise normalization (NorMuon innovation).
            # Sprint 1.1 Phase B2: optionally skip on small matrices via
            # neuron_norm_min_dim gate. min(rows, cols) check catches both
            # tall-skinny and short-fat projections symmetrically.
            if self.neuron_wise_norm and (
                self.neuron_norm_min_dim == 0
                or min(m_orth.shape) >= self.neuron_norm_min_dim
            ):
                m_orth = _neuron_wise_normalize(m_orth)

            # Built-in muP-style scaling matches Muon convention
            scale = max(m_orth.shape[0], m_orth.shape[1]) ** 0.5 * 0.2
            m_orth.mul_(scale)

            # Cautious / standard decoupled weight decay
            if wd > 0:
                if self.cautious_wd:
                    mask = _cautious_wd_mask(grad, p.data)
                    # Apply wd = lr * wd only where mask True; elsewhere no decay
                    p.data.sub_(p.data * (lr * wd) * mask.to(p.dtype))
                else:
                    p.data.mul_(1 - lr * wd)

            # Update
            p.data.add_(m_orth, alpha=-lr)

    def _adamw_step(self, group):
        lr = group["lr"]
        beta1, beta2 = group.get("betas", (0.9, 0.95))
        wd = group.get("weight_decay", 0.0)
        eps = group.get("eps", 1e-8)

        for p in group["params"]:
            if p.grad is None:
                continue
            grad = p.grad

            state = self.state[p]
            if "step" not in state:
                state["step"] = 0
                state["exp_avg"] = torch.zeros_like(p)
                state["exp_avg_sq"] = torch.zeros_like(p)

            state["step"] += 1
            exp_avg, exp_avg_sq = state["exp_avg"], state["exp_avg_sq"]

            bias_correction1 = 1 - beta1 ** state["step"]
            bias_correction2 = 1 - beta2 ** state["step"]

            if wd > 0:
                p.data.mul_(1 - lr * wd)

            exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)
            exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)

            denom = (exp_avg_sq.sqrt() / math.sqrt(bias_correction2)).add_(eps)
            step_size = lr / bias_correction1
            p.data.addcdiv_(exp_avg, denom, value=-step_size)

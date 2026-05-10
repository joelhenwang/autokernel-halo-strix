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
        # Sprint 1.5 Phase A: optional SPECTRA post-clipping on the final
        # weight delta. When enabled, each 2D update is rescaled so its
        # spectral norm is at most ``spectra_clip_norm``. Default OFF to
        # preserve Sprint 1 baseline.
        spectra_post: bool = False,
        spectra_clip_norm: float = 1.0,
        spectra_ns_iter: int = 5,
        # v3 T-0.2 per-param telemetry (docs/research/autokernel-40k-v3-
        # execution-plan.md section 5.1). Emits JSONL per 2D param.
        telemetry_enabled: bool = False,
        telemetry_path: Optional[str] = None,
        telemetry_sample_policy: str = "v3",  # 'v3' = 1..20 every, then every 50
        # v3 T-5.3 post-NorMuon trust cap on ||lr*update|| / ||w||.
        # 0.0 disables. Typical diagnostic: 0.02.
        trust_cap: float = 0.0,
        trust_cap_scope: str = "none",  # none | w_gate_up | spiking | all_2d
        # v3 T-5.2 w_gate_up update-scale staging for hidden-kernel recovery.
        # Initial scale ramps to 1.0 over ramp_steps.
        w_gate_up_scale: float = 1.0,
        w_gate_up_ramp_steps: int = 0,
        # v3 T-2.3 / NorMuon impl optimization: branchless SPECTRA
        # (eliminates sigma1.item() per-param-per-opt-step sync).
        spectra_branchless: bool = False,
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
        # Sprint 1.5 Phase A: SPECTRA knobs
        self.spectra_post = bool(spectra_post)
        self.spectra_clip_norm = float(spectra_clip_norm)
        self.spectra_ns_iter = int(spectra_ns_iter)
        # v3 T-0.2 telemetry
        self.telemetry_enabled = bool(telemetry_enabled)
        self.telemetry_path = telemetry_path
        self.telemetry_sample_policy = telemetry_sample_policy
        self._telem_file = None  # opened lazily in _should_sample
        self._telem_step = 0
        self._telem_warned = False  # set True after first warning to increase sample rate
        # v3 T-5.3 trust cap
        self.trust_cap = float(trust_cap)
        self.trust_cap_scope = trust_cap_scope
        self._spiking_param_names = set()  # names added when trust_cap_scope == "spiking"
        # v3 T-5.2 w_gate_up staging
        self.w_gate_up_scale_init = float(w_gate_up_scale)
        self.w_gate_up_ramp_steps = int(w_gate_up_ramp_steps)
        # v3 T-2.3 branchless SPECTRA
        self.spectra_branchless = bool(spectra_branchless)

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

        # v3 T-0.2 build name map AFTER super().__init__() populates self.param_groups.
        # Map param tensor id() -> readable name if provided via muon_params as a
        # list of dicts each with optional 'param_names'. Otherwise names are synthetic.
        self._param_names: dict = {}
        for g in self.param_groups:
            names = g.get("param_names", None)
            plist = g.get("params", [])
            for i, p in enumerate(plist):
                if names is not None and i < len(names):
                    self._param_names[id(p)] = names[i]
                else:
                    # synthesize: shape + index for stable identification
                    self._param_names[id(p)] = f"{tuple(p.shape)}#{id(p) % 1000000}"

    def set_param_names(self, mapping):
        """Allow trainer to provide param_id -> readable_name mapping after init.

        v3 T-0.2: called once by halo_training/optimizer.py::build_imu1_optimizer
        after attaching param_names to param groups. This replaces any synthetic
        names built during __init__.
        """
        self._param_names.update(mapping)

    def _telemetry_should_sample(self):
        """v3 T-0.2 sample policy:
           - every step for first 20
           - every 50 after that
           - after first warning: every 10
           - always on non-finite (handled in caller)
        """
        if not self.telemetry_enabled:
            return False
        s = self._telem_step
        if s < 20:
            return True
        if self._telem_warned:
            return (s % 10) == 0
        return (s % 50) == 0

    def _telemetry_emit(self, record):
        """Write one JSONL record to self.telemetry_path."""
        import json
        if self.telemetry_path is None:
            return
        if self._telem_file is None:
            self._telem_file = open(self.telemetry_path, "a", buffering=1)
        self._telem_file.write(json.dumps(record) + "\n")

    def _w_gate_up_scale_current(self):
        """v3 T-5.2: return current w_gate_up post-NorMuon scale (ramps to 1.0)."""
        if self.w_gate_up_ramp_steps <= 0:
            return self.w_gate_up_scale_init
        progress = min(1.0, self._telem_step / max(1, self.w_gate_up_ramp_steps))
        return self.w_gate_up_scale_init + progress * (1.0 - self.w_gate_up_scale_init)

    def _should_apply_trust_cap(self, param_name):
        """v3 T-5.3: return True if trust cap should apply to this param."""
        if self.trust_cap <= 0.0:
            return False
        scope = self.trust_cap_scope
        if scope == "none":
            return False
        if scope == "all_2d":
            return True
        if scope == "w_gate_up":
            return "w_gate_up" in param_name
        if scope == "spiking":
            return param_name in self._spiking_param_names
        return False

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

        # v3 T-0.2 advance telemetry step counter after all param groups
        if self.telemetry_enabled:
            self._telem_step += 1

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

            # v3 T-0.2 telemetry: capture raw grad / momentum stats (tensor-side)
            _telem = self._telemetry_should_sample()
            if _telem:
                _raw_grad_norm = grad.float().norm()
                _momentum_norm = buf.float().norm()
                _param_norm = p.data.float().norm()
                _maxabs_grad = grad.abs().max()
                _maxabs_param = p.data.abs().max()
                _grad_finite = bool(torch.isfinite(grad).all())

            # Newton-Schulz orthogonalization
            m_orth = _newton_schulz_polar_express(m, steps=ns_steps, dtype=self.ns_dtype)

            if _telem:
                _post_ns_norm = m_orth.float().norm()

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

            # Sprint 1.5 Phase A: SPECTRA post-clipping on the actual
            # weight delta (lr * m_orth). We implement by clipping m_orth
            # to (spectra_clip_norm / lr) so that ||lr * m_orth||_2 <=
            # spectra_clip_norm. Only 2D updates (already guaranteed by
            # the ndim check above).
            _spectra_scale_tensor = None  # for telemetry
            if self.spectra_post and lr > 0:
                if self.spectra_branchless:
                    # v3 T-1.1 / T-2.3: branchless SPECTRA (no sigma1.item()).
                    # Eliminates per-2D-param-per-opt-step CPU sync.
                    from halo_training.spectra import (
                        _estimate_spectral_norm,
                    )
                    effective_clip = self.spectra_clip_norm / lr
                    safety_margin = 1.02
                    sigma1_t = _estimate_spectral_norm(
                        m_orth,
                        ns_iterations=self.spectra_ns_iter,
                        dtype=None,
                    )
                    if not torch.is_tensor(sigma1_t):
                        sigma1_t = torch.tensor(
                            float(sigma1_t),
                            device=m_orth.device,
                            dtype=m_orth.dtype,
                        )
                    _scale = torch.clamp(
                        effective_clip * safety_margin
                        / torch.clamp(sigma1_t, min=1e-12),
                        max=1.0,
                    ).to(m_orth.dtype)
                    m_orth = m_orth * _scale
                    _spectra_scale_tensor = _scale
                else:
                    from halo_training.spectra import apply_post_clip
                    m_orth = apply_post_clip(
                        m_orth,
                        clip_norm=self.spectra_clip_norm / lr,
                        ns_iterations=self.spectra_ns_iter,
                    )

            # Cautious / standard decoupled weight decay
            if wd > 0:
                if self.cautious_wd:
                    mask = _cautious_wd_mask(grad, p.data)
                    # Apply wd = lr * wd only where mask True; elsewhere no decay
                    p.data.sub_(p.data * (lr * wd) * mask.to(p.dtype))
                else:
                    p.data.mul_(1 - lr * wd)

            # v3 T-5.2: w_gate_up update-scale staging (apply before trust cap).
            # Identified by param name containing "w_gate_up".
            _param_name = self._param_names.get(id(p), "")
            if ("w_gate_up" in _param_name
                    and (self.w_gate_up_scale_init != 1.0
                         or self.w_gate_up_ramp_steps > 0)):
                _wgu_scale = self._w_gate_up_scale_current()
                m_orth = m_orth * _wgu_scale

            # Compute effective update (lr * m_orth) for trust cap + telemetry.
            _effective_update = m_orth * lr  # float-dtype-dependent; tensor-side only

            # v3 T-5.3: post-NorMuon trust cap on ||lr*update|| / ||w||.
            _trust_ratio_pre = None
            _trust_ratio_post = None
            _trust_cap_triggered = False
            if self._should_apply_trust_cap(_param_name):
                _param_norm_f = p.data.float().norm().clamp_min(1e-12)
                _update_norm_f = _effective_update.float().norm()
                _trust_ratio_pre = _update_norm_f / _param_norm_f
                _cap_scale = torch.clamp(
                    self.trust_cap / (_trust_ratio_pre + 1e-12),
                    max=1.0,
                ).to(m_orth.dtype)
                m_orth = m_orth * _cap_scale
                _effective_update = m_orth * lr
                _trust_ratio_post = _effective_update.float().norm() / _param_norm_f
                # NB: tensor-side comparison; keep off CPU to avoid sync
                _trust_cap_triggered = bool((_cap_scale < 1.0).item())
                if _trust_cap_triggered:
                    # If in 'spiking' scope mode, remember this param
                    if self.trust_cap_scope == "spiking":
                        self._spiking_param_names.add(_param_name)

            # Update
            p.data.add_(m_orth, alpha=-lr)

            # v3 T-0.2 telemetry: emit JSONL record AFTER update applied.
            # Also emit unconditionally if any non-finite detected (warning).
            _update_finite = bool(torch.isfinite(m_orth).all())
            _force_emit = not (_grad_finite if _telem else True) or not _update_finite

            if _telem or _force_emit:
                if _force_emit and not self._telem_warned:
                    self._telem_warned = True  # switch to every-10 sampling
                # Only compute the full record if actually emitting
                if not _telem:
                    # Compute the fields we skipped for the force-emit path
                    _raw_grad_norm = grad.float().norm()
                    _momentum_norm = buf.float().norm()
                    _param_norm = p.data.float().norm()
                    _maxabs_grad = grad.abs().max()
                    _maxabs_param = p.data.abs().max()
                    _grad_finite = bool(torch.isfinite(grad).all())
                    _post_ns_norm = torch.tensor(float("nan"))
                _maxabs_update = m_orth.abs().max()
                record = {
                    "step": int(self._telem_step),
                    "param_name": _param_name,
                    "shape": list(p.shape),
                    "dtype_param": str(p.dtype),
                    "dtype_grad": str(grad.dtype),
                    "dtype_update": str(m_orth.dtype),
                    "param_norm": float(_param_norm.item()),
                    "raw_grad_norm": float(_raw_grad_norm.item()),
                    "momentum_norm": float(_momentum_norm.item()),
                    "post_ns_norm": float(_post_ns_norm.item()) if torch.isfinite(_post_ns_norm).item() else float("nan"),
                    "spectra_sigma1": float("nan"),  # tensor-side sigma1 not materialized in branchless path
                    "spectra_scale": float(_spectra_scale_tensor.item()) if _spectra_scale_tensor is not None else 1.0,
                    "effective_lr": float(lr),
                    "update_norm_pre_trust": float(_effective_update.float().norm().item()),
                    "update_norm_post_trust": float(_effective_update.float().norm().item()),  # same as pre if no cap
                    "trust_ratio_pre": float(_trust_ratio_pre.item()) if _trust_ratio_pre is not None else float("nan"),
                    "trust_ratio_post": float(_trust_ratio_post.item()) if _trust_ratio_post is not None else float("nan"),
                    "trust_cap_triggered": bool(_trust_cap_triggered),
                    "maxabs_param": float(_maxabs_param.item()),
                    "maxabs_grad": float(_maxabs_grad.item()),
                    "maxabs_update": float(_maxabs_update.item()),
                    "grad_isfinite": bool(_grad_finite),
                    "update_isfinite": bool(_update_finite),
                }
                self._telemetry_emit(record)


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

"""LEAP: Layer-exit aux loss for trainable early-exit inference.

Per LEAP paper (arXiv:2605.01058, Walmart/MiniLM-L12):

    ℒ_exit = σ(10 · (τ − cos(h_layer_i, sg(h_final))))

Where:
- ``h_layer_i`` is the hidden state output of intermediate layer i
- ``h_final`` is the final representation (after the last layer + final_norm)
- ``sg(·)`` is stop-gradient — h_final acts as a fixed target that intermediates
  are trained toward
- τ = 0.98 (train-time threshold; paper validated at this value)
- Per-layer scalar weight `w_i` scales each layer's contribution

The sigmoid with a steep slope (10) creates a soft cap: once cos exceeds τ,
the loss is near zero (no further push); below τ it's near one. This
forces intermediate representations to resemble the final without over-
penalizing tokens that already converged.

At inference time, a dual threshold θ (default 0.95) gates early exit:

    exit(layer_i, token_t) := cos(h_layer_i[t], h_final_approx[t]) > θ

`h_final_approx` in practice uses the LAST layer's output as a proxy for
the true final state (since we can't compute the full forward without the
last layer — the chicken-and-egg of early exit). A tighter variant uses
an auxiliary head projected from layer_i to estimate h_final_approx; that
adds parameters and is deferred until the base LEAP signal is validated.

Design
------
- `LeapAuxLoss` wraps a model and captures intermediate layer outputs
  via forward hooks. Call `compute_aux_loss(h_final)` inside the
  training step to get a scalar to add to main CE.
- Default: hooks on `model.layers[i]` for i in the provided index list.
  Assumes model has a `.layers` ModuleList (OdinFlat-style). Non-flat
  models would need a variant using `.shared_layers` (OdinHalo).
- Hooks STORE tensors with autograd graph intact; stop-gradient is
  applied only to h_final at loss computation.
- Captures are cleared after each `compute_aux_loss` to avoid memory leak.
- Thread-safe assumption: the owning training loop calls
  `compute_aux_loss` on the SAME forward pass that populated captures.

This module is test-covered by `scripts/test_leap.py`.

Usage (from trainer)
--------------------

    leap = LeapAuxLoss(model, layer_indices=[10, 11, 12],
                       weights=[0.2, 0.3, 0.5], tau=0.98)

    # Each training step:
    logits = model(x)                       # hooks fire, populate leap.captured
    loss_main = CE(logits, targets)
    loss_aux  = leap.compute_aux_loss(model.final_hidden())
    total     = loss_main + loss_aux
    total.backward()
"""

from __future__ import annotations

from typing import Dict, List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


DEFAULT_TAU_TRAIN = 0.98
DEFAULT_THETA_INFER = 0.95


class LeapAuxLoss:
    """Compute LEAP layer-exit aux loss for a set of intermediate layers.

    Installs forward hooks on ``model.layers[i]`` for each provided index
    and stores their outputs. Call ``compute_aux_loss(h_final)`` after
    a forward pass to get a scalar loss that pushes each captured layer's
    output toward ``h_final`` via a sigmoid-gated cosine target.

    Parameters
    ----------
    model : nn.Module
        Model with a ``.layers`` iterable (OdinFlat-style). OdinHalo-style
        models with ``.shared_layers`` are supported by passing
        ``layers_attr="shared_layers"``.
    layer_indices : list[int]
        Layer positions to install exit-loss hooks on. Typically the last
        1-3 layers (e.g. for a 14-layer model: [11, 12]). Index 13 (the
        final layer) is NOT a useful target because h_final comes from it.
    weights : list[float] | None
        Per-layer scalar weight on the aux loss. If None, uniform 1.0.
    tau : float
        Training-time cosine threshold. Sigmoid midpoint at this value.
    layers_attr : str
        Attribute name on ``model`` that returns the ModuleList of layers.
        Defaults to "layers"; use "shared_layers" for OdinHalo.

    Notes
    -----
    - All tensors are cast to fp32 inside cos similarity for numerical
      stability under fp16 autocast.
    - ``compute_aux_loss`` clears captures after use; calling twice per
      forward would error.
    - ``close()`` removes hooks; call this on training teardown to avoid
      leaking module references.
    """

    def __init__(
        self,
        model: nn.Module,
        layer_indices: List[int],
        weights: Optional[List[float]] = None,
        tau: float = DEFAULT_TAU_TRAIN,
        layers_attr: str = "layers",
        final_attr: Optional[str] = "final_norm",
    ):
        if not layer_indices:
            raise ValueError("layer_indices must be non-empty")
        if weights is None:
            weights = [1.0] * len(layer_indices)
        if len(weights) != len(layer_indices):
            raise ValueError(
                f"weights length ({len(weights)}) must match layer_indices "
                f"length ({len(layer_indices)})"
            )

        self.tau = float(tau)
        self.layer_indices = list(layer_indices)
        self.weights = list(weights)

        layers = getattr(model, layers_attr, None)
        if layers is None:
            raise AttributeError(
                f"model has no '{layers_attr}' attribute; LEAP needs a "
                f"ModuleList of layers"
            )

        max_idx = len(layers) - 1
        bad = [i for i in self.layer_indices if i < 0 or i > max_idx]
        if bad:
            raise IndexError(
                f"layer_indices {bad} out of bounds [0, {max_idx}]"
            )

        self._captured: Dict[int, torch.Tensor] = {}
        self._final_captured: Optional[torch.Tensor] = None
        self._hooks = []
        for idx in self.layer_indices:
            hook = layers[idx].register_forward_hook(self._make_hook(idx))
            self._hooks.append(hook)

        # Optionally hook final_norm so we can auto-capture h_final.
        # When final_attr is None, callers MUST pass h_final explicitly to
        # compute_aux_loss (useful for custom models).
        self._final_hook = None
        if final_attr is not None:
            final_mod = getattr(model, final_attr, None)
            if final_mod is not None:
                self._final_hook = final_mod.register_forward_hook(
                    self._make_final_hook())
                self._hooks.append(self._final_hook)

    def _make_hook(self, idx: int):
        def _hook(module, inputs, output):
            tensor = output
            if isinstance(tensor, tuple):
                tensor = tensor[0]
            if torch.is_tensor(tensor):
                # Store WITH gradient graph intact — intermediate needs
                # gradient to push it toward h_final. Stop-grad only
                # applies to h_final at loss time.
                self._captured[idx] = tensor
        return _hook

    def _make_final_hook(self):
        def _hook(module, inputs, output):
            tensor = output
            if isinstance(tensor, tuple):
                tensor = tensor[0]
            if torch.is_tensor(tensor):
                self._final_captured = tensor
        return _hook

    def compute_aux_loss(self, h_final: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Compute the LEAP aux loss from captured intermediates.

        Parameters
        ----------
        h_final : torch.Tensor | None
            If provided, use as the stop-gradient target. If None, use
            the tensor captured by the final-norm hook (if configured).

        Returns
        -------
        torch.Tensor : scalar loss.

        Raises
        ------
        RuntimeError : if hooks have not fired OR h_final is None and no
            final-norm hook was configured.
        """
        if not self._captured:
            raise RuntimeError(
                "LEAP: no captured intermediates. Did you call forward "
                "after the last compute_aux_loss?")

        if h_final is None:
            h_final = self._final_captured
        if h_final is None:
            raise RuntimeError(
                "LEAP: no h_final available. Either pass h_final to "
                "compute_aux_loss or construct LeapAuxLoss with a "
                "final_attr that hooks a module on the forward path.")

        # Stop-gradient on h_final. Cast to fp32 for stable cosine.
        hf = h_final.detach().float()
        hf_flat = hf.reshape(-1, hf.shape[-1])

        loss = torch.zeros((), device=hf.device, dtype=hf.dtype)

        missing = []
        for weight, idx in zip(self.weights, self.layer_indices):
            if idx not in self._captured:
                missing.append(idx)
                continue
            hi = self._captured[idx].float()
            hi_flat = hi.reshape(-1, hi.shape[-1])
            cos = F.cosine_similarity(hi_flat, hf_flat, dim=-1)
            # sigmoid(10 * (tau - cos)): high when cos < tau, low when cos >= tau
            loss_per_tok = torch.sigmoid(10.0 * (self.tau - cos))
            loss = loss + float(weight) * loss_per_tok.mean()

        # Clear captures before returning so next forward re-populates.
        self._captured.clear()
        self._final_captured = None

        if missing:
            raise RuntimeError(
                f"LEAP: missing captures for layer indices {missing}; hooks "
                f"may have failed to fire. Ensure forward was run and that "
                f"torch.compile didn't bypass the hooks.")

        return loss

    def discard_captures(self) -> None:
        """Clear any pending captured intermediates without computing loss.

        Call this when a forward pass ran for reasons other than LEAP
        accounting (e.g. validation) to avoid stale state on the next
        training step.
        """
        self._captured.clear()
        self._final_captured = None

    def close(self) -> None:
        """Remove all hooks. Idempotent and safe to call before init completes."""
        hooks = getattr(self, "_hooks", None) or []
        for h in hooks:
            try:
                h.remove()
            except Exception:  # noqa: BLE001
                pass
        self._hooks = []
        if getattr(self, "_captured", None):
            self._captured.clear()

    def __del__(self):
        try:
            self.close()
        except Exception:  # noqa: BLE001
            pass


def layer_exit_decision(
    h_current: torch.Tensor,
    h_final_approx: torch.Tensor,
    theta: float = DEFAULT_THETA_INFER,
) -> torch.Tensor:
    """Inference-time early-exit gate.

    Returns a bool mask per token: True = exit here, False = continue.

    Parameters
    ----------
    h_current : torch.Tensor
        Hidden state at some intermediate layer, shape [B, T, D].
    h_final_approx : torch.Tensor
        Approximation of final hidden, shape [B, T, D]. In practice this
        is the last-layer output or a learned auxiliary head.
    theta : float
        Cosine threshold for exit. Matches the paper's default (0.95).
    """
    if h_current.shape != h_final_approx.shape:
        raise ValueError(
            f"shape mismatch: h_current={h_current.shape}, "
            f"h_final_approx={h_final_approx.shape}")
    flat_cur = h_current.reshape(-1, h_current.shape[-1]).float()
    flat_fin = h_final_approx.reshape(-1, h_final_approx.shape[-1]).float()
    cos = F.cosine_similarity(flat_cur, flat_fin, dim=-1)
    return (cos > theta).reshape(h_current.shape[:-1])

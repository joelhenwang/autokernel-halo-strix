"""μP (maximal update parametrization) — init scaling + 3-way LR groups.

Implements Sprint 1.5 Phase A per
``docs/superpowers/specs/2026-05-06-sprint1.5-spectra-mup-design.md§5.3``.

Partial μP (Phase A) = init scaling + LR scaling.
Full μP (Phase E) adds 1/d_head attention scaling; not implemented here.

Usage::

    # At model construction time:
    model = OdinFlat()
    apply_mup_init(model, d_base=256)

    # When building the optimizer:
    groups = build_mup_param_groups(model, base_lr=0.0235, d_base=256)
    optimizer = torch.optim.AdamW(groups)

Design notes
------------
- We classify each 2D parameter into one of three μP categories
  (embedding / hidden / readout) by name-matching. This is pragmatic
  rather than elegant; OdinFlat's naming convention is stable enough
  that substring tests are robust.
- Default ``d_base=256`` matches the 30M probe model (OdinFlatMupProbe).
  OdinFlat's d_model=768 → d_ratio=3 → hidden LR = base_lr / 3.
- Embedding layers keep the standard init; hidden / readout matrices
  get the μP ``sqrt(d_base/d_in)`` correction factor.
- The 1D param group (bias / LN gain / tied embed) is untouched here;
  caller keeps it in the AdamW path with ``--lr-1d``.

Exports
-------
- ``apply_mup_init(model, d_base)``: in-place re-scale non-embedding 2D
  weights.
- ``build_mup_param_groups(model, base_lr, d_base)``: 3-tuple of param
  groups dicts suitable for ``torch.optim.*.__init__``.
- ``MUP_EMBEDDING_NAMES``, ``MUP_READOUT_NAMES``: classification tokens.
"""

from __future__ import annotations

from typing import Any, Dict, List

import torch
import torch.nn as nn


# Name substrings to classify parameters into μP categories.
# These match the naming conventions used by OdinFlat / OdinHalo / etc.
# Order matters: readout patterns take precedence over embedding because
# FactorizedLMHead often shares weight with tok_embeddings, and we want
# the head side (which is the true readout) to classify correctly.
MUP_EMBEDDING_NAMES = (
    "tok_embeddings",
    "embed_table",   # LowRankEmbedding / FactorizedEmbedding
    "input_embed",
)

MUP_READOUT_NAMES = (
    "lm_head",
    "output_proj",   # some models name it this
    "readout",
)


def _classify_param(name: str) -> str:
    """Classify a parameter name into {'embedding', 'readout', 'hidden'}.

    Priority: readout > embedding > hidden. A parameter matching both
    a readout and embedding token is treated as readout because
    FactorizedLMHead wraps lm_head.* around the tied embed weight.
    """
    for token in MUP_READOUT_NAMES:
        if token in name:
            return "readout"
    for token in MUP_EMBEDDING_NAMES:
        if token in name:
            return "embedding"
    return "hidden"


def apply_mup_init(model: nn.Module, d_base: int = 256) -> None:
    """Apply μP init scaling rules in-place on the instantiated model.

    Rules (partial μP, Phase A):
    - Embedding parameters: no rescaling. Standard init preserved.
    - Hidden and readout 2D parameters: rescale by ``sqrt(d_base / d_in)``
      where ``d_in`` is the parameter's last dimension.

    Non-2D parameters are left alone.

    This function assumes the model was already initialized with whatever
    policy the model itself applies (Xavier, depth-scaled, etc.). μP
    re-scales ON TOP of the existing init, per the spec.
    """
    if d_base <= 0:
        raise ValueError(f"d_base must be > 0; got {d_base}")

    for name, p in model.named_parameters():
        if p.dim() < 2:
            continue
        category = _classify_param(name)
        if category == "embedding":
            continue
        d_in = p.shape[-1]
        if d_in <= 0:
            continue
        scale_correction = (d_base / d_in) ** 0.5
        with torch.no_grad():
            p.mul_(scale_correction)


def build_mup_param_groups(
    model: nn.Module,
    base_lr: float,
    d_base: int = 256,
    weight_decay: float = 0.0,
) -> List[Dict[str, Any]]:
    """Build a 3-way μP parameter group list for the optimizer.

    Returns a list of exactly 3 param-group dicts in this order:

    1. Embedding (LR = ``base_lr``)
    2. Hidden   (LR = ``base_lr / d_ratio``)
    3. Readout  (LR = ``base_lr / d_ratio**2``)

    Each group has its own ``lr`` and includes the ``weight_decay``
    passed in. Consumers may further partition these by dimensionality
    if they want to also route 1D params through a separate AdamW group.

    Args:
        model: an instantiated model. Must have ``d_model`` attribute.
        base_lr: the hidden-layer peak LR (equivalent to Sprint 1's
            ``--lr-2d``). The other groups derive from this.
        d_base: μP base width.
        weight_decay: WD applied to every group identically.

    Raises:
        AttributeError: if model has no ``d_model`` attribute.
    """
    d_current = getattr(model, "d_model", None)
    if d_current is None:
        raise AttributeError(
            "build_mup_param_groups requires model.d_model to be set")
    if d_current <= 0:
        raise ValueError(f"model.d_model must be > 0; got {d_current}")
    d_ratio = float(d_current) / float(d_base)
    if d_ratio <= 0:
        raise ValueError(
            f"d_current / d_base <= 0: d_current={d_current} d_base={d_base}")

    emb_params: List[nn.Parameter] = []
    hidden_params: List[nn.Parameter] = []
    readout_params: List[nn.Parameter] = []
    emb_names: List[str] = []
    hidden_names: List[str] = []
    readout_names: List[str] = []

    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        # Only classify 2D-or-higher tensors into μP groups. 1D tensors
        # (biases, LN gains, etc.) are caller-responsibility.
        if p.dim() < 2:
            continue
        category = _classify_param(name)
        if category == "embedding":
            emb_params.append(p)
            emb_names.append(name)
        elif category == "readout":
            readout_params.append(p)
            readout_names.append(name)
        else:
            hidden_params.append(p)
            hidden_names.append(name)

    hidden_lr = base_lr / d_ratio
    readout_lr = base_lr / (d_ratio ** 2)

    return [
        {"params": emb_params, "param_names": emb_names, "lr": base_lr,
         "weight_decay": weight_decay, "_mup_group": "embedding"},
        {"params": hidden_params, "param_names": hidden_names, "lr": hidden_lr,
         "weight_decay": weight_decay, "_mup_group": "hidden"},
        {"params": readout_params, "param_names": readout_names, "lr": readout_lr,
         "weight_decay": weight_decay, "_mup_group": "readout"},
    ]

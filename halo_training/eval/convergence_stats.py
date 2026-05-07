"""Convergence / exit-readiness diagnostics (Phase A', LEAP-inspired).

Measures how quickly a model's hidden state converges toward its final
representation. Used as a gate for deciding whether to ship Phase B
(LEAP aux loss + adaptive iter count for OdinHalo).

Motivation
----------
LEAP (2605.01058) applies an auxiliary loss
    L_exit = sigma(10 * (tau - cos(h_intermediate, sg(h_final))))
at each layer, forcing intermediates to resemble the final representation.
At inference, layers whose cos(h_i, h_final) > theta can be skipped.

For OdinHalo (3-iter Parcae loop), the analogous question is: does iter-2's
output resemble iter-3's output enough that we can drop iter-3 adaptively?
This module measures exactly that, without modifying the model.

Metrics collected
-----------------
Flat models (OdinFlat et al., with ``.layers``):
    per_layer_cos_to_final            cos(h_after_layer_i, h_final)
                                      one value per layer
    per_layer_cos_to_final_frac_high  fraction of tokens with cos > 0.95
    inter_layer_transition_cos        cos(h_layer_i, h_layer_{i+1})
    effective_rank_final              sum(sv^2) / max(sv)^2 of h_final

Looped models (OdinHalo, with ``.shared_layers`` and ``mean_recurrence`` > 1):
    (above, plus)
    iter_k_cos_to_final               cos(h_after_iter_k, h_after_iter_last)
                                      one value per iter except the last
    iter_k_cos_to_final_frac_high     fraction of tokens with cos > tau
                                      (tau defaults to 0.95)
    iter_transition_cos               cos(h_iter_k, h_iter_{k+1})

The ``frac_high_cos`` fractions are the primary gate signal: if
``iter_(N-2)_cos_to_final_frac_high > 0.5`` we can consider Phase B.

Design
------
- Forward hooks on layers / shared_layers. For looped models each shared
  layer fires ``mean_recurrence`` times per forward; we split them into
  iter groups by call index modulo ``n_shared_layers``.
- Tokens are flattened to [N, D] and cosines computed per-token with
  ``F.cosine_similarity(a, b, dim=-1)``.
- Runs on the wikitext validation tail (or first available split) with
  small batch (2 x 512) to keep wall time < 60s and memory < 2 GB.
- Returns None if the model lacks a ``.layers`` or ``.shared_layers``
  attribute.

Output keys::

    {
        "num_batches": 3, "seq_len": 512, "batch_size": 2,
        "is_looped": true, "mean_recurrence": 3,
        "num_layers": 6,
        "tau": 0.95,
        "per_layer_cos_to_final":        [0.40, 0.55, 0.71, 0.83, 0.91, 1.00],
        "per_layer_cos_to_final_frac_high": [0.0, 0.02, 0.15, 0.47, 0.80, 1.0],
        "inter_layer_transition_cos":    [0.45, 0.58, 0.75, 0.88, 0.95],
        "effective_rank_final": 142.3,
        "iter_k_cos_to_final":                [0.68, 0.92],
        "iter_k_cos_to_final_frac_high":      [0.12, 0.62],
        "iter_transition_cos":                [0.71, 0.94],
        "_split_used": "wikitext_val"
    }
"""

from __future__ import annotations

import gc
from typing import Any, Dict, List, Optional

import numpy as np
import torch
import torch.nn.functional as F


DEFAULT_TAU = 0.95


def run(model, tokenizer, validation_splits, args=None,
        num_batches: int = 3, seq_len: int = 512, batch_size: int = 2,
        tau: float = DEFAULT_TAU) -> Optional[Dict[str, Any]]:
    """Collect convergence diagnostics. Returns None if unsupported."""
    layers, is_looped = _resolve_layers(model)
    if layers is None:
        print("  [convergence_stats] model has no .layers or .shared_layers; skipping")
        return None

    mean_recurrence = int(getattr(model, "mean_recurrence", 1)) if is_looped else 1
    n_shared = len(layers)

    split, split_name = _pick_split(validation_splits)
    if split is None:
        print("  [convergence_stats] no validation split available; skipping")
        return None
    path, offset, length_bytes = split
    mm = np.memmap(path, dtype=np.uint16, mode="r",
                   offset=offset, shape=(length_bytes // 2,))

    needed = batch_size * seq_len
    if len(mm) < needed * 2:
        print(f"  [convergence_stats] split {split_name} too small; skipping")
        return None

    device = next(model.parameters()).device

    # Accumulators across all batches
    per_layer_cos_accum: List[List[float]] = [[] for _ in range(n_shared)]
    per_layer_frac_high_accum: List[List[float]] = [[] for _ in range(n_shared)]
    inter_layer_cos_accum: List[List[float]] = [[] for _ in range(n_shared - 1)]
    eff_rank_accum: List[float] = []

    iter_cos_accum: List[List[float]] = [[] for _ in range(max(mean_recurrence - 1, 0))]
    iter_frac_high_accum: List[List[float]] = [[] for _ in range(max(mean_recurrence - 1, 0))]
    iter_trans_cos_accum: List[List[float]] = [[] for _ in range(max(mean_recurrence - 1, 0))]

    try:
        rng = np.random.default_rng(seed=13)
        for _ in range(num_batches):
            max_start = int(len(mm)) - needed - 1
            start = int(rng.integers(0, max_start))
            batch_np = np.stack([
                mm[start + i * seq_len: start + (i + 1) * seq_len].astype(np.int64)
                for i in range(batch_size)
            ])
            batch = torch.from_numpy(batch_np).to(device, non_blocking=True)

            # Per-batch collector: flat list of layer outputs in call order
            captured: List[torch.Tensor] = []
            hooks = _install_hooks(layers, captured)
            try:
                with torch.no_grad(), torch.amp.autocast("cuda", dtype=torch.float16):
                    _ = model(batch)
            finally:
                for h in hooks:
                    try:
                        h.remove()
                    except Exception:  # noqa: BLE001
                        pass

            # Expect captured length == n_shared * mean_recurrence for looped,
            # == n_shared for flat. If hooks fired wrong number of times, bail.
            expected = n_shared * mean_recurrence
            if len(captured) != expected:
                print(f"  [convergence_stats] expected {expected} hook captures, "
                      f"got {len(captured)}; skipping")
                return None

            # Final representation: last iter's last layer, flattened per-token
            final_layer_output = captured[-1]  # [B, T, D]
            h_final_flat = final_layer_output.reshape(-1, final_layer_output.shape[-1])

            # Within the LAST iter of the loop (or the single pass for flat),
            # compute per-layer metrics against h_final.
            last_iter_start = (mean_recurrence - 1) * n_shared if is_looped else 0
            last_iter_outputs = captured[last_iter_start: last_iter_start + n_shared]

            for i, layer_out in enumerate(last_iter_outputs):
                flat = layer_out.reshape(-1, layer_out.shape[-1])
                cos = F.cosine_similarity(flat.float(), h_final_flat.float(), dim=-1)
                per_layer_cos_accum[i].append(cos.mean().item())
                per_layer_frac_high_accum[i].append((cos > tau).float().mean().item())

            for i in range(n_shared - 1):
                a = last_iter_outputs[i].reshape(-1, last_iter_outputs[i].shape[-1])
                b = last_iter_outputs[i + 1].reshape(-1, last_iter_outputs[i + 1].shape[-1])
                cos = F.cosine_similarity(a.float(), b.float(), dim=-1)
                inter_layer_cos_accum[i].append(cos.mean().item())

            # Effective rank of final representation
            eff_rank = _effective_rank(h_final_flat.float())
            eff_rank_accum.append(eff_rank)

            # Iter-level metrics (looped only)
            if is_looped and mean_recurrence >= 2:
                # Per-iter "final" = the last shared_layer's output of that iter
                iter_final_outputs = [
                    captured[(k + 1) * n_shared - 1]  # shared_layers[-1] at iter k
                    for k in range(mean_recurrence)
                ]
                h_last_iter = iter_final_outputs[-1].reshape(
                    -1, iter_final_outputs[-1].shape[-1]).float()

                for k in range(mean_recurrence - 1):
                    hk = iter_final_outputs[k].reshape(
                        -1, iter_final_outputs[k].shape[-1]).float()
                    cos_to_last = F.cosine_similarity(hk, h_last_iter, dim=-1)
                    iter_cos_accum[k].append(cos_to_last.mean().item())
                    iter_frac_high_accum[k].append((cos_to_last > tau).float().mean().item())

                    hk1 = iter_final_outputs[k + 1].reshape(
                        -1, iter_final_outputs[k + 1].shape[-1]).float()
                    cos_trans = F.cosine_similarity(hk, hk1, dim=-1)
                    iter_trans_cos_accum[k].append(cos_trans.mean().item())

        # Aggregate across batches: mean of per-batch means
        def _mean(lst: List[float]) -> Optional[float]:
            if not lst:
                return None
            return round(float(np.mean(lst)), 4)

        result: Dict[str, Any] = {
            "num_batches": num_batches,
            "seq_len": seq_len,
            "batch_size": batch_size,
            "is_looped": bool(is_looped),
            "mean_recurrence": mean_recurrence,
            "num_layers": n_shared,
            "tau": tau,
            "per_layer_cos_to_final":
                [_mean(x) for x in per_layer_cos_accum],
            "per_layer_cos_to_final_frac_high":
                [_mean(x) for x in per_layer_frac_high_accum],
            "inter_layer_transition_cos":
                [_mean(x) for x in inter_layer_cos_accum],
            "effective_rank_final": _mean(eff_rank_accum),
            "_split_used": split_name,
        }
        if is_looped and mean_recurrence >= 2:
            result["iter_k_cos_to_final"] = [_mean(x) for x in iter_cos_accum]
            result["iter_k_cos_to_final_frac_high"] = [_mean(x) for x in iter_frac_high_accum]
            result["iter_transition_cos"] = [_mean(x) for x in iter_trans_cos_accum]
        else:
            result["iter_k_cos_to_final"] = None
            result["iter_k_cos_to_final_frac_high"] = None
            result["iter_transition_cos"] = None
        return result
    finally:
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def _resolve_layers(model):
    """Return (layers_module, is_looped_bool)."""
    # Prefer shared_layers (OdinHalo) to detect looped case
    shared = getattr(model, "shared_layers", None)
    if shared is not None and hasattr(shared, "__iter__"):
        return shared, True
    flat = getattr(model, "layers", None)
    if flat is not None and hasattr(flat, "__iter__"):
        return flat, False
    return None, False


def _install_hooks(layers, captured_out_list):
    """Install forward hooks that append each call's output tensor."""
    hooks = []
    for layer in layers:
        def _make_hook():
            def hook(module, inputs, output):
                tensor = output
                if isinstance(tensor, tuple):
                    tensor = tensor[0]
                if torch.is_tensor(tensor):
                    # Detach to avoid holding autograd state; do NOT move to CPU
                    # here since subsequent ops run on GPU.
                    captured_out_list.append(tensor.detach())
            return hook
        hooks.append(layer.register_forward_hook(_make_hook()))
    return hooks


def _effective_rank(h_flat: torch.Tensor, max_rows: int = 2048) -> float:
    """Stable rank approximation: sum(sv^2) / max(sv)^2.

    Subsamples rows if h_flat is huge to keep SVD tractable.
    Returns ``float('nan')`` on failure.
    """
    if h_flat.dim() != 2:
        return float("nan")
    n_rows = h_flat.shape[0]
    if n_rows > max_rows:
        idx = torch.randperm(n_rows, device=h_flat.device)[:max_rows]
        h_flat = h_flat[idx]
    try:
        # Use torch.linalg.svdvals for speed (only singular values needed)
        sv = torch.linalg.svdvals(h_flat)
        sv = sv[sv > 1e-8]
        if sv.numel() == 0:
            return float("nan")
        return float((sv.pow(2).sum() / sv[0].pow(2)).item())
    except Exception:  # noqa: BLE001 — SVD can fail on degenerate matrices
        return float("nan")


def _pick_split(validation_splits):
    """Pick the first available split (prefer wikitext_val for consistency)."""
    preferred = ["wikitext_val", "gpt_small_val", "stem_crawl_val", "dolma_val"]
    for name in preferred:
        spec = validation_splits.get(name)
        if spec is not None:
            return spec, name
    return None, None

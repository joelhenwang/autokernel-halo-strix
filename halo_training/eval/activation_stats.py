"""Per-layer activation diagnostics (Sprint 2 Phase 4).

Registers forward hooks on each ``model.layers[i]`` module, runs a small
number of validation batches, and reports:

    layer_kurtosis       — excess kurtosis of the layer output tensor
    layer_rms_norm       — RMS norm of the layer output
    mean_layer_kurtosis  — averaged across layers (headline number)

Uses the wikitext validation tail by default, falling back to any available
split. Hooks are scrupulously removed via a ``try/finally`` — memory leak
regression tested in ``scripts/test_eval_scorecard.py``.

Design decisions
----------------
- ``num_batches=5, batch_size=4, seq_len=512`` — keeps wall time under ~30s
  and GPU memory under ~1 GB even for 122M models. Total ~10K tokens
  per-layer of activation captured.
- Kurtosis / RMS computed on the full activation tensor flattened. For fp16
  training drift we look for values outside ~[1.5, 8.0] kurtosis range which
  indicate heavy-tail blowup (reported in knowledge/architectures/...).
- Attention entropy hooks are NOT registered in Phase 4 because the
  attention modules in OdinFlat / OdinHalo don't expose softmax output as
  a stable attribute. Adding entropy would require model-aware hook
  injection — deferred to a future sprint if the diagnostic is needed.

Output keys::

    {
        "num_layers": 14,
        "num_batches": 5,
        "seq_len": 512,
        "layer_kurtosis": [3.1, 3.4, 3.2, ..., 2.9],
        "layer_rms_norm": [1.12, 1.34, 1.41, ..., 1.08],
        "mean_layer_kurtosis": 3.08,
        "mean_layer_rms_norm": 1.23,
        "max_layer_kurtosis": 4.1,
        "max_layer_rms_norm": 1.89,
        "_split_used": "wikitext_val"
    }

Returns ``None`` if the model doesn't have a ``.layers`` ModuleList.
"""

from __future__ import annotations

import gc
from collections import defaultdict
from typing import Any, Dict, List, Optional

import numpy as np
import torch


def run(model, tokenizer, validation_splits, args=None,
        num_batches: int = 5, seq_len: int = 512, batch_size: int = 4) -> Optional[Dict[str, Any]]:
    """Collect per-layer activation statistics. Returns None if unsupported."""
    layers = getattr(model, "layers", None)
    if layers is None or not hasattr(layers, "__iter__"):
        print("  [activation_stats] model has no .layers iterable; skipping")
        return None

    # Pick a validation split for batch sampling
    split, split_name = _pick_split(validation_splits)
    if split is None:
        print("  [activation_stats] no validation split available; skipping")
        return None
    path, offset, length_bytes = split
    mm = np.memmap(path, dtype=np.uint16, mode="r",
                   offset=offset, shape=(length_bytes // 2,))

    needed = batch_size * seq_len
    if len(mm) < needed * 2:
        print(f"  [activation_stats] split {split_name} too small for "
              f"{batch_size}x{seq_len} batches; skipping")
        return None

    device = next(model.parameters()).device
    collector = _ActivationCollector(layers)

    try:
        rng = np.random.default_rng(seed=13)
        for _ in range(num_batches):
            max_start = int(len(mm)) - needed - 1
            start = int(rng.integers(0, max_start))
            batch_np = np.stack([
                mm[start + i * seq_len : start + (i + 1) * seq_len].astype(np.int64)
                for i in range(batch_size)
            ])
            batch = torch.from_numpy(batch_np).to(device, non_blocking=True)
            with torch.no_grad(), torch.amp.autocast("cuda", dtype=torch.float16):
                _ = model(batch)

        # Aggregate across captured activations
        n_layers = len(collector.per_layer_kurtosis)
        kurts: List[float] = []
        rmses: List[float] = []
        for i in range(n_layers):
            k_list = collector.per_layer_kurtosis[i]
            r_list = collector.per_layer_rms[i]
            kurts.append(
                float(np.mean(k_list)) if k_list else float("nan")
            )
            rmses.append(
                float(np.mean(r_list)) if r_list else float("nan")
            )

        def _round(v, n=3):
            return None if (v is None or (isinstance(v, float) and np.isnan(v))) else round(v, n)

        finite_k = [k for k in kurts if not np.isnan(k)]
        finite_r = [r for r in rmses if not np.isnan(r)]

        return {
            "num_layers": n_layers,
            "num_batches": num_batches,
            "seq_len": seq_len,
            "batch_size": batch_size,
            "layer_kurtosis": [_round(k) for k in kurts],
            "layer_rms_norm": [_round(r) for r in rmses],
            "mean_layer_kurtosis": _round(float(np.mean(finite_k))) if finite_k else None,
            "mean_layer_rms_norm": _round(float(np.mean(finite_r))) if finite_r else None,
            "max_layer_kurtosis": _round(float(np.max(finite_k))) if finite_k else None,
            "max_layer_rms_norm": _round(float(np.max(finite_r))) if finite_r else None,
            "_split_used": split_name,
        }
    finally:
        # Critical: always remove hooks, even on exception, to avoid leaking
        # references into the model's forward graph.
        collector.remove()
        gc.collect()
        torch.cuda.empty_cache()


def _pick_split(validation_splits):
    """Pick the first available split (prefer wikitext_val for consistency)."""
    preferred = ["wikitext_val", "gpt_small_val", "stem_crawl_val", "dolma_val"]
    for name in preferred:
        spec = validation_splits.get(name)
        if spec is not None:
            return spec, name
    return None, None


class _ActivationCollector:
    """Hooks each layer's forward_post; stores incremental kurtosis & rms.

    We avoid keeping full activation tensors around — even on CPU — because
    for n_layers=14, batch=4, seq=512, d_model=768 that's ~100 MB per
    batch, and we run 5 batches. Instead we compute kurtosis + RMS on-the-
    fly inside the hook and store only the resulting floats.
    """

    def __init__(self, layers):
        self.per_layer_kurtosis: Dict[int, List[float]] = defaultdict(list)
        self.per_layer_rms: Dict[int, List[float]] = defaultdict(list)
        self._hooks = []

        for i, layer in enumerate(layers):
            h = layer.register_forward_hook(self._make_hook(i))
            self._hooks.append(h)

    def _make_hook(self, idx: int):
        def hook(module, inputs, output):
            tensor = output
            if isinstance(tensor, tuple):
                tensor = tensor[0]
            if not torch.is_tensor(tensor):
                return
            # Move to float32 on the same device for stable moment computation
            flat = tensor.detach().float().flatten()
            mean = flat.mean()
            std = flat.std().clamp_min(1e-8)
            z = (flat - mean) / std
            # Excess kurtosis = E[z^4] - 3. Normal → 0, heavy-tail → positive.
            kurt = (z.pow(4).mean() - 3.0).item()
            rms = flat.pow(2).mean().sqrt().item()
            if not (np.isnan(kurt) or np.isinf(kurt)):
                self.per_layer_kurtosis[idx].append(float(kurt))
            if not (np.isnan(rms) or np.isinf(rms)):
                self.per_layer_rms[idx].append(float(rms))

        return hook

    def remove(self):
        for h in self._hooks:
            try:
                h.remove()
            except Exception:  # noqa: BLE001
                pass
        self._hooks = []

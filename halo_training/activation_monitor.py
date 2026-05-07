"""Per-layer max-abs activation tracker for fp16 stability diagnostics.

Opt-in via ``scripts/train_ddp.py --activation-monitor``. Registers
forward hooks on each child of ``model.layers`` (or any ``nn.ModuleList``
top-level attribute); samples the output tensor's max-abs every N
optimizer steps; writes one JSONL entry per sample step to
``$CKPT_DIR/activation_stats.jsonl``.

Schema (one JSON per line):
    {"step": int, "layer": str, "maxabs": float, "fp16_headroom": float,
     "dtype": "float16"|"float32"|..., "shape": [int, ...]}

``fp16_headroom`` = 65504 / maxabs; values <2 indicate danger of overflow.

Hooks are detached (``maxabs`` is computed without autograd state), so
the monitor does not pollute the backward graph.

**Sampling gate (2026-05-07 fix):** Hooks no-op entirely on non-sampling
steps — no reduction kernel, no GPU→CPU sync, no dict write. The trainer
calls ``monitor.set_step(global_step)`` once per opt step BEFORE the
microstep forwards to flip ``_should_sample``. On sampling steps, hooks
store the maxabs as a 0-d GPU tensor; ``.item()`` sync is deferred until
``monitor.step()`` commits the JSONL at end-of-cycle (one sync per tracked
layer per sample step, not per microstep per forward). This removes the
~10-15% steady-state overhead observed on OdinHalo. Default behavior with
no ``set_step`` call is "record every forward" for backward compatibility
with isolated-unit-test usage.

See ``knowledge/training/fp16_stability_gfx1151.md`` for interpretation.
"""

from __future__ import annotations

import json
import os
from typing import Dict, List, Optional

import torch
import torch.nn as nn


# fp16 max representable value (used for headroom calculation)
_FP16_MAX = 65504.0


class ActivationMonitor:
    """Forward-hook-based activation max-abs tracker.

    Usage:
        monitor = ActivationMonitor(
            model, output_path="ckpts/run/activation_stats.jsonl",
            sample_every=100,
        )
        monitor.attach()
        # ... training loop calls monitor.step(global_step) each opt step ...
        monitor.detach()

    The hooks store per-layer max-abs into ``self._current_stats`` on
    sampling-step forwards only (gated by ``_should_sample`` flag set by
    the trainer's ``set_step()`` call). ``step(N)`` commits stored stats
    to JSONL at end-of-cycle on sampling steps. Non-sampling forwards
    are zero-cost (no reduction, no sync). Between commits, only the
    most recent forward's values are kept — sampling is instantaneous,
    not a running max, to match ``torch.profiler`` semantics.
    """

    def __init__(
        self,
        model: nn.Module,
        output_path: Optional[str] = None,
        sample_every: int = 100,
        layer_attrs: Optional[List[str]] = None,
        include_top_level: bool = True,
    ):
        self.model = model
        self.output_path = output_path
        self.sample_every = max(1, int(sample_every))
        # ``layer_attrs`` lets callers pick specific submodules, e.g.
        # ["layers", "norm_out"]. Default: walk all nn.ModuleList children.
        self.layer_attrs = layer_attrs
        self.include_top_level = include_top_level

        self._hooks: List[torch.utils.hooks.RemovableHandle] = []
        self._tracked: Dict[str, Dict] = {}  # {name: {maxabs, dtype, shape}}
        self._current_stats: Dict[str, Dict] = {}
        self._attached = False

        # Sampling gate: trainer calls ``set_step(global_step)`` each opt
        # step to arm or disarm the forward hooks. Default True so that
        # isolated-unit-test usage (single forward, one step() call) still
        # records without the trainer contract. When False, hooks fully
        # no-op — no reduction, no sync, no dict write.
        self._current_step: int = -1
        self._should_sample: bool = True

    def _enumerate_tracked_modules(self) -> List[tuple]:
        """Return list of (name, module) pairs to hook.

        Walks the model's children; hooks every leaf of ``nn.ModuleList``
        top-level attributes. Also hooks named top-level modules whose
        name matches common pattern (norm_out, lm_head, tok_embeddings)
        so we see end-to-end range.
        """
        raw = self.model.module if hasattr(self.model, "module") else self.model
        raw = raw._orig_mod if hasattr(raw, "_orig_mod") else raw

        pairs = []

        if self.layer_attrs is not None:
            for attr in self.layer_attrs:
                if not hasattr(raw, attr):
                    continue
                obj = getattr(raw, attr)
                if isinstance(obj, nn.ModuleList):
                    for i, mod in enumerate(obj):
                        pairs.append((f"{attr}.{i}", mod))
                elif isinstance(obj, nn.Module):
                    pairs.append((attr, obj))
            return pairs

        # Default: auto-discover. Hook every item of every nn.ModuleList
        # child, plus top-level attrs named from _DEFAULT_TOP_LEVEL.
        for name, child in raw.named_children():
            if isinstance(child, nn.ModuleList):
                for i, leaf in enumerate(child):
                    pairs.append((f"{name}.{i}", leaf))
            elif self.include_top_level and name in _DEFAULT_TOP_LEVEL:
                pairs.append((name, child))
        return pairs

    def attach(self) -> None:
        if self._attached:
            return
        for name, mod in self._enumerate_tracked_modules():
            self._tracked[name] = {}
            self._hooks.append(mod.register_forward_hook(self._make_hook(name)))
        self._attached = True

    def detach(self) -> None:
        for h in self._hooks:
            h.remove()
        self._hooks.clear()
        self._attached = False

    def _make_hook(self, name: str):
        def hook(module, inputs, output):
            # Fast path: non-sampling step → zero work (no reduction, no
            # GPU sync, no dict write). This removes the per-forward
            # ``.item()`` sync that dominated monitor overhead on looped
            # models (~10-15% throughput on OdinHalo).
            if not self._should_sample:
                return
            # Output can be a tensor, tuple, or dict. We take the first
            # tensor we find.
            t = _first_tensor(output)
            if t is None:
                return
            with torch.no_grad():
                # Keep reduction on GPU — defer .item() until commit in
                # ``step()``. Allocates a 0-d tensor; no host transfer yet.
                maxabs_tensor = t.detach().abs().max()
            self._current_stats[name] = {
                "maxabs_tensor": maxabs_tensor,
                "dtype": str(t.dtype).replace("torch.", ""),
                "shape": list(t.shape),
            }
        return hook

    def set_step(self, global_step: int) -> None:
        """Arm (or disarm) the forward hooks for the upcoming opt step.

        Called by the trainer once per optimizer step, BEFORE the first
        microstep's forward. The flag remains in effect until the next
        ``set_step`` call. Hooks fully no-op on disarmed steps (no reduction,
        no sync).

        If the caller never invokes ``set_step`` the monitor defaults to
        "record every forward" for backward compatibility with isolated
        usage (e.g. unit tests with ``sample_every=1`` and a single step).
        """
        self._current_step = int(global_step)
        self._should_sample = (int(global_step) % self.sample_every == 0)

    def current_stats(self) -> Dict[str, Dict]:
        """Return a defensive copy of the most recent per-layer stats.

        Used by the NaN forensics dump (R1). Forces ``.item()`` on any
        pending 0-d tensors so downstream consumers see float values.
        Returns empty dict if no forward has completed yet.
        """
        out: Dict[str, Dict] = {}
        for k, v in self._current_stats.items():
            d = dict(v)
            t = d.pop("maxabs_tensor", None)
            if t is not None and isinstance(t, torch.Tensor):
                try:
                    d["maxabs"] = float(t.item())
                except Exception:
                    d["maxabs"] = float("nan")
            elif "maxabs" not in d:
                d["maxabs"] = float("nan")
            out[k] = d
        return out

    def step(self, global_step: int) -> None:
        """Commit current stats to JSONL if this step is a sampling step."""
        if global_step % self.sample_every != 0:
            # Not a sampling step. Drop anything the hook recorded
            # pre-``set_step`` (defensive; should be empty under correct
            # trainer contract).
            self._current_stats.clear()
            return
        if not self._current_stats:
            return
        if self.output_path is None:
            self._current_stats.clear()
            return
        os.makedirs(os.path.dirname(self.output_path) or ".", exist_ok=True)
        with open(self.output_path, "a", encoding="utf-8") as f:
            for layer, stats in sorted(self._current_stats.items()):
                maxabs_t = stats.get("maxabs_tensor")
                if maxabs_t is not None and isinstance(maxabs_t, torch.Tensor):
                    # Single GPU→CPU sync per tracked layer per sample step.
                    maxabs = float(maxabs_t.item())
                elif "maxabs" in stats:
                    # Back-compat path (shouldn't trigger under new hook).
                    maxabs = float(stats["maxabs"])
                else:
                    continue
                headroom = _FP16_MAX / maxabs if maxabs > 0 else float("inf")
                f.write(json.dumps({
                    "step": int(global_step),
                    "layer": layer,
                    "maxabs": maxabs,
                    "fp16_headroom": headroom,
                    "dtype": stats["dtype"],
                    "shape": stats["shape"],
                }) + "\n")
        # Clear after commit so a missed ``set_step`` on the next step
        # doesn't leak this cycle's data into the next JSONL entry.
        self._current_stats.clear()


_DEFAULT_TOP_LEVEL = {
    "tok_embeddings", "embeddings", "embed",
    "norm_out", "final_norm", "norm",
    "lm_head", "head", "output_head",
    "iter_norm",
}


def _first_tensor(x) -> Optional[torch.Tensor]:
    if isinstance(x, torch.Tensor):
        return x
    if isinstance(x, (tuple, list)):
        for item in x:
            t = _first_tensor(item)
            if t is not None:
                return t
        return None
    if isinstance(x, dict):
        for v in x.values():
            t = _first_tensor(v)
            if t is not None:
                return t
        return None
    return None

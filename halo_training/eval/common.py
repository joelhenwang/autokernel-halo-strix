"""Shared utilities for the Sprint 2 evaluation scorecard.

Provides:

- `load_model(model_path, class_name)`: instantiate a model from a .py file
- `load_checkpoint(model, checkpoint_path)`: load weights with `_orig_mod.`
  stripping (Sprint 1 fix) and fail-loud on key mismatch.
- `load_tokenizer(tokenizer_path)`: HF tokenizer loader
- `default_validation_splits()`: tail-slice offsets into existing `.bin` files
- `checkpoint_basename(path)`: canonical checkpoint identifier for filenames
- `current_timestamp_utc()`: ISO-8601 UTC timestamp with `Z` suffix
- `resolve_eval_machine()`: read `EVAL_MACHINE` env var; fall back to hostname

These helpers are intentionally narrow and have no dependency on any
evaluator module. Evaluators import from this module, not the other way
around, to keep Phase 1 standalone.
"""

from __future__ import annotations

import datetime
import importlib.util
import os
import socket
import sys
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import torch


def load_model(model_path: str, class_name: str) -> torch.nn.Module:
    """Load a model class from a .py file path and instantiate it (default ctor)."""
    spec = importlib.util.spec_from_file_location("user_model", model_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load spec for {model_path!r}")
    mod = importlib.util.module_from_spec(spec)
    sys.modules["user_model"] = mod
    spec.loader.exec_module(mod)
    cls = getattr(mod, class_name, None)
    if cls is None:
        raise AttributeError(
            f"Class {class_name!r} not found in {model_path!r}. "
            f"Available names: {[n for n in dir(mod) if not n.startswith('_')]}"
        )
    return cls()


def load_checkpoint(
    model: torch.nn.Module,
    checkpoint_path: str,
    device: str = "cuda",
) -> Tuple[torch.nn.Module, Optional[int]]:
    """Load a .pt checkpoint into `model` with `_orig_mod.` prefix stripping.

    Raises `RuntimeError` on any missing or unexpected keys after stripping.
    This is the same fail-loud policy used in `scripts/sample_odin_flat.py`.

    Returns
    -------
    model : the same model, with state dict loaded
    step  : the training step, if present in the checkpoint; else None
    """
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    state_dict = ckpt.get("model_state_dict", ckpt) if isinstance(ckpt, dict) else ckpt

    # Strip torch.compile's ._orig_mod. prefix from per-layer compile_zones wrappers
    cleaned = {k.replace("._orig_mod.", "."): v for k, v in state_dict.items()}
    missing, unexpected = model.load_state_dict(cleaned, strict=False)
    if missing or unexpected:
        raise RuntimeError(
            f"Checkpoint key mismatch after _orig_mod stripping: "
            f"{len(missing)} missing, {len(unexpected)} unexpected.\n"
            f"  First missing:    {list(missing)[:3]}\n"
            f"  First unexpected: {list(unexpected)[:3]}"
        )

    step = ckpt.get("step") if isinstance(ckpt, dict) else None
    return model, step


def load_tokenizer(tokenizer_path: str):
    """Load a HuggingFace fast tokenizer from tokenizer.json."""
    from tokenizers import Tokenizer

    return Tokenizer.from_file(tokenizer_path)


# Default validation splits: tail slices of the existing pre-tokenized .bin files.
# Returns {domain_name: (path, offset_bytes, length_bytes)} or None if unavailable.
# Offset is computed as file_size * (1 - TAIL_FRACTION).
_DEFAULT_DATASET_PATHS: Dict[str, str] = {
    "wikitext_val": "datasets/wikitext-103-odin32k.bin",
    "gpt_small_val": "datasets/gpt-training-small-odin32k.bin",
    "stem_crawl_val": "datasets/stem-crawl-solo-odin32k.bin",
    "dolma_val": "datasets/dolma-10b-odin32k.bin",
}
TAIL_FRACTION = 0.02  # last 2% of each .bin is held out as validation


def default_validation_splits() -> Dict[str, Optional[Tuple[str, int, int]]]:
    """Return {domain: (path, offset_bytes, length_bytes)} for validation tail slices.

    Unavailable files map to `None` so evaluators can gracefully skip them
    rather than failing the whole scorecard. Offsets are file-size-based; no
    separate split files are materialized.
    """
    result: Dict[str, Optional[Tuple[str, int, int]]] = {}
    for domain, path in _DEFAULT_DATASET_PATHS.items():
        if not os.path.exists(path):
            result[domain] = None
            continue
        file_size = os.path.getsize(path)
        tail_bytes = int(file_size * TAIL_FRACTION)
        # Align to even byte boundary (uint16 tokens occupy 2 bytes each)
        tail_bytes -= tail_bytes % 2
        offset = file_size - tail_bytes
        result[domain] = (path, offset, tail_bytes)
    return result


def checkpoint_basename(checkpoint_path: str) -> str:
    """Canonical identifier for scorecard output.

    ``checkpoints/odin-flat-wikitext-ddp/step_1869.pt`` → ``odin-flat-wikitext-ddp-step-1869``.

    Strips extension, joins parent directory and file stem with a hyphen,
    and replaces underscores in the stem with hyphens so the identifier is
    hyphen-consistent.
    """
    p = Path(checkpoint_path)
    stem = p.stem.replace("_", "-")
    return f"{p.parent.name}-{stem}"


def current_timestamp_utc() -> str:
    """ISO-8601 UTC timestamp with `Z` suffix (no microseconds)."""
    return (
        datetime.datetime.now(datetime.timezone.utc)
        .replace(microsecond=0)
        .isoformat()
        .replace("+00:00", "Z")
    )


def resolve_eval_machine(cli_value: Optional[str] = None) -> str:
    """Decide the `eval_machine` label for a scorecard.

    Priority:
      1. explicit CLI value (``--eval-machine``)
      2. ``EVAL_MACHINE`` env var (set by `run_remote.sh`, `run_remote_b.sh`, etc.)
      3. short hostname
    """
    if cli_value:
        return cli_value
    env = os.environ.get("EVAL_MACHINE")
    if env:
        return env
    return socket.gethostname().split(".")[0]


def summarise_model(model: torch.nn.Module) -> Dict[str, Any]:
    """Small metadata block about the model for the scorecard header."""
    total_params = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)

    def _try_get(name: str) -> Any:
        val = getattr(model, name, None)
        return int(val) if isinstance(val, (int,)) else val

    return {
        "total_params": int(total_params),
        "trainable_params": int(trainable),
        "d_model": _try_get("d_model"),
        "n_layers": _try_get("n_layers"),
    }

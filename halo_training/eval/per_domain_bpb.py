"""Per-domain BPB evaluator (Sprint 2 Phase 2).

Computes bits-per-byte on each validation split (tail slice of the
pre-tokenized ``.bin`` files) by averaging cross-entropy over fixed-size
chunks.

BPB convention
--------------
We use the same 3.6-bytes-per-token constant as ``scripts/train_ddp.py``
and ``halo_training/metrics.py`` so scorecard values are directly
comparable to training-time ``bpb=...`` log lines. Empirical bytes-per-token
(measured by decoding a sample of validation tokens) is also emitted as a
diagnostic field alongside each BPB value, so downstream tools can recompute
a corpus-specific BPB if preferred.

Output
------
Evaluator returns a dict of the form::

    {
        "wikitext_val": 1.575,
        "gpt_small_val": 1.98,
        "stem_crawl_val": 2.02,
        "dolma_val": 1.77,
        "_empirical_bytes_per_token": {
            "wikitext_val": 4.2,
            "gpt_small_val": 3.9,
            ...
        },
        "_raw_ce": {
            "wikitext_val": 3.93,
            ...
        },
    }

Missing or too-small splits map to ``None``.
"""

from __future__ import annotations

import math
from typing import Any, Dict, Optional

import numpy as np
import torch
import torch.nn.functional as F


_BYTES_PER_TOKEN_CONSTANT = 3.6  # matches scripts/train_ddp.py compute_bpb


@torch.no_grad()
def run(model, tokenizer, validation_splits, args=None) -> Dict[str, Any]:
    """Compute per-domain BPB using tail slices of validation .bin files.

    Parameters
    ----------
    model : torch.nn.Module, already in ``.eval()`` and on GPU
    tokenizer : HF Tokenizer (optional; used for empirical bytes/token)
    validation_splits : dict {domain: (path, offset, length_bytes) or None}
    args : argparse.Namespace (for ``max_tokens_per_domain``)

    Returns
    -------
    dict with one float BPB per domain, plus diagnostic nested dicts keyed
    with a leading underscore (``_raw_ce``, ``_empirical_bytes_per_token``).
    """
    max_tokens = getattr(args, "max_tokens_per_domain", 50_000) if args else 50_000
    batch_size = 8
    block_size = 512

    results: Dict[str, Optional[float]] = {}
    raw_ce: Dict[str, Optional[float]] = {}
    empirical_bpt: Dict[str, Optional[float]] = {}

    for domain, spec in validation_splits.items():
        if spec is None:
            results[domain] = None
            raw_ce[domain] = None
            empirical_bpt[domain] = None
            continue

        path, offset, length_bytes = spec
        # uint16 tokens → 2 bytes each
        n_tokens = length_bytes // 2
        mm = np.memmap(path, dtype=np.uint16, mode="r", offset=offset, shape=(n_tokens,))

        usable = min(int(mm.shape[0]), int(max_tokens))
        if usable < block_size + 1:
            results[domain] = None
            raw_ce[domain] = None
            empirical_bpt[domain] = None
            continue

        tokens = mm[:usable]

        mean_ce = _compute_mean_ce(model, tokens, batch_size=batch_size, block_size=block_size)

        bpt_empirical = _empirical_bytes_per_token(tokenizer, tokens) if tokenizer is not None else None

        # Canonical BPB: matches training-time formula (ce / ln2) / 3.6
        bpb = (mean_ce / math.log(2)) / _BYTES_PER_TOKEN_CONSTANT if math.isfinite(mean_ce) else None

        results[domain] = round(bpb, 4) if bpb is not None else None
        raw_ce[domain] = round(mean_ce, 4) if math.isfinite(mean_ce) else None
        empirical_bpt[domain] = round(bpt_empirical, 3) if bpt_empirical is not None else None

    return {
        **results,
        "_raw_ce": raw_ce,
        "_empirical_bytes_per_token": empirical_bpt,
        "_bpb_formula": "ce/ln(2)/3.6 (matches train_ddp.py)",
        "_block_size": block_size,
        "_batch_size": batch_size,
        "_max_tokens_per_domain": max_tokens,
    }


def _compute_mean_ce(model, tokens: np.ndarray, *, batch_size: int, block_size: int) -> float:
    """Mean CE over ``tokens`` using causal next-token prediction at block_size."""
    n = int(tokens.shape[0])
    # Number of full (block_size + 1)-token chunks available starting at non-overlapping indices
    n_chunks = (n - 1) // block_size
    if n_chunks <= 0:
        return float("nan")

    # Iterate through chunk indices in batches
    chunk_losses = []
    chunk_sizes = []
    device = next(model.parameters()).device

    for start in range(0, n_chunks, batch_size):
        chunk_indices = range(start, min(start + batch_size, n_chunks))
        # Build the batch: for each chunk i, take tokens[i*block_size : (i+1)*block_size + 1]
        batch_np = np.stack([
            tokens[i * block_size : (i + 1) * block_size + 1].astype(np.int64)
            for i in chunk_indices
        ])
        batch = torch.from_numpy(batch_np).to(device, non_blocking=True)
        input_ids = batch[:, :-1]
        targets = batch[:, 1:]

        with torch.amp.autocast("cuda", dtype=torch.float16):
            logits = model(input_ids)
        if isinstance(logits, tuple):
            logits = logits[0]

        loss = F.cross_entropy(
            logits.reshape(-1, logits.size(-1)).float(),
            targets.reshape(-1),
            reduction="mean",
        )
        if not torch.isfinite(loss):
            continue
        chunk_losses.append(float(loss.item()))
        chunk_sizes.append(int(targets.numel()))

    if not chunk_losses:
        return float("nan")

    # Token-weighted mean (all batches have the same target count, but preserve
    # the generality in case of a ragged final batch)
    total_tokens = sum(chunk_sizes)
    weighted = sum(l * s for l, s in zip(chunk_losses, chunk_sizes))
    return weighted / total_tokens if total_tokens > 0 else float("nan")


def _empirical_bytes_per_token(tokenizer, tokens: np.ndarray, sample: int = 5000) -> Optional[float]:
    """Estimate bytes per token by decoding a sample prefix."""
    if tokenizer is None:
        return None
    sample_ids = tokens[:sample].astype(np.int64).tolist()
    try:
        decoded = tokenizer.decode(sample_ids)
    except Exception:
        return None
    if not decoded:
        return None
    return len(decoded.encode("utf-8")) / len(sample_ids)

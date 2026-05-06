"""Inference throughput + peak memory evaluator (Sprint 2 Phase 3).

Measures forward-pass throughput (tokens per second) and peak VRAM at a
fixed set of sequence lengths with batch=1. This is NOT autoregressive
decode throughput — it's a forward-only upper bound suitable for
deployment-readiness tracking. Autoregressive decode (with KV-cache) will
be added in Sprint 4 when we have a post-trained model worth benchmarking.

Measurement protocol
--------------------
For each seq_len in (256, 512, 1024):

    1. Allocate a random prompt of shape (batch=1, seq_len) on GPU
    2. Warmup: 10 forward passes (JIT + kernel autotuning settles)
    3. Reset CUDA peak memory counter
    4. Measure: 30 forward passes, synchronize, record wall time
    5. tok/s = (seq_len * 30) / elapsed
    6. peak_mem = torch.cuda.max_memory_allocated

Output keys for each seq_len ``L``::

    tok_s_seqL_bs1
    peak_mem_gb_seqL

Plus diagnostics::

    _warmup_forwards
    _measure_forwards
    _batch_size
"""

from __future__ import annotations

import time
from typing import Any, Dict

import torch


def run(model, tokenizer, validation_splits, args=None) -> Dict[str, Any]:
    """Measure forward-pass throughput at seq={256, 512, 1024}, bs=1."""
    seq_lengths = (256, 512, 1024)
    batch_size = 1
    warmup = 10
    measure = 30

    vocab_size = tokenizer.get_vocab_size() if tokenizer is not None else 32768
    device = next(model.parameters()).device

    results: Dict[str, Any] = {
        "_warmup_forwards": warmup,
        "_measure_forwards": measure,
        "_batch_size": batch_size,
    }

    for seq_len in seq_lengths:
        tok_key = f"tok_s_seq{seq_len}_bs{batch_size}"
        mem_key = f"peak_mem_gb_seq{seq_len}"

        try:
            # Reset memory counters before each seq length so peak reflects
            # the cost of that particular context size, not the cumulative max.
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()

            prompt = torch.randint(
                0, vocab_size, (batch_size, seq_len),
                dtype=torch.long, device=device,
            )

            # Warmup
            with torch.no_grad():
                for _ in range(warmup):
                    with torch.amp.autocast("cuda", dtype=torch.float16):
                        _ = model(prompt)
            torch.cuda.synchronize()

            # Reset again so the peak measurement only covers the measure loop
            torch.cuda.reset_peak_memory_stats()

            t0 = time.time()
            with torch.no_grad():
                for _ in range(measure):
                    with torch.amp.autocast("cuda", dtype=torch.float16):
                        _ = model(prompt)
            torch.cuda.synchronize()
            elapsed = time.time() - t0

            tok_s = (batch_size * seq_len * measure) / elapsed if elapsed > 0 else float("nan")
            peak_mem_gb = torch.cuda.max_memory_allocated() / 1e9

            results[tok_key] = round(float(tok_s), 1)
            results[mem_key] = round(float(peak_mem_gb), 3)
        except torch.cuda.OutOfMemoryError:
            results[tok_key] = None
            results[mem_key] = None
            results.setdefault("_oom_seq_lengths", []).append(seq_len)
        except Exception as exc:  # noqa: BLE001
            results[tok_key] = None
            results[mem_key] = None
            results.setdefault("_errors", []).append(
                f"seq={seq_len}: {type(exc).__name__}: {exc}"
            )

    return results

"""Training metrics: BPB, throughput, MFU, structured logging."""

import json
import math
import os
import time
from typing import Any, Dict, Optional


def compute_bpb(ce_loss: float, tokenizer_name: str = "gpt2") -> float:
    """Compute bits-per-byte from cross-entropy loss.

    BPB = (CE_loss / ln2) * (tokens / bytes)
    The token/byte ratio depends on the tokenizer's compression.

    For tiktoken GPT-2 on English text: ~1.3 bytes per token (empirical).
    """
    # Compression ratios (bytes per token) for common tokenizers
    BYTES_PER_TOKEN = {
        "gpt2": 3.6,       # tiktoken GPT-2: ~3.6 bytes per token on English
        "cl100k_base": 3.7, # GPT-4 tokenizer
    }

    bpt = BYTES_PER_TOKEN.get(tokenizer_name, 3.6)
    bpb = (ce_loss / math.log(2)) / bpt
    return bpb


class ThroughputTracker:
    """Track tokens/second and model FLOP utilization (MFU).

    MFU = achieved FLOPS / peak hardware FLOPS.
    Strix Halo peak: ~59.4 TFLOPS FP16.
    Expected: eager ~16%, compiled ~30% (from llm_engineer playbook).

    FLOPS per token (forward + backward) ≈ 6 * model_params.
    """

    def __init__(self, model_params: int, peak_flops: float = 59.4e12):
        self.model_params = model_params
        self.peak_flops = peak_flops
        self.total_tokens = 0
        self.start_time = None
        self._last_tokens = 0
        self._last_time = None

    def start(self):
        self.start_time = time.time()
        self._last_time = self.start_time

    def update(self, tokens: int):
        self.total_tokens += tokens

    def get_stats(self) -> Dict[str, float]:
        if self.start_time is None:
            return {"tok_s": 0, "mfu": 0}

        elapsed = time.time() - self.start_time
        tok_s = self.total_tokens / elapsed if elapsed > 0 else 0

        # MFU: (tokens * 6 * params) / (elapsed * peak_flops)
        achieved_flops = self.total_tokens * 6 * self.model_params / elapsed if elapsed > 0 else 0
        mfu = achieved_flops / self.peak_flops if self.peak_flops > 0 else 0

        return {
            "tok_s": tok_s,
            "mfu": mfu,
            "total_tokens": self.total_tokens,
            "elapsed_s": elapsed,
        }

    def get_interval_stats(self) -> Dict[str, float]:
        """Get stats since last interval call (for per-log-interval reporting)."""
        now = time.time()
        dt = now - self._last_time if self._last_time else 1.0
        d_tokens = self.total_tokens - self._last_tokens

        tok_s = d_tokens / dt if dt > 0 else 0
        achieved_flops = d_tokens * 6 * self.model_params / dt if dt > 0 else 0
        mfu = achieved_flops / self.peak_flops if self.peak_flops > 0 else 0

        self._last_time = now
        self._last_tokens = self.total_tokens

        return {"tok_s": tok_s, "mfu": mfu}


class TrainingLogger:
    """Structured training logger — stdout + optional JSON file.

    Produces one JSON line per log event for easy parsing and charting.
    """

    def __init__(self, log_file: Optional[str] = None):
        self.log_file = log_file
        self.entries = []
        if log_file:
            os.makedirs(os.path.dirname(log_file) or ".", exist_ok=True)

    def log(self, **kwargs):
        """Log a training event with arbitrary key-value pairs."""
        entry = {k: _make_serializable(v) for k, v in kwargs.items()}
        self.entries.append(entry)

        if self.log_file:
            with open(self.log_file, "a") as f:
                f.write(json.dumps(entry) + "\n")

    def summary(self) -> Dict[str, Any]:
        """Return summary statistics from all logged entries."""
        if not self.entries:
            return {}

        losses = [e["loss"] for e in self.entries if "loss" in e]
        tok_s_vals = [e["tok_s"] for e in self.entries if "tok_s" in e]

        return {
            "total_entries": len(self.entries),
            "initial_loss": losses[0] if losses else None,
            "final_loss": losses[-1] if losses else None,
            "best_loss": min(losses) if losses else None,
            "avg_tok_s": sum(tok_s_vals) / len(tok_s_vals) if tok_s_vals else 0,
        }


def _make_serializable(v):
    """Convert torch tensors and other non-serializable types to Python primitives."""
    try:
        import torch
        if isinstance(v, torch.Tensor):
            return v.item() if v.numel() == 1 else v.tolist()
    except ImportError:
        pass
    if isinstance(v, float) and (math.isnan(v) or math.isinf(v)):
        return str(v)
    return v

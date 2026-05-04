"""Weighted cross-entropy loss for SFT and EOS warm-up phases.

Supports per-token weight multipliers (e.g., 5x weight on EOS token)
and ignore_index=-100 for loss masking on non-assistant tokens.
"""

from typing import Callable, Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class WeightedCrossEntropyLoss(nn.Module):
    """Cross-entropy with per-token weight multipliers.

    Args:
        token_weights: Maps token ID to weight multiplier (e.g., {50256: 5.0}).
            Tokens not in the dict get weight 1.0.
        ignore_index: Label value to ignore (default -100).
        label_smoothing: Label smoothing factor (default 0.0).
    """

    def __init__(
        self,
        token_weights: Optional[Dict[int, float]] = None,
        ignore_index: int = -100,
        label_smoothing: float = 0.0,
    ):
        super().__init__()
        self.token_weights = token_weights or {}
        self.ignore_index = ignore_index
        self.label_smoothing = label_smoothing

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """Compute weighted cross-entropy.

        Args:
            logits: (N, vocab_size) or (B, T, vocab_size)
            targets: (N,) or (B, T) with -100 for masked positions
        """
        logits_flat = logits.view(-1, logits.size(-1))
        targets_flat = targets.view(-1)

        # Per-token CE (no reduction)
        losses = F.cross_entropy(
            logits_flat, targets_flat,
            reduction="none",
            ignore_index=self.ignore_index,
            label_smoothing=self.label_smoothing,
        )

        # Build weight mask
        mask = targets_flat != self.ignore_index
        weights = torch.ones_like(losses)
        weights[~mask] = 0.0

        for token_id, weight in self.token_weights.items():
            token_mask = targets_flat == token_id
            weights[token_mask] = weight

        # Weighted mean (only over non-ignored positions)
        total_weight = weights.sum()
        if total_weight == 0:
            return losses.sum() * 0  # no valid targets, return zero with grad
        return (losses * weights).sum() / total_weight


def build_sft_loss_fn(
    eos_weight: float = 1.0,
    eos_token_id: int = 50256,
    ignore_index: int = -100,
    label_smoothing: float = 0.0,
) -> Callable:
    """Build a loss function matching the trainer's loss_fn(output, batch) interface.

    Args:
        eos_weight: Weight multiplier for EOS token. Use 5.0 for Phase 0.
        eos_token_id: EOS token ID (50256 for GPT-2, 0 for vidar-32k).
        ignore_index: Label value to ignore (default -100).
        label_smoothing: Label smoothing factor.

    Returns:
        Callable with signature loss_fn(output, batch) -> scalar loss.
    """
    token_weights = {}
    if eos_weight > 1.0:
        token_weights[eos_token_id] = eos_weight

    criterion = WeightedCrossEntropyLoss(
        token_weights=token_weights if token_weights else None,
        ignore_index=ignore_index,
        label_smoothing=label_smoothing,
    )

    def loss_fn(output, batch):
        _, targets = batch
        targets = targets.to(output.device if isinstance(output, torch.Tensor) else next(iter([])))

        if isinstance(output, torch.Tensor):
            logits = output
        elif hasattr(output, "logits"):
            logits = output.logits
        else:
            logits = output

        # Handle device — targets may not be on the same device
        if logits.device != targets.device:
            targets = targets.to(logits.device)

        return criterion(logits.view(-1, logits.size(-1)), targets.view(-1))

    return loss_fn

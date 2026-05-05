"""Multi-Token Prediction auxiliary head (de-duplicated from vidar_halo + tyr_halo).

Predicts token at position +depth+1 from hidden states.
Discarded after training; improves backbone representations via auxiliary loss.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class MTPHead(nn.Module):
    """Multi-Token Prediction auxiliary head. Shares embedding table (tied weights)."""

    def __init__(self, d_model: int, embed_rank: int, embed_table: nn.Embedding,
                 depth: int = 1):
        super().__init__()
        self.depth = depth
        self.proj = nn.Linear(d_model, embed_rank, bias=False)
        self.embed_table = embed_table

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        trimmed = h[:, :-(self.depth + 1)]
        return F.linear(self.proj(trimmed), self.embed_table.weight)
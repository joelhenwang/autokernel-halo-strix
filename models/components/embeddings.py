"""Factorized embeddings: low-rank embed + tied LM head (from chimera_halo)."""

import torch
import torch.nn as nn
import torch.nn.functional as F


class FactorizedEmbedding(nn.Module):
    """Low-rank embedding: Embedding(V, R) → Linear(R, D).

    Saves (V * D - V * R - R * D) params. With V=50260, R=256, D=768:
    saves ~25M params vs standard Embedding(V, D).
    """

    def __init__(self, vocab_size: int, rank: int, d_model: int):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, rank)
        self.proj_up = nn.Linear(rank, d_model, bias=False)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.proj_up(self.embed(input_ids))


class FactorizedLMHead(nn.Module):
    """Factorized output: Linear(D, R) → matmul with embed table^T.

    Shares the embedding table from FactorizedEmbedding (tied weights).

    When `use_chunked_ce` is True and `forward_hlow` is called, returns the
    low-rank projection `h_low` WITHOUT the final matmul. The caller is expected
    to compute logits+CE via ChunkedLinearCrossEntropyLoss(h_low, embed_table.weight).
    """

    def __init__(self, d_model: int, rank: int, embed_table: nn.Embedding):
        super().__init__()
        self.proj_down = nn.Linear(d_model, rank, bias=False)
        self.embed_table = embed_table
        self.use_chunked_ce = False

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        h_low = self.proj_down(h)
        return F.linear(h_low, self.embed_table.weight)

    def forward_hlow(self, h: torch.Tensor) -> torch.Tensor:
        """Return low-rank projection only; downstream chunked CE does the big matmul."""
        return self.proj_down(h)
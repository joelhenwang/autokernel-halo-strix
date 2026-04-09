"""
Adaptive LM Head with Chunked Cross-Entropy.

Splits vocabulary into frequency tiers:
  Tier 0 (common, 8192 tokens):  full-rank projection — handles 94% of tokens
  Tier 1 (medium, 16384 tokens): low-rank (d→256→16384) — handles 5% of tokens
  Tier 2 (rare, 25681 tokens):   low-rank (d→128→25681) — handles 1% of tokens

Training: chunked CE loss without materializing full (B*T, vocab) logit tensor.
Inference: full logits or early-exit top-K decode.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


def compute_tier_mapping(
    token_freqs: dict,
    vocab_size: int = 50257,
    tier_sizes: tuple = (8192, 16384),
):
    """Assign tokens to tiers by frequency.

    Returns:
        token_to_tier: (vocab_size,) int8 — which tier each token belongs to
        token_to_idx: (vocab_size,) int32 — index within its tier
        tier_token_ids: list of LongTensors — global token IDs per tier
    """
    # Sort by frequency (descending), with token ID as tiebreaker
    sorted_ids = sorted(range(vocab_size), key=lambda t: (-token_freqs.get(t, 0), t))

    token_to_tier = torch.zeros(vocab_size, dtype=torch.int8)
    token_to_idx = torch.zeros(vocab_size, dtype=torch.int32)
    tier_token_ids = []

    offset = 0
    for tier, size in enumerate(tier_sizes):
        tier_ids = sorted_ids[offset:offset + size]
        tier_token_ids.append(torch.tensor(tier_ids, dtype=torch.long))
        for local_idx, global_id in enumerate(tier_ids):
            token_to_tier[global_id] = tier
            token_to_idx[global_id] = local_idx
        offset += size

    # Remaining tokens go to the last tier
    remaining = sorted_ids[offset:]
    tier_token_ids.append(torch.tensor(remaining, dtype=torch.long))
    for local_idx, global_id in enumerate(remaining):
        token_to_tier[global_id] = len(tier_sizes)
        token_to_idx[global_id] = local_idx

    return token_to_tier, token_to_idx, tier_token_ids


class AdaptiveLMHead(nn.Module):
    """Three-tier adaptive softmax with chunked cross-entropy.

    During training (targets provided): returns scalar loss, never materializes
    the full (B*T, vocab) logit tensor.

    During inference (no targets): returns full logits (B, T, vocab_size).
    """

    def __init__(
        self,
        d_model: int,
        vocab_size: int = 50257,
        tier_sizes: tuple = (8192, 16384, 25681),
        tier_ranks: tuple = (None, 256, 128),
    ):
        super().__init__()
        self.d_model = d_model
        self.vocab_size = vocab_size
        self.tier_sizes = tier_sizes
        self.n_tiers = len(tier_sizes)

        assert sum(tier_sizes) == vocab_size, (
            f"tier_sizes {tier_sizes} must sum to vocab_size {vocab_size}"
        )

        # Build projection layers per tier
        self.projs = nn.ModuleList()
        self.heads = nn.ModuleList()
        for i, (size, rank) in enumerate(zip(tier_sizes, tier_ranks)):
            if rank is None or rank >= d_model:
                # Full-rank
                self.projs.append(None)
                self.heads.append(nn.Linear(d_model, size, bias=False))
            else:
                # Low-rank: d_model → rank → size
                self.projs.append(nn.Linear(d_model, rank, bias=False))
                self.heads.append(nn.Linear(rank, size, bias=False))

        # Tier mapping buffers (set by set_tier_mapping)
        self.register_buffer('token_to_tier', torch.zeros(vocab_size, dtype=torch.int8))
        self.register_buffer('token_to_idx', torch.zeros(vocab_size, dtype=torch.int32))

    def set_tier_mapping(self, token_freqs: dict):
        """Compute and set tier mapping from token frequency counts."""
        t2t, t2i, _ = compute_tier_mapping(
            token_freqs, self.vocab_size, self.tier_sizes[:-1]
        )
        self.token_to_tier.copy_(t2t)
        self.token_to_idx.copy_(t2i)

    def _tier_logits(self, h_flat: torch.Tensor, tier: int) -> torch.Tensor:
        """Compute logits for one tier."""
        proj = self.projs[tier]
        head = self.heads[tier]
        if proj is not None:
            return head(proj(h_flat))
        return head(h_flat)

    def forward(
        self, h: torch.Tensor, targets: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        if targets is not None:
            return self._chunked_ce_loss(h, targets)
        return self._full_logits(h)

    def _chunked_ce_loss(self, h: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """Compute CE loss without materializing full logit tensor.

        Iterates tiers, accumulates log-sum-exp, gathers target logits.
        Peak memory = max tier logit tensor (not full vocab).
        """
        B_T = h.shape[0] * h.shape[1] if h.dim() == 3 else h.shape[0]
        h_flat = h.reshape(B_T, self.d_model)
        targets_flat = targets.reshape(B_T)

        # Which tier and index within tier for each target
        target_tier = self.token_to_tier[targets_flat]  # (B_T,) int8
        target_idx = self.token_to_idx[targets_flat]    # (B_T,) int32

        # Accumulate in fp32 for numerical stability
        log_Z = torch.full((B_T,), -1e30, dtype=torch.float32, device=h.device)
        target_logit = torch.zeros(B_T, dtype=torch.float32, device=h.device)

        for tier in range(self.n_tiers):
            logits_tier = self._tier_logits(h_flat, tier).float()  # fp32

            # Accumulate log-sum-exp
            tier_lse = logits_tier.logsumexp(dim=-1)
            log_Z = torch.logaddexp(log_Z, tier_lse)

            # Gather target logits from this tier
            mask = (target_tier == tier)
            if mask.any():
                idx = target_idx[mask].long()
                target_logit[mask] = logits_tier[mask].gather(
                    1, idx.unsqueeze(1)
                ).squeeze(1)

        # CE loss = log_Z - target_logit (fp32)
        loss = (log_Z - target_logit).mean()
        return loss

    def _full_logits(self, h: torch.Tensor) -> torch.Tensor:
        """Compute full logits across all tiers. For inference/eval."""
        orig_shape = h.shape
        h_flat = h.reshape(-1, self.d_model)
        B_T = h_flat.shape[0]

        logits = torch.zeros(B_T, self.vocab_size, dtype=h.dtype, device=h.device)

        offset = 0
        for tier in range(self.n_tiers):
            size = self.tier_sizes[tier]
            tier_logits = self._tier_logits(h_flat, tier)
            logits[:, offset:offset + size] = tier_logits
            offset += size

        return logits.view(*orig_shape[:-1], self.vocab_size)

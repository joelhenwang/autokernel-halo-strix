"""
Engram: Hash-based N-gram Knowledge Tables.

Adapted from DeepSeek's engram_demo_v1.py (https://github.com/deepseek-ai/Engram).
Provides per-layer hash tables that store learned n-gram embeddings, enabling
factual knowledge injection with minimal compute overhead.

Key differences from our COOKBOOK spec (updated with DeepSeek's reference):
  - XOR-based hashing (not prime multiplication) — better distribution
  - abs().sqrt() * sign() gating (not plain sigmoid) — magnitude preservation
  - ShortConv on values (depthwise conv1d) — local smoothing

Usage:
    engram = EngramLayer(d_model=1024, d_engram=512, n_hash_heads=8,
                         ngram_sizes=[2, 3], vocab_size=50257)
    h = h + engram(h, input_ids)  # residual connection
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Optional

try:
    from kernels.hip.fused_engram_gate_conv import kernel_fn as _fused_engram_gate_conv_fn
    _HAS_FUSED_ENGRAM = True
except ImportError:
    _HAS_FUSED_ENGRAM = False


class NgramHashMapping(nn.Module):
    """XOR-based n-gram hash mapping with layer-specific multipliers.

    For each n-gram size and hash head, computes:
        mix = tokens[0] * mult[0]
        for k in 1..n: mix = mix ^ (tokens[k] * mult[k])
        index = mix % table_size

    Layer-specific multipliers ensure different layers learn different patterns.
    """

    def __init__(
        self,
        ngram_sizes: List[int],
        n_hash_heads: int,
        table_size: int,
        layer_id: int = 0,
        seed: int = 42,
    ):
        super().__init__()
        self.ngram_sizes = ngram_sizes
        self.n_hash_heads = n_hash_heads
        self.table_size = table_size

        # Generate deterministic, layer-specific multipliers
        max_n = max(ngram_sizes)
        multipliers = torch.zeros(len(ngram_sizes), n_hash_heads, max_n, dtype=torch.long)

        for ng_idx, n in enumerate(ngram_sizes):
            for h in range(n_hash_heads):
                gen = torch.Generator()
                gen.manual_seed(seed + 10007 * layer_id + 31 * ng_idx + 7 * h)
                for k in range(n):
                    multipliers[ng_idx, h, k] = torch.randint(
                        1, 2**16, (1,), generator=gen
                    ).item()

        self.register_buffer("multipliers", multipliers)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        """Compute hash indices for all n-gram sizes and heads.

        Args:
            input_ids: (batch, seq) token IDs

        Returns:
            hash_indices: (batch, seq, n_ngrams * n_hash_heads) indices into tables
        """
        B, T = input_ids.shape
        all_indices = []

        for ng_idx, n in enumerate(self.ngram_sizes):
            # Build shifted token windows: (B, T, n)
            padded = F.pad(input_ids, (n - 1, 0), value=0)  # left-pad
            windows = padded.unfold(1, n, 1)  # (B, T, n)

            for h in range(self.n_hash_heads):
                mults = self.multipliers[ng_idx, h, :n]  # (n,)
                # XOR-based hash
                mix = windows[:, :, 0] * mults[0]
                for k in range(1, n):
                    mix = mix ^ (windows[:, :, k] * mults[k])
                indices = mix % self.table_size  # (B, T)
                all_indices.append(indices)

        return torch.stack(all_indices, dim=-1)  # (B, T, n_ngrams * n_heads)


class MultiHeadEmbedding(nn.Module):
    """Multi-head embedding table with per-head offset indexing."""

    def __init__(self, n_heads: int, table_size: int, d_embed: int):
        super().__init__()
        self.n_heads = n_heads
        self.table_size = table_size
        self.d_embed = d_embed
        # Single large embedding table for all heads
        self.weight = nn.Parameter(torch.randn(n_heads * table_size, d_embed) * 0.02)

    def forward(self, indices: torch.Tensor) -> torch.Tensor:
        """Look up embeddings for multi-head indices.

        Args:
            indices: (batch, seq, n_heads) hash indices in [0, table_size)

        Returns:
            embeddings: (batch, seq, n_heads, d_embed)
        """
        B, T, H = indices.shape
        # Add per-head offset
        offsets = torch.arange(H, device=indices.device) * self.table_size
        flat_indices = (indices + offsets).view(-1)
        embs = self.weight[flat_indices].view(B, T, H, self.d_embed)
        return embs


class EngramLayer(nn.Module):
    """Engram knowledge layer with hash-based n-gram lookup and gated injection.

    Integrates at specific transformer layers via residual:
        h = h + engram(h, input_ids)

    Parameters follow COOKBOOK.md: Adam optimizer, 5x base LR, zero weight decay.
    """

    def __init__(
        self,
        d_model: int = 1024,
        d_engram: int = 512,
        n_hash_heads: int = 8,
        ngram_sizes: List[int] = None,
        vocab_size: int = 50257,
        table_size: int = 65536,
        layer_id: int = 0,
        conv_kernel: int = 3,
    ):
        super().__init__()
        if ngram_sizes is None:
            ngram_sizes = [2, 3]

        self.d_model = d_model
        self.d_engram = d_engram
        n_total_heads = len(ngram_sizes) * n_hash_heads

        # Hash mapping
        self.hash_map = NgramHashMapping(
            ngram_sizes, n_hash_heads, table_size, layer_id
        )

        # Embedding tables (the "knowledge store")
        self.embeddings = MultiHeadEmbedding(n_total_heads, table_size, d_engram // n_total_heads)

        # Projections: embed → key/value in model hidden space
        self.key_proj = nn.Linear(d_engram, d_model, bias=False)
        self.value_proj = nn.Linear(d_engram, d_model, bias=False)

        # Query from hidden state
        self.query_norm = nn.RMSNorm(d_model) if hasattr(nn, 'RMSNorm') else RMSNormSimple(d_model)

        # Gate: DeepSeek's abs().sqrt()*sign() + sigmoid
        # No learnable params — gate is purely input-dependent

        # Short conv on values (depthwise, local smoothing)
        self.short_conv = nn.Conv1d(
            d_model, d_model, kernel_size=conv_kernel,
            padding=conv_kernel - 1, groups=d_model, bias=True,
        )

    def forward(self, hidden_states: torch.Tensor, input_ids: torch.Tensor) -> torch.Tensor:
        """Compute engram output to be added as residual.

        Args:
            hidden_states: (batch, seq, d_model) — current hidden state
            input_ids: (batch, seq) — token IDs for hash computation

        Returns:
            engram_output: (batch, seq, d_model) — add to hidden_states as residual
        """
        B, T, D = hidden_states.shape

        # 1. Hash and look up n-gram embeddings
        hash_indices = self.hash_map(input_ids)            # (B, T, n_total_heads)
        embs = self.embeddings(hash_indices)               # (B, T, n_total_heads, d_head)
        embs_flat = embs.reshape(B, T, -1)                 # (B, T, d_engram)

        # 2. Project to key/value space
        key = self.key_proj(embs_flat)                     # (B, T, d_model)
        value = self.value_proj(embs_flat)                 # (B, T, d_model)

        # 3. Query from hidden state
        query = self.query_norm(hidden_states)              # (B, T, d_model)

        # 4-5. Fused gate + gated value + conv (7.4x speedup)
        if _HAS_FUSED_ENGRAM and hidden_states.dtype == torch.float16:
            conv_w = self.short_conv.weight.squeeze(1)      # (D, 1, K) -> (D, K)
            conv_b = self.short_conv.bias                   # (D,)
            return _fused_engram_gate_conv_fn(
                query.reshape(-1, D), key.reshape(-1, D), value.reshape(-1, D),
                conv_w, conv_b, T,
            ).reshape(B, T, D)

        # Fallback: PyTorch ops
        # 4. Gate computation (DeepSeek style: magnitude-preserving)
        gate_raw = (query * key).sum(dim=-1, keepdim=True) / (D ** 0.5)
        gate = gate_raw.abs().clamp(min=1e-6).sqrt() * gate_raw.sign()
        gate = torch.sigmoid(gate)                          # (B, T, 1)

        # 5. Gated value + short conv
        gated_value = gate * value                          # (B, T, d_model)
        conv_value = self.short_conv(value.transpose(1, 2))[:, :, :T].transpose(1, 2)

        return gated_value + conv_value


class RMSNormSimple(nn.Module):
    """Fallback RMSNorm for PyTorch < 2.4."""
    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps) * self.weight

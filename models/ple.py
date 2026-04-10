"""
Per-Layer Embeddings (PLE) — Composable Module.

Inspired by Google's Gemma 4. Each decoder layer receives a dedicated
embedding signal, solving the "frontloading problem" where a single initial
embedding degrades with depth. Three ablation modes:

  mode="a"   — Context-aware projection only (hidden state → bottleneck → up)
  mode="b"   — Factored token-identity only (shared table + per-layer mixing)
  mode="a+b" — Both paths combined

Usage:
    ple = PLEModule(PLEConfig(ple_mode="a+b"))
    for i, layer in enumerate(model.layers):
        h = h + ple(h, input_ids, i)
        h = layer(h)
"""

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.tempest import RMSNorm


@dataclass
class PLEConfig:
    vocab_size: int = 50257
    d_model: int = 1024
    n_layers: int = 16
    ple_mode: str = "a+b"       # "a" | "b" | "a+b"
    ple_dim: int = 64           # Path A bottleneck dimension
    ple_table_rank: int = 32    # Path B shared table rank
    ple_table_dim: int = 64     # Path B per-layer embedding dimension
    inject_point: str = "before"  # "before" or "after" block


class PLEModule(nn.Module):
    """Per-Layer Embeddings with ablatable dual-path design.

    Path A (context-aware): Projects the current hidden state through a
    per-layer gated bottleneck. Each layer learns a unique "lens" on the
    current representation.

    Path B (factored token-identity): A shared embedding table with per-layer
    learned mixing matrices. Each layer gets a specialized view of the
    original token identity without depending on the hidden state.

    Args:
        config: PLEConfig with mode, dimensions, and layer count.
    """

    def __init__(self, config: PLEConfig):
        super().__init__()
        self.config = config
        self.use_a = config.ple_mode in ("a", "a+b")
        self.use_b = config.ple_mode in ("b", "a+b")

        # Path A: context-aware projection
        if self.use_a:
            self.context_down = nn.ModuleList([
                nn.Linear(config.d_model, config.ple_dim, bias=False)
                for _ in range(config.n_layers)
            ])
            self.context_up = nn.ModuleList([
                nn.Linear(config.ple_dim, config.d_model, bias=False)
                for _ in range(config.n_layers)
            ])

        # Path B: factored token-identity table
        if self.use_b:
            self.shared_table = nn.Embedding(config.vocab_size, config.ple_table_rank)
            nn.init.normal_(self.shared_table.weight, std=0.02)

            self.layer_mixing = nn.Parameter(
                torch.randn(config.n_layers, config.ple_table_rank, config.ple_table_dim) * 0.02
            )
            self.table_up = nn.ModuleList([
                nn.Linear(config.ple_table_dim, config.d_model, bias=False)
                for _ in range(config.n_layers)
            ])

        # Per-layer output normalization
        self.ple_norm = nn.ModuleList([
            RMSNorm(config.d_model) for _ in range(config.n_layers)
        ])

        self._init_weights()

    def _init_weights(self):
        for name, p in self.named_parameters():
            if p.dim() >= 2 and "shared_table" not in name and "layer_mixing" not in name:
                nn.init.xavier_uniform_(p)

    def forward(
        self,
        h: torch.Tensor,
        token_ids: torch.Tensor,
        layer_idx: int,
    ) -> torch.Tensor:
        """Compute PLE contribution for one layer.

        Args:
            h: (B, T, d_model) current hidden state
            token_ids: (B, T) original input token IDs
            layer_idx: which layer is calling

        Returns:
            (B, T, d_model) to ADD to the residual stream
        """
        out = torch.zeros_like(h) if (self.use_a and self.use_b) else None

        # Path A: context-aware
        if self.use_a:
            ctx = F.gelu(self.context_down[layer_idx](h))
            a_out = self.context_up[layer_idx](ctx)
            out = a_out if out is None else out + a_out

        # Path B: factored token-identity
        if self.use_b:
            base = self.shared_table(token_ids)  # (B, T, table_rank)
            mixed = torch.einsum(
                "btr,rp->btp", base, self.layer_mixing[layer_idx]
            )  # (B, T, table_dim)
            b_out = self.table_up[layer_idx](mixed)
            out = b_out if out is None else out + b_out

        return self.ple_norm[layer_idx](out)

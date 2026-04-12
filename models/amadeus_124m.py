"""
AMADEUS-124M: Parameter-matched variant for fair comparison with LlamaModel 124.7M.

d_model=768, 12 layers, d_conv=512, d_mamba=256, ffn=2048, vocab=32000.
~124M parameters.
"""

import math
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.amadeus import (
    RMSNorm, SwiGLU, GatedConv, Mamba3SISO,
    AmadeusConfig, FiLMConditioner, _scan_dispatch, selective_scan_chunked,
)


class Amadeus124M(nn.Module):
    """AMADEUS scaled to ~124M params for fair comparison with LlamaModel."""

    def __init__(
        self,
        vocab_size: int = 32000,
        d_model: int = 768,
        n_layers: int = 12,
        d_conv: int = 512,
        d_mamba: int = 256,
        dstate: int = 64,
        n_ssm_heads: int = 4,
        ffn_inner: int = 2048,
        d_film: int = 64,
        film_start: int = 6,
        max_seq_len: int = 1024,
        conv_kernel: int = 3,
    ):
        super().__init__()
        assert d_conv + d_mamba == d_model

        self.film_start = film_start
        self.n_layers = n_layers

        cfg = AmadeusConfig(
            vocab_size=vocab_size, d_model=d_model, n_layers=n_layers,
            d_conv=d_conv, d_mamba=d_mamba, dstate=dstate,
            n_ssm_heads=n_ssm_heads, ffn_inner=ffn_inner,
            d_film=d_film, film_start=film_start,
            max_seq_len=max_seq_len, conv_kernel=conv_kernel,
        )

        from models.amadeus import ParallelHybridBlock

        self.tok_embeddings = nn.Embedding(vocab_size, d_model)
        self.layers = nn.ModuleList([
            ParallelHybridBlock(cfg) for _ in range(n_layers)
        ])
        self.film = FiLMConditioner(d_model, d_film, n_layers - film_start)
        self.norm = RMSNorm(d_model)
        self.output = nn.Linear(d_model, vocab_size, bias=False)
        self.output.weight = self.tok_embeddings.weight

        self._init_weights()
        n_params = sum(p.numel() for p in self.parameters())
        print(f"Amadeus124M: {n_params / 1e6:.1f}M parameters")

    def _init_weights(self):
        for name, p in self.named_parameters():
            if p.dim() >= 2:
                nn.init.xavier_uniform_(p)

    def forward(self, input_ids: torch.Tensor, targets=None) -> torch.Tensor:
        B, T = input_ids.shape
        h = self.tok_embeddings(input_ids)

        context = None
        for i, layer in enumerate(self.layers):
            if i == self.film_start:
                context = self.film.compute_context(h)
            if i >= self.film_start and context is not None:
                h = self.film.apply(h, context, i - self.film_start)
            h = layer(h)

        return self.output(self.norm(h))

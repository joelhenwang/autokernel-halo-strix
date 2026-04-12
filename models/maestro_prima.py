"""
MAESTRO-PRIMA: AMADEUS + Conductor Network (Component Dynamics).

The conductor reads the input once and controls HOW LOUD each component
plays at each layer. ~137K params overhead (0.06% of model).

Three signals per layer: conv_scale, mamba_scale, ffn_scale ∈ [0, 2].
Init at 1.0 = identity (untrained conductor = standard AMADEUS).

Usage:
    python -m halo_training --model models/maestro_prima.py --class-name MaestroPrima --dataset babylm
    python -m halo_training --model models/maestro_prima.py --class-name MaestroPrima --smoke
"""

import math
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

# Reuse AMADEUS components
from models.amadeus import (
    AmadeusConfig, RMSNorm, SwiGLU, GatedConv, Mamba3SISO,
    FiLMConditioner, _scan_dispatch, selective_scan_chunked,
)


class Conductor(nn.Module):
    """Tiny network that controls component volumes per layer.

    Reads the sequence once (mean-pool), produces 3 scaling signals
    per layer: conv_scale, mamba_scale, ffn_scale.
    """

    def __init__(self, d_model: int = 1024, d_cond: int = 128,
                 n_layers: int = 16, n_signals: int = 3):
        super().__init__()
        self.n_layers = n_layers
        self.n_signals = n_signals
        self.input_proj = nn.Linear(d_model, d_cond, bias=True)
        self.score_proj = nn.Linear(d_cond, n_layers * n_signals, bias=True)

        # Init so sigmoid(0) * 2 = 1.0 → identity at start
        nn.init.zeros_(self.input_proj.weight)
        nn.init.zeros_(self.input_proj.bias)
        nn.init.zeros_(self.score_proj.weight)
        nn.init.zeros_(self.score_proj.bias)

    def forward(self, embed: torch.Tensor) -> torch.Tensor:
        """Returns (B, n_layers, 3) scaling factors in [0, 2]."""
        summary = embed.mean(dim=1)                         # (B, d)
        cond = F.relu(self.input_proj(summary))             # (B, d_cond)
        score = torch.sigmoid(self.score_proj(cond)) * 2.0  # (B, n_layers * 3)
        return score.view(-1, self.n_layers, self.n_signals)


class ConductedBlock(nn.Module):
    """AMADEUS layer with conductor scaling on each component."""

    def __init__(self, cfg: AmadeusConfig):
        super().__init__()
        self.pre_norm = RMSNorm(cfg.d_model)
        self.conv = GatedConv(cfg.d_model, cfg.d_conv, cfg.conv_kernel)
        self.ssm = Mamba3SISO(cfg.d_model, cfg.d_mamba, cfg.dstate, cfg.n_ssm_heads)
        self.out_proj = nn.Linear(cfg.d_model, cfg.d_model, bias=False)
        self.ffn_norm = RMSNorm(cfg.d_model)
        self.ffn = SwiGLU(cfg.d_model, cfg.ffn_inner)

    def forward(self, x: torch.Tensor, score_i: torch.Tensor = None) -> torch.Tensor:
        x_norm = self.pre_norm(x)
        conv_out = self.conv(x_norm)
        ssm_out = self.ssm(x_norm)

        if score_i is not None:
            # score_i: (B, 3) — conv, mamba, ffn scales
            conv_out = conv_out * score_i[:, 0:1].unsqueeze(1)
            ssm_out = ssm_out * score_i[:, 1:2].unsqueeze(1)

        mixed = self.out_proj(torch.cat([conv_out, ssm_out], dim=-1))
        x = x + mixed

        ffn_out = self.ffn(self.ffn_norm(x))
        if score_i is not None:
            ffn_out = ffn_out * score_i[:, 2:3].unsqueeze(1)
        x = x + ffn_out
        return x


class MaestroPrima(nn.Module):
    """MAESTRO-PRIMA: AMADEUS + Conductor.

    16 Conducted Blocks + FiLM conditioning + Conductor network.
    ~241.7M parameters (241.6M AMADEUS + 137K Conductor).
    """

    def __init__(
        self,
        vocab_size: int = 50257,
        d_model: int = 1024,
        n_layers: int = 16,
        d_conv: int = 640,
        d_mamba: int = 384,
        dstate: int = 64,
        n_ssm_heads: int = 6,
        ffn_inner: int = 2560,
        d_film: int = 64,
        film_start: int = 8,
        d_cond: int = 128,
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

        self.tok_embeddings = nn.Embedding(vocab_size, d_model)
        self.layers = nn.ModuleList([ConductedBlock(cfg) for _ in range(n_layers)])
        self.film = FiLMConditioner(d_model, d_film, n_layers - film_start)
        self.conductor = Conductor(d_model, d_cond, n_layers)
        self.norm = RMSNorm(d_model)
        self.output = nn.Linear(d_model, vocab_size, bias=False)
        self.output.weight = self.tok_embeddings.weight

        self._init_weights()
        n_params = sum(p.numel() for p in self.parameters())
        print(f"MaestroPrima: {n_params / 1e6:.1f}M parameters")

    def _init_weights(self):
        for name, p in self.named_parameters():
            if 'conductor' in name:
                continue  # Conductor has its own init
            if p.dim() >= 2:
                nn.init.xavier_uniform_(p)

    def forward(self, input_ids: torch.Tensor, targets=None) -> torch.Tensor:
        B, T = input_ids.shape
        h = self.tok_embeddings(input_ids)

        # Conductor reads embedding once
        scores = self.conductor(h)  # (B, n_layers, 3)

        context = None
        for i, layer in enumerate(self.layers):
            if i == self.film_start:
                context = self.film.compute_context(h)
            if i >= self.film_start and context is not None:
                h = self.film.apply(h, context, i - self.film_start)
            h = layer(h, scores[:, i])

        return self.output(self.norm(h))

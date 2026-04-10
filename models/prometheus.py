"""
PROMETHEUS: Steal Fire from the Gods — 14 Griffin + 2 Attention + Momentum.

Hybrid architecture: mostly Griffin (element-wise, fast) with 2 attention
layers at strategic positions (4 and 12) for global context.

Attention backend selection (in order of preference):
  1. hybrid_attention (flash fwd + SDPA bwd, 8.9% faster than SDPA)
  2. PyTorch SDPA (F.scaled_dot_product_attention)

Usage:
    python -m halo_training --model models/prometheus.py --class-name Prometheus --dataset babylm
    python -m halo_training --model models/prometheus.py --class-name Prometheus --smoke
"""

import math
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

# Import shared components from tempest (identical Griffin/Conv/Momentum)
from models.tempest import (
    RMSNorm, SwiGLU, GatedConv, GriffinRecurrence, MomentumResidual,
)


# --------------------------------------------------------------------------
# Attention backend auto-detection
# --------------------------------------------------------------------------

_ATTN_BACKEND = None


def _detect_attn_backend():
    global _ATTN_BACKEND
    if _ATTN_BACKEND is not None:
        return _ATTN_BACKEND

    # Priority 1: hybrid_attention (flash fwd + SDPA bwd, 8.9% faster)
    try:
        from kernels.hip.hybrid_attention import hybrid_flash_sdpa_attention
        _ATTN_BACKEND = "hybrid"
        print("[Prometheus] Attention backend: hybrid (flash fwd + SDPA bwd)")
        return _ATTN_BACKEND
    except ImportError:
        pass

    # Fall back to SDPA
    _ATTN_BACKEND = "sdpa"
    print("[Prometheus] Attention backend: PyTorch SDPA")
    return _ATTN_BACKEND


def _attention_forward(q, k, v, causal=True):
    """Dispatch to best available attention backend.

    q, k, v are (B, H, T, D) from GQAAttentionLayer.
    hybrid_attention expects (B, T, H, D) — transpose as needed.
    """
    backend = _detect_attn_backend()
    if backend == "hybrid":
        from kernels.hip.hybrid_attention import hybrid_flash_sdpa_attention
        # hybrid expects (B, T, H, D); q/k/v are (B, H, T, D)
        out = hybrid_flash_sdpa_attention(
            q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2), causal=causal
        )
        return out.transpose(1, 2)  # back to (B, H, T, D)
    else:
        return F.scaled_dot_product_attention(q, k, v, is_causal=causal)


# --------------------------------------------------------------------------
# Architecture
# --------------------------------------------------------------------------

@dataclass
class PrometheusConfig:
    vocab_size: int = 50257
    d_model: int = 1024
    n_layers: int = 16
    d_conv: int = 640
    d_griffin: int = 384
    ffn_inner: int = 2560
    conv_kernel: int = 3
    n_attn_heads: int = 8
    n_kv_heads: int = 2
    head_dim: int = 128
    attn_layers: tuple = (3, 11)  # 0-indexed: layers 4 and 12
    momentum_beta_init: float = 0.5
    max_seq_len: int = 1024


def precompute_freqs_cis(dim: int, end: int, theta: float = 10000.0) -> torch.Tensor:
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2).float() / dim))
    t = torch.arange(end, dtype=torch.float32)
    freqs = torch.outer(t, freqs)
    return torch.polar(torch.ones_like(freqs), freqs)


def apply_rotary_emb(
    xq: torch.Tensor, xk: torch.Tensor, freqs_cis: torch.Tensor
):
    xq_ = torch.view_as_complex(xq.float().reshape(*xq.shape[:-1], -1, 2))
    xk_ = torch.view_as_complex(xk.float().reshape(*xk.shape[:-1], -1, 2))
    freqs = freqs_cis[None, :xq_.shape[1], None, :]
    xq_out = torch.view_as_real(xq_ * freqs).flatten(3)
    xk_out = torch.view_as_real(xk_ * freqs).flatten(3)
    return xq_out.type_as(xq), xk_out.type_as(xk)


class GQAAttentionLayer(nn.Module):
    """Grouped Query Attention with auto-detected backend."""

    def __init__(self, d_model: int, n_heads: int, n_kv_heads: int, head_dim: int):
        super().__init__()
        self.n_heads = n_heads
        self.n_kv_heads = n_kv_heads
        self.head_dim = head_dim
        self.n_rep = n_heads // n_kv_heads

        self.wq = nn.Linear(d_model, n_heads * head_dim, bias=False)
        self.wk = nn.Linear(d_model, n_kv_heads * head_dim, bias=False)
        self.wv = nn.Linear(d_model, n_kv_heads * head_dim, bias=False)
        self.wo = nn.Linear(n_heads * head_dim, d_model, bias=False)

    def forward(self, x: torch.Tensor, freqs_cis: torch.Tensor) -> torch.Tensor:
        B, T, _ = x.shape

        q = self.wq(x).view(B, T, self.n_heads, self.head_dim)
        k = self.wk(x).view(B, T, self.n_kv_heads, self.head_dim)
        v = self.wv(x).view(B, T, self.n_kv_heads, self.head_dim)

        q, k = apply_rotary_emb(q, k, freqs_cis)

        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        if self.n_rep > 1:
            k = k.repeat_interleave(self.n_rep, dim=1)
            v = v.repeat_interleave(self.n_rep, dim=1)

        y = _attention_forward(q, k, v, causal=True)
        y = y.transpose(1, 2).contiguous().view(B, T, -1)
        return self.wo(y)


class GriffinBlock(nn.Module):
    """Griffin hybrid block with momentum residual."""

    def __init__(self, cfg: PrometheusConfig):
        super().__init__()
        self.pre_norm = RMSNorm(cfg.d_model)
        self.conv = GatedConv(cfg.d_model, cfg.d_conv, cfg.conv_kernel)
        self.griffin = GriffinRecurrence(cfg.d_model, cfg.d_griffin)
        self.out_proj = nn.Linear(cfg.d_model, cfg.d_model, bias=False)
        self.momentum = MomentumResidual(cfg.momentum_beta_init)
        self.ffn_norm = RMSNorm(cfg.d_model)
        self.ffn = SwiGLU(cfg.d_model, cfg.ffn_inner)

    def forward(self, x, velocity, freqs_cis=None):
        x_norm = self.pre_norm(x)
        conv_out = self.conv(x_norm)
        griffin_out = self.griffin(x_norm)
        mixer_out = self.out_proj(torch.cat([conv_out, griffin_out], dim=-1))
        x, velocity = self.momentum(x, mixer_out, velocity)
        x = x + self.ffn(self.ffn_norm(x))
        return x, velocity


class AttentionBlock(nn.Module):
    """Attention block with momentum residual."""

    def __init__(self, cfg: PrometheusConfig):
        super().__init__()
        self.pre_norm = RMSNorm(cfg.d_model)
        self.attn = GQAAttentionLayer(
            cfg.d_model, cfg.n_attn_heads, cfg.n_kv_heads, cfg.head_dim
        )
        self.momentum = MomentumResidual(cfg.momentum_beta_init)
        self.ffn_norm = RMSNorm(cfg.d_model)
        self.ffn = SwiGLU(cfg.d_model, cfg.ffn_inner)

    def forward(self, x, velocity, freqs_cis):
        attn_out = self.attn(self.pre_norm(x), freqs_cis)
        x, velocity = self.momentum(x, attn_out, velocity)
        x = x + self.ffn(self.ffn_norm(x))
        return x, velocity


class Prometheus(nn.Module):
    """PROMETHEUS: 14 Griffin + 2 Attention + Residual Momentum.

    Layers 1-3:   Griffin hybrid (local)
    Layer 4:      GQA Attention (global context)
    Layers 5-11:  Griffin hybrid (local+medium)
    Layer 12:     GQA Attention (late refinement)
    Layers 13-16: Griffin hybrid (output prep)

    All layers use momentum residual on the mixer output.
    ~216M parameters.
    """

    def __init__(
        self,
        vocab_size: int = 50257,
        d_model: int = 1024,
        n_layers: int = 16,
        d_conv: int = 640,
        d_griffin: int = 384,
        ffn_inner: int = 2560,
        conv_kernel: int = 3,
        n_attn_heads: int = 8,
        n_kv_heads: int = 2,
        head_dim: int = 128,
        attn_layers: tuple = (3, 11),
        momentum_beta_init: float = 0.5,
        max_seq_len: int = 1024,
    ):
        super().__init__()
        assert d_conv + d_griffin == d_model

        cfg = PrometheusConfig(
            vocab_size=vocab_size, d_model=d_model, n_layers=n_layers,
            d_conv=d_conv, d_griffin=d_griffin, ffn_inner=ffn_inner,
            conv_kernel=conv_kernel, n_attn_heads=n_attn_heads,
            n_kv_heads=n_kv_heads, head_dim=head_dim,
            attn_layers=attn_layers, momentum_beta_init=momentum_beta_init,
            max_seq_len=max_seq_len,
        )

        self.attn_layer_indices = set(attn_layers)

        self.tok_embeddings = nn.Embedding(vocab_size, d_model)
        self.layers = nn.ModuleList()
        for i in range(n_layers):
            if i in self.attn_layer_indices:
                self.layers.append(AttentionBlock(cfg))
            else:
                self.layers.append(GriffinBlock(cfg))

        self.norm = RMSNorm(d_model)
        self.output = nn.Linear(d_model, vocab_size, bias=False)
        self.output.weight = self.tok_embeddings.weight

        self.register_buffer(
            "freqs_cis",
            precompute_freqs_cis(head_dim, max_seq_len * 2),
            persistent=False,
        )

        self._init_weights()
        n_params = sum(p.numel() for p in self.parameters())
        n_griffin = n_layers - len(attn_layers)
        print(f"Prometheus: {n_params / 1e6:.1f}M params "
              f"({n_griffin} Griffin + {len(attn_layers)} attention layers)")

    def _init_weights(self):
        for name, p in self.named_parameters():
            if p.dim() >= 2:
                nn.init.xavier_uniform_(p)

    def forward(self, input_ids: torch.Tensor, targets=None) -> torch.Tensor:
        B, T = input_ids.shape
        h = self.tok_embeddings(input_ids)
        velocity = torch.zeros_like(h)
        freqs = self.freqs_cis[:T]

        for i, layer in enumerate(self.layers):
            if i in self.attn_layer_indices:
                h, velocity = layer(h, velocity, freqs)
            else:
                h, velocity = layer(h, velocity)

        logits = self.output(self.norm(h))
        return logits

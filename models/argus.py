"""
ARGUS: Adaptive Retrieval-Guided Unified System.

Novel architecture combining 6 mechanisms for maximum throughput + adaptive learning:
  1. ShortConv (GatedConv) — cheap local pattern mixing (12 layers)
  2. Sparse GQA with RoPE — global context (4 layers, every 4th)
  3. In-Place TTT — weight adaptation via cumsum on FFN down_proj (ByteDance approach)
  4. Engram — N-gram hash table pattern retrieval
  5. Momentum Residual — inertia in residual stream
  6. MatFormer — nested submodel training for elastic deployment

Inspired by LFM2.5-350M (3:1 ShortConv/GQA ratio) + In-Place TTT (ByteDance/arXiv 2604.06169).

Usage:
    python -m halo_training --model models/argus.py --class-name Argus --smoke
    python -m halo_training --model models/argus.py --class-name Argus --dataset babylm --compile --optimize-kernels
"""

import math
from dataclasses import dataclass, field
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.amadeus import RMSNorm, GatedConv
from models.engram import EngramLayer
from models.matformer import MatFormerSwiGLU, MatFormerConfig


# ---------------------------------------------------------------------------
# Copied from llama_7b.py (avoids RMSNorm name conflict)
# ---------------------------------------------------------------------------

def precompute_freqs_cis(dim: int, end: int, theta: float = 10000.0) -> torch.Tensor:
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2).float() / dim))
    t = torch.arange(end, dtype=torch.float32)
    freqs = torch.outer(t, freqs)
    return torch.polar(torch.ones_like(freqs), freqs)


def apply_rotary_emb(xq, xk, freqs_cis):
    xq_ = torch.view_as_complex(xq.float().reshape(*xq.shape[:-1], -1, 2))
    xk_ = torch.view_as_complex(xk.float().reshape(*xk.shape[:-1], -1, 2))
    freqs = freqs_cis[None, :xq_.shape[1], None, :]
    xq_out = torch.view_as_real(xq_ * freqs).flatten(3)
    xk_out = torch.view_as_real(xk_ * freqs).flatten(3)
    return xq_out.type_as(xq), xk_out.type_as(xk)


# ---------------------------------------------------------------------------
# Copied from tempest.py
# ---------------------------------------------------------------------------

class MomentumResidual(nn.Module):
    """velocity = beta * velocity + layer_output; h = h + velocity."""

    def __init__(self, beta_init: float = 0.5):
        super().__init__()
        self.log_beta = nn.Parameter(
            torch.tensor(math.log(beta_init / (1 - beta_init)))
        )

    @property
    def beta(self):
        return torch.sigmoid(self.log_beta)

    def forward(self, h, layer_output, velocity):
        velocity = self.beta * velocity + layer_output
        return h + velocity, velocity


# ---------------------------------------------------------------------------
# GQA Attention (from llama_7b.py)
# ---------------------------------------------------------------------------

class Attention(nn.Module):
    """Grouped-Query Attention with RoPE."""

    def __init__(self, dim: int, n_heads: int, n_kv_heads: int):
        super().__init__()
        self.n_heads = n_heads
        self.n_kv_heads = n_kv_heads
        self.head_dim = dim // n_heads
        self.n_rep = n_heads // n_kv_heads

        self.wq = nn.Linear(dim, n_heads * self.head_dim, bias=False)
        self.wk = nn.Linear(dim, n_kv_heads * self.head_dim, bias=False)
        self.wv = nn.Linear(dim, n_kv_heads * self.head_dim, bias=False)
        self.wo = nn.Linear(n_heads * self.head_dim, dim, bias=False)

    def forward(self, x: torch.Tensor, freqs_cis: torch.Tensor) -> torch.Tensor:
        B, T, _ = x.shape
        q = self.wq(x).view(B, T, self.n_heads, self.head_dim)
        k = self.wk(x).view(B, T, self.n_kv_heads, self.head_dim)
        v = self.wv(x).view(B, T, self.n_kv_heads, self.head_dim)
        q, k = apply_rotary_emb(q, k, freqs_cis)
        q, k, v = q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)
        if self.n_rep > 1:
            k = k.repeat_interleave(self.n_rep, dim=1)
            v = v.repeat_interleave(self.n_rep, dim=1)
        y = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        return self.wo(y.transpose(1, 2).contiguous().view(B, T, -1))


# ---------------------------------------------------------------------------
# In-Place TTT SwiGLU (ByteDance cumsum approach)
# ---------------------------------------------------------------------------

class TTTSwiGLU(nn.Module):
    """SwiGLU with In-Place TTT on down_proj via chunked cumsum.

    The down_proj weight adapts per-chunk: chunks are processed in parallel
    via cumulative sum of outer-product updates. Each chunk sees the weight
    updated by all *previous* chunks (causal).

    Based on ByteDance In-Place TTT (arXiv 2604.06169).
    """

    _skip_autokernel = True  # prevent FusedSwiGLU from replacing this

    def __init__(self, d_model: int, ffn_inner: int,
                 ttt_chunk: int = 256, ttt_lr_init: float = 0.01,
                 ttt_conv_kernel: int = 5):
        super().__init__()
        self.d_model = d_model
        self.ffn_inner = ffn_inner
        self.ttt_chunk = ttt_chunk

        # Standard SwiGLU weights
        self.w_gate_up = nn.Linear(d_model, 2 * ffn_inner, bias=False)
        self.w_down = nn.Linear(ffn_inner, d_model, bias=False)

        # TTT components
        self.ttt_proj = nn.Linear(d_model, d_model, bias=False)
        self.ttt_conv = nn.Conv1d(
            d_model, d_model, kernel_size=ttt_conv_kernel,
            padding=ttt_conv_kernel - 1,  # causal: left-pad only
            groups=d_model, bias=True,
        )
        # Non-learnable TTT LR — 0.01 for from-scratch, 0.3 for fine-tuning
        self.register_buffer('ttt_lr', torch.tensor(ttt_lr_init))

    def _init_ttt_weights(self):
        """Zero-init conv, diagonal-init projection (near-identity start)."""
        nn.init.zeros_(self.ttt_conv.weight)
        nn.init.zeros_(self.ttt_conv.bias)
        # Diagonal init for ttt_proj: starts as scaled identity
        with torch.no_grad():
            nn.init.zeros_(self.ttt_proj.weight)
            diag = torch.randn(self.d_model) * 0.02
            self.ttt_proj.weight.diagonal().copy_(diag)

    def _pad_and_chunk(self, x, chunk_size):
        """Pad to multiple of chunk_size and reshape to (B, n_chunks, C, D)."""
        B, T, D = x.shape
        pad_len = (chunk_size - T % chunk_size) % chunk_size
        if pad_len > 0:
            x = F.pad(x, (0, 0, 0, pad_len))
        return x.reshape(B, -1, chunk_size, D)

    def forward(self, x, ttt_target=None):
        """
        Args:
            x: (B, T, d_model) — input to FFN
            ttt_target: (B, T, d_model) — target signal for TTT (post-attention hidden states)
        """
        B, T, _ = x.shape

        # Standard SwiGLU forward
        gate, up = self.w_gate_up(x).chunk(2, dim=-1)
        h = F.silu(gate) * up  # (B, T, ffn_inner)

        # Non-TTT path
        if ttt_target is None:
            return self.w_down(h)

        # === In-Place TTT path ===

        # Smooth target via causal depthwise conv1d
        t = self.ttt_conv(ttt_target.transpose(1, 2))[:, :, :T].transpose(1, 2)

        # Pad and chunk
        C = self.ttt_chunk
        h_chunked = self._pad_and_chunk(h, C)      # (B, nc, C, ffn_inner)
        t_chunked = self._pad_and_chunk(t, C)       # (B, nc, C, d_model)
        nc = h_chunked.shape[1]

        # Compute weight deltas from all chunks except the last
        # delta[i] = outer_product(h[i], proj(t[i])) — gradient of reconstruction
        if nc > 1:
            t_proj = F.linear(t_chunked[:, :-1], self.ttt_proj.weight)  # (B, nc-1, C, d_model)
            # Batched outer product: (B, nc-1, d_model, ffn_inner)
            d_down = torch.einsum('btch,btcd->btdh', h_chunked[:, :-1], t_proj)
        else:
            d_down = h_chunked.new_zeros(B, 0, self.d_model, self.ffn_inner)

        # Prepend original w_down weight, scale deltas by lr
        w_orig = self.w_down.weight.unsqueeze(0).expand(B, -1, -1).unsqueeze(1)  # (B,1,d,h)
        d_down_scaled = torch.cat([w_orig, d_down * self.ttt_lr], dim=1)  # (B, nc, d, h)

        # Cumsum → adapted weight per chunk (parallel!)
        w_adapted = d_down_scaled.cumsum(dim=1)  # (B, nc, d_model, ffn_inner)

        # Apply adapted weight to each chunk: output = h @ W^T
        output = torch.einsum('btdh,btch->btcd', w_adapted, h_chunked)  # (B, nc, C, d_model)

        # Reshape and trim padding
        return output.reshape(B, -1, self.d_model)[:, :T]


# ---------------------------------------------------------------------------
# Block types
# ---------------------------------------------------------------------------

class ShortConvBlock(nn.Module):
    """ShortConv mixer + MomentumResidual + MatFormerSwiGLU FFN."""

    def __init__(self, d_model: int, d_conv: int, ffn_inner: int,
                 conv_kernel: int = 3, momentum_beta: float = 0.5,
                 mf_config: Optional[MatFormerConfig] = None):
        super().__init__()
        self.pre_norm = RMSNorm(d_model)
        self.conv = GatedConv(d_model, d_conv, conv_kernel)
        self.out_proj = nn.Linear(d_conv, d_model, bias=False)
        self.momentum = MomentumResidual(momentum_beta)
        self.ffn_norm = RMSNorm(d_model)
        if mf_config is not None:
            self.ffn = MatFormerSwiGLU(d_model, ffn_inner, mf_config)
        else:
            from models.amadeus import SwiGLU
            self.ffn = SwiGLU(d_model, ffn_inner)

    def forward(self, x, velocity):
        mixer_out = self.out_proj(self.conv(self.pre_norm(x)))
        x, velocity = self.momentum(x, mixer_out, velocity)
        x = x + self.ffn(self.ffn_norm(x))
        return x, velocity


class GQABlock(nn.Module):
    """GQA Attention + MomentumResidual + TTTSwiGLU FFN."""

    def __init__(self, d_model: int, ffn_inner: int,
                 n_heads: int = 12, n_kv_heads: int = 4,
                 momentum_beta: float = 0.5,
                 ttt_chunk: int = 256, ttt_lr_init: float = 0.01,
                 ttt_conv_kernel: int = 5):
        super().__init__()
        self.pre_norm = RMSNorm(d_model)
        self.attn = Attention(d_model, n_heads, n_kv_heads)
        self.momentum = MomentumResidual(momentum_beta)
        self.ffn_norm = RMSNorm(d_model)
        self.ffn = TTTSwiGLU(d_model, ffn_inner, ttt_chunk, ttt_lr_init, ttt_conv_kernel)

    def forward(self, x, velocity, freqs_cis, ttt_target):
        attn_out = self.attn(self.pre_norm(x), freqs_cis)
        x, velocity = self.momentum(x, attn_out, velocity)
        ffn_in = self.ffn_norm(x)
        x = x + self.ffn(ffn_in, ttt_target=ttt_target)
        return x, velocity


# ---------------------------------------------------------------------------
# ARGUS Model
# ---------------------------------------------------------------------------

@dataclass
class ArgusConfig:
    vocab_size: int = 50257
    d_model: int = 768
    n_layers: int = 16
    d_conv: int = 768
    ffn_inner: int = 2048
    conv_kernel: int = 3
    # GQA
    n_heads: int = 12
    n_kv_heads: int = 4
    gqa_layers: tuple = (3, 7, 11, 15)
    # TTT
    ttt_chunk: int = 256
    ttt_lr_init: float = 0.01  # conservative for from-scratch training
    ttt_conv_kernel: int = 5
    # Engram
    use_engram: bool = True
    engram_layer: int = 1
    engram_d: int = 512
    engram_n_hash_heads: int = 2
    engram_table_size: int = 8192
    engram_ngram_sizes: tuple = (2, 3)
    # MatFormer
    matformer_granularities: tuple = (0.25, 0.5, 0.75, 1.0)
    # General
    momentum_beta_init: float = 0.5
    max_seq_len: int = 1024


class Argus(nn.Module):
    """ARGUS: 3:1 ShortConv/GQA + In-Place TTT + Engram + Momentum + MatFormer.

    ~170M parameters. Targets 50% MFU / 20K+ tok/s on gfx1151.
    """

    def __init__(
        self,
        vocab_size: int = 50257,
        d_model: int = 768,
        n_layers: int = 16,
        d_conv: int = 768,
        ffn_inner: int = 2048,
        conv_kernel: int = 3,
        n_heads: int = 12,
        n_kv_heads: int = 4,
        gqa_layers: tuple = (3, 7, 11, 15),
        ttt_chunk: int = 256,
        ttt_lr_init: float = 0.01,
        ttt_conv_kernel: int = 5,
        use_engram: bool = True,
        engram_layer: int = 1,
        engram_d: int = 512,
        engram_n_hash_heads: int = 2,
        engram_table_size: int = 8192,
        engram_ngram_sizes: tuple = (2, 3),
        matformer_granularities: tuple = (0.25, 0.5, 0.75, 1.0),
        momentum_beta_init: float = 0.5,
        max_seq_len: int = 1024,
    ):
        super().__init__()
        self.d_model = d_model
        self.gqa_set = set(gqa_layers)
        self.engram_layer = engram_layer if use_engram else -1

        mf_config = MatFormerConfig(granularities=matformer_granularities)

        # Token embedding + tied output
        self.tok_embeddings = nn.Embedding(vocab_size, d_model)
        self.norm = RMSNorm(d_model)
        self.output = nn.Linear(d_model, vocab_size, bias=False)
        self.output.weight = self.tok_embeddings.weight

        # RoPE frequencies (for GQA layers)
        head_dim = d_model // n_heads
        self.register_buffer(
            "freqs_cis",
            precompute_freqs_cis(head_dim, max_seq_len * 2),
            persistent=False,
        )

        # Build layers
        self.layers = nn.ModuleList()
        for i in range(n_layers):
            if i in self.gqa_set:
                self.layers.append(GQABlock(
                    d_model, ffn_inner, n_heads, n_kv_heads,
                    momentum_beta_init, ttt_chunk, ttt_lr_init, ttt_conv_kernel,
                ))
            else:
                self.layers.append(ShortConvBlock(
                    d_model, d_conv, ffn_inner, conv_kernel,
                    momentum_beta_init, mf_config,
                ))

        # Engram (N-gram hash table pattern retrieval)
        if use_engram:
            self.engram = EngramLayer(
                d_model=d_model, d_engram=engram_d,
                n_hash_heads=engram_n_hash_heads,
                ngram_sizes=list(engram_ngram_sizes),
                vocab_size=vocab_size, table_size=engram_table_size,
                layer_id=0,
            )
            # Replace nn.RMSNorm with our RMSNorm (autokernel-compatible)
            if hasattr(self.engram, 'query_norm'):
                self.engram.query_norm = RMSNorm(d_model)
        else:
            self.engram = None

        self._init_weights()
        n_params = sum(p.numel() for p in self.parameters())
        print(f"Argus: {n_params / 1e6:.1f}M parameters")

    def _init_weights(self):
        for name, p in self.named_parameters():
            if p.dim() >= 2:
                nn.init.xavier_uniform_(p)
        # Special TTT init: zero conv, diagonal proj
        for layer in self.layers:
            if isinstance(layer, GQABlock):
                layer.ffn._init_ttt_weights()

    def forward(self, input_ids: torch.Tensor, targets=None) -> torch.Tensor:
        B, T = input_ids.shape
        h = self.tok_embeddings(input_ids)
        velocity = torch.zeros_like(h)
        freqs_cis = self.freqs_cis[:T]

        for i, layer in enumerate(self.layers):
            # Engram augmentation at designated layer
            if i == self.engram_layer and self.engram is not None:
                h = h + self.engram(h, input_ids)

            if isinstance(layer, GQABlock):
                h, velocity = layer(h, velocity, freqs_cis, ttt_target=h)
            else:
                h, velocity = layer(h, velocity)

        return self.output(self.norm(h))


# ---------------------------------------------------------------------------
# Mini config for smoke testing
# ---------------------------------------------------------------------------

class ArgusMini(Argus):
    """Tiny ARGUS for smoke testing (~3M params)."""

    def __init__(self):
        super().__init__(
            vocab_size=1000,
            d_model=128,
            n_layers=4,
            d_conv=128,
            ffn_inner=256,
            conv_kernel=3,
            n_heads=4,
            n_kv_heads=2,
            gqa_layers=(3,),
            ttt_chunk=64,
            ttt_lr_init=0.3,
            ttt_conv_kernel=5,
            use_engram=False,  # skip engram for mini
            matformer_granularities=(0.5, 1.0),
            max_seq_len=1024,  # must cover smoke test block_size=512
        )

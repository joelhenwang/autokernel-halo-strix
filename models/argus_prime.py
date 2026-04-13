"""
ARGUS-PRIME: Streamlined LFM2.5 skeleton + surgical TTT.

Evolution from ARGUS: strip Engram/MatFormer, align to LFM2's actual 10:6
ShortConv/GQA ratio, bigger FFN (3.7x), QK-Norm, hybrid attention.
TTT placed surgically: 1 layer at position 16 (the "sniper").

6 ablation variants:
  B0: 1 TTT single-step (sniper, no support)
  B1: 1 TTT single-step + FiLM (sniper + air support)
  B2: 1 TTT multi-step 3x (deep sniper)
  B3: 1 TTT multi-step 3x + FiLM (THE HYPOTHESIS)
  B4: 2 TTT single-step at layers 8,16 (bracket)
  B5: 2 TTT single-step + FiLM (bracket + air support)

Usage:
    python -m halo_training --model models/argus_prime.py --class-name ArgusPrime --smoke
    python -m halo_training --model models/argus_prime.py --class-name ArgusPrimeMultiFiLM --dataset babylm --compile --optimize-kernels
"""

import math
from dataclasses import dataclass
from typing import Optional, Set

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.amadeus import RMSNorm, SwiGLU, GatedConv, FiLMConditioner
from models.argus import TTTSwiGLU, precompute_freqs_cis, apply_rotary_emb

try:
    from kernels.hip.hybrid_attention import hybrid_flash_sdpa_attention
    _HAS_HYBRID_ATTN = True
except ImportError:
    _HAS_HYBRID_ATTN = False

try:
    from kernels.hip.fused_gated_conv import kernel_fn as fused_gated_conv_fn
    _HAS_FUSED_GATED_CONV = True
except Exception:
    _HAS_FUSED_GATED_CONV = False

try:
    from causal_conv1d import causal_conv1d_fn
except ImportError:
    pass


# ---------------------------------------------------------------------------
# GQA Attention with QK-Norm (LFM2 technique)
# ---------------------------------------------------------------------------

class Attention(nn.Module):
    """GQA with RoPE + QK-Norm (L2-normalize Q,K before attention)."""

    def __init__(self, dim: int, n_heads: int, n_kv_heads: int, qk_norm: bool = True):
        super().__init__()
        self.n_heads = n_heads
        self.n_kv_heads = n_kv_heads
        self.head_dim = dim // n_heads
        self.n_rep = n_heads // n_kv_heads
        self.qk_norm = qk_norm

        self.wq = nn.Linear(dim, n_heads * self.head_dim, bias=False)
        self.wk = nn.Linear(dim, n_kv_heads * self.head_dim, bias=False)
        self.wv = nn.Linear(dim, n_kv_heads * self.head_dim, bias=False)
        self.wo = nn.Linear(n_heads * self.head_dim, dim, bias=False)

        if qk_norm:
            self.q_scale = nn.Parameter(torch.ones(n_heads, 1, 1) * math.sqrt(self.head_dim))
            self.k_scale = nn.Parameter(torch.ones(n_kv_heads, 1, 1) * math.sqrt(self.head_dim))

    def forward(self, x: torch.Tensor, freqs_cis: torch.Tensor) -> torch.Tensor:
        B, T, _ = x.shape
        q = self.wq(x).view(B, T, self.n_heads, self.head_dim)
        k = self.wk(x).view(B, T, self.n_kv_heads, self.head_dim)
        v = self.wv(x).view(B, T, self.n_kv_heads, self.head_dim)
        q, k = apply_rotary_emb(q, k, freqs_cis)
        q, k, v = q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)

        # QK-Norm: L2-normalize then scale (LFM2 technique)
        if self.qk_norm:
            q = F.normalize(q, dim=-1) * self.q_scale
            k = F.normalize(k, dim=-1) * self.k_scale

        if self.n_rep > 1:
            k = k.repeat_interleave(self.n_rep, dim=1)
            v = v.repeat_interleave(self.n_rep, dim=1)

        # Hybrid attention: flash_attn forward + SDPA backward (8.9% faster)
        if _HAS_HYBRID_ATTN and q.dtype == torch.float16:
            y = hybrid_flash_sdpa_attention(
                q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2), causal=True
            ).transpose(1, 2)
        else:
            y = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        return self.wo(y.transpose(1, 2).contiguous().view(B, T, -1))


# ---------------------------------------------------------------------------
# Multi-Step TTT SwiGLU (3 steps unrolled — no for loop)
# ---------------------------------------------------------------------------

class MultiStepTTTSwiGLU(TTTSwiGLU):
    """TTTSwiGLU with 3 unrolled gradient steps per chunk.

    3 steps x 1 layer ~ cost of 1.5 single-step layers.
    Steps are unrolled (no for loop) for torch.compile safety.
    """

    _skip_autokernel = True

    def __init__(self, d_model: int, ffn_inner: int,
                 ttt_chunk: int = 512, ttt_lr_init: float = 0.01,
                 ttt_conv_kernel: int = 5):
        super().__init__(d_model, ffn_inner, ttt_chunk, ttt_lr_init, ttt_conv_kernel)

    def _single_chunk_multistep(self, h_c, t_c, w_current):
        """3 unrolled gradient steps on one chunk. Returns (output, w_updated)."""
        lr = self.ttt_lr / 3.0  # divide by n_steps to keep total update magnitude stable

        # Step 1
        out1 = h_c @ w_current.T
        t_proj1 = F.linear(t_c, self.ttt_proj.weight)
        grad1 = torch.einsum('bch,bcd->dh', h_c, t_proj1 - out1) / (h_c.shape[0] * h_c.shape[1])
        w1 = w_current + lr * grad1.detach()

        # Step 2
        out2 = h_c @ w1.T
        t_proj2 = F.linear(t_c, self.ttt_proj.weight)
        grad2 = torch.einsum('bch,bcd->dh', h_c, t_proj2 - out2) / (h_c.shape[0] * h_c.shape[1])
        w2 = w1 + lr * grad2.detach()

        # Step 3
        out3 = h_c @ w2.T
        t_proj3 = F.linear(t_c, self.ttt_proj.weight)
        grad3 = torch.einsum('bch,bcd->dh', h_c, t_proj3 - out3) / (h_c.shape[0] * h_c.shape[1])
        w3 = w2 + lr * grad3.detach()

        # Final output with fully adapted weight
        return h_c @ w3.T, w3

    def forward(self, x, ttt_target=None):
        B, T, _ = x.shape
        gate, up = self.w_gate_up(x).chunk(2, dim=-1)
        h = F.silu(gate) * up

        if ttt_target is None:
            return self.w_down(h)

        # Smooth target
        t = self.ttt_conv(ttt_target.transpose(1, 2))[:, :, :T].transpose(1, 2)

        C = self.ttt_chunk
        h_chunked = self._pad_and_chunk(h, C)
        t_chunked = self._pad_and_chunk(t, C)
        nc = h_chunked.shape[1]

        # Sequential multi-step across chunks (each chunk adapts from previous)
        w_current = self.w_down.weight.clone()
        outputs = []
        for chunk_idx in range(nc):
            out_c, w_current = self._single_chunk_multistep(
                h_chunked[:, chunk_idx], t_chunked[:, chunk_idx], w_current
            )
            outputs.append(out_c)

        return torch.stack(outputs, dim=1).reshape(B, -1, self.d_model)[:, :T]


# ---------------------------------------------------------------------------
# Block types
# ---------------------------------------------------------------------------

class ShortConvBlock(nn.Module):
    """GatedConv mixer + inlined momentum + SwiGLU FFN. Compile-friendly."""

    def __init__(self, d_model: int, d_conv: int, ffn_inner: int,
                 conv_kernel: int = 3, momentum_beta: float = 0.5):
        super().__init__()
        self.pre_norm = RMSNorm(d_model)
        self.conv = GatedConv(d_model, d_conv, conv_kernel)
        self.out_proj = nn.Linear(d_conv, d_model, bias=False)
        self.log_beta = nn.Parameter(
            torch.tensor(math.log(momentum_beta / (1 - momentum_beta)))
        )
        self.ffn_norm = RMSNorm(d_model)
        self.ffn = SwiGLU(d_model, ffn_inner)

    def forward(self, x, velocity):
        # GatedConv with optional fused forward kernel
        normed = self.pre_norm(x)
        if _HAS_FUSED_GATED_CONV and normed.dtype == torch.float16:
            proj_out = self.conv.proj(normed)
            conv_out = fused_gated_conv_fn(
                proj_out, self.conv.conv_weight, self.conv.conv_bias, x.shape[1]
            )
        else:
            conv_out = self.conv(normed)
        mixer_out = self.out_proj(conv_out)

        # Inlined momentum (compile-friendly)
        beta = torch.sigmoid(self.log_beta)
        velocity = beta * velocity + mixer_out

        # Inlined residual + RMSNorm (Inductor fuses element-wise)
        x = x + velocity
        rms = torch.rsqrt(x.float().pow(2).mean(-1, keepdim=True) + 1e-6)
        normed = (x.float() * rms).to(x.dtype) * self.ffn_norm.weight

        x = x + self.ffn(normed)
        return x, velocity


class GQABlock(nn.Module):
    """GQA Attention + QK-Norm + inlined momentum + SwiGLU/TTT FFN."""

    def __init__(self, d_model: int, ffn_inner: int,
                 n_heads: int = 12, n_kv_heads: int = 4,
                 momentum_beta: float = 0.5,
                 ttt_mode: str = "none",
                 ttt_chunk: int = 512, ttt_lr_init: float = 0.01):
        super().__init__()
        self.pre_norm = RMSNorm(d_model)
        self.attn = Attention(d_model, n_heads, n_kv_heads, qk_norm=True)
        self.log_beta = nn.Parameter(
            torch.tensor(math.log(momentum_beta / (1 - momentum_beta)))
        )
        self.ffn_norm = RMSNorm(d_model)
        self.ttt_mode = ttt_mode

        if ttt_mode == "single":
            self.ffn = TTTSwiGLU(d_model, ffn_inner, ttt_chunk, ttt_lr_init)
        elif ttt_mode == "multi":
            self.ffn = MultiStepTTTSwiGLU(d_model, ffn_inner, ttt_chunk, ttt_lr_init)
        else:
            self.ffn = SwiGLU(d_model, ffn_inner)

    def forward(self, x, velocity, freqs_cis, ttt_target=None):
        attn_out = self.attn(self.pre_norm(x), freqs_cis)

        # Inlined momentum
        beta = torch.sigmoid(self.log_beta)
        velocity = beta * velocity + attn_out

        # Inlined residual + RMSNorm
        x = x + velocity
        rms = torch.rsqrt(x.float().pow(2).mean(-1, keepdim=True) + 1e-6)
        normed = (x.float() * rms).to(x.dtype) * self.ffn_norm.weight

        if self.ttt_mode != "none" and ttt_target is not None:
            x = x + self.ffn(normed, ttt_target=ttt_target)
        else:
            x = x + self.ffn(normed)

        return x, velocity


# ---------------------------------------------------------------------------
# ARGUS-PRIME Model
# ---------------------------------------------------------------------------

class ArgusPrimeBase(nn.Module):
    """Base ARGUS-PRIME: LFM2.5-aligned 10:6 ShortConv/GQA + surgical TTT.

    Subclass and set ttt_layers, ttt_mode, use_film to create variants.
    """

    def __init__(
        self,
        vocab_size: int = 50257,
        d_model: int = 768,
        n_layers: int = 16,
        d_conv: int = 512,  # smaller conv = less memory-bound ops, more time in FFN matmuls
        ffn_inner: int = 2816,  # 3.7x (LFM2-aligned)
        conv_kernel: int = 3,
        n_heads: int = 12,
        n_kv_heads: int = 4,  # GQA 3:1 (LFM2 uses 16/8 but 12/8 isn't integer)
        gqa_layers: tuple = (2, 5, 7, 9, 12, 15),  # 0-indexed: 6 GQA layers
        ttt_layers: Set[int] = None,  # which GQA layers get TTT
        ttt_mode: str = "single",  # "single" or "multi"
        ttt_chunk: int = 512,
        ttt_lr_init: float = 0.01,
        use_film: bool = False,
        d_film: int = 64,
        film_start: int = 7,  # 0-indexed layer where FiLM context is computed
        momentum_beta_init: float = 0.5,
        max_seq_len: int = 1024,
    ):
        super().__init__()
        self.d_model = d_model
        self.gqa_set = set(gqa_layers)
        self.ttt_layers = ttt_layers or set()
        self.use_film = use_film
        self.film_start = film_start

        # Embedding + tied output
        self.tok_embeddings = nn.Embedding(vocab_size, d_model)
        self.norm = RMSNorm(d_model)
        self.output = nn.Linear(d_model, vocab_size, bias=False)
        self.output.weight = self.tok_embeddings.weight

        # RoPE
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
                layer_ttt = ttt_mode if i in self.ttt_layers else "none"
                self.layers.append(GQABlock(
                    d_model, ffn_inner, n_heads, n_kv_heads,
                    momentum_beta_init, layer_ttt, ttt_chunk, ttt_lr_init,
                ))
            else:
                self.layers.append(ShortConvBlock(
                    d_model, d_conv, ffn_inner, conv_kernel, momentum_beta_init,
                ))

        # FiLM conditioning (from AMADEUS)
        if use_film:
            n_conditioned = n_layers - film_start
            self.film = FiLMConditioner(d_model, d_film, n_conditioned)
        else:
            self.film = None

        self._init_weights()
        n_params = sum(p.numel() for p in self.parameters())
        print(f"{self.__class__.__name__}: {n_params / 1e6:.1f}M parameters")

    def _init_weights(self):
        for name, p in self.named_parameters():
            if p.dim() >= 2:
                nn.init.xavier_uniform_(p)
        # TTT-specific init
        for layer in self.layers:
            if isinstance(layer, GQABlock) and hasattr(layer.ffn, '_init_ttt_weights'):
                layer.ffn._init_ttt_weights()

    def forward(self, input_ids: torch.Tensor, targets=None) -> torch.Tensor:
        B, T = input_ids.shape
        h = self.tok_embeddings(input_ids)
        velocity = torch.zeros_like(h)
        freqs_cis = self.freqs_cis[:T]

        context = None
        for i, layer in enumerate(self.layers):
            # FiLM: compute context at film_start
            if self.use_film and self.film is not None and i == self.film_start:
                context = self.film.compute_context(h)

            if isinstance(layer, GQABlock):
                ttt_target = h if layer.ttt_mode != "none" else None
                h, velocity = layer(h, velocity, freqs_cis, ttt_target=ttt_target)
            else:
                h, velocity = layer(h, velocity)

            # FiLM: apply AFTER layer forward (safer — avoids amplifying FFN gradients)
            if self.use_film and context is not None and i >= self.film_start:
                h = self.film.apply(h, context, i - self.film_start)

        return self.output(self.norm(h))


# ---------------------------------------------------------------------------
# 6 Ablation Variants
# ---------------------------------------------------------------------------

# GQA positions (0-indexed): 2, 5, 7, 9, 12, 15
# (1-indexed: layers 3, 6, 8, 10, 13, 16)

class ArgusPrime(ArgusPrimeBase):
    """B0: 1 TTT single-step at layer 16 (sniper, no support)."""
    def __init__(self, **kw):
        super().__init__(ttt_layers={15}, ttt_mode="single", use_film=False, **kw)


class ArgusPrimeFiLM(ArgusPrimeBase):
    """B1: 1 TTT single-step + FiLM (sniper + air support)."""
    def __init__(self, **kw):
        super().__init__(ttt_layers={15}, ttt_mode="single", use_film=True, **kw)


class ArgusPrimeMulti(ArgusPrimeBase):
    """B2: 1 TTT multi-step 3x at layer 16 (deep sniper)."""
    def __init__(self, **kw):
        super().__init__(ttt_layers={15}, ttt_mode="multi", use_film=False, **kw)


class ArgusPrimeMultiFiLM(ArgusPrimeBase):
    """B3: 1 TTT multi-step 3x + FiLM (THE HYPOTHESIS)."""
    def __init__(self, **kw):
        super().__init__(ttt_layers={15}, ttt_mode="multi", use_film=True, **kw)


class ArgusPrimeTTT2(ArgusPrimeBase):
    """B4: 2 TTT single-step at layers 8,16 (bracket)."""
    def __init__(self, **kw):
        super().__init__(ttt_layers={7, 15}, ttt_mode="single", use_film=False, **kw)


class ArgusPrimeTTT2FiLM(ArgusPrimeBase):
    """B5: 2 TTT single-step + FiLM (bracket + air support)."""
    def __init__(self, **kw):
        super().__init__(ttt_layers={7, 15}, ttt_mode="single", use_film=True, **kw)


# ---------------------------------------------------------------------------
# Throughput optimization variants
# ---------------------------------------------------------------------------

class ArgusPrimeWide(ArgusPrimeBase):
    """B0 with bigger FFN (3328 = 4.3x) — more compute in FFN matmuls."""
    def __init__(self, **kw):
        super().__init__(
            ffn_inner=3328, ttt_layers={15}, ttt_mode="single", use_film=False, **kw
        )


class ArgusPrime14L(ArgusPrimeBase):
    """B0 with 14 layers (8 ShortConv + 6 GQA) + bigger FFN (3328).

    Fewer layers = less sequential overhead. Same 6 GQA but only 8 ShortConv.
    GQA at positions 2, 4, 6, 8, 11, 13 (0-indexed in 14 layers).
    TTT on layer 13 (last GQA).
    """
    def __init__(self, **kw):
        super().__init__(
            n_layers=14, ffn_inner=3328,
            gqa_layers=(2, 4, 6, 8, 11, 13),
            ttt_layers={13}, ttt_mode="single", use_film=False,
            film_start=6,  # adjusted for 14 layers
            **kw,
        )


# ---------------------------------------------------------------------------
# Mini for smoke testing
# ---------------------------------------------------------------------------

class ArgusPrimeMini(ArgusPrimeBase):
    """Tiny ARGUS-PRIME for smoke testing (~2M params)."""
    def __init__(self):
        super().__init__(
            vocab_size=1000, d_model=128, n_layers=4,
            d_conv=128, ffn_inner=256, conv_kernel=3,
            n_heads=4, n_kv_heads=2,
            gqa_layers=(1, 3),  # 2 GQA in 4 layers (50%)
            ttt_layers={3}, ttt_mode="single",
            ttt_chunk=64, use_film=False,
            max_seq_len=1024,
        )

"""ARGUS-PRIME standalone model — no ROCm/autokernel dependencies.

Self-contained model definition for running on NVIDIA GPU servers.
All components inlined from models/amadeus.py, models/argus.py, models/argus_prime.py.

Checkpoint-compatible: state_dict keys match the original argus_prime.py exactly.

Dependencies: torch (only). Optional: causal_conv1d (pip install causal-conv1d).
"""

import math
from typing import Optional, Set

import torch
import torch.nn as nn
import torch.nn.functional as F

# Optional fast causal conv1d (10x vs nn.Conv1d, builds CUDA kernels on NVIDIA)
try:
    from causal_conv1d import causal_conv1d_fn
    _HAS_CAUSAL_CONV1D = True
except ImportError:
    _HAS_CAUSAL_CONV1D = False


# ---------------------------------------------------------------------------
# Inlined from models/amadeus.py
# ---------------------------------------------------------------------------

class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        norm = torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        return x * norm * self.weight


class SwiGLU(nn.Module):
    def __init__(self, d_model: int, ffn_inner: int):
        super().__init__()
        self.w_gate_up = nn.Linear(d_model, 2 * ffn_inner, bias=False)
        self.w_down = nn.Linear(ffn_inner, d_model, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gate, up = self.w_gate_up(x).chunk(2, dim=-1)
        return self.w_down(F.silu(gate) * up)


class GatedConv(nn.Module):
    def __init__(self, d_model: int, d_conv: int, kernel_size: int = 3):
        super().__init__()
        self.d_conv = d_conv
        self.kernel_size = kernel_size
        self.proj = nn.Linear(d_model, 3 * d_conv, bias=False)
        if _HAS_CAUSAL_CONV1D:
            self.conv_weight = nn.Parameter(torch.randn(d_conv, kernel_size))
            self.conv_bias = nn.Parameter(torch.zeros(d_conv))
        else:
            self.conv = nn.Conv1d(
                d_conv, d_conv, kernel_size=kernel_size,
                padding=kernel_size - 1, groups=d_conv, bias=True,
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, _ = x.shape
        b, c, h_tilde = self.proj(x).chunk(3, dim=-1)
        y = b * h_tilde
        if _HAS_CAUSAL_CONV1D:
            z = causal_conv1d_fn(
                y.transpose(1, 2), self.conv_weight, self.conv_bias
            ).transpose(1, 2)
        else:
            z = self.conv(y.transpose(1, 2))[:, :, :T].transpose(1, 2)
        return c * z


class FiLMConditioner(nn.Module):
    def __init__(self, d_model: int, d_film: int, n_conditioned_layers: int):
        super().__init__()
        self.context_proj = nn.Linear(d_model, d_film, bias=True)
        self.gamma_projs = nn.ModuleList([
            nn.Linear(d_film, d_model, bias=True)
            for _ in range(n_conditioned_layers)
        ])
        self.beta_projs = nn.ModuleList([
            nn.Linear(d_film, d_model, bias=True)
            for _ in range(n_conditioned_layers)
        ])
        self._init_identity()

    def _init_identity(self):
        for proj in self.gamma_projs:
            nn.init.zeros_(proj.weight)
            nn.init.zeros_(proj.bias)
        for proj in self.beta_projs:
            nn.init.zeros_(proj.weight)
            nn.init.zeros_(proj.bias)

    def compute_context(self, h: torch.Tensor) -> torch.Tensor:
        return self.context_proj(h.mean(dim=1))

    def apply(self, h: torch.Tensor, context: torch.Tensor, layer_idx: int) -> torch.Tensor:
        gamma = self.gamma_projs[layer_idx](context) + 1.0
        beta = self.beta_projs[layer_idx](context)
        return gamma.unsqueeze(1) * h + beta.unsqueeze(1)


# ---------------------------------------------------------------------------
# Inlined from models/argus.py
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


class TTTSwiGLU(nn.Module):
    """SwiGLU with In-Place TTT on down_proj via chunked cumsum."""

    _skip_autokernel = True

    def __init__(self, d_model: int, ffn_inner: int,
                 ttt_chunk: int = 512, ttt_lr_init: float = 0.01,
                 ttt_conv_kernel: int = 5):
        super().__init__()
        self.d_model = d_model
        self.ffn_inner = ffn_inner
        self.ttt_chunk = ttt_chunk
        self.w_gate_up = nn.Linear(d_model, 2 * ffn_inner, bias=False)
        self.w_down = nn.Linear(ffn_inner, d_model, bias=False)
        self.ttt_proj = nn.Linear(d_model, d_model, bias=False)
        self.ttt_conv = nn.Conv1d(
            d_model, d_model, kernel_size=ttt_conv_kernel,
            padding=ttt_conv_kernel - 1, groups=d_model, bias=True,
        )
        self.register_buffer('ttt_lr', torch.tensor(ttt_lr_init))

    def _init_ttt_weights(self):
        nn.init.zeros_(self.ttt_conv.weight)
        nn.init.zeros_(self.ttt_conv.bias)
        with torch.no_grad():
            nn.init.zeros_(self.ttt_proj.weight)
            diag = torch.randn(self.d_model) * 0.02
            self.ttt_proj.weight.diagonal().copy_(diag)

    def _pad_and_chunk(self, x, chunk_size):
        B, T, D = x.shape
        pad_len = (chunk_size - T % chunk_size) % chunk_size
        if pad_len > 0:
            x = F.pad(x, (0, 0, 0, pad_len))
        return x.reshape(B, -1, chunk_size, D)

    def forward(self, x, ttt_target=None):
        B, T, _ = x.shape
        gate, up = self.w_gate_up(x).chunk(2, dim=-1)
        h = F.silu(gate) * up
        if ttt_target is None:
            return self.w_down(h)

        t = self.ttt_conv(ttt_target.transpose(1, 2))[:, :, :T].transpose(1, 2)
        C = self.ttt_chunk
        h_chunked = self._pad_and_chunk(h, C)
        t_chunked = self._pad_and_chunk(t, C)
        nc = h_chunked.shape[1]

        if nc > 1:
            t_proj = F.linear(t_chunked[:, :-1], self.ttt_proj.weight)
            d_down = torch.einsum('btch,btcd->btdh', h_chunked[:, :-1], t_proj)
        else:
            d_down = h_chunked.new_zeros(B, 0, self.d_model, self.ffn_inner)

        w_orig = self.w_down.weight.unsqueeze(0).expand(B, -1, -1).unsqueeze(1)
        d_down_scaled = torch.cat([w_orig, d_down * self.ttt_lr], dim=1)
        w_adapted = d_down_scaled.detach().cumsum(dim=1)
        output = torch.einsum('btdh,btch->btcd', w_adapted, h_chunked)
        return output.reshape(B, -1, self.d_model)[:, :T]


# ---------------------------------------------------------------------------
# ARGUS-PRIME components (from models/argus_prime.py)
# ---------------------------------------------------------------------------

class Attention(nn.Module):
    """GQA with RoPE + QK-Norm. SDPA only (no HIP hybrid path)."""

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
        if self.qk_norm:
            q = F.normalize(q, dim=-1) * self.q_scale
            k = F.normalize(k, dim=-1) * self.k_scale
        if self.n_rep > 1:
            k = k.repeat_interleave(self.n_rep, dim=1)
            v = v.repeat_interleave(self.n_rep, dim=1)
        y = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        return self.wo(y.transpose(1, 2).contiguous().view(B, T, -1))


class MultiStepTTTSwiGLU(TTTSwiGLU):
    """TTTSwiGLU with 3 unrolled gradient steps per chunk."""

    _skip_autokernel = True

    def __init__(self, d_model: int, ffn_inner: int,
                 ttt_chunk: int = 512, ttt_lr_init: float = 0.01,
                 ttt_conv_kernel: int = 5):
        super().__init__(d_model, ffn_inner, ttt_chunk, ttt_lr_init, ttt_conv_kernel)

    def _single_chunk_multistep(self, h_c, t_c, w_current):
        lr = self.ttt_lr / 3.0
        out1 = h_c @ w_current.T
        t_proj1 = F.linear(t_c, self.ttt_proj.weight)
        grad1 = torch.einsum('bch,bcd->dh', h_c, t_proj1 - out1) / (h_c.shape[0] * h_c.shape[1])
        w1 = w_current + lr * grad1.detach()
        out2 = h_c @ w1.T
        t_proj2 = F.linear(t_c, self.ttt_proj.weight)
        grad2 = torch.einsum('bch,bcd->dh', h_c, t_proj2 - out2) / (h_c.shape[0] * h_c.shape[1])
        w2 = w1 + lr * grad2.detach()
        out3 = h_c @ w2.T
        t_proj3 = F.linear(t_c, self.ttt_proj.weight)
        grad3 = torch.einsum('bch,bcd->dh', h_c, t_proj3 - out3) / (h_c.shape[0] * h_c.shape[1])
        w3 = w2 + lr * grad3.detach()
        return h_c @ w3.T, w3

    def forward(self, x, ttt_target=None):
        B, T, _ = x.shape
        gate, up = self.w_gate_up(x).chunk(2, dim=-1)
        h = F.silu(gate) * up
        if ttt_target is None:
            return self.w_down(h)
        t = self.ttt_conv(ttt_target.transpose(1, 2))[:, :, :T].transpose(1, 2)
        C = self.ttt_chunk
        h_chunked = self._pad_and_chunk(h, C)
        t_chunked = self._pad_and_chunk(t, C)
        nc = h_chunked.shape[1]
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
        normed = self.pre_norm(x)
        conv_out = self.conv(normed)
        mixer_out = self.out_proj(conv_out)
        beta = torch.sigmoid(self.log_beta)
        velocity = beta * velocity + mixer_out
        x = x + velocity
        rms = torch.rsqrt(x.float().pow(2).mean(-1, keepdim=True) + 1e-6)
        normed = (x.float() * rms).to(x.dtype) * self.ffn_norm.weight
        x = x + self.ffn(normed)
        return x, velocity


class GQABlock(nn.Module):
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
        beta = torch.sigmoid(self.log_beta)
        velocity = beta * velocity + attn_out
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
    def __init__(
        self,
        vocab_size: int = 50257,
        d_model: int = 768,
        n_layers: int = 16,
        d_conv: int = 512,
        ffn_inner: int = 2816,
        conv_kernel: int = 3,
        n_heads: int = 12,
        n_kv_heads: int = 4,
        gqa_layers: tuple = (2, 5, 7, 9, 12, 15),
        ttt_layers: Set[int] = None,
        ttt_mode: str = "single",
        ttt_chunk: int = 512,
        ttt_lr_init: float = 0.01,
        use_film: bool = False,
        d_film: int = 64,
        film_start: int = 7,
        momentum_beta_init: float = 0.5,
        max_seq_len: int = 1024,
    ):
        super().__init__()
        self.d_model = d_model
        self.gqa_set = set(gqa_layers)
        self.ttt_layers = ttt_layers or set()
        self.use_film = use_film
        self.film_start = film_start

        self.tok_embeddings = nn.Embedding(vocab_size, d_model)
        self.norm = RMSNorm(d_model)
        self.output = nn.Linear(d_model, vocab_size, bias=False)
        self.output.weight = self.tok_embeddings.weight

        head_dim = d_model // n_heads
        self.register_buffer(
            "freqs_cis",
            precompute_freqs_cis(head_dim, max_seq_len * 2),
            persistent=False,
        )

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
            if self.use_film and self.film is not None and i == self.film_start:
                context = self.film.compute_context(h)
            if isinstance(layer, GQABlock):
                ttt_target = h if layer.ttt_mode != "none" else None
                h, velocity = layer(h, velocity, freqs_cis, ttt_target=ttt_target)
            else:
                h, velocity = layer(h, velocity)
            if self.use_film and context is not None and i >= self.film_start:
                h = self.film.apply(h, context, i - self.film_start)

        return self.output(self.norm(h))


# ---------------------------------------------------------------------------
# 6 Ablation Variants + Throughput Variants
# ---------------------------------------------------------------------------

class ArgusPrime(ArgusPrimeBase):
    """B0: 1 TTT single-step at layer 16 (sniper)."""
    def __init__(self, **kw):
        super().__init__(ttt_layers={15}, ttt_mode="single", use_film=False, **kw)

class ArgusPrimeFiLM(ArgusPrimeBase):
    """B1: 1 TTT single-step + FiLM."""
    def __init__(self, **kw):
        super().__init__(ttt_layers={15}, ttt_mode="single", use_film=True, **kw)

class ArgusPrimeMulti(ArgusPrimeBase):
    """B2: 1 TTT multi-step 3x."""
    def __init__(self, **kw):
        super().__init__(ttt_layers={15}, ttt_mode="multi", use_film=False, **kw)

class ArgusPrimeMultiFiLM(ArgusPrimeBase):
    """B3: 1 TTT multi-step 3x + FiLM."""
    def __init__(self, **kw):
        super().__init__(ttt_layers={15}, ttt_mode="multi", use_film=True, **kw)

class ArgusPrimeTTT2(ArgusPrimeBase):
    """B4: 2 TTT single-step at layers 8,16."""
    def __init__(self, **kw):
        super().__init__(ttt_layers={7, 15}, ttt_mode="single", use_film=False, **kw)

class ArgusPrimeTTT2FiLM(ArgusPrimeBase):
    """B5: 2 TTT single-step + FiLM."""
    def __init__(self, **kw):
        super().__init__(ttt_layers={7, 15}, ttt_mode="single", use_film=True, **kw)

class ArgusPrimeWide(ArgusPrimeBase):
    """B0 with bigger FFN (3328 = 4.3x)."""
    def __init__(self, **kw):
        super().__init__(ffn_inner=3328, ttt_layers={15}, ttt_mode="single", use_film=False, **kw)

class ArgusPrime14L(ArgusPrimeBase):
    """B0 with 14 layers + bigger FFN (3328)."""
    def __init__(self, **kw):
        super().__init__(
            n_layers=14, ffn_inner=3328,
            gqa_layers=(2, 4, 6, 8, 11, 13),
            ttt_layers={13}, ttt_mode="single", use_film=False,
            film_start=6, **kw,
        )

class ArgusPrimeMini(ArgusPrimeBase):
    """Tiny ARGUS-PRIME for smoke testing (~2M params)."""
    def __init__(self):
        super().__init__(
            vocab_size=1000, d_model=128, n_layers=4,
            d_conv=128, ffn_inner=256, conv_kernel=3,
            n_heads=4, n_kv_heads=2,
            gqa_layers=(1, 3), ttt_layers={3}, ttt_mode="single",
            ttt_chunk=64, use_film=False, max_seq_len=1024,
        )

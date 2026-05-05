"""
SPECTRAL-HYDRA: Multi-Scale Gated Recurrence with Heterogeneous Temporal Decay.

16 recurrent heads per layer, each initialized to a structurally different
temporal decay rate (fast/medium/slow). Cross-head linear mixer combines outputs.
All ops element-wise — no attention, no O(T²). Maximum MFU potential.

Usage:
    python -m halo_training --model models/spectral_hydra.py --class-name SpectralHydra --dataset babylm
    python -m halo_training --model models/spectral_hydra.py --class-name SpectralHydra --smoke
"""

import math
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from causal_conv1d import causal_conv1d_fn
    _HAS_CAUSAL_CONV1D = True
except ImportError:
    _HAS_CAUSAL_CONV1D = False


@dataclass
class SpectralHydraConfig:
    vocab_size: int = 50257
    d_model: int = 1024
    n_layers: int = 16
    n_heads: int = 16        # 5 fast + 6 medium + 5 slow
    d_head: int = 64         # d_model // n_heads
    ffn_inner: int = 2560
    conv_kernel: int = 4
    max_seq_len: int = 1024


from models._components import RMSNorm, SwiGLU


class DepthwiseConv(nn.Module):
    """Depthwise causal conv1d with optional fast backend."""

    def __init__(self, d: int, kernel_size: int = 4):
        super().__init__()
        self.d = d
        self.kernel_size = kernel_size
        if _HAS_CAUSAL_CONV1D:
            self.conv_weight = nn.Parameter(torch.randn(d, kernel_size))
            self.conv_bias = nn.Parameter(torch.zeros(d))
        else:
            self.conv = nn.Conv1d(
                d, d, kernel_size=kernel_size,
                padding=kernel_size - 1, groups=d, bias=True,
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, D = x.shape
        if _HAS_CAUSAL_CONV1D:
            return causal_conv1d_fn(
                x.transpose(1, 2), self.conv_weight, self.conv_bias
            ).transpose(1, 2)
        else:
            y = self.conv(x.transpose(1, 2))[:, :, :T]
            return y.transpose(1, 2)


class MultiScaleGatedRecurrence(nn.Module):
    """Multi-scale gated recurrence with heterogeneous decay spectrum.

    16 heads: 5 fast (decay ~0.10), 6 medium (~0.50), 5 slow (~0.99).
    Griffin-style coupling: input = sqrt(1 - a²) * (i * v).
    """

    def __init__(self, d_model: int, n_heads: int = 16, d_head: int = 64):
        super().__init__()
        self.n_heads = n_heads
        self.d_head = d_head
        d_rec = n_heads * d_head  # = d_model

        # Fused projection: single GEMM instead of 3 separate ones
        self.w_aiv = nn.Linear(d_model, 3 * d_rec, bias=False)

        # Per-head decay bias: the spectral initialization
        self.decay_bias = nn.Parameter(torch.zeros(d_rec))
        with torch.no_grad():
            n_fast = 5
            n_medium = 6
            n_slow = 5
            fast_end = n_fast * d_head
            medium_end = (n_fast + n_medium) * d_head
            self.decay_bias[:fast_end].fill_(-2.2)       # fast: local N-grams
            self.decay_bias[fast_end:medium_end].fill_(0.0)  # medium: clause structure
            self.decay_bias[medium_end:].fill_(4.6)      # slow: topic tracking

        # Cross-head mixer: combines all heads' outputs
        self.mixer = nn.Linear(d_rec, d_model, bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        aiv = self.w_aiv(x)
        a_proj, i_proj, v = aiv.split(self.n_heads * self.d_head, dim=-1)
        a = torch.sigmoid(a_proj + self.decay_bias)
        i = torch.sigmoid(i_proj)

        # Griffin coupling: bounded input signal
        input_signal = torch.sqrt((1 - a * a).clamp(min=1e-8)) * (i * v)

        # Chunked linear recurrence (parallel within chunks)
        h = self._chunked_scan(a, input_signal)

        # Cross-head mixer
        return self.mixer(h)

    def _chunked_scan(self, decay, value, chunk_size=64):
        batch, seqlen, d = decay.shape
        decay_f = decay.float()
        value_f = value.float()

        pad = (chunk_size - seqlen % chunk_size) % chunk_size
        if pad > 0:
            decay_f = F.pad(decay_f, (0, 0, 0, pad), value=1.0)
            value_f = F.pad(value_f, (0, 0, 0, pad), value=0.0)

        total_len = seqlen + pad
        n_chunks = total_len // chunk_size

        dc = decay_f.view(batch, n_chunks, chunk_size, d)
        vc = value_f.view(batch, n_chunks, chunk_size, d)

        log_dc = torch.log(dc.clamp(min=1e-10))
        cum_log = torch.cumsum(log_dc, dim=2)
        cum_decay = torch.exp(cum_log)

        weighted = vc / cum_decay.clamp(min=1e-10)
        cum_weighted = torch.cumsum(weighted, dim=2)
        intra_state = cum_decay * cum_weighted

        # Cross-chunk propagation — vectorized (no Python loop, enables torch.compile)
        chunk_final_state = intra_state[:, :, -1, :]
        chunk_total_decay = cum_decay[:, :, -1, :]

        log_chunk_decay = torch.log(chunk_total_decay.clamp(min=1e-10))
        cum_log_chunk = torch.cumsum(log_chunk_decay, dim=1)
        cum_chunk_decay = torch.exp(cum_log_chunk)

        weighted_chunk = chunk_final_state / cum_chunk_decay.clamp(min=1e-10)
        cum_weighted_chunk = torch.cumsum(weighted_chunk, dim=1)

        zeros = torch.zeros(batch, 1, d, dtype=torch.float32, device=decay.device)
        incoming_cum_weighted = torch.cat([zeros, cum_weighted_chunk[:, :-1]], dim=1)
        incoming_cum_decay = torch.cat([
            torch.ones(batch, 1, d, dtype=torch.float32, device=decay.device),
            cum_chunk_decay[:, :-1]
        ], dim=1)
        incoming_state = incoming_cum_decay * incoming_cum_weighted

        cross_contrib = cum_decay * incoming_state.unsqueeze(2)
        states = (intra_state + cross_contrib).reshape(batch, total_len, d)[:, :seqlen]
        return states.to(decay.dtype)


class SpectralHydraBlock(nn.Module):
    """Single layer: Conv1d → Multi-Scale Recurrence → residual → SwiGLU → residual."""

    def __init__(self, cfg: SpectralHydraConfig):
        super().__init__()
        self.pre_norm = RMSNorm(cfg.d_model)
        self.conv = DepthwiseConv(cfg.d_model, cfg.conv_kernel)
        self.recurrence = MultiScaleGatedRecurrence(cfg.d_model, cfg.n_heads, cfg.d_head)
        self.ffn_norm = RMSNorm(cfg.d_model)
        self.ffn = SwiGLU(cfg.d_model, cfg.ffn_inner)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.pre_norm(x)
        h = self.conv(h)
        h = self.recurrence(h)
        x = x + h
        x = x + self.ffn(self.ffn_norm(x))
        return x


class SpectralHydra(nn.Module):
    """SPECTRAL-HYDRA: Multi-Scale Gated Recurrence.

    16 layers, 16 heads per layer with heterogeneous decay spectrum.
    All non-FFN ops are element-wise. Maximum MFU potential.
    ~245M parameters.
    """

    def __init__(
        self,
        vocab_size: int = 50257,
        d_model: int = 1024,
        n_layers: int = 16,
        n_heads: int = 16,
        d_head: int = 64,
        ffn_inner: int = 2560,
        conv_kernel: int = 4,
        max_seq_len: int = 1024,
    ):
        super().__init__()
        cfg = SpectralHydraConfig(
            vocab_size=vocab_size, d_model=d_model, n_layers=n_layers,
            n_heads=n_heads, d_head=d_head, ffn_inner=ffn_inner,
            conv_kernel=conv_kernel, max_seq_len=max_seq_len,
        )

        self.tok_embeddings = nn.Embedding(vocab_size, d_model)
        self.layers = nn.ModuleList([SpectralHydraBlock(cfg) for _ in range(n_layers)])
        self.norm = RMSNorm(d_model)
        self.output = nn.Linear(d_model, vocab_size, bias=False)
        self.output.weight = self.tok_embeddings.weight

        self._init_weights()
        n_params = sum(p.numel() for p in self.parameters())
        print(f"SpectralHydra: {n_params / 1e6:.1f}M parameters")

    def _init_weights(self):
        for name, p in self.named_parameters():
            if p.dim() >= 2:
                nn.init.xavier_uniform_(p)

    def forward(self, input_ids: torch.Tensor, targets=None) -> torch.Tensor:
        h = self.tok_embeddings(input_ids)
        for layer in self.layers:
            h = layer(h)
        return self.output(self.norm(h))

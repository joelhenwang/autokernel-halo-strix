"""
OBSIDIAN: BitNet b1.58 Reflex + Caveman LFM Genius + Engram.

Dual-path architecture:
- Reflex: BitNet b1.58 (ternary weights), d=320, 8 layers, L2-resident
- Genius: Caveman LFM (fp16 standard), d=1024, 10 layers
- Routing: state delta norm (soft gate during training)

~224M parameters.

Usage:
    python -m halo_training --model models/obsidian.py --class-name Obsidian --dataset babylm
    python -m halo_training --model models/obsidian.py --class-name Obsidian --smoke
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


class BitNetLinear(nn.Module):
    """BitNet b1.58: ternary weight quantization {-1, 0, +1}.

    Forward: absmax quantization of activations (8-bit) and weights (ternary).
    Gradients flow through the scaling factors (smooth, no STE).
    """

    def __init__(self, in_features: int, out_features: int):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(out_features, in_features) * 0.02)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Activation quantization: absmax → 8-bit
        x_scale = x.abs().max(dim=-1, keepdim=True).values.clamp(min=1e-5)
        x_quant = (x / x_scale * 127).round().clamp(-128, 127)

        # Weight quantization: centralize + absmax → {-1, 0, +1}
        w_mean = self.weight.mean()
        w_centered = self.weight - w_mean
        w_scale = w_centered.abs().mean().clamp(min=1e-5)
        w_quant = (w_centered / w_scale).round().clamp(-1, 1)

        # Scaled ternary matmul
        y = F.linear(x_quant.float(), w_quant) * (w_scale * x_scale / 127)
        return y.to(x.dtype)


class GatedConvBlock(nn.Module):
    """Gated conv block for Caveman LFM genius path."""

    def __init__(self, d_model: int, ffn_inner: int, conv_kernel: int = 3):
        super().__init__()
        self.pre_norm = RMSNorm(d_model)
        self.proj = nn.Linear(d_model, 3 * d_model, bias=False)
        if _HAS_CAUSAL_CONV1D:
            self.conv_weight = nn.Parameter(torch.randn(d_model, conv_kernel))
            self.conv_bias = nn.Parameter(torch.zeros(d_model))
            self._use_fast_conv = True
        else:
            self.conv = nn.Conv1d(
                d_model, d_model, kernel_size=conv_kernel,
                padding=conv_kernel - 1, groups=d_model, bias=True,
            )
            self._use_fast_conv = False
        self.out_proj = nn.Linear(d_model, d_model, bias=False)
        self.ffn_norm = RMSNorm(d_model)
        self.ffn = SwiGLU(d_model, ffn_inner)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, _ = x.shape
        h = self.pre_norm(x)
        b, c, h_tilde = self.proj(h).chunk(3, dim=-1)
        y = b * h_tilde
        if self._use_fast_conv:
            z = causal_conv1d_fn(
                y.transpose(1, 2), self.conv_weight, self.conv_bias
            ).transpose(1, 2)
        else:
            z = self.conv(y.transpose(1, 2))[:, :, :T].transpose(1, 2)
        x = x + self.out_proj(c * z)
        x = x + self.ffn(self.ffn_norm(x))
        return x


class GriffinRecBlock(nn.Module):
    """Griffin recurrence block for Caveman LFM genius path."""

    def __init__(self, d_model: int, d_rec: int, ffn_inner: int):
        super().__init__()
        self.pre_norm = RMSNorm(d_model)
        self.w_a = nn.Linear(d_model, d_rec, bias=False)
        self.w_i = nn.Linear(d_model, d_rec, bias=False)
        self.w_v = nn.Linear(d_model, d_rec, bias=False)
        self.out_proj = nn.Linear(d_rec, d_model, bias=False)

        self.decay_bias = nn.Parameter(torch.zeros(d_rec))
        with torch.no_grad():
            quarter = d_rec // 4
            self.decay_bias[:quarter].fill_(-2.2)
            self.decay_bias[quarter:3 * quarter].fill_(0.0)
            self.decay_bias[3 * quarter:].fill_(4.6)

        self.ffn_norm = RMSNorm(d_model)
        self.ffn = SwiGLU(d_model, ffn_inner)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.pre_norm(x)
        a = torch.sigmoid(self.w_a(h) + self.decay_bias)
        i_gate = torch.sigmoid(self.w_i(h))
        v = self.w_v(h)
        input_signal = torch.sqrt((1 - a * a).clamp(min=1e-8)) * (i_gate * v)
        rec_out = _chunked_scan(a, input_signal)
        x = x + self.out_proj(rec_out)
        x = x + self.ffn(self.ffn_norm(x))
        return x


class BitNetReflexBlock(nn.Module):
    """Reflex block with BitNet ternary weights + Griffin recurrence."""

    def __init__(self, d: int = 320, d_ffn: int = 640):
        super().__init__()
        self.pre_norm = RMSNorm(d)
        # BitNet gate FFN
        self.gate_proj = BitNetLinear(d, d_ffn)
        self.down_proj = BitNetLinear(d_ffn, d)

        # Griffin recurrence with BitNet gates
        self.w_a = BitNetLinear(d, d)
        self.w_i = BitNetLinear(d, d)
        self.w_v = BitNetLinear(d, d)
        self.decay_bias = nn.Parameter(torch.zeros(d))
        with torch.no_grad():
            quarter = d // 4
            self.decay_bias[:quarter].fill_(-2.2)
            self.decay_bias[quarter:3 * quarter].fill_(0.0)
            self.decay_bias[3 * quarter:].fill_(4.6)

        self.post_norm = RMSNorm(d)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.pre_norm(x)

        # Griffin recurrence
        a = torch.sigmoid(self.w_a(h) + self.decay_bias)
        i_gate = torch.sigmoid(self.w_i(h))
        v = self.w_v(h)
        input_signal = torch.sqrt((1 - a * a).clamp(min=1e-8)) * (i_gate * v)
        rec_out = _chunked_scan(a, input_signal)

        # BitNet SiLU gate FFN
        ffn_out = self.down_proj(F.silu(self.gate_proj(self.post_norm(rec_out))))

        return x + rec_out + ffn_out


def _chunked_scan(decay, value, chunk_size=64):
    """Shared chunked scan for all Griffin blocks."""
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

    state = torch.zeros(batch, d, dtype=torch.float32, device=decay.device)
    all_states = []
    for c in range(n_chunks):
        prev_contrib = cum_decay[:, c] * state.unsqueeze(1)
        chunk_states = intra_state[:, c] + prev_contrib
        all_states.append(chunk_states)
        state = chunk_states[:, -1]

    states = torch.cat(all_states, dim=1)[:, :seqlen]
    return states.to(decay.dtype)


class Obsidian(nn.Module):
    """OBSIDIAN: BitNet b1.58 Reflex + Caveman LFM Genius.

    Reflex: 8 BitNet layers, d=320 (L2-resident)
    Genius: 10 Caveman LFM layers, d=1024 (pattern: C C R C C R C R C R)
    Routing: state delta soft gate
    ~224M parameters.
    """

    def __init__(
        self,
        vocab_size: int = 50257,
        d_model: int = 1024,
        d_reflex: int = 320,
        n_reflex_layers: int = 8,
        d_reflex_ffn: int = 640,
        n_genius_layers: int = 10,
        d_genius_rec: int = 1024,
        ffn_genius: int = 2240,     # 2.2x for param budget
        conv_kernel: int = 3,
        lambda_compute: float = 0.01,
        max_seq_len: int = 1024,
    ):
        super().__init__()
        self.d_reflex = d_reflex
        self.lambda_compute = lambda_compute

        self.tok_embeddings = nn.Embedding(vocab_size, d_model)

        # Reflex path
        self.embed_to_reflex = nn.Linear(d_model, d_reflex, bias=False)
        self.reflex_layers = nn.ModuleList([
            BitNetReflexBlock(d_reflex, d_reflex_ffn)
            for _ in range(n_reflex_layers)
        ])

        # Genius path: Caveman LFM pattern C C R C C R C R C R
        genius_pattern = ['C', 'C', 'R', 'C', 'C', 'R', 'C', 'R', 'C', 'R']
        self.reflex_to_genius = nn.Linear(d_reflex, d_model, bias=False)
        self.genius_layers = nn.ModuleList()
        for p in genius_pattern[:n_genius_layers]:
            if p == 'C':
                self.genius_layers.append(
                    GatedConvBlock(d_model, ffn_genius, conv_kernel)
                )
            else:
                self.genius_layers.append(
                    GriffinRecBlock(d_model, d_genius_rec, ffn_genius)
                )
        self.genius_to_reflex = nn.Linear(d_model, d_reflex, bias=False)

        # Routing: state delta gate
        self.gate_mlp = nn.Sequential(
            nn.Linear(d_reflex, 64, bias=True),
            nn.ReLU(),
            nn.Linear(64, 1, bias=True),
        )
        nn.init.zeros_(self.gate_mlp[2].weight)
        nn.init.constant_(self.gate_mlp[2].bias, 0.0)

        # Output projection
        self.reflex_to_embed = nn.Linear(d_reflex, d_model, bias=False)
        self.norm = RMSNorm(d_model)
        self.output = nn.Linear(d_model, vocab_size, bias=False)
        self.output.weight = self.tok_embeddings.weight

        self._init_weights()
        n_params = sum(p.numel() for p in self.parameters())
        reflex_params = sum(p.numel() for p in self.reflex_layers.parameters())
        genius_params = sum(p.numel() for p in self.genius_layers.parameters())
        print(f"Obsidian: {n_params / 1e6:.1f}M total "
              f"(reflex: {reflex_params / 1e6:.1f}M BitNet, genius: {genius_params / 1e6:.1f}M fp16)")

    def _init_weights(self):
        for name, p in self.named_parameters():
            if 'gate_mlp' in name:
                continue
            if p.dim() >= 2:
                nn.init.xavier_uniform_(p)

    def forward(self, input_ids: torch.Tensor, targets=None) -> torch.Tensor:
        B, T = input_ids.shape
        h_embed = self.tok_embeddings(input_ids)

        # Reflex path (always runs)
        h_reflex = self.embed_to_reflex(h_embed)
        h_reflex_init = h_reflex

        for layer in self.reflex_layers:
            h_reflex = layer(h_reflex)

        # State delta gate
        delta = (h_reflex - h_reflex_init).detach()
        gate = torch.sigmoid(self.gate_mlp(delta))  # (B, T, 1)
        self._slow_fraction = gate.mean()

        # Genius path (conditional via soft gate)
        h_genius = self.reflex_to_genius(h_reflex)
        for layer in self.genius_layers:
            h_genius = layer(h_genius)
        genius_correction = self.genius_to_reflex(h_genius)

        # Merge
        h_out = h_reflex + gate * genius_correction

        # LM head
        h_out = self.reflex_to_embed(h_out)
        logits = self.output(self.norm(h_out))

        self._compute_cost = self.lambda_compute * self._slow_fraction
        return logits

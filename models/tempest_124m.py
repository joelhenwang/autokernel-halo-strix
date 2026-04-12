"""
TEMPEST-124M: Parameter-matched variant for fair comparison with LlamaModel 124.7M.

Same architecture as Tempest but scaled down:
- d_model=768, 14 layers, d_conv=512, d_griffin=256, ffn=2048, vocab=32000
- ~124M parameters (matching LlamaModel)

This isolates architecture overhead from model size effects.

Usage:
    python -m halo_training --model models/tempest_124m.py --class-name Tempest124M --smoke
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


class GatedConv(nn.Module):
    def __init__(self, d_model: int, d_conv: int, kernel_size: int = 3):
        super().__init__()
        self.d_conv = d_conv
        self.proj = nn.Linear(d_model, 3 * d_conv, bias=False)
        if _HAS_CAUSAL_CONV1D:
            self.conv_weight = nn.Parameter(torch.randn(d_conv, kernel_size))
            self.conv_bias = nn.Parameter(torch.zeros(d_conv))
            self._fast = True
        else:
            self.conv = nn.Conv1d(
                d_conv, d_conv, kernel_size=kernel_size,
                padding=kernel_size - 1, groups=d_conv, bias=True,
            )
            self._fast = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, _ = x.shape
        b, c, h_tilde = self.proj(x).chunk(3, dim=-1)
        y = b * h_tilde
        if self._fast:
            z = causal_conv1d_fn(
                y.transpose(1, 2), self.conv_weight, self.conv_bias
            ).transpose(1, 2)
        else:
            z = self.conv(y.transpose(1, 2))[:, :, :T].transpose(1, 2)
        return c * z


class GriffinRecurrence(nn.Module):
    def __init__(self, d_model: int, d_rec: int):
        super().__init__()
        self.d_rec = d_rec
        self.w_aiv = nn.Linear(d_model, 3 * d_rec, bias=False)

        self.decay_bias = nn.Parameter(torch.zeros(d_rec))
        with torch.no_grad():
            quarter = d_rec // 4
            self.decay_bias[:quarter].fill_(-2.2)
            self.decay_bias[quarter:3 * quarter].fill_(0.0)
            self.decay_bias[3 * quarter:].fill_(4.6)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        aiv = self.w_aiv(x)
        a_proj, i_proj, v = aiv.split(self.d_rec, dim=-1)
        a = torch.sigmoid(a_proj + self.decay_bias)
        i = torch.sigmoid(i_proj)
        input_signal = torch.sqrt((1 - a * a).clamp(min=1e-8)) * (i * v)
        return self._chunked_scan(a, input_signal)

    def _chunked_scan(self, decay, value, chunk_size=64):
        """Fully vectorized chunked scan — no Python loops for torch.compile."""
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

        # Cross-chunk propagation — vectorized
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


class MomentumResidual(nn.Module):
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


class TempestBlock(nn.Module):
    def __init__(self, d_model, d_conv, d_griffin, ffn_inner, conv_kernel=3):
        super().__init__()
        self.pre_norm = RMSNorm(d_model)
        self.conv = GatedConv(d_model, d_conv, conv_kernel)
        self.griffin = GriffinRecurrence(d_model, d_griffin)
        self.out_proj = nn.Linear(d_model, d_model, bias=False)
        self.momentum = MomentumResidual()
        self.ffn_norm = RMSNorm(d_model)
        self.ffn = SwiGLU(d_model, ffn_inner)

    def forward(self, x, velocity):
        x_norm = self.pre_norm(x)
        conv_out = self.conv(x_norm)
        griffin_out = self.griffin(x_norm)
        mixer_out = self.out_proj(torch.cat([conv_out, griffin_out], dim=-1))
        x, velocity = self.momentum(x, mixer_out, velocity)
        x = x + self.ffn(self.ffn_norm(x))
        return x, velocity


class Tempest124M(nn.Module):
    """TEMPEST scaled to ~124M params for fair comparison with LlamaModel.

    d=768, 14 layers, d_conv=512, d_griffin=256, ffn=2048, vocab=32000.
    """

    def __init__(
        self,
        vocab_size: int = 32000,
        d_model: int = 768,
        n_layers: int = 14,
        d_conv: int = 512,
        d_griffin: int = 256,
        ffn_inner: int = 2048,
        conv_kernel: int = 3,
        max_seq_len: int = 1024,
    ):
        super().__init__()
        assert d_conv + d_griffin == d_model

        self.tok_embeddings = nn.Embedding(vocab_size, d_model)
        self.layers = nn.ModuleList([
            TempestBlock(d_model, d_conv, d_griffin, ffn_inner, conv_kernel)
            for _ in range(n_layers)
        ])
        self.norm = RMSNorm(d_model)
        self.output = nn.Linear(d_model, vocab_size, bias=False)
        self.output.weight = self.tok_embeddings.weight

        self._init_weights()
        n_params = sum(p.numel() for p in self.parameters())
        print(f"Tempest124M: {n_params / 1e6:.1f}M parameters")

    def _init_weights(self):
        for name, p in self.named_parameters():
            if p.dim() >= 2:
                nn.init.xavier_uniform_(p)

    def forward(self, input_ids: torch.Tensor, targets=None) -> torch.Tensor:
        h = self.tok_embeddings(input_ids)
        velocity = torch.zeros_like(h)
        for layer in self.layers:
            h, velocity = layer(h, velocity)
        return self.output(self.norm(h))

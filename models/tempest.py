"""
TEMPEST: Pure Element-Wise Storm — 16 Griffin Layers + Residual Momentum.

Maximum MFU architecture. Every non-FFN operation is element-wise.
Griffin recurrence (simpler than Mamba-3) + momentum residual for depth-wise inertia.

Usage:
    python -m halo_training --model models/tempest.py --class-name Tempest --dataset babylm
    python -m halo_training --model models/tempest.py --class-name Tempest --smoke
"""

import math
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

# Fast causal conv1d backend (10x vs nn.Conv1d)
try:
    from causal_conv1d import causal_conv1d_fn
    _HAS_CAUSAL_CONV1D = True
except ImportError:
    _HAS_CAUSAL_CONV1D = False


@dataclass
class TempestConfig:
    vocab_size: int = 50257
    d_model: int = 1024
    n_layers: int = 16
    d_conv: int = 640
    d_griffin: int = 384
    ffn_inner: int = 2560
    conv_kernel: int = 3
    momentum_beta_init: float = 0.5
    max_seq_len: int = 1024


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
            # causal_conv1d_fn expects weight (D, K) and optional bias (D,)
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
            # causal_conv1d_fn: input (B, D, L), weight (D, K) → output (B, D, L)
            z = causal_conv1d_fn(
                y.transpose(1, 2), self.conv_weight, self.conv_bias
            ).transpose(1, 2)
        else:
            z = self.conv(y.transpose(1, 2))[:, :, :T]
            z = z.transpose(1, 2)
        return c * z


class GriffinRecurrence(nn.Module):
    """Pure element-wise recurrence with decay bias spectrum.

    h[t] = a[t] * h[t-1] + sqrt(1 - a[t]²) * (i[t] * v[t])

    All operations are element-wise — sigmoid, multiply, sqrt, add.
    Zero matmuls in the recurrence itself.
    """

    def __init__(self, d_model: int, d_rec: int = 384):
        super().__init__()
        self.d_rec = d_rec
        # Fused projection: single GEMM instead of 3 separate ones
        self.w_aiv = nn.Linear(d_model, 3 * d_rec, bias=False)

        # Decay bias spectrum: multi-scale temporal dynamics
        self.decay_bias = nn.Parameter(torch.zeros(d_rec))
        with torch.no_grad():
            quarter = d_rec // 4
            self.decay_bias[:quarter].fill_(-2.2)           # fast: local N-grams
            self.decay_bias[quarter:3 * quarter].fill_(0.0) # medium: clause structure
            self.decay_bias[3 * quarter:].fill_(4.6)        # slow: topic tracking

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        aiv = self.w_aiv(x)
        a_proj, i_proj, v = aiv.split(self.d_rec, dim=-1)
        a = torch.sigmoid(a_proj + self.decay_bias)
        i = torch.sigmoid(i_proj)

        # Bounded input signal (Griffin coupling)
        input_signal = torch.sqrt((1 - a * a).clamp(min=1e-8)) * (i * v)

        # Chunked linear recurrence (same pattern as AMADEUS)
        y = self._chunked_scan(a, input_signal)
        return y

    def _chunked_scan(self, decay, value, chunk_size=64):
        """Chunked linear recurrence — fully vectorized, no Python loops.

        Two-level scan: within-chunk via cumsum (parallel), cross-chunk via
        cumulative product on boundary states (also parallel).
        """
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

        # Within-chunk cumulative decay (log-domain for stability)
        log_dc = torch.log(dc.clamp(min=1e-10))
        cum_log = torch.cumsum(log_dc, dim=2)
        cum_decay = torch.exp(cum_log)

        # Within-chunk state (intra_state[c, t] = state from positions within chunk c up to t)
        weighted = vc / cum_decay.clamp(min=1e-10)
        cum_weighted = torch.cumsum(weighted, dim=2)
        intra_state = cum_decay * cum_weighted  # (batch, n_chunks, chunk_size, d)

        # Cross-chunk propagation — vectorized (no Python loop)
        # Each chunk's final state: intra_state[:, c, -1] (contribution from within chunk c)
        # Each chunk's total decay: cum_decay[:, c, -1] (total decay across chunk c)
        chunk_final_state = intra_state[:, :, -1, :]   # (batch, n_chunks, d)
        chunk_total_decay = cum_decay[:, :, -1, :]      # (batch, n_chunks, d)

        # Cross-chunk scan: state[c] = sum_{j<c} (prod_{k=j+1}^{c-1} total_decay[k]) * final_state[j]
        # This is another linear recurrence on the chunk boundaries.
        # With n_chunks typically 4-16, a cumulative product approach works.
        log_chunk_decay = torch.log(chunk_total_decay.clamp(min=1e-10))  # (batch, n_chunks, d)
        cum_log_chunk = torch.cumsum(log_chunk_decay, dim=1)              # (batch, n_chunks, d)
        cum_chunk_decay = torch.exp(cum_log_chunk)                        # (batch, n_chunks, d)

        # Weighted chunk states for cross-chunk propagation
        weighted_chunk = chunk_final_state / cum_chunk_decay.clamp(min=1e-10)
        cum_weighted_chunk = torch.cumsum(weighted_chunk, dim=1)

        # cross_state[c] = state entering chunk c from previous chunks
        # Shift right by 1: chunk 0 gets 0, chunk c gets cum up to c-1
        cross_state_at_boundary = cum_chunk_decay * cum_weighted_chunk  # (batch, n_chunks, d)
        # But this includes the current chunk's own contribution — subtract it
        cross_state_at_boundary = cross_state_at_boundary - chunk_final_state
        # Shift: chunk c's incoming state = cross_state_at_boundary[c-1]'s outgoing
        # Actually, let's compute it properly:
        # incoming_state[0] = 0
        # incoming_state[c] = total_decay[c-1] * incoming_state[c-1] + final_state[c-1]
        # This is: cum_chunk_decay shifted right by 1 * cum_weighted shifted right by 1
        zeros = torch.zeros(batch, 1, d, dtype=torch.float32, device=decay.device)
        incoming_cum_weighted = torch.cat([zeros, cum_weighted_chunk[:, :-1]], dim=1)
        incoming_cum_decay = torch.cat([
            torch.ones(batch, 1, d, dtype=torch.float32, device=decay.device),
            cum_chunk_decay[:, :-1]
        ], dim=1)
        incoming_state = incoming_cum_decay * incoming_cum_weighted  # (batch, n_chunks, d)

        # Apply cross-chunk state to each position within each chunk
        # position t in chunk c: total_state = intra_state[c,t] + cum_decay[c,t] * incoming_state[c]
        cross_contrib = cum_decay * incoming_state.unsqueeze(2)  # (batch, n_chunks, chunk_size, d)
        states = (intra_state + cross_contrib).reshape(batch, total_len, d)[:, :seqlen]

        return states.to(decay.dtype)


class MomentumResidual(nn.Module):
    """Residual connection with depth-wise momentum.

    velocity = beta * velocity + layer_output
    h = h + velocity

    Gives the mixer signal inertia across depth.
    """

    def __init__(self, beta_init: float = 0.5):
        super().__init__()
        # Parametrize via sigmoid for (0, 1) range
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
    """Single TEMPEST layer: parallel conv + Griffin, momentum residual, SwiGLU."""

    def __init__(self, cfg: TempestConfig):
        super().__init__()
        self.pre_norm = RMSNorm(cfg.d_model)
        self.conv = GatedConv(cfg.d_model, cfg.d_conv, cfg.conv_kernel)
        self.griffin = GriffinRecurrence(cfg.d_model, cfg.d_griffin)
        self.out_proj = nn.Linear(cfg.d_model, cfg.d_model, bias=False)
        self.momentum = MomentumResidual(cfg.momentum_beta_init)
        self.ffn_norm = RMSNorm(cfg.d_model)
        self.ffn = SwiGLU(cfg.d_model, cfg.ffn_inner)

    def forward(self, x: torch.Tensor, velocity: torch.Tensor):
        x_norm = self.pre_norm(x)
        conv_out = self.conv(x_norm)
        griffin_out = self.griffin(x_norm)
        mixer_out = self.out_proj(torch.cat([conv_out, griffin_out], dim=-1))

        # Momentum residual on mixer
        x, velocity = self.momentum(x, mixer_out, velocity)
        # Standard residual on FFN
        x = x + self.ffn(self.ffn_norm(x))
        return x, velocity


class Tempest(nn.Module):
    """TEMPEST: Pure element-wise storm.

    16 identical blocks: GatedConv || Griffin → momentum residual → SwiGLU.
    No attention. No complex SSM. Maximum MFU.
    ~245M parameters.
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
        momentum_beta_init: float = 0.5,
        max_seq_len: int = 1024,
    ):
        super().__init__()
        assert d_conv + d_griffin == d_model

        cfg = TempestConfig(
            vocab_size=vocab_size, d_model=d_model, n_layers=n_layers,
            d_conv=d_conv, d_griffin=d_griffin, ffn_inner=ffn_inner,
            conv_kernel=conv_kernel, momentum_beta_init=momentum_beta_init,
            max_seq_len=max_seq_len,
        )

        self.tok_embeddings = nn.Embedding(vocab_size, d_model)
        self.layers = nn.ModuleList([TempestBlock(cfg) for _ in range(n_layers)])
        self.norm = RMSNorm(d_model)
        self.output = nn.Linear(d_model, vocab_size, bias=False)
        self.output.weight = self.tok_embeddings.weight

        self._init_weights()
        n_params = sum(p.numel() for p in self.parameters())
        print(f"Tempest: {n_params / 1e6:.1f}M parameters")

    def _init_weights(self):
        for name, p in self.named_parameters():
            if p.dim() >= 2:
                nn.init.xavier_uniform_(p)

    def forward(self, input_ids: torch.Tensor, targets=None) -> torch.Tensor:
        B, T = input_ids.shape
        h = self.tok_embeddings(input_ids)
        velocity = torch.zeros_like(h)

        for layer in self.layers:
            h, velocity = layer(h, velocity)

        logits = self.output(self.norm(h))
        return logits

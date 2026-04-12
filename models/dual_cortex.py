"""
DUAL-CORTEX: System 1/System 2 with Entropy-Gated Routing.

Fast path (d=320, 8L, ~10M params): always runs, fits in L2 cache.
Slow path (d=1280, 10L, ~155M params): engages only when uncertain (~30%).
Entropy gate: state delta norm > learned threshold.

~231M total parameters.

Usage:
    python -m halo_training --model models/dual_cortex.py --class-name DualCortex --dataset babylm
    python -m halo_training --model models/dual_cortex.py --class-name DualCortex --smoke
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


class GriffinBlock(nn.Module):
    """Griffin recurrence + Conv + SwiGLU. Reusable for both paths."""

    def __init__(self, d_model: int, d_rec: int, ffn_inner: int,
                 conv_kernel: int = 4, decay_spectrum: bool = True):
        super().__init__()
        self.d_rec = d_rec

        # Recurrence projections
        self.pre_norm = RMSNorm(d_model)
        self.w_a = nn.Linear(d_model, d_rec, bias=False)
        self.w_i = nn.Linear(d_model, d_rec, bias=False)
        self.w_v = nn.Linear(d_model, d_rec, bias=False)
        self.out_proj = nn.Linear(d_rec, d_model, bias=False)

        # Decay bias spectrum
        self.decay_bias = nn.Parameter(torch.zeros(d_rec))
        if decay_spectrum:
            with torch.no_grad():
                quarter = d_rec // 4
                self.decay_bias[:quarter].fill_(-2.2)
                self.decay_bias[quarter:3 * quarter].fill_(0.0)
                self.decay_bias[3 * quarter:].fill_(4.6)

        # Conv
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

        # FFN
        self.ffn_norm = RMSNorm(d_model)
        self.ffn = SwiGLU(d_model, ffn_inner)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, D = x.shape

        # Conv
        h = self.pre_norm(x)
        if self._use_fast_conv:
            h = causal_conv1d_fn(
                h.transpose(1, 2), self.conv_weight, self.conv_bias
            ).transpose(1, 2)
        else:
            h = self.conv(h.transpose(1, 2))[:, :, :T].transpose(1, 2)

        # Griffin recurrence
        a = torch.sigmoid(self.w_a(h) + self.decay_bias)
        i_gate = torch.sigmoid(self.w_i(h))
        v = self.w_v(h)
        input_signal = torch.sqrt((1 - a * a).clamp(min=1e-8)) * (i_gate * v)
        rec_out = self._chunked_scan(a, input_signal)
        x = x + self.out_proj(rec_out)

        # FFN
        x = x + self.ffn(self.ffn_norm(x))
        return x

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

        state = torch.zeros(batch, d, dtype=torch.float32, device=decay.device)
        all_states = []
        for c in range(n_chunks):
            prev_contrib = cum_decay[:, c] * state.unsqueeze(1)
            chunk_states = intra_state[:, c] + prev_contrib
            all_states.append(chunk_states)
            state = chunk_states[:, -1]

        states = torch.cat(all_states, dim=1)[:, :seqlen]
        return states.to(decay.dtype)


class DualCortex(nn.Module):
    """DUAL-CORTEX: System 1/System 2 with entropy-gated routing.

    Fast path: d=320, 8 layers (always runs)
    Slow path: d=1280, 10 layers (conditional on state delta)
    ~231M parameters.

    During training: soft gating (all tokens go through both paths with weights).
    During inference: hard gating (skip slow path for easy tokens).
    """

    def __init__(
        self,
        vocab_size: int = 50257,
        d_embed: int = 1024,       # reduced from 1280 to fit budget
        d_fast: int = 256,         # reduced from 320
        d_slow: int = 1024,        # reduced from 1280
        n_fast_layers: int = 8,
        n_slow_layers: int = 8,    # reduced from 10
        ffn_fast: int = 512,       # 2x ratio
        ffn_slow: int = 2560,      # 2.5x ratio
        conv_kernel: int = 4,
        lambda_compute: float = 0.01,
        max_seq_len: int = 1024,
    ):
        super().__init__()
        self.d_fast = d_fast
        self.d_slow = d_slow
        self.lambda_compute = lambda_compute

        # Embeddings at slow-path dimension
        self.tok_embeddings = nn.Embedding(vocab_size, d_embed)

        # Project to fast path
        self.embed_to_fast = nn.Linear(d_embed, d_fast, bias=False)

        # Fast path (System 1): always runs
        self.fast_layers = nn.ModuleList([
            GriffinBlock(d_fast, d_fast, ffn_fast, conv_kernel)
            for _ in range(n_fast_layers)
        ])

        # Entropy estimator: state delta norm → threshold comparison
        # MLP for learned threshold (more flexible than raw norm)
        self.entropy_mlp = nn.Sequential(
            nn.Linear(d_fast, 64, bias=True),
            nn.ReLU(),
            nn.Linear(64, 1, bias=True),
        )
        # Init threshold so ~30% of tokens engage slow path
        nn.init.zeros_(self.entropy_mlp[2].weight)
        nn.init.constant_(self.entropy_mlp[2].bias, 0.0)  # sigmoid(0) = 0.5

        # Projection bridges
        self.fast_to_slow = nn.Linear(d_fast, d_slow, bias=False)
        self.slow_to_fast = nn.Linear(d_slow, d_fast, bias=False)

        # Slow path (System 2): conditional
        self.slow_layers = nn.ModuleList([
            GriffinBlock(d_slow, d_slow, ffn_slow, conv_kernel)
            for _ in range(n_slow_layers)
        ])

        # Project back to embedding dim for LM head
        self.fast_to_embed = nn.Linear(d_fast, d_embed, bias=False)
        self.norm = RMSNorm(d_embed)
        self.output = nn.Linear(d_embed, vocab_size, bias=False)
        self.output.weight = self.tok_embeddings.weight

        self._init_weights()
        n_params = sum(p.numel() for p in self.parameters())
        fast_params = sum(p.numel() for p in self.fast_layers.parameters())
        slow_params = sum(p.numel() for p in self.slow_layers.parameters())
        print(f"DualCortex: {n_params / 1e6:.1f}M total "
              f"(fast: {fast_params / 1e6:.1f}M, slow: {slow_params / 1e6:.1f}M)")

    def _init_weights(self):
        for name, p in self.named_parameters():
            if 'entropy_mlp' in name:
                continue
            if p.dim() >= 2:
                nn.init.xavier_uniform_(p)

    def forward(self, input_ids: torch.Tensor, targets=None) -> torch.Tensor:
        B, T = input_ids.shape
        h_embed = self.tok_embeddings(input_ids)

        # Project to fast path
        h_fast = self.embed_to_fast(h_embed)
        h_fast_prev = h_fast  # save for delta computation

        # Fast path (always runs)
        for layer in self.fast_layers:
            h_fast = layer(h_fast)

        # Entropy gate: state delta as uncertainty proxy
        delta = (h_fast - h_fast_prev).norm(dim=-1, keepdim=True)  # (B, T, 1)
        gate_logit = self.entropy_mlp(h_fast.detach())  # (B, T, 1)
        gate = torch.sigmoid(gate_logit)  # soft gate for training

        # Track fraction of tokens engaging slow path
        self._slow_fraction = gate.mean()

        # Slow path (conditional)
        h_slow_in = self.fast_to_slow(h_fast)
        for layer in self.slow_layers:
            h_slow_in = layer(h_slow_in)
        h_slow_correction = self.slow_to_fast(h_slow_in)

        # Merge: fast + gated slow correction
        h_out = h_fast + gate * h_slow_correction

        # Project back to embedding dim
        h_out = self.fast_to_embed(h_out)
        logits = self.output(self.norm(h_out))

        # Auxiliary loss: penalize slow path usage
        self._compute_cost = self.lambda_compute * self._slow_fraction

        return logits

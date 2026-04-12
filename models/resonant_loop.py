"""
RESONANT-LOOP: Cache-Resident Iterative Shared Block with Adaptive Halting.

Single shared block (~7.3M params) iterated 8-16 times per token.
SCORE-style damped residuals + ACT adaptive halting.
In int4, block fits in L2 (3.7 MB < 6 MB) — iterations 2-16 read from cache.

~58.8M unique parameters, ~168M effective (16 iterations).

Usage:
    python -m halo_training --model models/resonant_loop.py --class-name ResonantLoop --dataset babylm
    python -m halo_training --model models/resonant_loop.py --class-name ResonantLoop --smoke
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
class ResonantLoopConfig:
    vocab_size: int = 50257
    d_model: int = 1024
    ffn_inner: int = 2048        # smaller SwiGLU (2x ratio, not 2.5x)
    conv_kernel: int = 4
    max_iterations: int = 16
    damping_init: float = -2.2   # sigmoid(-2.2) ≈ 0.1
    lambda_ponder: float = 0.01
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


class SharedBlock(nn.Module):
    """The single shared block that gets iterated.

    RMSNorm → Conv1d → Channel Mixer → residual
    RMSNorm → SwiGLU → residual
    ~7.3M params.
    """

    def __init__(self, cfg: ResonantLoopConfig):
        super().__init__()
        d = cfg.d_model

        # Conv path
        self.conv_norm = RMSNorm(d)
        if _HAS_CAUSAL_CONV1D:
            self.conv_weight = nn.Parameter(torch.randn(d, cfg.conv_kernel))
            self.conv_bias = nn.Parameter(torch.zeros(d))
            self._use_fast_conv = True
        else:
            self.conv = nn.Conv1d(
                d, d, kernel_size=cfg.conv_kernel,
                padding=cfg.conv_kernel - 1, groups=d, bias=True,
            )
            self._use_fast_conv = False

        # Channel mixer (d → d)
        self.channel_mixer = nn.Linear(d, d, bias=True)

        # FFN path
        self.ffn_norm = RMSNorm(d)
        self.ffn = SwiGLU(d, cfg.ffn_inner)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, D = x.shape

        # Conv + mixer path
        h = self.conv_norm(x)
        if self._use_fast_conv:
            h = causal_conv1d_fn(
                h.transpose(1, 2), self.conv_weight, self.conv_bias
            ).transpose(1, 2)
        else:
            h = self.conv(h.transpose(1, 2))[:, :, :T].transpose(1, 2)
        h = self.channel_mixer(h)
        x = x + h

        # SwiGLU FFN
        x = x + self.ffn(self.ffn_norm(x))
        return x


class ResonantLoop(nn.Module):
    """RESONANT-LOOP: Iterative shared block.

    Single SharedBlock iterated max_iterations times with:
    - Per-iteration embeddings for depth conditioning
    - SCORE damped residuals (h = (1-d)*h + d*block(h))
    - ACT halting gates for adaptive depth (training: always run all iters)

    ~58.8M unique params, ~168M effective at 16 iterations.
    """

    def __init__(
        self,
        vocab_size: int = 50257,
        d_model: int = 1024,
        ffn_inner: int = 2048,
        conv_kernel: int = 4,
        max_iterations: int = 16,
        damping_init: float = -2.2,
        lambda_ponder: float = 0.01,
        max_seq_len: int = 1024,
    ):
        super().__init__()
        self.max_iterations = max_iterations
        self.lambda_ponder = lambda_ponder

        cfg = ResonantLoopConfig(
            vocab_size=vocab_size, d_model=d_model, ffn_inner=ffn_inner,
            conv_kernel=conv_kernel, max_iterations=max_iterations,
            damping_init=damping_init, lambda_ponder=lambda_ponder,
            max_seq_len=max_seq_len,
        )

        self.tok_embeddings = nn.Embedding(vocab_size, d_model)

        # Single shared block
        self.block = SharedBlock(cfg)

        # Per-iteration embeddings (NOT shared — each gives depth info)
        self.iter_embeddings = nn.Parameter(torch.zeros(max_iterations, d_model))
        nn.init.normal_(self.iter_embeddings, std=0.02)

        # SCORE damping scalar (learned, shared across iterations)
        self.damping_logit = nn.Parameter(torch.tensor(damping_init))

        # ACT halting gates (per-iteration, NOT shared)
        self.halt_projs = nn.ModuleList([
            nn.Linear(d_model, 1, bias=True) for _ in range(max_iterations)
        ])
        # Init halting bias: early iterations unlikely to halt, late ones likely
        with torch.no_grad():
            for i, proj in enumerate(self.halt_projs):
                nn.init.zeros_(proj.weight)
                if i < 4:
                    proj.bias.fill_(-2.0)    # unlikely halt
                elif i < 8:
                    proj.bias.fill_(-0.5)
                elif i < 12:
                    proj.bias.fill_(0.0)
                else:
                    proj.bias.fill_(1.0)     # likely halt

        self.norm = RMSNorm(d_model)
        self.output = nn.Linear(d_model, vocab_size, bias=False)
        self.output.weight = self.tok_embeddings.weight

        self._init_weights()
        n_params = sum(p.numel() for p in self.parameters())
        print(f"ResonantLoop: {n_params / 1e6:.1f}M unique parameters "
              f"(~{n_params * max_iterations / 1e6:.0f}M effective at {max_iterations} iterations)")

    def _init_weights(self):
        for name, p in self.named_parameters():
            if 'halt_projs' in name or 'iter_embeddings' in name or 'damping_logit' in name:
                continue
            if p.dim() >= 2:
                nn.init.xavier_uniform_(p)

    def forward(self, input_ids: torch.Tensor, targets=None) -> torch.Tensor:
        B, T = input_ids.shape
        h = self.tok_embeddings(input_ids)

        d = torch.sigmoid(self.damping_logit)  # SCORE damping ∈ (0, 1)

        # ACT: accumulate halting probabilities
        cum_halt = torch.zeros(B, T, 1, device=h.device)
        remainder = torch.ones(B, T, 1, device=h.device)
        output = torch.zeros_like(h)

        for i in range(self.max_iterations):
            # Depth conditioning
            h_in = h + self.iter_embeddings[i]

            # Shared block
            h_out = self.block(h_in)

            # SCORE damped residual
            h = (1 - d) * h + d * h_out

            # ACT halting
            p_halt = torch.sigmoid(self.halt_projs[i](h))  # (B, T, 1)

            # Weight this iteration's output by halting probability
            if i < self.max_iterations - 1:
                still_running = (cum_halt < 1.0).float()
                p_halt = p_halt * still_running
                halted_now = torch.min(p_halt, remainder)
                output = output + halted_now * h
                cum_halt = cum_halt + halted_now
                remainder = remainder - halted_now
            else:
                # Last iteration: dump all remaining probability
                output = output + remainder * h

        logits = self.output(self.norm(output))

        # During training, add ponder cost as auxiliary loss
        # The trainer handles the main CE loss; we store ponder cost for access
        self._ponder_cost = self.lambda_ponder * cum_halt.mean()

        return logits

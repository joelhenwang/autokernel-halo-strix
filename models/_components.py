"""Shared model components extracted from amadeus.py, argus.py, and llama_7b.py.

Single source of truth for:
- RMSNorm
- SwiGLU
- GatedConv
- precompute_freqs_cis
- apply_rotary_emb

Previously defined redundantly in 9+ model files.
When a component is defined in 3+ files, it belongs here.
"""

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
    """SwiGLU FFN with fused gate+up projection."""

    def __init__(self, d_model: int, ffn_inner: int):
        super().__init__()
        self.w_gate_up = nn.Linear(d_model, 2 * ffn_inner, bias=False)
        self.w_down = nn.Linear(ffn_inner, d_model, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gate, up = self.w_gate_up(x).chunk(2, dim=-1)
        return self.w_down(F.silu(gate) * up)


class GatedConv(nn.Module):
    """Gated short convolution for local pattern matching.

    B, C, h_tilde = proj(x).chunk(3)
    y = B * h_tilde  (element-wise gate)
    z = causal_conv1d(y)
    out = C * z       (output gate)
    """

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
            z = self.conv(y.transpose(1, 2))[:, :, :T]
            z = z.transpose(1, 2)
        return c * z


def precompute_freqs_cis(dim: int, end: int, theta: float = 10000.0) -> torch.Tensor:
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2).float() / dim))
    t = torch.arange(end, dtype=torch.float32)
    freqs = torch.outer(t, freqs)
    return torch.polar(torch.ones_like(freqs), freqs)


def apply_rotary_emb(
    xq: torch.Tensor, xk: torch.Tensor, freqs_cis: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    xq_ = torch.view_as_complex(xq.float().reshape(*xq.shape[:-1], -1, 2))
    xk_ = torch.view_as_complex(xk.float().reshape(*xk.shape[:-1], -1, 2))
    freqs = freqs_cis[None, :xq_.shape[1], None, :]
    xq_out = torch.view_as_real(xq_ * freqs).flatten(3)
    xk_out = torch.view_as_real(xk_ * freqs).flatten(3)
    return xq_out.type_as(xq), xk_out.type_as(xk)
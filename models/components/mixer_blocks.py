"""Mixer blocks: GriffinGlobalBlock.

GriffinGlobalBlock: Multi-scale element-wise linear recurrence (O(T)) + SwiGLU FFN.
Drop-in replacement for NoPEGQABlock at the same-interface level.
Uses GriffinRecurrence from models/tempest.py with multi-scale decay spectrum.
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from models._components import RMSNorm, SwiGLU
from models.tempest import GriffinRecurrence


class GriffinGlobalBlock(nn.Module):
    """Global context mixer using element-wise linear recurrence (O(T)).

    Architecture: pre_norm -> GriffinRecurrence -> out_proj -> residual
                  ffn_norm -> SwiGLU FFN -> residual

    This replaces quadratic attention (NoPEGQABlock) with an O(T) alternative.
    The GriffinRecurrence uses multi-scale decay bias (fast/medium/slow) to
    capture N-grams, clause structure, and topic tracking respectively.

    Interface is compatible with HyPEShortConvBlock: forward(x, freqs_cis) -> Tensor.
    The freqs_cis argument is accepted but ignored (no positional encoding needed).
    """

    _skip_autokernel = True  # Non-standard forward signature for pattern matching

    def __init__(self, d_model: int, ffn_inner: int, d_rec: int = None):
        """
        Args:
            d_model: Hidden dimension (768 for OdinFlat-scale)
            ffn_inner: SwiGLU inner dimension (2816 for OdinFlat-scale)
            d_rec: Recurrence hidden dimension. Default d_model (full-rank).
        """
        super().__init__()
        if d_rec is None:
            d_rec = d_model
        self.d_model = d_model
        self.d_rec = d_rec

        self.pre_norm = RMSNorm(d_model)
        self.griffin = GriffinRecurrence(d_model, d_rec)
        self.out_proj = nn.Linear(d_rec, d_model, bias=False)
        self.ffn_norm = RMSNorm(d_model)
        self.ffn = SwiGLU(d_model, ffn_inner)

    def forward(self, x: torch.Tensor, freqs_cis=None, **kwargs) -> torch.Tensor:
        """Forward pass. freqs_cis accepted for interface compatibility but unused."""
        # Global recurrence
        h = self.griffin(self.pre_norm(x))
        x = x + self.out_proj(h)
        # FFN
        x = x + self.ffn(self.ffn_norm(x))
        return x

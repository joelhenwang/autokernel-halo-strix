"""
MatFormer — Composable Nested Submodel Training.

Trains one model containing nested submodels as strict weight prefixes.
At inference, extract smaller models by slicing weight matrices — zero
post-training cost. Based on the MatFormer paper (arXiv 2310.07707).

Components:
  MatFormerSwiGLU     — Drop-in SwiGLU replacement with nested FFN width
  MatFormerAttention   — GQA attention with nested query head counts
  extract_submodel()   — Extract a clean model at a given granularity

Usage:
    # Wrap existing SwiGLU
    mf_ffn = MatFormerSwiGLU.from_swiglu(existing_ffn, MatFormerConfig())

    # Training: random granularity sampled per step
    logits = model(input_ids)

    # Inference: extract 1/4 submodel
    small_ffn = mf_ffn.extract_submodel(0.25)
"""

import random
from dataclasses import dataclass, field
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.tempest import SwiGLU


def _round_to_multiple(n: int, multiple: int = 128) -> int:
    """Round up to nearest multiple for Tensile tile alignment."""
    return ((n + multiple - 1) // multiple) * multiple


@dataclass
class MatFormerConfig:
    granularities: Tuple[float, ...] = (0.125, 0.25, 0.5, 1.0)
    mode: str = "train"                     # "train" | "eval" | "extract"
    fixed_granularity: Optional[float] = None  # for eval/extract


class MatFormerSwiGLU(nn.Module):
    """SwiGLU with MatFormer nested granularity training.

    Zero extra parameters. During training, randomly samples an FFN width
    from the granularity set. The weight prefix property means smaller
    submodels use the first m neurons of w_gate_up and w_down.

    For fused gate+up layout (w_gate_up has shape [2*ffn_inner, d_model]):
        granularity g → m = round_to_128(g * ffn_inner)
        slice w_gate_up.weight[:2*m, :] (first m gate + first m up)
        slice w_down.weight[:, :m]      (first m input dims)
    """

    def __init__(self, d_model: int, ffn_inner: int, config: MatFormerConfig):
        super().__init__()
        self.d_model = d_model
        self.ffn_inner = ffn_inner
        self.config = config

        # Same weights as standard SwiGLU
        self.w_gate_up = nn.Linear(d_model, 2 * ffn_inner, bias=False)
        self.w_down = nn.Linear(ffn_inner, d_model, bias=False)

        # Pre-compute aligned slice widths
        self.slice_widths = []
        for g in config.granularities:
            m = _round_to_multiple(int(g * ffn_inner), 128)
            m = min(m, ffn_inner)
            self.slice_widths.append(m)

    @classmethod
    def from_swiglu(cls, swiglu: SwiGLU, config: MatFormerConfig) -> "MatFormerSwiGLU":
        """Create MatFormerSwiGLU from an existing SwiGLU, copying weights."""
        d_model = swiglu.w_gate_up.in_features
        ffn_inner = swiglu.w_down.in_features
        mf = cls(d_model, ffn_inner, config)
        mf.w_gate_up.weight.data.copy_(swiglu.w_gate_up.weight.data)
        mf.w_down.weight.data.copy_(swiglu.w_down.weight.data)
        return mf

    def _get_active_width(self) -> int:
        """Select FFN width for this forward pass."""
        if self.training and self.config.mode == "train":
            idx = random.randint(0, len(self.slice_widths) - 1)
            return self.slice_widths[idx]
        if self.config.fixed_granularity is not None:
            m = _round_to_multiple(int(self.config.fixed_granularity * self.ffn_inner), 128)
            return min(m, self.ffn_inner)
        return self.ffn_inner

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        m = self._get_active_width()

        if m == self.ffn_inner:
            # Full forward — no slicing overhead
            gate, up = self.w_gate_up(x).chunk(2, dim=-1)
            return self.w_down(F.silu(gate) * up)

        # Sliced forward: first 2*m columns of gate_up, first m columns of down
        gate_up = F.linear(x, self.w_gate_up.weight[:2 * m, :])
        gate, up = gate_up.chunk(2, dim=-1)
        hidden = F.silu(gate) * up
        return F.linear(hidden, self.w_down.weight[:, :m])

    def extract_submodel(self, granularity: float) -> SwiGLU:
        """Extract a plain SwiGLU with sliced weights. Zero overhead at inference."""
        m = _round_to_multiple(int(granularity * self.ffn_inner), 128)
        m = min(m, self.ffn_inner)

        ffn = SwiGLU(self.d_model, m)
        ffn.w_gate_up.weight.data = self.w_gate_up.weight.data[:2 * m, :].clone()
        ffn.w_down.weight.data = self.w_down.weight.data[:, :m].clone()
        return ffn


class MatFormerAttention(nn.Module):
    """GQA Attention with MatFormer nested query head counts.

    During training, randomly activates a subset of query heads. KV heads
    stay fixed (already small). Enables shared KV cache for speculative
    decoding: draft model's heads are a prefix of verifier's.

    Args:
        d_model: Model dimension
        n_heads: Total number of query heads (e.g., 8)
        n_kv_heads: Number of KV heads (e.g., 2)
        head_dim: Per-head dimension (e.g., 128)
        config: MatFormerConfig with granularities
    """

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        n_kv_heads: int,
        head_dim: int,
        config: MatFormerConfig,
    ):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.n_kv_heads = n_kv_heads
        self.head_dim = head_dim
        self.config = config

        # Full-size projections
        self.wq = nn.Linear(d_model, n_heads * head_dim, bias=False)
        self.wk = nn.Linear(d_model, n_kv_heads * head_dim, bias=False)
        self.wv = nn.Linear(d_model, n_kv_heads * head_dim, bias=False)
        self.wo = nn.Linear(n_heads * head_dim, d_model, bias=False)

        # Pre-compute valid head counts (minimum = n_kv_heads)
        self.head_counts = sorted(set(
            max(n_kv_heads, int(g * n_heads)) for g in config.granularities
        ))

    def _get_active_heads(self) -> int:
        if self.training and self.config.mode == "train":
            return random.choice(self.head_counts)
        if self.config.fixed_granularity is not None:
            return max(self.n_kv_heads, int(self.config.fixed_granularity * self.n_heads))
        return self.n_heads

    def forward(self, x: torch.Tensor, freqs_cis: torch.Tensor) -> torch.Tensor:
        B, T, _ = x.shape
        n_active = self._get_active_heads()
        n_rep = n_active // self.n_kv_heads

        # Q: compute all, mask inactive
        q_full = self.wq(x).view(B, T, self.n_heads, self.head_dim)
        q = q_full[:, :, :n_active, :]  # prefix slice

        # K, V: always full (KV heads are small and shared)
        k = self.wk(x).view(B, T, self.n_kv_heads, self.head_dim)
        v = self.wv(x).view(B, T, self.n_kv_heads, self.head_dim)

        # RoPE
        from models.prometheus import apply_rotary_emb
        q, k = apply_rotary_emb(q, k, freqs_cis)

        q = q.transpose(1, 2)  # (B, n_active, T, head_dim)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        # GQA expand
        if n_rep > 1:
            k = k.repeat_interleave(n_rep, dim=1)
            v = v.repeat_interleave(n_rep, dim=1)

        # Attention
        y = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        y = y.transpose(1, 2).contiguous().view(B, T, n_active * self.head_dim)

        # Output projection: slice wo to match active heads
        if n_active == self.n_heads:
            return self.wo(y)
        return F.linear(y, self.wo.weight[:, :n_active * self.head_dim])

    def extract_submodel(self, granularity: float):
        """Extract a plain GQAAttentionLayer at the given head count."""
        from models.prometheus import GQAAttentionLayer
        n_active = max(self.n_kv_heads, int(granularity * self.n_heads))
        hd = self.head_dim

        attn = GQAAttentionLayer(self.d_model, n_active, self.n_kv_heads, hd)
        attn.wq.weight.data = self.wq.weight.data[:n_active * hd, :].clone()
        attn.wk.weight.data = self.wk.weight.data.clone()
        attn.wv.weight.data = self.wv.weight.data.clone()
        attn.wo.weight.data = self.wo.weight.data[:, :n_active * hd].clone()
        return attn


def extract_model_at_granularity(model: nn.Module, granularity: float) -> nn.Module:
    """Walk a model and extract all MatFormer modules at the given granularity.

    Replaces MatFormerSwiGLU → SwiGLU, MatFormerAttention → GQAAttentionLayer.
    Returns a clean model with no MatFormer overhead for inference.
    """
    import copy
    model = copy.deepcopy(model)

    replacements = []
    for name, module in model.named_modules():
        if isinstance(module, (MatFormerSwiGLU, MatFormerAttention)):
            replacements.append((name, module.extract_submodel(granularity)))

    for name, new_module in replacements:
        parts = name.split(".")
        parent = model
        for part in parts[:-1]:
            parent = getattr(parent, part)
        setattr(parent, parts[-1], new_module)

    return model

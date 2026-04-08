"""
AutoKernel — Optimization pattern definitions.

Each Pattern knows how to detect and replace a specific module type
with an optimized HIP kernel wrapper.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Pattern base
# ---------------------------------------------------------------------------

@dataclass
class Pattern:
    name: str
    priority: int       # higher = applied first
    op_speedup: float   # benchmark speedup vs PyTorch (informational)

    def matches(self, name: str, module: nn.Module, model: nn.Module) -> bool:
        raise NotImplementedError

    def apply(self, name: str, module: nn.Module, model: nn.Module) -> nn.Module:
        """Return the replacement module."""
        raise NotImplementedError


# ---------------------------------------------------------------------------
# Detection helpers
# ---------------------------------------------------------------------------

# Attribute name aliases for HuggingFace model compatibility
_QKV_ALIASES = [
    ("wq", "wk", "wv", "wo"),                           # LLaMA (ours)
    ("q_proj", "k_proj", "v_proj", "o_proj"),            # HuggingFace LLaMA/Mistral
]

_SWIGLU_ALIASES = [
    ("w1", "w2", "w3"),                                  # LLaMA (ours): gate, down, up
    ("gate_proj", "down_proj", "up_proj"),                # HuggingFace LLaMA/Mistral
]

_BLOCK_ALIASES = [
    ("attention", "attention_norm", "feed_forward", "ffn_norm"),            # ours
    ("self_attn", "input_layernorm", "mlp", "post_attention_layernorm"),   # HuggingFace
]

_RMSNORM_CLASS_NAMES = {
    "RMSNorm", "LlamaRMSNorm", "MistralRMSNorm",
    "T5LayerNorm", "GemmaRMSNorm", "Qwen2RMSNorm",
}


def _has_linear_attrs(module: nn.Module, attrs: tuple) -> bool:
    """Check if module has all named attributes and at least the first is nn.Linear."""
    return (
        all(hasattr(module, a) for a in attrs)
        and isinstance(getattr(module, attrs[0], None), nn.Linear)
    )


def _find_qkv_attrs(module: nn.Module) -> Optional[tuple]:
    """Return (q, k, v, o) attribute names if module has QKV projections."""
    for alias in _QKV_ALIASES:
        if _has_linear_attrs(module, alias):
            return alias
    return None


def _find_swiglu_attrs(module: nn.Module) -> Optional[tuple]:
    """Return (gate, down, up) attribute names if module has SwiGLU pattern."""
    for alias in _SWIGLU_ALIASES:
        if _has_linear_attrs(module, alias):
            return alias
    return None


def _find_block_attrs(module: nn.Module) -> Optional[tuple]:
    """Return (attn, attn_norm, ffn, ffn_norm) names if module is a TransformerBlock."""
    for alias in _BLOCK_ALIASES:
        attn, attn_norm, ffn, ffn_norm = alias
        if (hasattr(module, attn) and hasattr(module, attn_norm)
                and hasattr(module, ffn) and hasattr(module, ffn_norm)
                and hasattr(getattr(module, ffn_norm, None), "weight")):
            return alias
    return None


def _find_complex_freqs(model: nn.Module) -> Optional[torch.Tensor]:
    """Find complex freqs_cis buffer for RoPE."""
    for bname, buf in model.named_buffers():
        if "freqs_cis" in bname and buf.is_complex():
            return buf
    return None


# ---------------------------------------------------------------------------
# Wrapper classes (lightweight, self-contained)
# ---------------------------------------------------------------------------
# These mirror verify.py's wrappers but are independent so autokernel/
# has no import dependency on verify.py.


class _RMSNormReplacement(nn.Module):
    def __init__(self, original: nn.Module, kernel_fn: Callable):
        super().__init__()
        self.weight = original.weight
        self.eps = getattr(original, "eps", 1e-6)
        self.kernel_fn = kernel_fn

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        orig_shape = x.shape
        if x.dtype == torch.float16:
            return self.kernel_fn(x.view(-1, x.shape[-1]), self.weight).view(orig_shape)
        norm = torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        return x * norm * self.weight


class _LayerNormReplacement(nn.Module):
    def __init__(self, original: nn.LayerNorm, kernel_fn: Callable):
        super().__init__()
        self.weight = original.weight
        self.bias = original.bias
        self.eps = original.eps
        self.normalized_shape = original.normalized_shape
        self.kernel_fn = kernel_fn

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dtype == torch.float16:
            return self.kernel_fn(x, self.weight, self.bias)
        return F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)


class _SiluGateMulReplacement(nn.Module):
    def __init__(self, original: nn.Module, kernel_fn: Callable,
                 gate_attr: str, down_attr: str, up_attr: str):
        super().__init__()
        self.gate_proj = getattr(original, gate_attr)
        self.down_proj = getattr(original, down_attr)
        self.up_proj = getattr(original, up_attr)
        self.kernel_fn = kernel_fn

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gate = self.gate_proj(x)
        up = self.up_proj(x)
        if gate.dtype == torch.float16:
            activated = self.kernel_fn(gate.contiguous(), up.contiguous())
        else:
            activated = F.silu(gate) * up
        return self.down_proj(activated)


class _FusedQKVAttentionReplacement(nn.Module):
    def __init__(self, original: nn.Module,
                 q_attr: str, k_attr: str, v_attr: str, o_attr: str,
                 rotary_fn: Optional[Callable] = None,
                 freqs_cis_complex: Optional[torch.Tensor] = None):
        super().__init__()
        q_proj = getattr(original, q_attr)
        k_proj = getattr(original, k_attr)
        v_proj = getattr(original, v_attr)
        self.wo = getattr(original, o_attr)

        self.n_heads = getattr(original, "n_heads", getattr(original, "num_heads", 1))
        self.n_kv_heads = getattr(original, "n_kv_heads", getattr(original, "num_key_value_heads", self.n_heads))
        self.head_dim = getattr(original, "head_dim", q_proj.out_features // self.n_heads)
        self.n_rep = self.n_heads // self.n_kv_heads

        # Fused QKV weight
        self.w_qkv = nn.Linear(q_proj.in_features,
                                q_proj.out_features + k_proj.out_features + v_proj.out_features,
                                bias=False)
        self.w_qkv.weight.data = torch.cat([
            q_proj.weight.data, k_proj.weight.data, v_proj.weight.data,
        ], dim=0)
        self.q_size = q_proj.out_features
        self.k_size = k_proj.out_features

        self.rotary_fn = rotary_fn
        if freqs_cis_complex is not None and freqs_cis_complex.is_complex():
            self.register_buffer("_cos", freqs_cis_complex.real.clone(), persistent=False)
            self.register_buffer("_sin", freqs_cis_complex.imag.clone(), persistent=False)
        else:
            self._cos = None
            self._sin = None

    def forward(self, x: torch.Tensor, freqs_cis: torch.Tensor) -> torch.Tensor:
        B, T, _ = x.shape
        qkv = self.w_qkv(x)
        q = qkv[..., :self.q_size].view(B, T, self.n_heads, self.head_dim)
        k = qkv[..., self.q_size:self.q_size + self.k_size].view(B, T, self.n_kv_heads, self.head_dim)
        v = qkv[..., self.q_size + self.k_size:].view(B, T, self.n_kv_heads, self.head_dim)

        if self.rotary_fn is not None:
            q = q.transpose(1, 2).contiguous()
            k = k.transpose(1, 2).contiguous()
            v = v.transpose(1, 2)
            if freqs_cis.is_complex():
                cos, sin = freqs_cis[:T].real, freqs_cis[:T].imag
            elif self._cos is not None:
                cos, sin = self._cos[:T], self._sin[:T]
            else:
                raise RuntimeError("No complex freqs_cis available for rotary")
            q = self.rotary_fn(q, cos, sin)
            k = self.rotary_fn(k, cos, sin)
        else:
            try:
                from models.llama_7b import apply_rotary_emb
                q, k = apply_rotary_emb(q, k, freqs_cis)
            except ImportError:
                pass
            q = q.transpose(1, 2)
            k = k.transpose(1, 2)
            v = v.transpose(1, 2)

        if self.n_rep > 1:
            k = k.repeat_interleave(self.n_rep, dim=1)
            v = v.repeat_interleave(self.n_rep, dim=1)
        y = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        return self.wo(y.transpose(1, 2).contiguous().view(B, T, -1))


class _FusedResidualRMSNormBlockReplacement(nn.Module):
    def __init__(self, original: nn.Module, kernel_fn_dual: Callable,
                 attn_attr: str, attn_norm_attr: str,
                 ffn_attr: str, ffn_norm_attr: str):
        super().__init__()
        self.attention = getattr(original, attn_attr)
        self.attention_norm = getattr(original, attn_norm_attr)
        self.feed_forward = getattr(original, ffn_attr)
        self.ffn_norm_weight = getattr(original, ffn_norm_attr).weight
        self.kernel_fn_dual = kernel_fn_dual

    def forward(self, x: torch.Tensor, freqs_cis: torch.Tensor) -> torch.Tensor:
        attn_out = self.attention(self.attention_norm(x), freqs_cis)
        if attn_out.dtype == torch.float16:
            hidden, normed = self.kernel_fn_dual(
                attn_out.view(-1, attn_out.shape[-1]),
                x.view(-1, x.shape[-1]),
                self.ffn_norm_weight,
            )
            hidden = hidden.view(attn_out.shape)
            normed = normed.view(attn_out.shape)
        else:
            hidden = x + attn_out
            rms = torch.rsqrt(hidden.pow(2).mean(-1, keepdim=True) + 1e-6)
            normed = hidden * rms * self.ffn_norm_weight
        return hidden + self.feed_forward(normed)


# ---------------------------------------------------------------------------
# Pattern implementations
# ---------------------------------------------------------------------------

class FusedResidualRMSNormPattern(Pattern):
    def __init__(self):
        super().__init__("fused_residual_rmsnorm", priority=100, op_speedup=6.6)

    def matches(self, name: str, module: nn.Module, model: nn.Module) -> bool:
        return _find_block_attrs(module) is not None

    def apply(self, name: str, module: nn.Module, model: nn.Module) -> nn.Module:
        from kernels.hip.fused_residual_add_rmsnorm import kernel_fn_dual
        attrs = _find_block_attrs(module)
        return _FusedResidualRMSNormBlockReplacement(
            module, kernel_fn_dual, *attrs
        )


class FusedQKVPattern(Pattern):
    def __init__(self):
        super().__init__("fused_qkv", priority=90, op_speedup=1.3)
        self._freqs_cis = None

    def matches(self, name: str, module: nn.Module, model: nn.Module) -> bool:
        return _find_qkv_attrs(module) is not None

    def apply(self, name: str, module: nn.Module, model: nn.Module) -> nn.Module:
        attrs = _find_qkv_attrs(module)
        rotary_fn = None
        try:
            from kernels.hip.rotary_embedding import kernel_fn_fp32
            rotary_fn = kernel_fn_fp32
        except ImportError:
            pass
        freqs = _find_complex_freqs(model)
        return _FusedQKVAttentionReplacement(module, *attrs, rotary_fn, freqs)


class RMSNormPattern(Pattern):
    def __init__(self):
        super().__init__("rmsnorm", priority=50, op_speedup=3.3)

    def matches(self, name: str, module: nn.Module, model: nn.Module) -> bool:
        cls_name = type(module).__name__
        return (
            cls_name in _RMSNORM_CLASS_NAMES
            or (hasattr(module, "weight")
                and hasattr(module, "eps")
                and not hasattr(module, "bias")
                and cls_name.lower().endswith("norm")
                and not isinstance(module, nn.LayerNorm))
        )

    def apply(self, name: str, module: nn.Module, model: nn.Module) -> nn.Module:
        from kernels.hip.rmsnorm import kernel_fn
        return _RMSNormReplacement(module, kernel_fn)


class LayerNormPattern(Pattern):
    def __init__(self):
        super().__init__("layernorm", priority=50, op_speedup=1.06)

    def matches(self, name: str, module: nn.Module, model: nn.Module) -> bool:
        return isinstance(module, nn.LayerNorm)

    def apply(self, name: str, module: nn.Module, model: nn.Module) -> nn.Module:
        from kernels.hip.layernorm import kernel_fn
        return _LayerNormReplacement(module, kernel_fn)


class SiluGateMulPattern(Pattern):
    def __init__(self):
        super().__init__("silu_gate_mul", priority=40, op_speedup=1.6)

    def matches(self, name: str, module: nn.Module, model: nn.Module) -> bool:
        return _find_swiglu_attrs(module) is not None

    def apply(self, name: str, module: nn.Module, model: nn.Module) -> nn.Module:
        from kernels.hip.silu_gate_mul import kernel_fn
        attrs = _find_swiglu_attrs(module)
        return _SiluGateMulReplacement(module, kernel_fn, *attrs)


class RotaryEmbeddingPattern(Pattern):
    def __init__(self):
        super().__init__("rotary_embedding", priority=30, op_speedup=3.7)

    def matches(self, name: str, module: nn.Module, model: nn.Module) -> bool:
        # Skip if already replaced by FusedQKV (which integrates rotary)
        if isinstance(module, _FusedQKVAttentionReplacement):
            return False
        return (_find_qkv_attrs(module) is not None
                and _find_complex_freqs(model) is not None)

    def apply(self, name: str, module: nn.Module, model: nn.Module) -> nn.Module:
        # Standalone rotary — only used when fused_qkv is excluded.
        # Import the verify.py wrapper for this (it's the most complex one).
        from kernels.hip.rotary_embedding import kernel_fn_fp32
        from verify import _RotaryAttentionWrapper
        freqs = _find_complex_freqs(model)
        return _RotaryAttentionWrapper(module, kernel_fn_fp32, freqs)


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

ALL_PATTERNS: List[Pattern] = [
    FusedResidualRMSNormPattern(),
    FusedQKVPattern(),
    RMSNormPattern(),
    LayerNormPattern(),
    SiluGateMulPattern(),
    RotaryEmbeddingPattern(),
]

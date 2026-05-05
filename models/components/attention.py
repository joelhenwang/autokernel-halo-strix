"""Attention modules: GQA, XSA/Coda, NoPE (HyPE).

Attention:       Standard GQA with RoPE + QK-Norm (from argus_prime).
CodaAttention:   Attention + XSA (exclusive self-attention) + MoDA depth KVs (from jormungandr_halo).
NoPECodaAttention: Content-only GQA — no RoPE, QK-Norm mandatory. For HyPE length generalization (ODIN-HALO).
"""

import math
from typing import List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from models._components import apply_rotary_emb

_HAS_HYBRID_ATTN = False  # disabled: flash_attn requires aiter (not on Machine A)


class Attention(nn.Module):
    """GQA with RoPE + QK-Norm. Uses separate wq/wk/wv/wo for autokernel FusedQKV pattern."""

    def __init__(self, dim: int, n_heads: int, n_kv_heads: int, qk_norm: bool = True):
        super().__init__()
        self.n_heads = n_heads
        self.n_kv_heads = n_kv_heads
        self.head_dim = dim // n_heads
        self.n_rep = n_heads // n_kv_heads
        self.qk_norm = qk_norm

        self.wq = nn.Linear(dim, n_heads * self.head_dim, bias=False)
        self.wk = nn.Linear(dim, n_kv_heads * self.head_dim, bias=False)
        self.wv = nn.Linear(dim, n_kv_heads * self.head_dim, bias=False)
        self.wo = nn.Linear(n_heads * self.head_dim, dim, bias=False)

        if qk_norm:
            self.q_scale = nn.Parameter(torch.ones(n_heads, 1, 1) * math.sqrt(self.head_dim))
            self.k_scale = nn.Parameter(torch.ones(n_kv_heads, 1, 1) * math.sqrt(self.head_dim))

    def forward(self, x: torch.Tensor, freqs_cis: torch.Tensor) -> torch.Tensor:
        B, T, _ = x.shape
        q = self.wq(x).view(B, T, self.n_heads, self.head_dim)
        k = self.wk(x).view(B, T, self.n_kv_heads, self.head_dim)
        v = self.wv(x).view(B, T, self.n_kv_heads, self.head_dim)
        q, k = apply_rotary_emb(q, k, freqs_cis)
        q, k, v = q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)

        if self.qk_norm:
            q = F.normalize(q, dim=-1) * self.q_scale
            k = F.normalize(k, dim=-1) * self.k_scale

        if self.n_rep > 1:
            k = k.repeat_interleave(self.n_rep, dim=1)
            v = v.repeat_interleave(self.n_rep, dim=1)

        if _HAS_HYBRID_ATTN and q.dtype == torch.float16:
            y = hybrid_flash_sdpa_attention(
                q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2), causal=True
            ).transpose(1, 2)
        else:
            y = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        return self.wo(y.transpose(1, 2).contiguous().view(B, T, -1))


class CodaAttention(Attention):
    """GQA Attention with optional ValueEmbedding bias, MoDA depth KVs, and XSA.

    XSA removes the self-value projection from attention output, forcing attention
    to capture information orthogonal to each token's own value — reducing redundancy
    with the FFN layer and improving context utilization.

    MoDA depth KVs: prepended prefix KVs from prior loop iterations, enabling
    cross-iteration depth-attention.
    """

    def __init__(self, dim: int, n_heads: int, n_kv_heads: int,
                 qk_norm: bool = True, exclusive: bool = False):
        super().__init__(dim, n_heads, n_kv_heads, qk_norm)
        self.exclusive = exclusive

    def forward(self, x: torch.Tensor, freqs_cis: torch.Tensor,
                value_bias: Optional[torch.Tensor] = None,
                depth_kvs: Optional[List[Tuple[torch.Tensor, torch.Tensor]]] = None,
                ) -> torch.Tensor:
        B, T, _ = x.shape
        q = self.wq(x).view(B, T, self.n_heads, self.head_dim)
        k = self.wk(x).view(B, T, self.n_kv_heads, self.head_dim)
        v = self.wv(x).view(B, T, self.n_kv_heads, self.head_dim)

        if value_bias is not None:
            v = v + value_bias.view(B, T, self.n_kv_heads, self.head_dim)

        q, k = apply_rotary_emb(q, k, freqs_cis)
        q, k, v = q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)

        if self.qk_norm:
            q = F.normalize(q, dim=-1) * self.q_scale
            k = F.normalize(k, dim=-1) * self.k_scale

        if self.n_rep > 1:
            k = k.repeat_interleave(self.n_rep, dim=1)
            v = v.repeat_interleave(self.n_rep, dim=1)

        has_depth = depth_kvs is not None and len(depth_kvs) > 0
        if has_depth:
            depth_k_list, depth_v_list = [], []
            for dk, dv in depth_kvs:
                if dk.shape[1] != k.shape[1]:
                    dk = dk.repeat_interleave(k.shape[1] // dk.shape[1], dim=1)
                    dv = dv.repeat_interleave(v.shape[1] // dv.shape[1], dim=1)
                depth_k_list.append(dk)
                depth_v_list.append(dv)
            k = torch.cat(depth_k_list + [k], dim=2)
            v = torch.cat(depth_v_list + [v], dim=2)

        if _HAS_HYBRID_ATTN and q.dtype == torch.float16 and not has_depth:
            try:
                y = hybrid_flash_sdpa_attention(
                    q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2), causal=True
                ).transpose(1, 2)
            except Exception:
                y = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        else:
            y = F.scaled_dot_product_attention(q, k, v, is_causal=True)

        if self.exclusive:
            v_seq = v[:, :, -T:, :]
            dot = (y * v_seq).sum(dim=-1, keepdim=True)
            v_norm_sq = (v_seq * v_seq).sum(dim=-1, keepdim=True).clamp(min=1e-8)
            y = y - (dot / v_norm_sq) * v_seq

        return self.wo(y.transpose(1, 2).contiguous().view(B, T, -1))


class NoPECodaAttention(Attention):
    """Content-only CodaAttention — no RoPE on Q/K, QK-Norm mandatory.

    Designed for HyPE (ODIN-HALO): attention is purely content-based, enabling
    length generalization without positional encoding bias. QK-Norm prevents
    exploding logits without the positional anchor that RoPE provides.

    Supports XSA (exclusive self-attention) and MoDA depth KVs.
    """

    def __init__(self, dim: int, n_heads: int, n_kv_heads: int,
                 exclusive: bool = False):
        super().__init__(dim, n_heads, n_kv_heads, qk_norm=True)
        self.exclusive = exclusive

    def forward(self, x: torch.Tensor,
                depth_kvs: Optional[List[Tuple[torch.Tensor, torch.Tensor]]] = None,
                ) -> torch.Tensor:
        """No RoPE — pure content attention with QK-Norm.

        Args:
            x: (B, T, dim) input hidden states
            depth_kvs: optional MoDA depth KV pairs from prior iterations
        """
        B, T, _ = x.shape
        q = self.wq(x).view(B, T, self.n_heads, self.head_dim)
        k = self.wk(x).view(B, T, self.n_kv_heads, self.head_dim)
        v = self.wv(x).view(B, T, self.n_kv_heads, self.head_dim)

        q, k, v = q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)

        q = F.normalize(q, dim=-1) * self.q_scale
        k = F.normalize(k, dim=-1) * self.k_scale

        if self.n_rep > 1:
            k = k.repeat_interleave(self.n_rep, dim=1)
            v = v.repeat_interleave(self.n_rep, dim=1)

        has_depth = depth_kvs is not None and len(depth_kvs) > 0
        if has_depth:
            depth_k_list, depth_v_list = [], []
            for dk, dv in depth_kvs:
                if dk.shape[1] != k.shape[1]:
                    dk = dk.repeat_interleave(k.shape[1] // dk.shape[1], dim=1)
                    dv = dv.repeat_interleave(v.shape[1] // dv.shape[1], dim=1)
                depth_k_list.append(dk)
                depth_v_list.append(dv)
            k = torch.cat(depth_k_list + [k], dim=2)
            v = torch.cat(depth_v_list + [v], dim=2)

        if _HAS_HYBRID_ATTN and q.dtype == torch.float16 and not has_depth:
            try:
                y = hybrid_flash_sdpa_attention(
                    q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2), causal=True
                ).transpose(1, 2)
            except Exception:
                y = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        else:
            y = F.scaled_dot_product_attention(q, k, v, is_causal=True)

        if self.exclusive:
            v_seq = v[:, :, -T:, :]
            dot = (y * v_seq).sum(dim=-1, keepdim=True)
            v_norm_sq = (v_seq * v_seq).sum(dim=-1, keepdim=True).clamp(min=1e-8)
            y = y - (dot / v_norm_sq) * v_seq

        return self.wo(y.transpose(1, 2).contiguous().view(B, T, -1))
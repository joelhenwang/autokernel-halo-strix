"""
Minimal LLaMA-style model for AutoKernel profiling.

Self-contained implementation -- no transformers library needed.
Uses RMSNorm, RoPE, SwiGLU MLP, and grouped-query attention
(the same architecture primitives as LLaMA 2 / 3).

Usage:
    # LLaMA-160M (fast, for testing)
    uv run profile.py --model models/llama_7b.py --class-name LlamaModel --input-shape 1,512 --dtype float16

    # LLaMA-7B scale (needs ~14GB VRAM in fp16)
    uv run profile.py --model models/llama_7b.py --class-name LlamaModel7B --input-shape 1,2048 --dtype float16
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


from models._components import RMSNorm, precompute_freqs_cis, apply_rotary_emb


class Attention(nn.Module):
    def __init__(self, dim: int, n_heads: int, n_kv_heads: int):
        super().__init__()
        self.n_heads = n_heads
        self.n_kv_heads = n_kv_heads
        self.head_dim = dim // n_heads
        self.n_rep = n_heads // n_kv_heads

        self.wq = nn.Linear(dim, n_heads * self.head_dim, bias=False)
        self.wk = nn.Linear(dim, n_kv_heads * self.head_dim, bias=False)
        self.wv = nn.Linear(dim, n_kv_heads * self.head_dim, bias=False)
        self.wo = nn.Linear(n_heads * self.head_dim, dim, bias=False)

    def forward(self, x: torch.Tensor, freqs_cis: torch.Tensor) -> torch.Tensor:
        B, T, _ = x.shape

        q = self.wq(x).view(B, T, self.n_heads, self.head_dim)
        k = self.wk(x).view(B, T, self.n_kv_heads, self.head_dim)
        v = self.wv(x).view(B, T, self.n_kv_heads, self.head_dim)

        q, k = apply_rotary_emb(q, k, freqs_cis)

        q = q.transpose(1, 2)  # (B, n_heads, T, head_dim)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        # Repeat KV heads for GQA
        if self.n_rep > 1:
            k = k.repeat_interleave(self.n_rep, dim=1)
            v = v.repeat_interleave(self.n_rep, dim=1)

        # Scaled dot-product attention (uses FlashAttention when available)
        y = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        y = y.transpose(1, 2).contiguous().view(B, T, -1)
        return self.wo(y)


class FeedForward(nn.Module):
    """SwiGLU MLP (LLaMA-style)."""

    def __init__(self, dim: int, hidden_dim: int):
        super().__init__()
        self.w1 = nn.Linear(dim, hidden_dim, bias=False)  # gate
        self.w2 = nn.Linear(hidden_dim, dim, bias=False)  # down
        self.w3 = nn.Linear(dim, hidden_dim, bias=False)  # up

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.w2(F.silu(self.w1(x)) * self.w3(x))


class TransformerBlock(nn.Module):
    def __init__(self, dim: int, n_heads: int, n_kv_heads: int, hidden_dim: int,
                 rms_norm_eps: float = 1e-6):
        super().__init__()
        self.attention = Attention(dim, n_heads, n_kv_heads)
        self.feed_forward = FeedForward(dim, hidden_dim)
        self.attention_norm = RMSNorm(dim, eps=rms_norm_eps)
        self.ffn_norm = RMSNorm(dim, eps=rms_norm_eps)

    def forward(self, x: torch.Tensor, freqs_cis: torch.Tensor) -> torch.Tensor:
        x = x + self.attention(self.attention_norm(x), freqs_cis)
        x = x + self.feed_forward(self.ffn_norm(x))
        return x


class LlamaModel(nn.Module):
    """
    Compact LLaMA (160M params) -- fits on any GPU, good for testing AutoKernel.

    Config: dim=768, n_layers=12, n_heads=12, n_kv_heads=4, hidden_dim=2048
    """

    def __init__(
        self,
        vocab_size: int = 32000,
        dim: int = 768,
        n_layers: int = 12,
        n_heads: int = 12,
        n_kv_heads: int = 4,
        hidden_dim: int = 2048,
        max_seq_len: int = 2048,
        rope_theta: float = 10000.0,
        rms_norm_eps: float = 1e-6,
        tie_embeddings: bool = False,
    ):
        super().__init__()
        self.tok_embeddings = nn.Embedding(vocab_size, dim)
        self.layers = nn.ModuleList([
            TransformerBlock(dim, n_heads, n_kv_heads, hidden_dim, rms_norm_eps)
            for _ in range(n_layers)
        ])
        self.norm = RMSNorm(dim, eps=rms_norm_eps)
        self.output = nn.Linear(dim, vocab_size, bias=False)
        self._tie = tie_embeddings
        if tie_embeddings:
            self.output.weight = self.tok_embeddings.weight

        # RoPE frequencies
        self.register_buffer(
            "freqs_cis",
            precompute_freqs_cis(dim // n_heads, max_seq_len * 2, theta=rope_theta),
            persistent=False,
        )

        n_params = sum(p.numel() for p in self.parameters())
        print(f"{self.__class__.__name__}: {n_params / 1e6:.1f}M parameters")

    def forward(self, input_ids: torch.Tensor, targets=None) -> torch.Tensor:
        B, T = input_ids.shape
        h = self.tok_embeddings(input_ids)
        if self._tie:
            h = h.clone()
        freqs = self.freqs_cis[:T]

        for layer in self.layers:
            h = layer(h, freqs)

        h = self.norm(h)
        logits = self.output(h)
        return logits


class LlamaModel7B(nn.Module):
    """
    LLaMA-7B scale -- requires ~14GB VRAM in fp16.

    Config: dim=4096, n_layers=32, n_heads=32, n_kv_heads=8, hidden_dim=11008
    """

    def __init__(
        self,
        vocab_size: int = 32000,
        dim: int = 4096,
        n_layers: int = 32,
        n_heads: int = 32,
        n_kv_heads: int = 8,
        hidden_dim: int = 11008,
        max_seq_len: int = 4096,
    ):
        super().__init__()
        self.tok_embeddings = nn.Embedding(vocab_size, dim)
        self.layers = nn.ModuleList([
            TransformerBlock(dim, n_heads, n_kv_heads, hidden_dim)
            for _ in range(n_layers)
        ])
        self.norm = RMSNorm(dim)
        self.output = nn.Linear(dim, vocab_size, bias=False)

        self.register_buffer(
            "freqs_cis",
            precompute_freqs_cis(dim // n_heads, max_seq_len * 2),
            persistent=False,
        )

        n_params = sum(p.numel() for p in self.parameters())
        print(f"LlamaModel7B: {n_params / 1e6:.1f}M parameters")

    def forward(self, input_ids: torch.Tensor, targets=None) -> torch.Tensor:
        B, T = input_ids.shape
        h = self.tok_embeddings(input_ids)
        freqs = self.freqs_cis[:T]

        for layer in self.layers:
            h = layer(h, freqs)

        h = self.norm(h)
        logits = self.output(h)
        return logits


class Llama3_175M(LlamaModel):
    """Llama-3 architecture at ~175M params for Strix Halo pretraining.

    Llama-3 differences from Llama-2: theta=500k, rms_eps=1e-5, GQA, tied embeddings.
    Uses GPT-2 tokenizer (50257) for compatibility with halo_training stack.

    Config: dim=768, n_layers=22, n_heads=12, n_kv_heads=4, hidden_dim=2048
    """

    def __init__(self, **kw):
        super().__init__(
            vocab_size=kw.pop("vocab_size", 50257),
            dim=768,
            n_layers=22,
            n_heads=12,
            n_kv_heads=4,
            hidden_dim=2048,
            max_seq_len=2048,
            rope_theta=500000.0,
            rms_norm_eps=1e-5,
            tie_embeddings=True,
            **kw,
        )

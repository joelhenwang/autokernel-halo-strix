"""ODIN-PHANTOM-LOOP: Shared weights + per-position affine modulation (Idea A).

Architecture: Same 4 shared core layers as OdinHaloSlim, but instead of a Python
loop with Parcae injection, forward() is fully unrolled into 12 explicit positions
(3 virtual iterations x 4 layers). Each position applies a cheap learned affine
transform (scale + bias) to differentiate the layer's behavior at that depth.

Key innovation: captures both advantages of looped models—
  1. Weight sharing (L2 cache efficiency + parameter efficiency)
  2. Refresh signal (input re-injection at virtual iteration boundaries)
—while presenting a fully static computation graph to torch.compile.

Per-position modulation adds only ~12K params (0.02% of total) but allows each
of the 12 depth positions to specialize its representation.

Usage:
    python -m halo_training --model models/odin_phantom_loop.py --class-name OdinPhantomLoop --smoke
"""

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from models._components import precompute_freqs_cis, RMSNorm, SwiGLU
from models.components import (
    FactorizedEmbedding, FactorizedLMHead,
    HyPEShortConvBlock,
    NoPECodaAttention,
)


class NoPEGQABlock(nn.Module):
    """NoPE GQA block for coda."""

    _skip_autokernel = True

    def __init__(self, d_model: int, ffn_inner: int,
                 n_heads: int = 12, n_kv_heads: int = 4,
                 use_xsa: bool = True):
        super().__init__()
        self.pre_norm = RMSNorm(d_model)
        self.attn = NoPECodaAttention(d_model, n_heads, n_kv_heads,
                                      exclusive=use_xsa)
        self.ffn_norm = RMSNorm(d_model)
        self.ffn = SwiGLU(d_model, ffn_inner)

    def forward(self, x: torch.Tensor, **kwargs) -> torch.Tensor:
        x = x + self.attn(self.pre_norm(x))
        x = x + self.ffn(self.ffn_norm(x))
        return x


class OdinPhantomLoop(nn.Module):
    """ODIN-PHANTOM-LOOP: Shared layers with per-position modulation.

    12 positions = 3 virtual cycles x 4 shared layers, fully unrolled.
    Each position gets its own affine modulation (scale + bias).
    Refresh signal injected at virtual iteration boundaries (pos 4 and 8).

    Same param count as OdinHaloSlim (~49.4M + 12K modulation params).
    """

    def __init__(
        self,
        vocab_size: int = 32768,
        d_outer: int = 768,
        d_core: int = 512,
        embed_rank: int = 256,
        n_heads_outer: int = 12,
        n_kv_heads_outer: int = 4,
        n_heads_core: int = 8,
        ffn_outer: int = 2816,
        ffn_core: int = 1792,
        d_conv_core: int = 512,
        conv_kernel: int = 3,
        max_seq_len: int = 2048,
        use_xsa: bool = True,
        use_softcap: bool = True,
    ):
        super().__init__()
        self.d_outer = d_outer
        self.d_core = d_core
        self.vocab_size = vocab_size
        self.use_softcap = use_softcap
        self.logit_softcap = 30.0 if use_softcap else 0.0
        self.head_dim_outer = d_outer // n_heads_outer
        self.head_dim_core = d_core // n_heads_core

        # Embedding + LM Head
        self.tok_embeddings = FactorizedEmbedding(vocab_size, embed_rank, d_outer)
        self.final_norm = RMSNorm(d_outer)
        self.lm_head = FactorizedLMHead(d_outer, embed_rank, self.tok_embeddings.embed)

        # Prelude
        self.prelude = HyPEShortConvBlock(
            d_outer, d_conv_core, ffn_outer, conv_kernel,
            head_dim=self.head_dim_outer)

        # Projections
        self.proj_down = nn.Linear(d_outer, d_core, bias=False)
        self.proj_up = nn.Linear(d_core, d_outer, bias=False)

        # 4 shared core layers
        self.core_0 = HyPEShortConvBlock(d_core, d_conv_core, ffn_core, conv_kernel,
                                         head_dim=self.head_dim_core)
        self.core_1 = HyPEShortConvBlock(d_core, d_conv_core, ffn_core, conv_kernel,
                                         head_dim=self.head_dim_core)
        self.core_2 = HyPEShortConvBlock(d_core, d_conv_core, ffn_core, conv_kernel,
                                         head_dim=self.head_dim_core)
        self.core_3 = HyPEShortConvBlock(d_core, d_conv_core, ffn_core, conv_kernel,
                                         head_dim=self.head_dim_core)

        # Per-position additive modulation: 12 positions (3 virtual iters x 4 layers)
        # Additive only — no multiplicative scale (prevents gradient amplification
        # compounding across 12 positions). Init to zeros = pass-through at init.
        self.pos_bias = nn.Parameter(torch.zeros(12, d_core))

        # Refresh signal at virtual iteration boundaries
        # h = h + alpha * input_embed_down (lightweight re-injection)
        self.refresh_alpha_1 = nn.Parameter(torch.tensor(0.1))
        self.refresh_alpha_2 = nn.Parameter(torch.tensor(0.1))

        # Coda
        self.coda_gqa = NoPEGQABlock(
            d_outer, ffn_outer, n_heads_outer, n_kv_heads_outer, use_xsa=use_xsa)
        self.coda_conv = HyPEShortConvBlock(
            d_outer, d_conv_core, ffn_outer, conv_kernel,
            head_dim=self.head_dim_outer)

        # RoPE
        freqs_cis = precompute_freqs_cis(self.head_dim_core, max_seq_len)
        self.register_buffer("freqs_cos", freqs_cis.real.float(), persistent=False)
        self.register_buffer("freqs_sin", freqs_cis.imag.float(), persistent=False)

        self._init_weights()
        n_params = sum(p.numel() for p in self.parameters())
        print(f"{self.__class__.__name__}: {n_params / 1e6:.1f}M params")

    def _init_weights(self):
        n_eff = 12 + 3  # 12 core passes + prelude + 2 coda
        depth_scale = math.sqrt(2 * n_eff)
        for name, p in self.named_parameters():
            if p.dim() >= 2:
                nn.init.xavier_uniform_(p)
                if "wo." in name or "w_down." in name or "out_proj." in name:
                    with torch.no_grad():
                        p.div_(depth_scale)
            elif p.dim() == 1 and "bias" in name:
                nn.init.zeros_(p)
        # Don't touch pos_scale/pos_bias — already init to ones/zeros

    def forward(self, input_ids: torch.Tensor, targets=None, doc_ids=None):
        B, T = input_ids.shape
        h = self.tok_embeddings(input_ids)
        freqs_cis = torch.polar(self.freqs_cos[:T].float(), self.freqs_sin[:T].float())

        # Prelude
        h = self.prelude(h, freqs_cis)

        # Project down
        input_embed_down = self.proj_down(h)
        h = input_embed_down

        # ============================================================
        # VIRTUAL ITERATION 0 — positions 0-3
        # ============================================================
        h = self.core_0(h, freqs_cis) + self.pos_bias[0]
        h = self.core_1(h, freqs_cis) + self.pos_bias[1]
        h = self.core_2(h, freqs_cis) + self.pos_bias[2]
        h = self.core_3(h, freqs_cis) + self.pos_bias[3]

        # ============================================================
        # REFRESH 1 — re-inject input signal
        # ============================================================
        h = h + self.refresh_alpha_1 * input_embed_down

        # ============================================================
        # VIRTUAL ITERATION 1 — positions 4-7
        # ============================================================
        h = self.core_0(h, freqs_cis) + self.pos_bias[4]
        h = self.core_1(h, freqs_cis) + self.pos_bias[5]
        h = self.core_2(h, freqs_cis) + self.pos_bias[6]
        h = self.core_3(h, freqs_cis) + self.pos_bias[7]

        # ============================================================
        # REFRESH 2 — re-inject input signal
        # ============================================================
        h = h + self.refresh_alpha_2 * input_embed_down

        # ============================================================
        # VIRTUAL ITERATION 2 — positions 8-11
        # ============================================================
        h = self.core_0(h, freqs_cis) + self.pos_bias[8]
        h = self.core_1(h, freqs_cis) + self.pos_bias[9]
        h = self.core_2(h, freqs_cis) + self.pos_bias[10]
        h = self.core_3(h, freqs_cis) + self.pos_bias[11]

        # ============================================================
        # Project up + Coda
        # ============================================================
        h = self.proj_up(h)
        h = self.coda_gqa(h)
        h = self.coda_conv(h, freqs_cis)

        # Output
        normed = self.final_norm(h)
        logits = self.lm_head(normed)
        if self.use_softcap:
            logits = 30.0 * torch.tanh(logits / 30.0)
        return {"logits": logits}

    def compile_zones(self, mode: str = "default"):
        """Per-layer compilation — 4 unique core layers compiled once each."""
        import torch as _torch
        self.prelude = _torch.compile(self.prelude, mode=mode)
        self.core_0 = _torch.compile(self.core_0, mode=mode)
        self.core_1 = _torch.compile(self.core_1, mode=mode)
        self.core_2 = _torch.compile(self.core_2, mode=mode)
        self.core_3 = _torch.compile(self.core_3, mode=mode)
        self.coda_gqa = _torch.compile(self.coda_gqa, mode=mode)
        self.coda_conv = _torch.compile(self.coda_conv, mode=mode)

    def compile_full(self, mode: str = "default"):
        """Whole-model compilation — single graph, maximum fusion potential.

        Since forward() is fully unrolled with no Python control flow, Dynamo
        can trace the entire forward as one static graph. This enables cross-layer
        fusion opportunities that per-zone compile misses.

        Use via: model = model.compile_full(mode="max-autotune-no-cudagraphs")
        Returns self for chaining.
        """
        import torch as _torch
        # Compile the entire forward method as a single graph
        self.forward = _torch.compile(self.forward, mode=mode, fullgraph=False)
        return self

"""ODIN-HALO-SLIM-UNROLLED: Diagnostic variant — fully static-unrolled forward.

Same architecture and params as OdinHaloSlim (4 shared core layers × 3 iters),
but with the Python loops manually unrolled so torch.compile/Dynamo sees a
completely static computation graph with no loop variables.

Purpose: measure pure overhead of Python loop + ParcaeInjectionSlim module.
If this matches OdinHaloSlim's ~50k tok/s, the loop isn't the bottleneck.
If it beats it, explicit unrolling is worth pursuing.

Usage:
    python -m halo_training --model models/odin_halo_slim_unrolled.py --class-name OdinHaloSlimUnrolled --smoke
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


class OdinHaloSlimUnrolled(nn.Module):
    """ODIN-HALO-SLIM with fully unrolled forward (no Python loops).

    Architecture identical to OdinHaloSlim:
      Prelude(768) -> proj_down -> 4 shared core layers × 3 iters -> proj_up -> Coda(768)

    Difference: forward() has no for-loops. All 12 layer calls (4 layers × 3 iters)
    are written out explicitly. Parcae injection replaced by inline scalar refresh.
    """

    def __init__(
        self,
        vocab_size: int = 32768,
        d_outer: int = 768,
        d_core: int = 512,
        embed_rank: int = 256,
        n_core_layers: int = 4,
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

        # 4 shared core layers (registered once, called 3x each in forward)
        assert n_core_layers == 4, "Unrolled variant hardcodes 4 core layers × 3 iters"
        self.core_0 = HyPEShortConvBlock(d_core, d_conv_core, ffn_core, conv_kernel,
                                         head_dim=self.head_dim_core)
        self.core_1 = HyPEShortConvBlock(d_core, d_conv_core, ffn_core, conv_kernel,
                                         head_dim=self.head_dim_core)
        self.core_2 = HyPEShortConvBlock(d_core, d_conv_core, ffn_core, conv_kernel,
                                         head_dim=self.head_dim_core)
        self.core_3 = HyPEShortConvBlock(d_core, d_conv_core, ffn_core, conv_kernel,
                                         head_dim=self.head_dim_core)

        # Per-iteration norm + scale (same as OdinHaloSlim)
        self.iter_norm = RMSNorm(d_core)
        self.iter_scales = nn.Parameter(torch.ones(3))
        self.loop_pos_embeds = nn.Parameter(torch.zeros(3, d_core))

        # Refresh scalars (replace ParcaeInjectionSlim)
        # At iter boundaries 1 and 2: h = alpha * h + beta * input_embed_down
        self.refresh_alpha_1 = nn.Parameter(torch.tensor(0.9))
        self.refresh_beta_1 = nn.Parameter(torch.tensor(0.1))
        self.refresh_alpha_2 = nn.Parameter(torch.tensor(0.9))
        self.refresh_beta_2 = nn.Parameter(torch.tensor(0.1))

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
        n_eff = 4 * 3 + 3  # 12 core passes + prelude + 2 coda
        depth_scale = math.sqrt(2 * n_eff)
        for name, p in self.named_parameters():
            if p.dim() >= 2:
                nn.init.xavier_uniform_(p)
                if "wo." in name or "w_down." in name or "out_proj." in name:
                    with torch.no_grad():
                        p.div_(depth_scale)
            elif p.dim() == 1 and "bias" in name:
                nn.init.zeros_(p)

    def _apply_iter_norm(self, h: torch.Tensor, idx: int) -> torch.Tensor:
        scale = self.iter_scales[idx].clamp(-4.0, 4.0)
        return self.iter_norm(h) * scale + self.loop_pos_embeds[idx]

    def forward(self, input_ids: torch.Tensor, targets=None, doc_ids=None):
        B, T = input_ids.shape
        h = self.tok_embeddings(input_ids)
        freqs_cis = torch.polar(self.freqs_cos[:T].float(), self.freqs_sin[:T].float())

        # Prelude at d_outer
        h = self.prelude(h, freqs_cis)

        # Project down
        input_embed_down = self.proj_down(h)
        h_core = input_embed_down

        # ============================================================
        # ITERATION 0: 4 unrolled core layer calls
        # ============================================================
        h_core = self.core_0(h_core, freqs_cis)
        h_core = self.core_1(h_core, freqs_cis)
        h_core = self.core_2(h_core, freqs_cis)
        h_core = self.core_3(h_core, freqs_cis)
        h_core = self._apply_iter_norm(h_core, 0)

        # ============================================================
        # ITERATION 1: refresh + 4 unrolled core layer calls
        # ============================================================
        h_core = self.refresh_alpha_1 * h_core + self.refresh_beta_1 * input_embed_down
        h_core = self.core_0(h_core, freqs_cis)
        h_core = self.core_1(h_core, freqs_cis)
        h_core = self.core_2(h_core, freqs_cis)
        h_core = self.core_3(h_core, freqs_cis)
        h_core = self._apply_iter_norm(h_core, 1)

        # ============================================================
        # ITERATION 2: refresh + 4 unrolled core layer calls
        # ============================================================
        h_core = self.refresh_alpha_2 * h_core + self.refresh_beta_2 * input_embed_down
        h_core = self.core_0(h_core, freqs_cis)
        h_core = self.core_1(h_core, freqs_cis)
        h_core = self.core_2(h_core, freqs_cis)
        h_core = self.core_3(h_core, freqs_cis)
        h_core = self._apply_iter_norm(h_core, 2)

        # ============================================================
        # Project up + Coda
        # ============================================================
        h = self.proj_up(h_core)
        h = self.coda_gqa(h)
        h = self.coda_conv(h, freqs_cis)

        # Output
        normed = self.final_norm(h)
        logits = self.lm_head(normed)
        if self.use_softcap:
            logits = 30.0 * torch.tanh(logits / 30.0)
        return {"logits": logits}

    def compile_zones(self, mode: str = "default"):
        """Per-layer compilation."""
        import torch as _torch
        self.prelude = _torch.compile(self.prelude, mode=mode)
        self.core_0 = _torch.compile(self.core_0, mode=mode)
        self.core_1 = _torch.compile(self.core_1, mode=mode)
        self.core_2 = _torch.compile(self.core_2, mode=mode)
        self.core_3 = _torch.compile(self.core_3, mode=mode)
        self.coda_gqa = _torch.compile(self.coda_gqa, mode=mode)
        self.coda_conv = _torch.compile(self.coda_conv, mode=mode)

    def compile_full(self, mode: str = "default"):
        """Whole-model compilation — single graph for maximum fusion.

        Returns self for chaining.
        """
        import torch as _torch
        self.forward = _torch.compile(self.forward, mode=mode, fullgraph=False)
        return self

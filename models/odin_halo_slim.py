"""ODIN-HALO-SLIM: Heterogeneous-dimension looped hybrid LM.

Architecture: FactorizedEmbed(768) -> Prelude(768) -> proj_down(768->512) ->
              [4 shared core layers @ d=512 x 3 iterations] ->
              proj_up(512->768) -> Coda(768) -> FactorizedLMHead(768)

Design rationale: The core loop operates at d=512 (55% fewer bytes per matmul
vs d=768). On bandwidth-bound gfx1151, this directly reduces DRAM traffic per
iteration. With 4 layers x 3 iters = 12 effective core passes, plus prelude +
coda = 15 effective layers total.

Target: ~34M unique params, ~80M effective. Higher tok/s than OdinHalo (58M/156M).

Usage:
    python -m halo_training --model models/odin_halo_slim.py --class-name OdinHaloSlim --smoke
"""

import math
from typing import Dict, List, Optional, Tuple

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
    """NoPE GQA block for coda. No MoDA depth-KVs (simplified for Slim)."""

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


class ParcaeInjectionSlim(nn.Module):
    """Parcae re-injection with dimension projection (768 -> 512).

    On first call, projects input_embed from d_outer to d_core.
    On subsequent iterations: h_core = alpha * h_core + beta * proj_down(input_embed)
    where alpha/beta are learned per-iteration.
    """

    def __init__(self, d_outer: int, d_core: int):
        super().__init__()
        self.proj_down = nn.Linear(d_outer, d_core, bias=False)
        # Learned blending: init alpha=0.9 (retain), beta=0.1 (inject)
        self.alpha = nn.Parameter(torch.tensor(0.9))
        self.beta = nn.Parameter(torch.tensor(0.1))

    def forward(self, h_core: torch.Tensor, input_embed_down: torch.Tensor) -> torch.Tensor:
        """Re-inject input signal into core state.

        Args:
            h_core: [B, T, d_core] current core hidden state
            input_embed_down: [B, T, d_core] pre-projected input embedding
        """
        return self.alpha * h_core + self.beta * input_embed_down


class OdinHaloSlim(nn.Module):
    """ODIN-HALO-SLIM: Heterogeneous-dim looped hybrid.

    Args:
        vocab_size: 32768 (custom 32K BPE)
        d_outer: Prelude/coda hidden dim (768)
        d_core: Core loop hidden dim (512)
        embed_rank: Factorized embedding rank (256)
        n_core_layers: Shared layers in core loop (4)
        mean_recurrence: Loop iterations (3)
        n_heads_outer: Attention heads at d_outer (12, head_dim=64)
        n_kv_heads_outer: KV heads at d_outer (4, GQA 3:1)
        n_heads_core: RoPE heads at d_core (8, head_dim=64)
        ffn_outer: SwiGLU inner at d_outer (2816)
        ffn_core: SwiGLU inner at d_core (1792)
        d_conv_core: Conv dim in core (512 = d_core)
    """

    def __init__(
        self,
        vocab_size: int = 32768,
        d_outer: int = 768,
        d_core: int = 512,
        embed_rank: int = 256,
        n_core_layers: int = 4,
        mean_recurrence: int = 3,
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
        self.mean_recurrence = mean_recurrence
        self.use_softcap = use_softcap
        self.logit_softcap = 30.0 if use_softcap else 0.0
        self.head_dim_outer = d_outer // n_heads_outer  # 64
        self.head_dim_core = d_core // n_heads_core     # 64

        # Embedding + LM Head (at d_outer)
        self.tok_embeddings = FactorizedEmbedding(vocab_size, embed_rank, d_outer)
        self.final_norm = RMSNorm(d_outer)
        self.lm_head = FactorizedLMHead(d_outer, embed_rank, self.tok_embeddings.embed)

        # Prelude: 1 HyPEShortConvBlock at d_outer
        self.prelude = HyPEShortConvBlock(
            d_outer, d_conv_core, ffn_outer, conv_kernel,
            head_dim=self.head_dim_outer)

        # Dimension projections
        self.proj_down = nn.Linear(d_outer, d_core, bias=False)
        self.proj_up = nn.Linear(d_core, d_outer, bias=False)

        # Core loop: 4 HyPEShortConvBlocks at d_core (SHARED across iterations)
        core_layers = []
        for _ in range(n_core_layers):
            core_layers.append(HyPEShortConvBlock(
                d_core, d_conv_core, ffn_core, conv_kernel,
                head_dim=self.head_dim_core))
        self.core_layers = nn.ModuleList(core_layers)

        # Parcae injection (d_outer -> d_core)
        self.injection = ParcaeInjectionSlim(d_outer, d_core)

        # Per-iteration norms and scales
        self.iter_norm = RMSNorm(d_core)
        self.iter_scales = nn.Parameter(torch.ones(mean_recurrence))
        self.loop_pos_embeds = nn.Parameter(torch.zeros(mean_recurrence, d_core))

        # Coda: 1 NoPE-GQA block + 1 HyPEShortConvBlock at d_outer
        self.coda_gqa = NoPEGQABlock(
            d_outer, ffn_outer, n_heads_outer, n_kv_heads_outer, use_xsa=use_xsa)
        self.coda_conv = HyPEShortConvBlock(
            d_outer, d_conv_core, ffn_outer, conv_kernel,
            head_dim=self.head_dim_outer)

        # RoPE frequencies (shared for both d_outer and d_core since head_dim=64 in both)
        freqs_cis = precompute_freqs_cis(self.head_dim_core, max_seq_len)
        self.register_buffer("freqs_cos", freqs_cis.real.float(), persistent=False)
        self.register_buffer("freqs_sin", freqs_cis.imag.float(), persistent=False)

        self._init_weights()
        n_params = sum(p.numel() for p in self.parameters())
        print(f"{self.__class__.__name__}: {n_params / 1e6:.1f}M params")

    def _init_weights(self):
        n_eff = len(self.core_layers) * self.mean_recurrence + 3  # core + prelude + 2 coda
        depth_scale = math.sqrt(2 * n_eff)
        for name, p in self.named_parameters():
            if p.dim() >= 2:
                nn.init.xavier_uniform_(p)
                if "wo." in name or "w_down." in name or "out_proj." in name:
                    with torch.no_grad():
                        p.div_(depth_scale)
            elif p.dim() == 1 and "bias" in name:
                nn.init.zeros_(p)

    def _apply_iter_norm(self, h: torch.Tensor, iter_idx: int) -> torch.Tensor:
        scale = self.iter_scales[iter_idx].clamp(-4.0, 4.0)
        return self.iter_norm(h) * scale + self.loop_pos_embeds[iter_idx]

    def _run_core_block(self, h: torch.Tensor, freqs_cis: torch.Tensor) -> torch.Tensor:
        """Run all core layers once (one iteration)."""
        for layer in self.core_layers:
            h = layer(h, freqs_cis)
        return h

    def forward(self, input_ids: torch.Tensor,
                targets=None, doc_ids=None):
        B, T = input_ids.shape
        h = self.tok_embeddings(input_ids)
        freqs_cis = torch.polar(self.freqs_cos[:T].float(), self.freqs_sin[:T].float())

        # Prelude at d_outer
        h = self.prelude(h, freqs_cis)

        # Project down to d_core
        input_embed_down = self.proj_down(h)  # Cache for re-injection
        h_core = input_embed_down

        # Core loop (3 iterations at d_core)
        # Iteration 0
        h_core = self._run_core_block(h_core, freqs_cis)
        h_core = self._apply_iter_norm(h_core, 0)

        # Iterations 1..N-1 (with Parcae re-injection)
        for i in range(1, self.mean_recurrence):
            h_core = self.injection(h_core, input_embed_down)
            h_core = self._run_core_block(h_core, freqs_cis)
            h_core = self._apply_iter_norm(h_core, i)

        # Project back up to d_outer
        h = self.proj_up(h_core)

        # Coda at d_outer
        h = self.coda_gqa(h)
        h = self.coda_conv(h, freqs_cis)

        # Output
        normed = self.final_norm(h)
        logits = self.lm_head(normed)
        if self.use_softcap:
            logits = 30.0 * torch.tanh(logits / 30.0)
        return {"logits": logits}

    def compile_zones(self, mode: str = "default"):
        """Per-layer compilation for the trainer."""
        import torch
        self.prelude = torch.compile(self.prelude, mode=mode)
        for i, layer in enumerate(self.core_layers):
            self.core_layers[i] = torch.compile(layer, mode=mode)
        self.coda_gqa = torch.compile(self.coda_gqa, mode=mode)
        self.coda_conv = torch.compile(self.coda_conv, mode=mode)

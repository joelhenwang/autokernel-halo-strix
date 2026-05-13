"""ODIN-FLAT-SLIM: Flat hetero-dim hybrid LM (d_core=512, no looping).

Architecture: FactorizedEmbed(768) -> Prelude(768) -> proj_down(768->512) ->
              [12 unique HyPEShortConvBlocks @ d=512] ->
              proj_up(512->768) -> Coda(768) -> FactorizedLMHead(768)

Design rationale: Takes OdinHaloSlim's throughput trick (narrow d=512 core for
reduced DRAM bandwidth) but removes the looping machinery entirely. Each core
layer has unique weights — no Parcae injection, no iter_norm, no weight sharing.

Benefits over looped variant:
  - Same throughput (same total bytes moved: 12 × d=512 matmuls)
  - More capacity (12 unique layers vs 4 shared)
  - Better compile (no Python loop breaking Dynamo)
  - Simpler code (OdinFlat-level simplicity)

Target: ~82M params, ~50k tok/s DDP aggregate.

Usage:
    python -m halo_training --model models/odin_flat_slim.py --class-name OdinFlatSlim --smoke
"""

import math
from typing import Optional, Tuple

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
    """NoPE GQA block for coda (same as odin_flat's)."""

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


class OdinFlatSlim(nn.Module):
    """ODIN-FLAT-SLIM: Flat hetero-dim hybrid.

    Args:
        vocab_size: 32768 (custom 32K BPE)
        d_outer: Prelude/coda hidden dim (768)
        d_core: Core hidden dim (512)
        embed_rank: Factorized embedding rank (256)
        n_prelude: Number of prelude layers at d_outer (1)
        n_core: Number of unique core layers at d_core (12)
        n_coda: Number of coda layers at d_outer (2: 1 GQA + 1 Conv)
        n_heads_outer: Attention heads at d_outer (12, head_dim=64)
        n_kv_heads_outer: KV heads at d_outer (4, GQA 3:1)
        n_heads_core: RoPE heads at d_core (8, head_dim=64)
        ffn_outer: SwiGLU inner at d_outer (2816)
        ffn_core: SwiGLU inner at d_core (1792)
    """

    def __init__(
        self,
        vocab_size: int = 32768,
        d_outer: int = 768,
        d_core: int = 512,
        embed_rank: int = 256,
        n_prelude: int = 1,
        n_core: int = 12,
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
        self.head_dim_outer = d_outer // n_heads_outer  # 64
        self.head_dim_core = d_core // n_heads_core     # 64

        # Embedding + LM Head (at d_outer)
        self.tok_embeddings = FactorizedEmbedding(vocab_size, embed_rank, d_outer)
        self.final_norm = RMSNorm(d_outer)
        self.lm_head = FactorizedLMHead(d_outer, embed_rank, self.tok_embeddings.embed)

        # Prelude: HyPEShortConvBlocks at d_outer
        self.prelude = nn.ModuleList([
            HyPEShortConvBlock(d_outer, d_conv_core, ffn_outer, conv_kernel,
                               head_dim=self.head_dim_outer)
            for _ in range(n_prelude)
        ])

        # Dimension projections
        self.proj_down = nn.Linear(d_outer, d_core, bias=False)
        self.proj_up = nn.Linear(d_core, d_outer, bias=False)

        # Core: 12 UNIQUE HyPEShortConvBlocks at d_core (flat, no sharing)
        self.core_layers = nn.ModuleList([
            HyPEShortConvBlock(d_core, d_conv_core, ffn_core, conv_kernel,
                               head_dim=self.head_dim_core)
            for _ in range(n_core)
        ])

        # Coda: 1 NoPE-GQA + 1 HyPEShortConvBlock at d_outer
        self.coda_gqa = NoPEGQABlock(
            d_outer, ffn_outer, n_heads_outer, n_kv_heads_outer, use_xsa=use_xsa)
        self.coda_conv = HyPEShortConvBlock(
            d_outer, d_conv_core, ffn_outer, conv_kernel,
            head_dim=self.head_dim_outer)

        # RoPE frequencies (head_dim=64 for both outer and core)
        freqs_cis = precompute_freqs_cis(self.head_dim_core, max_seq_len)
        self.register_buffer("freqs_cos", freqs_cis.real.float(), persistent=False)
        self.register_buffer("freqs_sin", freqs_cis.imag.float(), persistent=False)

        self._init_weights()
        n_params = sum(p.numel() for p in self.parameters())
        print(f"{self.__class__.__name__}: {n_params / 1e6:.1f}M params")

    def _init_weights(self):
        n_eff = len(self.prelude) + len(self.core_layers) + 2  # prelude + core + 2 coda
        depth_scale = math.sqrt(2 * n_eff)
        for name, p in self.named_parameters():
            if p.dim() >= 2:
                nn.init.xavier_uniform_(p)
                if "wo." in name or "w_down." in name or "out_proj." in name:
                    with torch.no_grad():
                        p.div_(depth_scale)
            elif p.dim() == 1 and "bias" in name:
                nn.init.zeros_(p)

        # IMU-1 LayerNorm-gamma scaling
        from models._components import RMSNorm as _RMSNorm
        all_layers = list(self.prelude) + list(self.core_layers) + [self.coda_gqa, self.coda_conv]
        for layer_idx, layer in enumerate(all_layers):
            scale = 1.0 / math.sqrt(layer_idx + 1)
            for submodule in layer.modules():
                if isinstance(submodule, _RMSNorm):
                    with torch.no_grad():
                        submodule.weight.data.mul_(scale)

    def forward(self, input_ids: torch.Tensor,
                targets=None, doc_ids=None):
        B, T = input_ids.shape
        h = self.tok_embeddings(input_ids)
        freqs_cis = torch.polar(self.freqs_cos[:T].float(), self.freqs_sin[:T].float())

        # Prelude at d_outer
        for layer in self.prelude:
            h = layer(h, freqs_cis)

        # Project down to d_core
        h = self.proj_down(h)

        # Core: flat forward through 12 unique d=512 layers
        for layer in self.core_layers:
            h = layer(h, freqs_cis)

        # Project back up to d_outer
        h = self.proj_up(h)

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
        import torch as _torch
        for i, layer in enumerate(self.prelude):
            self.prelude[i] = _torch.compile(layer, mode=mode)
        for i, layer in enumerate(self.core_layers):
            self.core_layers[i] = _torch.compile(layer, mode=mode)
        self.coda_gqa = _torch.compile(self.coda_gqa, mode=mode)
        self.coda_conv = _torch.compile(self.coda_conv, mode=mode)

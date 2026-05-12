"""ODIN-FLAT-GRIFFIN: 14-layer flat hybrid LM with linear recurrence (O(T)).

Architecture: FactorizedEmbed -> [14 unique layers] -> FactorizedLMHead
  12 HyPEShortConvBlocks (RoPE gate, causal conv, SwiGLU)
  2 GriffinGlobalBlocks at positions 6 and 13 (replacing quadratic attention)

The GriffinGlobalBlock uses multi-scale element-wise linear recurrence
(O(T) per token) instead of O(T^2) softmax attention. This enables:
  - Better scaling to longer sequences (T=1024+)
  - Reduced peak memory (no attention score materialization)
  - Slightly higher throughput at T=512 (~3-5% from fewer FLOPs)

Same param count as OdinFlat (~122M). Same d_model, FFN, conv architecture.
Only difference: attention layers -> GriffinRecurrence layers.

Usage:
    python -m halo_training --model models/odin_flat_griffin.py --class-name OdinFlatGriffin --smoke
"""

import math
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from models._components import precompute_freqs_cis, RMSNorm, SwiGLU
from models.components import (
    FactorizedEmbedding, FactorizedLMHead,
    HyPEShortConvBlock,
    GriffinGlobalBlock,
)


class OdinFlatGriffin(nn.Module):
    """ODIN-FLAT-GRIFFIN: Non-looped hybrid with linear recurrence.

    Drop-in replacement for OdinFlat where the 2 NoPEGQABlocks are replaced
    with GriffinGlobalBlocks. Everything else is identical.
    """

    def __init__(
        self,
        vocab_size: int = 32768,
        d_model: int = 768,
        embed_rank: int = 256,
        n_layers: int = 14,
        griffin_positions: Tuple[int, ...] = (6, 13),
        n_heads: int = 12,
        ffn_inner: int = 2816,
        d_conv: int = 512,
        conv_kernel: int = 3,
        max_seq_len: int = 2048,
        use_softcap: bool = True,
        use_mup: bool = False,
        mup_base_width: int = 256,
    ):
        super().__init__()
        self.d_model = d_model
        self.vocab_size = vocab_size
        self.use_softcap = use_softcap
        self.logit_softcap = 30.0 if use_softcap else 0.0
        self.head_dim = d_model // n_heads

        self.tok_embeddings = FactorizedEmbedding(vocab_size, embed_rank, d_model)
        self.final_norm = RMSNorm(d_model)
        self.lm_head = FactorizedLMHead(d_model, embed_rank, self.tok_embeddings.embed)

        griffin_set = set(griffin_positions)
        self._is_griffin = [i in griffin_set for i in range(n_layers)]
        layers = []
        for i in range(n_layers):
            if i in griffin_set:
                # GriffinGlobalBlock: O(T) linear recurrence + SwiGLU FFN
                layers.append(GriffinGlobalBlock(d_model, ffn_inner, d_rec=d_model))
            else:
                # HyPEShortConvBlock: causal conv + RoPE gate + SwiGLU FFN
                layers.append(HyPEShortConvBlock(
                    d_model, d_conv, ffn_inner, conv_kernel,
                    head_dim=self.head_dim))
        self.layers = nn.ModuleList(layers)

        # RoPE frequencies for conv blocks (Griffin blocks ignore freqs_cis)
        freqs_cis = precompute_freqs_cis(self.head_dim, max_seq_len)
        self.register_buffer("freqs_cos", freqs_cis.real.float(), persistent=False)
        self.register_buffer("freqs_sin", freqs_cis.imag.float(), persistent=False)

        self._init_weights()

        if use_mup:
            from halo_training.mup import apply_mup_init
            apply_mup_init(self, d_base=mup_base_width)

        n_params = sum(p.numel() for p in self.parameters())
        print(f"{self.__class__.__name__}: {n_params / 1e6:.1f}M params")

    def _init_weights(self):
        n_layers = len(self.layers)
        depth_scale = math.sqrt(2 * n_layers)
        for name, p in self.named_parameters():
            if p.dim() >= 2:
                nn.init.xavier_uniform_(p)
                if "wo." in name or "w_down." in name or "out_proj." in name:
                    with torch.no_grad():
                        p.div_(depth_scale)
            elif p.dim() == 1 and "bias" in name:
                nn.init.zeros_(p)

        # IMU-1 LayerNorm-gamma scaling (same as OdinFlat)
        from models._components import RMSNorm
        for layer_idx, layer in enumerate(self.layers):
            scale = 1.0 / math.sqrt(layer_idx + 1)
            for submodule in layer.modules():
                if isinstance(submodule, RMSNorm):
                    with torch.no_grad():
                        submodule.weight.data.mul_(scale)

    def forward(self, input_ids: torch.Tensor,
                targets=None, doc_ids=None):
        B, T = input_ids.shape
        h = self.tok_embeddings(input_ids)
        freqs_cis = torch.polar(self.freqs_cos[:T].float(), self.freqs_sin[:T].float())

        for layer in self.layers:
            h = layer(h, freqs_cis)

        normed = self.final_norm(h)
        logits = self.lm_head(normed)
        if self.use_softcap:
            logits = 30.0 * torch.tanh(logits / 30.0)
        return {"logits": logits}

    def compile_zones(self, mode: str = "default"):
        """Per-layer compilation for the trainer."""
        import torch
        for i, layer in enumerate(self.layers):
            self.layers[i] = torch.compile(layer, mode=mode)

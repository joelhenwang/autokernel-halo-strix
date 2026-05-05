"""
BALDR-HALO: Fast Flat Hybrid LLM, 118M params.

12-layer flat hybrid (9 ShortConv + 3 MoDA-GQA). No loop, no mHC.
All novelty is low-overhead: MoDA depth-attention, XSA, QK-Norm, MTP, momentum residuals.
Designed for maximum tok/s with autokernel + full-model torch.compile.

Usage:
    python -m halo_training --model models/baldr_halo.py --class-name BaldrHaloMini --smoke
    python -m halo_training --model models/baldr_halo.py --class-name BaldrHalo \
        --dataset datasets/stem-crawl-solo.bin --compile --optimize-kernels --muon --mtp
"""

import math
from typing import Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.amadeus import RMSNorm, SwiGLU, GatedConv
from models.argus import precompute_freqs_cis, apply_rotary_emb
from models.components import Attention, ShortConvBlock
from models.components import FactorizedEmbedding, FactorizedLMHead
from models.components import CodaAttention
from models.components import MTPHead


from models.components.conv_blocks import MoDAGQABlock


class BaldrHaloBase(nn.Module):
    """BALDR-HALO: 12-layer flat hybrid. No loop — pure speed."""

    def __init__(
        self,
        vocab_size: int = 50257,
        d_model: int = 768,
        embed_rank: int = 256,
        n_layers: int = 12,
        gqa_positions: Tuple[int, ...] = (3, 7, 11),
        d_conv: int = 512,
        ffn_inner: int = 3072,
        n_heads: int = 12,
        n_kv_heads: int = 4,
        conv_kernel: int = 3,
        use_xsa: bool = True,
        use_moda: bool = True,
        use_mtp: bool = True,
        mtp_depth: int = 1,
        momentum_beta_init: float = 0.5,
        max_seq_len: int = 1024,
        use_chunked_ce: bool = False,
    ):
        super().__init__()
        self.d_model = d_model
        self.gqa_positions = set(gqa_positions)
        self.use_moda = use_moda
        self.use_mtp = use_mtp
        self.use_chunked_ce = use_chunked_ce
        self.logit_softcap = 0.0

        self.tok_embeddings = FactorizedEmbedding(vocab_size, embed_rank, d_model)
        self.norm = RMSNorm(d_model)
        self.lm_head = FactorizedLMHead(d_model, embed_rank, self.tok_embeddings.embed)
        self.lm_head.use_chunked_ce = use_chunked_ce

        head_dim = d_model // n_heads
        self.register_buffer(
            "freqs_cis",
            precompute_freqs_cis(head_dim, max_seq_len * 2),
            persistent=False,
        )

        self.layers = nn.ModuleList()
        self._is_gqa = []
        self._gqa_layer_indices = []
        for i in range(n_layers):
            if i in self.gqa_positions:
                self.layers.append(MoDAGQABlock(
                    d_model, ffn_inner, n_heads, n_kv_heads,
                    momentum_beta_init, use_xsa=use_xsa,
                ))
                self._is_gqa.append(True)
                self._gqa_layer_indices.append(i)
            else:
                self.layers.append(ShortConvBlock(
                    d_model, d_conv, ffn_inner, conv_kernel, momentum_beta_init,
                ))
                self._is_gqa.append(False)

        if use_mtp:
            self.mtp_head = MTPHead(d_model, embed_rank, self.tok_embeddings.embed,
                                    depth=mtp_depth)

        self._init_weights()
        n_params = sum(p.numel() for p in self.parameters())
        print(f"{self.__class__.__name__}: {n_params / 1e6:.1f}M parameters")

    def _init_weights(self):
        n_layers = len(self.layers)
        for name, p in self.named_parameters():
            if p.dim() >= 2:
                nn.init.xavier_uniform_(p)
                if "wo." in name or "w_down." in name or "out_proj." in name:
                    with torch.no_grad():
                        p.div_(math.sqrt(2 * n_layers))
            elif p.dim() == 1 and "bias" in name:
                nn.init.zeros_(p)

    def forward(self, input_ids: torch.Tensor, targets=None) -> Union[torch.Tensor, Dict[str, torch.Tensor]]:
        B, T = input_ids.shape

        h = self.tok_embeddings(input_ids)
        freqs_cis = self.freqs_cis[:T]
        velocity = torch.zeros_like(h)

        depth_kv_buffer: List[Tuple[torch.Tensor, torch.Tensor]] = []

        for layer, is_gqa in zip(self.layers, self._is_gqa):
            if is_gqa:
                prior_kvs = depth_kv_buffer if (self.use_moda and depth_kv_buffer) else None
                h, velocity = layer(h, velocity, freqs_cis, depth_kvs=prior_kvs)
                if self.use_moda:
                    depth_kv_buffer.append(layer.compute_depth_kv(h.detach()))
            else:
                h, velocity = layer(h, velocity)

        normed = self.norm(h)
        if self.use_chunked_ce and self.training:
            return self.lm_head.forward_hlow(normed)

        logits = self.lm_head(normed)

        if self.training and self.use_mtp and hasattr(self, 'mtp_head'):
            return {"logits": logits, "mtp1": self.mtp_head(h)}
        return logits


class BaldrHalo(BaldrHaloBase):
    """Default: MoDA + XSA + MTP."""
    pass


class BaldrHaloBare(BaldrHaloBase):
    """Ablation: no MoDA, no XSA, no MTP."""
    def __init__(self, **kw):
        super().__init__(use_xsa=False, use_moda=False, use_mtp=False, **kw)


class BaldrHaloMini(BaldrHaloBase):
    """Smoke test variant."""
    def __init__(self):
        super().__init__(
            vocab_size=50257,
            d_model=128,
            embed_rank=32,
            n_layers=6,
            gqa_positions=(1, 3, 5),
            d_conv=128,
            ffn_inner=256,
            n_heads=4,
            n_kv_heads=2,
            conv_kernel=3,
            use_xsa=False,
            use_moda=True,
            use_mtp=True,
            mtp_depth=1,
            momentum_beta_init=0.5,
            max_seq_len=512,
        )

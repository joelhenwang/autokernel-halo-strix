"""ODIN-FLAT: 14-layer flat (non-looped) hybrid LM, ~122M params.

Architecture: FactorizedEmbed → [14 unique layers] → FactorizedLMHead
  12 HyPEShortConvBlocks (RoPE gate, causal conv, SwiGLU)
  2 NoPEGQABlocks at positions 6 and 13 (NoPE attention, XSA, SwiGLU)
  HyPE: NoPE attention for length generalization + RoPE conv gate for local position
  No looping, no Parcae injection, no MoDA depth KVs, no skip gates

Comparison counterpart to OdinHalo (58M unique / ~157M effective, 6 shared × 3 iters).
Same d_model=768, same block internals, same factorized embed/head.
Difference: all 14 layers are unique (no weight sharing), no iteration machinery.

Usage:
    python -m halo_training --model models/odin_flat.py --class-name OdinFlat --smoke
    TORCH_COMPILE_MODE=max-autotune python -m halo_training \\
        --model models/odin_flat.py --class-name OdinFlat \\
        --compile --smoke
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
    NoPECodaAttention,
)


class NoPEGQABlock(nn.Module):
    """NoPE GQA block for flat (non-looped) models.

    Content-only attention (NoPECodaAttention) with XSA + SwiGLU FFN.
    No MoDA depth KVs — flat model has no prior-iteration state.
    """

    def __init__(self, d_model: int, ffn_inner: int,
                 n_heads: int = 12, n_kv_heads: int = 4,
                 use_xsa: bool = True):
        super().__init__()
        self.pre_norm = RMSNorm(d_model)
        self.attn = NoPECodaAttention(d_model, n_heads, n_kv_heads,
                                      exclusive=use_xsa)
        self.ffn_norm = RMSNorm(d_model)
        self.ffn = SwiGLU(d_model, ffn_inner)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.pre_norm(x))
        x = x + self.ffn(self.ffn_norm(x))
        return x


class OdinFlatBase(nn.Module):
    """ODIN-FLAT: Non-looped hybrid with HyPE positional encoding.

    Args:
        vocab_size: Tokenizer vocabulary size (32768 for custom 32K BPE)
        d_model: Hidden dimension
        embed_rank: Factorized embedding rank
        n_layers: Total number of layers
        gqa_positions: Which layer indices use NoPE-GQA (rest use conv)
        n_heads: Query heads
        n_kv_heads: KV heads (GQA ratio)
        ffn_inner: SwiGLU hidden dim
        d_conv: Conv channel dim
        conv_kernel: Causal conv1d kernel size
        max_seq_len: Max sequence length for RoPE precomputation
        use_xsa: Exclusive Self Attention in GQA blocks
        use_softcap: Logit softcap (30 * tanh(x/30))
        use_chunked_ce: If True, forward() returns h_low for ChunkedLinearCE
    """

    def __init__(
        self,
        vocab_size: int = 32768,
        d_model: int = 768,
        embed_rank: int = 256,
        n_layers: int = 14,
        gqa_positions: Tuple[int, ...] = (6, 13),
        n_heads: int = 12,
        n_kv_heads: int = 4,
        ffn_inner: int = 2816,
        d_conv: int = 512,
        conv_kernel: int = 3,
        max_seq_len: int = 2048,
        use_xsa: bool = True,
        use_softcap: bool = True,
        use_chunked_ce: bool = False,
    ):
        super().__init__()
        self.d_model = d_model
        self.vocab_size = vocab_size
        self.use_softcap = use_softcap
        self.use_chunked_ce = use_chunked_ce
        self.logit_softcap = 30.0 if use_softcap else 0.0
        self.head_dim = d_model // n_heads

        self.tok_embeddings = FactorizedEmbedding(vocab_size, embed_rank, d_model)
        self.final_norm = RMSNorm(d_model)
        self.lm_head = FactorizedLMHead(d_model, embed_rank, self.tok_embeddings.embed)
        self.lm_head.use_chunked_ce = use_chunked_ce

        gqa_set = set(gqa_positions)
        self._is_gqa = [i in gqa_set for i in range(n_layers)]
        layers = []
        for i in range(n_layers):
            if i in gqa_set:
                layers.append(NoPEGQABlock(
                    d_model, ffn_inner, n_heads, n_kv_heads,
                    use_xsa=use_xsa))
            else:
                layers.append(HyPEShortConvBlock(
                    d_model, d_conv, ffn_inner, conv_kernel,
                    head_dim=self.head_dim))
        self.layers = nn.ModuleList(layers)

        # RoPE frequencies for conv blocks
        freqs_cis = precompute_freqs_cis(self.head_dim, max_seq_len)
        self.register_buffer("freqs_cos", freqs_cis.real.float(), persistent=False)
        self.register_buffer("freqs_sin", freqs_cis.imag.float(), persistent=False)

        self._init_weights()
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

    def forward(self, input_ids: torch.Tensor, targets: Optional[torch.Tensor] = None):
        B, T = input_ids.shape
        h = self.tok_embeddings(input_ids)
        freqs_cis = torch.polar(self.freqs_cos[:T].float(), self.freqs_sin[:T].float())

        for layer, is_gqa in zip(self.layers, self._is_gqa):
            if is_gqa:
                h = layer(h)
            else:
                h = layer(h, freqs_cis)

        normed = self.final_norm(h)

        if self.use_chunked_ce and self.training:
            return self.lm_head.forward_hlow(normed)

        logits = self.lm_head(normed)
        if self.use_softcap:
            logits = 30.0 * torch.tanh(logits / 30.0)
        return logits

    def compile_zones(self, mode: str = None):
        """Per-layer compilation for ROCm kernel fusion.

        Args:
            mode: torch.compile mode. Defaults to env var TORCH_COMPILE_MODE,
                or "default" if unset.
        """
        import os
        if mode is None:
            mode = os.environ.get("TORCH_COMPILE_MODE", "default")
        if mode == "reduce-overhead":
            print("NOTE: reduce-overhead redirected to default (HIP graph capture "
                  "fails silently on this backend).")
            mode = "default"
        try:
            from kernels.hip._torch_ops import disable_hip_backward
            disable_hip_backward()
        except ImportError:
            pass
        print(f"compile_zones: per-layer torch.compile mode={mode} "
              f"({len(self.layers)} layers)")
        for i in range(len(self.layers)):
            self.layers[i] = torch.compile(self.layers[i], mode=mode)
        return self

    def compile_zones_friendly(self, mode: str = None):
        """Per-layer compilation using compile-friendly inner paths."""
        import os
        if mode is None:
            mode = os.environ.get("TORCH_COMPILE_MODE", "default")
        if mode == "reduce-overhead":
            mode = "default"
        from models.components.conv_blocks import HyPEShortConvBlock
        for layer in self.layers:
            if isinstance(layer, HyPEShortConvBlock):
                layer._compile_friendly = True
        for i in range(len(self.layers)):
            self.layers[i] = torch.compile(self.layers[i], mode=mode)
        return self

    def param_count(self) -> Dict[str, int]:
        embed = sum(p.numel() for p in self.tok_embeddings.parameters())
        head = sum(p.numel() for n, p in self.lm_head.named_parameters()
                   if "embed_table" not in n)
        layer_params = sum(p.numel() for p in self.layers.parameters())
        misc = sum(p.numel() for p in self.final_norm.parameters())
        unique = embed + head + layer_params + misc
        return {
            "embed": embed,
            "head": head,
            "layers": layer_params,
            "misc": misc,
            "unique_total": unique,
            "effective_total": unique,  # flat: unique == effective
        }


# ---------------------------------------------------------------------------
# Variant classes
# ---------------------------------------------------------------------------

class OdinFlat(OdinFlatBase):
    """Production: d=768, 14 layers (12 conv + 2 GQA), ~122M params."""
    pass


class OdinFlatAblation(OdinFlatBase):
    """Ablation: d=384, 8 layers (7 conv + 1 GQA), ~15M params."""

    def __init__(self, **kw):
        kw.setdefault("d_model", 384)
        kw.setdefault("embed_rank", 128)
        kw.setdefault("n_layers", 8)
        kw.setdefault("gqa_positions", (4,))
        kw.setdefault("n_heads", 6)
        kw.setdefault("n_kv_heads", 2)
        kw.setdefault("ffn_inner", 1408)
        kw.setdefault("d_conv", 320)
        super().__init__(**kw)


class OdinFlatMini(OdinFlatBase):
    """Smoke test: d=128, 6 layers (5 conv + 1 GQA), ~1.5M params."""

    def __init__(self, **kw):
        kw.setdefault("vocab_size", 1000)
        kw.setdefault("d_model", 128)
        kw.setdefault("embed_rank", 64)
        kw.setdefault("n_layers", 6)
        kw.setdefault("gqa_positions", (3,))
        kw.setdefault("n_heads", 4)
        kw.setdefault("n_kv_heads", 2)
        kw.setdefault("ffn_inner", 256)
        kw.setdefault("d_conv", 128)
        super().__init__(**kw)

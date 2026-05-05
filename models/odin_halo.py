"""ODIN-HALO: 58M unique / ~156M effective looped hybrid LM.

Architecture: FactorizedEmbed → [6 shared layers × 3 Parcae iterations] → FactorizedLMHead
  6 layers: 5 ShortConv (RoPE gate) + 1 NoPE-GQA (position 3, center)
  ×3 iterations = 18 effective layers
  HyPE: NoPE attention for length generalization + RoPE conv gate for local position
  MoDA depth-attention, iteration skip connections, logit softcap

Design spec: docs/superpowers/specs/2026-05-04-odin-halo-design.md

Usage:
    python -m halo_training --model models/odin_halo.py --class-name OdinHaloMini --smoke
    python -m halo_training --model models/odin_halo.py --class-name OdinHalo \\
        --dataset datasets/dolma-10b-vidar32k.bin --epochs 1 \\
        --compile --polar-ns --muon --scheduler wsd --min-lr-ratio 0.1 \\
        --ema --z-loss 1e-4 --block-size 256 \\
        --tokenizer-path tokenizers/vidar-32k/tokenizer.json
"""

import math
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from models._components import precompute_freqs_cis
from models.components import (
    FactorizedEmbedding, FactorizedLMHead,
    SimpleParcaeInjection,
    HyPEShortConvBlock,
    NoPECodaAttention,
)
from models._components import RMSNorm, SwiGLU


class NoPEMoDAGQABlock(nn.Module):
    """Momentum-free GQA with NoPE attention, XSA, MoDA depth KVs, SwiGLU FFN.

    Designed for ODIN-HALO's HyPE architecture: position 3 of the 6-layer
    shared block. Uses NoPECodaAttention (content-only, QK-Norm mandatory)
    for length generalization.
    """

    def __init__(self, d_model: int, ffn_inner: int,
                 n_heads: int = 12, n_kv_heads: int = 4,
                 use_xsa: bool = True):
        super().__init__()
        self.head_dim = d_model // n_heads
        self.n_kv_heads = n_kv_heads
        self.pre_norm = RMSNorm(d_model)
        self.attn = NoPECodaAttention(d_model, n_heads, n_kv_heads,
                                      exclusive=use_xsa)
        self.use_xsa = use_xsa
        self.depth_kv_proj = nn.Linear(
            d_model, n_kv_heads * self.head_dim * 2, bias=False)
        self.ffn_norm = RMSNorm(d_model)
        self.ffn = SwiGLU(d_model, ffn_inner)

    def forward(self, x: torch.Tensor,
                depth_kvs: Optional[List[Tuple[torch.Tensor, torch.Tensor]]] = None,
                ) -> torch.Tensor:
        x = x + self.attn(self.pre_norm(x), depth_kvs=depth_kvs)
        x = x + self.ffn(self.ffn_norm(x))
        return x

    def compute_depth_kv(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        B, T, _ = x.shape
        kv = self.depth_kv_proj(x)
        k, v = kv.chunk(2, dim=-1)
        k = k.reshape(B, T, self.n_kv_heads, self.head_dim).permute(0, 2, 1, 3)
        v = v.reshape(B, T, self.n_kv_heads, self.head_dim).permute(0, 2, 1, 3)
        return k, v


class OdinHaloBase(nn.Module):
    """ODIN-HALO: Looped hybrid with HyPE positional encoding.

    Args:
        vocab_size: Tokenizer vocabulary size (32768 for custom 32K BPE)
        d_model: Hidden dimension
        embed_rank: Factorized embedding rank (256 = aggressive)
        n_shared_layers: Shared layers in Parcae loop
        gqa_positions: Which layer indices use NoPE-GQA (rest use conv)
        n_heads: Query heads (12, head_dim=64)
        n_kv_heads: KV heads (4, 3:1 GQA)
        ffn_inner: SwiGLU hidden dim (2816 = 3.67× d, 256-multiple)
        d_conv: Conv channel dim (512 = 0.67× d)
        conv_kernel: Causal conv1d kernel size
        mean_recurrence: Parcae loop iterations (deterministic)
        backprop_depth: Gradient depth (equal to mean_recurrence for unrolled)
        max_seq_len: Max sequence length for RoPE precomputation
        use_xsa: Exclusive Self Attention
        use_moda: MoDA depth-attention
        use_skip_connections: Iteration skip connections with sigmoid gates
        use_softcap: Logit softcap (30 * tanh(x/30))
        use_chunked_ce: If True, forward() returns h_low instead of logits when
            training, and the trainer uses ChunkedLinearCrossEntropyLoss to compute
            loss without materializing the full logits tensor. logit_softcap is
            handled inside the chunked CE kernel. Defaults to False.
    """

    def __init__(
        self,
        vocab_size: int = 32768,
        d_model: int = 768,
        embed_rank: int = 256,
        n_shared_layers: int = 6,
        gqa_positions: Tuple[int, ...] = (3,),
        n_heads: int = 12,
        n_kv_heads: int = 4,
        ffn_inner: int = 2816,
        d_conv: int = 512,
        conv_kernel: int = 3,
        mean_recurrence: int = 3,
        backprop_depth: int = 3,
        max_seq_len: int = 2048,
        use_xsa: bool = True,
        use_moda: bool = True,
        use_skip_connections: bool = True,
        use_softcap: bool = True,
        use_chunked_ce: bool = False,
    ):
        super().__init__()
        self.d_model = d_model
        self.vocab_size = vocab_size
        self.mean_recurrence = mean_recurrence
        self.backprop_depth = backprop_depth
        self.use_moda = use_moda
        self.use_softcap = use_softcap
        self.use_skip_connections = use_skip_connections
        self.use_chunked_ce = use_chunked_ce
        self.logit_softcap = 30.0 if use_softcap else 0.0  # exposed for chunked CE
        self.head_dim = d_model // n_heads

        self.tok_embeddings = FactorizedEmbedding(vocab_size, embed_rank, d_model)
        self.final_norm = RMSNorm(d_model)
        self.lm_head = FactorizedLMHead(d_model, embed_rank, self.tok_embeddings.embed)
        self.lm_head.use_chunked_ce = use_chunked_ce

        gqa_set = set(gqa_positions)
        self._is_gqa = [i in gqa_set for i in range(n_shared_layers)]
        self._gqa_indices = [i for i in range(n_shared_layers) if i in gqa_set]

        layers = []
        for i in range(n_shared_layers):
            if i in gqa_set:
                layers.append(NoPEMoDAGQABlock(
                    d_model, ffn_inner, n_heads, n_kv_heads,
                    use_xsa=use_xsa))
            else:
                layers.append(HyPEShortConvBlock(
                    d_model, d_conv, ffn_inner, conv_kernel,
                    head_dim=self.head_dim))
        self.shared_layers = nn.ModuleList(layers)

        self.injection = SimpleParcaeInjection(d_model)
        self.iter_norm = RMSNorm(d_model)

        # Per-iteration learned scales
        self.iter_scales = nn.Parameter(torch.ones(mean_recurrence))

        # Loop position embeddings
        self.loop_pos_embeds = nn.Parameter(torch.zeros(mean_recurrence, d_model))

        # Iteration skip gates
        if use_skip_connections:
            self.skip_gates = nn.Parameter(torch.zeros(mean_recurrence - 1, d_model))

        # RoPE frequencies for conv blocks (stored as cos/sin floats to avoid
        # complex128→fp16 casting issues when model.to(dtype=float16) is called)
        freqs_cis = precompute_freqs_cis(self.head_dim, max_seq_len)
        self.register_buffer("freqs_cos", freqs_cis.real.float(), persistent=False)
        self.register_buffer("freqs_sin", freqs_cis.imag.float(), persistent=False)

        self._init_weights()
        n_params = sum(p.numel() for p in self.parameters())
        print(f"{self.__class__.__name__}: {n_params / 1e6:.1f}M params")

    def _init_weights(self):
        n_layers = len(self.shared_layers)
        depth_scale = math.sqrt(2 * n_layers * self.mean_recurrence)
        for name, p in self.named_parameters():
            if p.dim() >= 2:
                nn.init.xavier_uniform_(p)
                if "wo." in name or "w_down." in name or "out_proj." in name:
                    with torch.no_grad():
                        p.div_(depth_scale)
            elif p.dim() == 1 and "bias" in name:
                nn.init.zeros_(p)

    def _run_shared_block(
        self,
        h: torch.Tensor,
        freqs_cis: torch.Tensor,
        depth_kv_buffer: List[Dict[int, Tuple[torch.Tensor, torch.Tensor]]],
    ) -> Tuple[torch.Tensor, Dict[int, Tuple[torch.Tensor, torch.Tensor]]]:
        current_kvs: Dict[int, Tuple[torch.Tensor, torch.Tensor]] = {}
        for idx, (layer, is_gqa) in enumerate(zip(self.shared_layers, self._is_gqa)):
            if is_gqa:
                prior_kvs = None
                if self.use_moda and depth_kv_buffer:
                    prior_kvs = [buf[idx] for buf in depth_kv_buffer if idx in buf]
                    if not prior_kvs:
                        prior_kvs = None
                h = layer(h, depth_kvs=prior_kvs)
                if self.use_moda:
                    current_kvs[idx] = layer.compute_depth_kv(h.detach())
            else:
                h = layer(h, freqs_cis)
        return h, current_kvs

    def _apply_iter_norm(self, h: torch.Tensor, iter_idx: int) -> torch.Tensor:
        return self.iter_norm(h) * self.iter_scales[iter_idx] + self.loop_pos_embeds[iter_idx]

    def _forward_unrolled(self, input_ids: torch.Tensor):
        B, T = input_ids.shape
        h = self.tok_embeddings(input_ids)
        freqs_cis = torch.polar(self.freqs_cos[:T].float(), self.freqs_sin[:T].float())
        input_embed = h

        depth_kv_buffer: List[Dict[int, Tuple[torch.Tensor, torch.Tensor]]] = []

        # Iteration 0
        h, current_kvs = self._run_shared_block(h, freqs_cis, depth_kv_buffer)
        h = self._apply_iter_norm(h, 0)
        skip_state = h if self.use_skip_connections else None
        if current_kvs:
            depth_kv_buffer.append(current_kvs)

        # Iterations 1..N-1
        for i in range(1, self.mean_recurrence):
            h = self.injection(h, input_embed)
            if self.use_skip_connections and skip_state is not None:
                h = h + torch.sigmoid(self.skip_gates[i - 1]) * skip_state
            h, current_kvs = self._run_shared_block(h, freqs_cis, depth_kv_buffer)
            h = self._apply_iter_norm(h, i)
            if self.use_skip_connections:
                skip_state = h
            if current_kvs:
                depth_kv_buffer.append(current_kvs)

        normed = self.final_norm(h)
        if self.use_chunked_ce and self.training:
            # Return low-rank hidden state; trainer will use ChunkedLinearCrossEntropyLoss
            # to compute logits+CE+softcap without materializing [N, V] tensor.
            # Trainer is responsible for cloning when reduce-overhead mode is active.
            return self.lm_head.forward_hlow(normed)

        logits = self.lm_head(normed)

        if self.use_softcap:
            logits = 30.0 * torch.tanh(logits / 30.0)

        return logits

    def forward(self, input_ids: torch.Tensor, targets: Optional[torch.Tensor] = None):
        return self._forward_unrolled(input_ids)

    def compile_zones(self):
        """Per-layer compilation for ROCm kernel fusion."""
        try:
            from kernels.hip._torch_ops import disable_hip_backward
            disable_hip_backward()
        except ImportError:
            pass
        for i in range(len(self.shared_layers)):
            self.shared_layers[i] = torch.compile(self.shared_layers[i], mode="default")
        return self

    def compile_zones_friendly(self):
        """Per-layer compilation using compile-friendly inner paths.

        - Enables native PyTorch RoPE+gate (no HIP kernel break-in)
        - Result: Inductor sees full layer graph and can fuse
        - Trade: loses custom HIP fused_rope_gate_mul kernel (but Inductor
          usually fuses these ops anyway)
        """
        # Mark all HyPEShortConvBlocks to use compile-friendly RoPE
        from models.components.conv_blocks import HyPEShortConvBlock
        for layer in self.shared_layers:
            if isinstance(layer, HyPEShortConvBlock):
                layer._compile_friendly = True
        for i in range(len(self.shared_layers)):
            self.shared_layers[i] = torch.compile(self.shared_layers[i], mode="default")
        return self

    def param_count(self) -> Dict[str, int]:
        embed = sum(p.numel() for p in self.tok_embeddings.parameters())
        head = sum(p.numel() for n, p in self.lm_head.named_parameters()
                   if "embed_table" not in n)
        shared = sum(p.numel() for p in self.shared_layers.parameters())
        inject = sum(p.numel() for p in self.injection.parameters())
        misc = (self.loop_pos_embeds.numel() +
                sum(p.numel() for p in self.iter_norm.parameters()) +
                sum(p.numel() for p in self.final_norm.parameters()) +
                self.iter_scales.numel())
        if self.use_skip_connections:
            misc += self.skip_gates.numel()
        unique = embed + head + shared + inject + misc
        return {
            "embed": embed, "head": head, "shared": shared,
            "injection": inject, "misc": misc,
            "unique_total": unique,
            "effective_total": (embed + head + shared * self.mean_recurrence +
                                inject + misc),
        }


# ---------------------------------------------------------------------------
# Variant classes
# ---------------------------------------------------------------------------

class OdinHalo(OdinHaloBase):
    """Production: d=768, 6 layers × 3 iters, ~58M unique, ~156M effective."""
    pass


class OdinHaloAblation(OdinHaloBase):
    """Ablation: d=384, 6 layers, ~20M unique. Tier S screening."""

    def __init__(self, **kw):
        kw.setdefault("d_model", 384)
        kw.setdefault("embed_rank", 128)
        kw.setdefault("n_heads", 6)
        kw.setdefault("n_kv_heads", 2)
        kw.setdefault("ffn_inner", 1408)
        kw.setdefault("d_conv", 320)
        super().__init__(**kw)


class OdinHaloMini(OdinHaloBase):
    """Smoke test: d=128, ~2M params."""

    def __init__(self, **kw):
        kw.setdefault("vocab_size", 1000)
        kw.setdefault("d_model", 128)
        kw.setdefault("embed_rank", 64)
        kw.setdefault("n_heads", 4)
        kw.setdefault("n_kv_heads", 2)
        kw.setdefault("ffn_inner", 256)
        kw.setdefault("d_conv", 128)
        kw.setdefault("mean_recurrence", 2)
        kw.setdefault("backprop_depth", 2)
        super().__init__(**kw)
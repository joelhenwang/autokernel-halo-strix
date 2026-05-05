"""VIDAR-HALO: Lean looped hybrid LM.

~47.5M unique / ~95M effective params. d=768, 4 shared layers × 2 Parcae iterations.
No momentum — direct residuals. Custom 32K tokenizer. MoDA depth-attention.

Usage:
    python -m halo_training --model models/vidar_halo.py --class-name VidarHalo \
        --compile --muon --mtp --ema --scheduler wsd --z-loss 1e-4
"""

import math
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.amadeus import SwiGLU, GatedConv
from models._components import precompute_freqs_cis
from models.components import FactorizedEmbedding, FactorizedLMHead
from models.components import SimpleParcaeInjection
from models.components import CodaAttention


class NativeRMSNorm(torch.nn.Module):
    """RMSNorm using torch.rms_norm. Weight stored fp32, cast to input dtype per call.

    Named NativeRMSNorm to avoid autokernel pattern matching (which replaces
    classes named 'RMSNorm' with a slower HIP kernel).
    """

    _native_norm = True

    def __init__(self, d: int, eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        self.weight = torch.nn.Parameter(torch.ones(d))

    def forward(self, x):
        return torch.rms_norm(x, (x.size(-1),), self.weight.to(x.dtype), self.eps)

RMSNorm = NativeRMSNorm


class VidarShortConvBlock(nn.Module):
    """Momentum-free ShortConv + SwiGLU FFN with direct residual."""

    _skip_autokernel = True

    def __init__(self, d_model: int, d_conv: int, ffn_inner: int, conv_kernel: int = 3):
        super().__init__()
        self.pre_norm = RMSNorm(d_model)
        self.conv = GatedConv(d_model, d_conv, conv_kernel)
        self.out_proj = nn.Linear(d_conv, d_model, bias=False)
        self.ffn_norm = RMSNorm(d_model)
        self.ffn = SwiGLU(d_model, ffn_inner)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.out_proj(self.conv(self.pre_norm(x)))
        x = x + self.ffn(self.ffn_norm(x))
        return x


class VidarMoDAGQABlock(nn.Module):
    """Momentum-free GQA with MoDA depth-attention, XSA, SwiGLU FFN."""

    _skip_autokernel = True

    def __init__(self, d_model: int, ffn_inner: int,
                 n_heads: int = 12, n_kv_heads: int = 4, use_xsa: bool = True):
        super().__init__()
        self.head_dim = d_model // n_heads
        self.n_kv_heads = n_kv_heads
        self.pre_norm = RMSNorm(d_model)
        self.attn = CodaAttention(d_model, n_heads, n_kv_heads,
                                  qk_norm=True, exclusive=use_xsa)
        self.depth_kv_proj = nn.Linear(d_model, n_kv_heads * self.head_dim * 2, bias=False)
        self.ffn_norm = RMSNorm(d_model)
        self.ffn = SwiGLU(d_model, ffn_inner)

    def forward(self, x: torch.Tensor, freqs_cis: torch.Tensor,
                depth_kvs: Optional[List[Tuple[torch.Tensor, torch.Tensor]]] = None
                ) -> torch.Tensor:
        x = x + self.attn(self.pre_norm(x), freqs_cis, depth_kvs=depth_kvs)
        x = x + self.ffn(self.ffn_norm(x))
        return x

    def compute_depth_kv(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        B, T, _ = x.shape
        kv = self.depth_kv_proj(x)
        k, v = kv.chunk(2, dim=-1)
        k = k.reshape(B, T, self.n_kv_heads, self.head_dim).permute(0, 2, 1, 3)
        v = v.reshape(B, T, self.n_kv_heads, self.head_dim).permute(0, 2, 1, 3)
        return k, v


from models.components.mtp import MTPHead


class VidarHaloBase(nn.Module):
    """Base class for VIDAR-HALO variants.

    Args:
        vocab_size: Tokenizer vocabulary size (32000 for custom BPE)
        d_model: Hidden dimension
        embed_rank: Factorized embedding rank
        n_shared_layers: Number of shared layers in Parcae loop
        gqa_positions: Tuple of layer indices that use MoDA-GQA (rest use ShortConv)
        n_heads: Query attention heads
        n_kv_heads: Key-value heads (GQA)
        ffn_inner: SwiGLU hidden dimension
        d_conv: Gated conv channel dimension
        conv_kernel: Causal conv1d kernel size
        mean_recurrence: Parcae loop iterations (deterministic)
        backprop_depth: Iterations with gradient (must equal mean_recurrence for unrolled)
        max_seq_len: Maximum sequence length for RoPE precomputation
        use_xsa: Exclusive Self Attention
        use_moda: MoDA depth-attention (cross-iteration KVs)
        use_mtp: Multi-Token Prediction auxiliary head
    """

    def __init__(
        self,
        vocab_size: int = 32000,
        d_model: int = 768,
        embed_rank: int = 384,
        n_shared_layers: int = 4,
        gqa_positions: Tuple[int, ...] = (2,),
        n_heads: int = 12,
        n_kv_heads: int = 4,
        ffn_inner: int = 2816,
        d_conv: int = 640,
        conv_kernel: int = 3,
        mean_recurrence: int = 2,
        backprop_depth: int = 2,
        max_seq_len: int = 2048,
        use_xsa: bool = True,
        use_moda: bool = True,
        use_mtp: bool = False,
        use_chunked_ce: bool = False,
    ):
        super().__init__()
        self.d_model = d_model
        self.vocab_size = vocab_size
        self.mean_recurrence = mean_recurrence
        self.backprop_depth = backprop_depth
        self.use_moda = use_moda
        self.use_mtp = use_mtp
        self.use_chunked_ce = use_chunked_ce
        self.logit_softcap = 0.0  # VidarHalo doesn't use softcap

        self.tok_embeddings = FactorizedEmbedding(vocab_size, embed_rank, d_model)
        self.norm = RMSNorm(d_model)
        self.lm_head = FactorizedLMHead(d_model, embed_rank, self.tok_embeddings.embed)
        self.lm_head.use_chunked_ce = use_chunked_ce

        gqa_set = set(gqa_positions)
        self._is_gqa = [i in gqa_set for i in range(n_shared_layers)]
        layers = []
        for i in range(n_shared_layers):
            if i in gqa_set:
                layers.append(VidarMoDAGQABlock(
                    d_model, ffn_inner, n_heads, n_kv_heads, use_xsa=use_xsa))
            else:
                layers.append(VidarShortConvBlock(d_model, d_conv, ffn_inner, conv_kernel))
        self.shared_layers = nn.ModuleList(layers)

        self.injection = SimpleParcaeInjection(d_model)
        self.iter_norm = RMSNorm(d_model)
        self.loop_pos_embeds = nn.Parameter(torch.zeros(mean_recurrence, d_model))

        head_dim = d_model // n_heads
        self.register_buffer(
            "freqs_cis", precompute_freqs_cis(head_dim, max_seq_len), persistent=False)

        if use_mtp:
            self.mtp_head = MTPHead(d_model, embed_rank, self.tok_embeddings.embed, depth=1)

        self._init_weights()

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
                h = layer(h, freqs_cis, depth_kvs=prior_kvs)
                if self.use_moda:
                    current_kvs[idx] = layer.compute_depth_kv(h.detach())
            else:
                h = layer(h)
        return h, current_kvs

    def _forward_unrolled(self, input_ids: torch.Tensor):
        B, T = input_ids.shape
        h = self.tok_embeddings(input_ids)
        freqs_cis = self.freqs_cis[:T]
        input_embed = h

        depth_kv_buffer: List[Dict[int, Tuple[torch.Tensor, torch.Tensor]]] = []

        h, current_kvs = self._run_shared_block(h, freqs_cis, depth_kv_buffer)
        h = self.iter_norm(h) + self.loop_pos_embeds[0]
        if current_kvs:
            depth_kv_buffer.append(current_kvs)

        for i in range(1, self.mean_recurrence):
            h = self.injection(h, input_embed)
            h, current_kvs = self._run_shared_block(h, freqs_cis, depth_kv_buffer)
            h = self.iter_norm(h) + self.loop_pos_embeds[i]
            if current_kvs:
                depth_kv_buffer.append(current_kvs)

        normed = self.norm(h)
        if self.use_chunked_ce and self.training:
            # Return low-rank hidden state for trainer-side ChunkedLinearCrossEntropyLoss.
            return self.lm_head.forward_hlow(normed)

        logits = self.lm_head(normed)

        if self.training and self.use_mtp and hasattr(self, "mtp_head"):
            return {"logits": logits, "mtp1": self.mtp_head(h)}
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

    def param_count(self) -> Dict[str, int]:
        embed = sum(p.numel() for p in self.tok_embeddings.parameters())
        head = sum(p.numel() for n, p in self.lm_head.named_parameters()
                   if "embed_table" not in n)
        shared = sum(p.numel() for p in self.shared_layers.parameters())
        inject = sum(p.numel() for p in self.injection.parameters())
        mtp = sum(p.numel() for n, p in self.mtp_head.named_parameters()
                  if "embed_table" not in n) if hasattr(self, "mtp_head") else 0
        misc = self.loop_pos_embeds.numel() + sum(
            p.numel() for p in self.iter_norm.parameters()) + sum(
            p.numel() for p in self.norm.parameters())
        unique = embed + head + shared + inject + mtp + misc
        return {
            "embed": embed, "head": head, "shared": shared,
            "injection": inject, "mtp": mtp, "misc": misc,
            "unique_total": unique,
            "effective_total": embed + head + shared * self.mean_recurrence + inject + mtp + misc,
        }


class VidarHalo(VidarHaloBase):
    """Production (custom 32K tokenizer): d=768, 4 layers × 2 iters, ~47M unique."""
    pass


class VidarHaloGPT2(VidarHaloBase):
    """GPT-2 tokenizer variant: vocab=50257, ~54M unique. For pre-tokenized .bin files."""

    def __init__(self, **kw):
        kw.setdefault("vocab_size", 50257)
        kw.setdefault("embed_rank", 448)
        super().__init__(**kw)


class VidarHaloMini(VidarHaloBase):
    """Smoke test: d=128, 4 layers, ~2M params."""

    def __init__(self, **kw):
        kw.setdefault("d_model", 128)
        kw.setdefault("embed_rank", 32)
        kw.setdefault("n_shared_layers", 4)
        kw.setdefault("gqa_positions", (2,))
        kw.setdefault("n_heads", 4)
        kw.setdefault("n_kv_heads", 2)
        kw.setdefault("ffn_inner", 256)
        kw.setdefault("d_conv", 128)
        kw.setdefault("mean_recurrence", 2)
        kw.setdefault("backprop_depth", 2)
        super().__init__(**kw)


class VidarHaloAblation(VidarHaloBase):
    """Ablation variant: d=768, 2 layers (1 ShortConv + 1 GQA) × 2 iters, ~30M unique.

    Same width as production VidarHalo — keeps GEMM shapes, autokernel patterns,
    and vocab matmul ratio identical. Depth-reduced for fast screening.

    All technique flags default off (baseline = standard VidarHalo at d=768, fewer layers).
    """

    def __init__(self, iter_scales_enabled=False, softcap=False,
                 delayed_recurrence=False, delayed_soft=False,
                 parallel_residuals=False, skip_connection=False, **kw):
        kw.setdefault("d_model", 768)
        kw.setdefault("embed_rank", 384)
        kw.setdefault("n_shared_layers", 2)
        kw.setdefault("gqa_positions", (1,))
        kw.setdefault("n_heads", 12)
        kw.setdefault("n_kv_heads", 4)
        kw.setdefault("ffn_inner", 2816)
        kw.setdefault("d_conv", 640)
        kw.setdefault("conv_kernel", 3)
        kw.setdefault("mean_recurrence", 2)
        kw.setdefault("backprop_depth", 2)
        super().__init__(**kw)

        self._softcap = softcap
        self._delayed_recurrence = delayed_recurrence
        self._delayed_soft = delayed_soft
        self._parallel_residuals = parallel_residuals
        self._skip_connection = skip_connection

        if iter_scales_enabled:
            self.iter_output_scales = nn.Parameter(torch.ones(self.mean_recurrence))

        if skip_connection:
            self.skip_gate = nn.Parameter(torch.zeros(self.d_model))

        if parallel_residuals:
            self.par_mix = nn.Parameter(torch.tensor([1.0, 0.0, 0.0, 1.0]))
            self.par_res_scale = nn.Parameter(torch.ones(2))

        self._current_step = 0
        self._total_steps = 1

    def set_training_progress(self, current_step: int, total_steps: int):
        self._current_step = current_step
        self._total_steps = total_steps

    def _run_shared_block(self, h, freqs_cis, depth_kv_buffer):
        current_kvs = {}
        for idx, (layer, is_gqa) in enumerate(zip(self.shared_layers, self._is_gqa)):
            if is_gqa:
                prior_kvs = None
                if self.use_moda and depth_kv_buffer:
                    prior_kvs = [buf[idx] for buf in depth_kv_buffer if idx in buf]
                    if not prior_kvs:
                        prior_kvs = None
                if self._parallel_residuals:
                    attn_out = layer.attn(layer.pre_norm(h), freqs_cis, depth_kvs=prior_kvs)
                    ffn_out = layer.ffn(layer.ffn_norm(h))
                    m = self.par_mix
                    lane0 = self.par_res_scale[0] * h + m[0] * attn_out + m[2] * ffn_out
                    lane1 = self.par_res_scale[1] * h + m[1] * attn_out + m[3] * ffn_out
                    h = 0.5 * (lane0 + lane1)
                else:
                    h = layer(h, freqs_cis, depth_kvs=prior_kvs)
                if self.use_moda:
                    current_kvs[idx] = layer.compute_depth_kv(h.detach())
            else:
                h = layer(h)
        return h, current_kvs

    def _apply_iter_norm(self, h, i):
        h = self.iter_norm(h)
        if hasattr(self, "iter_output_scales"):
            h = h * self.iter_output_scales[i]
        return h + self.loop_pos_embeds[i]

    def _forward_unrolled(self, input_ids):
        B, T = input_ids.shape
        h = self.tok_embeddings(input_ids)
        freqs_cis = self.freqs_cis[:T]
        input_embed = h
        depth_kv_buffer = []

        flat_phase = (self._delayed_recurrence and
                      self._current_step < 0.35 * self._total_steps)

        if flat_phase:
            h, current_kvs = self._run_shared_block(h, freqs_cis, depth_kv_buffer)
            h = self._apply_iter_norm(h, 0)
        else:
            h, current_kvs = self._run_shared_block(h, freqs_cis, depth_kv_buffer)
            h = self._apply_iter_norm(h, 0)
            if current_kvs:
                depth_kv_buffer.append(current_kvs)

            h0_skip = h if self._skip_connection else None

            for i in range(1, self.mean_recurrence):
                if self._delayed_recurrence and self._delayed_soft:
                    act_step = int(0.35 * self._total_steps)
                    ramp = max(0.0, min(1.0, (self._current_step - act_step) / 500.0))
                    h_inj = self.injection(h, input_embed)
                    h = h + ramp * (h_inj - h)
                else:
                    h = self.injection(h, input_embed)

                if self._skip_connection and h0_skip is not None:
                    h = h + torch.sigmoid(self.skip_gate) * h0_skip

                h, current_kvs = self._run_shared_block(h, freqs_cis, depth_kv_buffer)
                h = self._apply_iter_norm(h, i)
                if current_kvs:
                    depth_kv_buffer.append(current_kvs)

        logits = self.lm_head(self.norm(h))

        if self._softcap:
            logits = 30.0 * torch.tanh(logits / 30.0)

        if self.training and self.use_mtp and hasattr(self, "mtp_head"):
            return {"logits": logits, "mtp1": self.mtp_head(h)}
        return logits

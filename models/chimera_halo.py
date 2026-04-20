"""
CHIMERA-HALO: Unified Looped Hybrid with Factorized Embeddings.

Combines insights from 5 papers:
  - LFM2: 75:25 conv:attention hybrid ratio, hardware-optimized
  - Parcae: Stable looping via spectral constraint rho(A)<1, scaling laws
  - XSA: Exclusive Self Attention (zero-param quality boost)
  - Nandi: Factorized embeddings + layer sharing for parameter efficiency
  - Attention-to-Mamba: Identity initialization for new components

Architecture: FactorizedEmbed → Prelude GQA → [Shared Block x2 repeat, Parcae loop] → Coda GQA → FactorizedLMHead
  - 8 unique shared layers (6 ShortConv + 2 GQA), repeated 2x = 16 effective
  - Parcae loop with Poisson depth sampling (mean=3) = 48 effective layers
  - ~94M unique params, ~158M effective

Baselines: SmolLM2-135M (HellaSwag 42.1), Nandi-150M (avg 25.63)

Usage:
    python -m halo_training --model models/chimera_halo.py --class-name ChimeraHaloMini --smoke
    python -m halo_training --model models/chimera_halo.py --class-name ChimeraHalo --dataset babylm --compile --optimize-kernels --muon
"""

import math
from typing import List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.amadeus import RMSNorm, SwiGLU, GatedConv
from models.argus import precompute_freqs_cis, apply_rotary_emb
from models.argus_prime import Attention, ShortConvBlock
from models.griffin_halo import SimpleParcaeInjection
from models.jormungandr_halo import DepthMemoryCache, CodaAttention

try:
    from kernels.hip.hybrid_attention import hybrid_flash_sdpa_attention
    _HAS_HYBRID_ATTN = True
except ImportError:
    _HAS_HYBRID_ATTN = False


# ---------------------------------------------------------------------------
# Factorized Embedding (Nandi-style: vocab → rank → d_model)
# ---------------------------------------------------------------------------

class FactorizedEmbedding(nn.Module):
    """Low-rank embedding: Embedding(V, R) → Linear(R, D).

    Saves (V * D - V * R - R * D) params. With V=50260, R=256, D=768:
    saves ~25M params vs standard Embedding(V, D).
    """

    def __init__(self, vocab_size: int, rank: int, d_model: int):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, rank)
        self.proj_up = nn.Linear(rank, d_model, bias=False)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.proj_up(self.embed(input_ids))


class FactorizedLMHead(nn.Module):
    """Factorized output: Linear(D, R) → matmul with embed table^T.

    Shares the embedding table from FactorizedEmbedding (tied weights).
    """

    def __init__(self, d_model: int, rank: int, embed_table: nn.Embedding):
        super().__init__()
        self.proj_down = nn.Linear(d_model, rank, bias=False)
        self.embed_table = embed_table

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        h_low = self.proj_down(h)
        return F.linear(h_low, self.embed_table.weight)


# ---------------------------------------------------------------------------
# XSA GQA Block (GQA + Exclusive Self Attention + momentum + SwiGLU)
# ---------------------------------------------------------------------------

class XSAGQABlock(nn.Module):
    """GQA block with Exclusive Self Attention, inlined momentum, SwiGLU FFN.

    Uses CodaAttention(exclusive=True) for XSA. When XSA is disabled, falls
    back to standard Attention for full autokernel FusedQKV compatibility.
    """

    def __init__(self, d_model: int, ffn_inner: int,
                 n_heads: int = 12, n_kv_heads: int = 4,
                 momentum_beta: float = 0.5, use_xsa: bool = True):
        super().__init__()
        self.pre_norm = RMSNorm(d_model)
        self.use_xsa = use_xsa
        if use_xsa:
            self.attn = CodaAttention(d_model, n_heads, n_kv_heads,
                                      qk_norm=True, exclusive=True)
        else:
            self.attn = Attention(d_model, n_heads, n_kv_heads, qk_norm=True)
        self.log_beta = nn.Parameter(
            torch.tensor(math.log(momentum_beta / (1 - momentum_beta)))
        )
        self.ffn_norm = RMSNorm(d_model)
        self.ffn = SwiGLU(d_model, ffn_inner)

    def forward(self, x: torch.Tensor, velocity: torch.Tensor,
                freqs_cis: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        attn_out = self.attn(self.pre_norm(x), freqs_cis)

        beta = torch.sigmoid(self.log_beta)
        velocity = beta * velocity + attn_out

        x = x + velocity
        rms = torch.rsqrt(x.float().pow(2).mean(-1, keepdim=True) + 1e-6)
        normed = (x.float() * rms).to(x.dtype) * self.ffn_norm.weight

        x = x + self.ffn(normed)
        return x, velocity


# ---------------------------------------------------------------------------
# CHIMERA-HALO Model
# ---------------------------------------------------------------------------

class ChimeraHaloBase(nn.Module):
    """CHIMERA-HALO: Uniform d looped hybrid with factorized embeddings.

    Prelude (1 GQA) → Parcae loop [Shared Block × n_repeat, iterated] → Coda (1 GQA) → output.
    Shared block: 6 ShortConv + 2 GQA with XSA, repeated n_repeat times per iteration.
    """

    def __init__(
        self,
        vocab_size: int = 50257,
        d_model: int = 768,
        embed_rank: int = 256,
        n_shared_layers: int = 8,
        gqa_positions: Tuple[int, ...] = (3, 6),
        n_repeat: int = 2,
        d_conv: int = 512,
        ffn_inner: int = 2816,
        n_heads: int = 12,
        n_kv_heads: int = 4,
        conv_kernel: int = 3,
        mean_recurrence: int = 3,
        backprop_depth: int = 2,
        curriculum_steps: int = 5000,
        use_xsa: bool = True,
        use_depth_cache: bool = True,
        d_gate: int = 64,
        momentum_beta_init: float = 0.5,
        max_seq_len: int = 1024,
        use_prelude: bool = True,
        use_coda: bool = True,
    ):
        super().__init__()
        self.d_model = d_model
        self.n_repeat = n_repeat
        self.mean_recurrence = mean_recurrence
        self.backprop_depth = backprop_depth
        self.curriculum_steps = curriculum_steps
        self.max_seq_len = max_seq_len
        self.gqa_positions = set(gqa_positions)
        self.use_prelude = use_prelude
        self.use_coda = use_coda

        # === FACTORIZED EMBEDDINGS ===
        self.tok_embeddings = FactorizedEmbedding(vocab_size, embed_rank, d_model)
        self.norm = RMSNorm(d_model)
        self.lm_head = FactorizedLMHead(d_model, embed_rank, self.tok_embeddings.embed)

        # RoPE
        head_dim = d_model // n_heads
        self.register_buffer(
            "freqs_cis",
            precompute_freqs_cis(head_dim, max_seq_len * 2),
            persistent=False,
        )

        # Step counter for curriculum
        self.register_buffer("step_counter", torch.tensor(0, dtype=torch.long))

        # === PRELUDE (unshared entry GQA) ===
        if use_prelude:
            self.prelude = XSAGQABlock(
                d_model, ffn_inner, n_heads, n_kv_heads,
                momentum_beta_init, use_xsa=use_xsa,
            )

        # === PARCAE INJECTION ===
        self.injection = SimpleParcaeInjection(d_model)

        # === SHARED BLOCK (8 unique layers, iterated n_repeat times) ===
        self.shared_layers = nn.ModuleList()
        for i in range(n_shared_layers):
            if i in self.gqa_positions:
                self.shared_layers.append(XSAGQABlock(
                    d_model, ffn_inner, n_heads, n_kv_heads,
                    momentum_beta_init, use_xsa=use_xsa,
                ))
            else:
                self.shared_layers.append(ShortConvBlock(
                    d_model, d_conv, ffn_inner, conv_kernel, momentum_beta_init,
                ))

        # Norm after each loop iteration (stabilizes shared-weight gradient flow)
        self.iter_norm = RMSNorm(d_model)

        # === DEPTH MEMORY CACHE ===
        self.depth_cache = DepthMemoryCache(d_model, d_gate) if use_depth_cache else None

        # === CODA (unshared exit GQA) ===
        if use_coda:
            self.coda = XSAGQABlock(
                d_model, ffn_inner, n_heads, n_kv_heads,
                momentum_beta_init, use_xsa=use_xsa,
            )

        self._init_weights()
        n_params = sum(p.numel() for p in self.parameters())
        print(f"{self.__class__.__name__}: {n_params / 1e6:.1f}M parameters")

    def _init_weights(self):
        for name, p in self.named_parameters():
            if p.dim() >= 2:
                nn.init.xavier_uniform_(p)

    def compile_zones(self):
        """Compile each layer independently for per-zone fusion.

        Call AFTER autokernel.optimize() and BEFORE training.
        Python loops (repeat, Parcae) stay uncompiled; each layer = fused kernel.
        """
        if self.use_prelude:
            self.prelude = torch.compile(self.prelude, mode="default")
        for i in range(len(self.shared_layers)):
            self.shared_layers[i] = torch.compile(self.shared_layers[i], mode="default")
        if self.use_coda:
            self.coda = torch.compile(self.coda, mode="default")
        return self

    def sample_loop_depth(self, step: int) -> Tuple[int, int]:
        """Parcae-style Poisson depth sampling with 1-sqrt curriculum."""
        progress = min(step / max(self.curriculum_steps, 1), 1.0)
        effective_progress = 1 - math.sqrt(1 - progress)

        t_full = max(self.mean_recurrence - self.backprop_depth, 0)
        t = max(math.ceil(effective_progress * t_full), 0)

        n_detached = torch.poisson(torch.tensor([float(t)])).long().item()
        n_detached = min(n_detached, 2 * max(t, 1))
        n_grad = self.backprop_depth

        return n_detached, n_grad

    def _run_shared_block(self, h: torch.Tensor, velocity: torch.Tensor,
                          freqs_cis: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Run all shared layers n_repeat times."""
        for _repeat in range(self.n_repeat):
            for layer in self.shared_layers:
                if isinstance(layer, XSAGQABlock):
                    h, velocity = layer(h, velocity, freqs_cis)
                else:
                    h, velocity = layer(h, velocity)
        return h, velocity

    def forward(self, input_ids: torch.Tensor, targets=None) -> torch.Tensor:
        B, T = input_ids.shape

        if torch.is_autocast_enabled() and torch.get_autocast_gpu_dtype() == torch.bfloat16:
            raise RuntimeError(
                "bf16 is 24% slower on gfx1151 and crashes compile with RoPE. Use fp16."
            )

        h = self.tok_embeddings(input_ids)
        freqs_cis = self.freqs_cis[:T]

        # === PRELUDE ===
        velocity = torch.zeros_like(h)
        if self.use_prelude:
            h, velocity = self.prelude(h, velocity, freqs_cis)
        input_embed = h

        # === PARCAE LOOP ===
        if self.training:
            step = self.step_counter.item()
            n_detached, n_grad = self.sample_loop_depth(step)
            self.step_counter += 1
        else:
            n_detached = 0
            n_grad = self.mean_recurrence

        use_dc = self.depth_cache is not None
        cached_states = []
        total_iters = n_detached + n_grad

        # First iteration: no re-injection (h == input_embed)
        h, velocity = self._run_shared_block(h, velocity, freqs_cis)
        h = self.iter_norm(h)
        if total_iters > 1:
            if n_detached > 0:
                if use_dc or not self.training:
                    cached_states.append(h.detach())
                n_detached -= 1
            else:
                if use_dc:
                    cached_states.append(h)
                elif not self.training:
                    cached_states.append(h.detach())
                n_grad -= 1

        # Detached re-entry iterations
        for _t in range(n_detached):
            with torch.no_grad():
                h = self.injection(h, input_embed)
                h, velocity = self._run_shared_block(h, velocity, freqs_cis)
                h = self.iter_norm(h)
                if use_dc or not self.training:
                    cached_states.append(h.detach())

        # Gradient-tracked re-entry iterations
        for _t in range(n_grad):
            h = self.injection(h, input_embed)
            h, velocity = self._run_shared_block(h, velocity, freqs_cis)
            h = self.iter_norm(h)
            if use_dc:
                cached_states.append(h)
            elif not self.training:
                cached_states.append(h.detach())

        # Depth Memory Cache: gated aggregation over iteration states
        if use_dc and len(cached_states) > 1:
            h = self.depth_cache(cached_states)

        # === CODA ===
        velocity = torch.zeros_like(h)
        if self.use_coda:
            h, velocity = self.coda(h, velocity, freqs_cis)

        return self.lm_head(self.norm(h))


# ---------------------------------------------------------------------------
# Variant classes
# ---------------------------------------------------------------------------

class ChimeraHalo(ChimeraHaloBase):
    """Default: mean_recurrence=3, XSA on, depth cache on."""
    def __init__(self, **kw):
        super().__init__(mean_recurrence=3, **kw)


class ChimeraHaloDeep(ChimeraHaloBase):
    """Deep: 5 iterations for maximum effective capacity."""
    def __init__(self, **kw):
        super().__init__(mean_recurrence=5, backprop_depth=3, **kw)


class ChimeraHaloBare(ChimeraHaloBase):
    """Ablation: no XSA, no depth cache."""
    def __init__(self, **kw):
        super().__init__(
            mean_recurrence=3,
            use_xsa=False, use_depth_cache=False, **kw,
        )


class ChimeraHaloNoLoop(ChimeraHaloBase):
    """Ablation: no Parcae loop (single pass through shared block x2)."""
    def __init__(self, **kw):
        super().__init__(
            mean_recurrence=1, backprop_depth=1,
            use_depth_cache=False, **kw,
        )


class ChimeraHaloMini(ChimeraHaloBase):
    """Tiny config for smoke testing (~3M params)."""
    def __init__(self):
        super().__init__(
            vocab_size=1000,
            d_model=128,
            embed_rank=32,
            n_shared_layers=4,
            gqa_positions=(1, 3),
            n_repeat=2,
            d_conv=128,
            ffn_inner=256,
            n_heads=4,
            n_kv_heads=2,
            conv_kernel=3,
            mean_recurrence=2,
            backprop_depth=2,
            curriculum_steps=100,
            use_xsa=False,
            use_depth_cache=False,
            momentum_beta_init=0.5,
            max_seq_len=512,
            use_prelude=False,
            use_coda=False,
        )

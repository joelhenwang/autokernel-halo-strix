"""
FENRIR-HALO: Clean-Sheet Parcae-Looped Hybrid for Strix Halo DDP.

Architecture: FactorizedEmbed -> Prelude GQA -> [10-layer Shared Block, Parcae loop] -> Coda GQA -> FactorizedLMHead
  - 10 unique shared layers (8 ShortConv + 2 GQA/XSA), NO repeat
  - Parcae loop with Poisson depth sampling (mean=3) = 30 effective layers
  - ~80M unique params, ~160M Parcae-equivalent
  - Proven mechanisms only: XSA, DepthMemoryCache, factorized embeddings

Targets: Beat Portimbria-150M, compete with SmolLM2-135M on 12B tokens (2-machine DDP).

Usage:
    python -m halo_training --model models/fenrir_halo.py --class-name FenrirHaloMini --smoke
    python -m halo_training --model models/fenrir_halo.py --class-name FenrirHalo --dataset babylm --compile --optimize-kernels --muon
"""

import math
from typing import List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.amadeus import RMSNorm, SwiGLU, GatedConv
from models.argus import precompute_freqs_cis, apply_rotary_emb
from models.argus_prime import Attention, ShortConvBlock
from models.chimera_halo import FactorizedEmbedding, FactorizedLMHead
from models.griffin_halo import SimpleParcaeInjection
from models.jormungandr_halo import DepthMemoryCache, CodaAttention

try:
    from kernels.hip.hybrid_attention import hybrid_flash_sdpa_attention
    _HAS_HYBRID_ATTN = True
except ImportError:
    _HAS_HYBRID_ATTN = False


# ---------------------------------------------------------------------------
# XSA GQA Block (GQA + Exclusive Self Attention + momentum + SwiGLU)
# ---------------------------------------------------------------------------

class XSAGQABlock(nn.Module):
    """GQA block with Exclusive Self Attention, inlined momentum, SwiGLU FFN.

    Uses CodaAttention(exclusive=True) for XSA. When XSA is disabled, falls
    back to standard Attention for full autokernel FusedQKV compatibility.
    """

    def __init__(self, d_model: int, ffn_inner: int,
                 n_heads: int = 10, n_kv_heads: int = 2,
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
        velocity = velocity.clamp(-8.0, 8.0)

        x = x + velocity
        rms = torch.rsqrt(x.float().pow(2).mean(-1, keepdim=True) + 1e-6)
        normed = (x.float() * rms).to(x.dtype) * self.ffn_norm.weight

        x = x + self.ffn(normed)
        return x, velocity


# ---------------------------------------------------------------------------
# FENRIR-HALO Model
# ---------------------------------------------------------------------------

class FenrirHaloBase(nn.Module):
    """FENRIR-HALO: Clean-sheet Parcae loop with uniform d=640.

    Prelude (1 GQA) -> Parcae loop [10-layer Shared Block, iterated] -> Coda (1 GQA) -> output.
    Shared block: 8 ShortConv + 2 GQA with XSA. No layer repeat -- Parcae loop alone provides depth.
    """

    def __init__(
        self,
        vocab_size: int = 50257,
        d_model: int = 640,
        embed_rank: int = 256,
        n_shared_layers: int = 10,
        gqa_positions: Tuple[int, ...] = (4, 9),
        d_conv: int = 512,
        ffn_inner: int = 2304,
        n_heads: int = 10,
        n_kv_heads: int = 2,
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

        # === SHARED BLOCK (10 unique layers, iterated via Parcae loop) ===
        self.shared_layers = nn.ModuleList()
        self._is_gqa = []
        for i in range(n_shared_layers):
            if i in self.gqa_positions:
                self.shared_layers.append(XSAGQABlock(
                    d_model, ffn_inner, n_heads, n_kv_heads,
                    momentum_beta_init, use_xsa=use_xsa,
                ))
                self._is_gqa.append(True)
            else:
                self.shared_layers.append(ShortConvBlock(
                    d_model, d_conv, ffn_inner, conv_kernel, momentum_beta_init,
                ))
                self._is_gqa.append(False)

        # Norm after each loop iteration
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
        n_layers = len(self.shared_layers)
        for name, p in self.named_parameters():
            if p.dim() >= 2:
                nn.init.xavier_uniform_(p)
                if "wo." in name or "w_down." in name or "out_proj." in name:
                    with torch.no_grad():
                        p.div_(math.sqrt(2 * n_layers))
            elif p.dim() == 1 and "bias" in name:
                nn.init.zeros_(p)

    def compile_zones(self):
        """Compile each layer independently for per-zone fusion.

        Call AFTER autokernel.optimize() and BEFORE training.
        Python loops (Parcae) stay uncompiled; each layer = fused kernel.
        """
        try:
            from kernels.hip._torch_ops import disable_hip_backward
            disable_hip_backward()
        except ImportError:
            pass
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
        """Run all 10 shared layers once. No repeat -- Parcae loop provides depth."""
        for layer, is_gqa in zip(self.shared_layers, self._is_gqa):
            if is_gqa:
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

class FenrirHalo(FenrirHaloBase):
    """Default: mean_recurrence=3, XSA on, depth cache on."""
    def __init__(self, **kw):
        super().__init__(mean_recurrence=3, **kw)


class FenrirHaloDeep(FenrirHaloBase):
    """Deep: 5 iterations for maximum effective capacity."""
    def __init__(self, **kw):
        super().__init__(mean_recurrence=5, backprop_depth=3, **kw)


class FenrirHaloBare(FenrirHaloBase):
    """Ablation: no XSA, no depth cache."""
    def __init__(self, **kw):
        super().__init__(
            mean_recurrence=3,
            use_xsa=False, use_depth_cache=False, **kw,
        )


class FenrirHaloNoLoop(FenrirHaloBase):
    """Ablation: no Parcae loop (single pass through shared block)."""
    def __init__(self, **kw):
        super().__init__(
            mean_recurrence=1, backprop_depth=1,
            use_depth_cache=False, **kw,
        )


class FenrirHaloMini(FenrirHaloBase):
    """Tiny config for smoke testing + CLIMB proxy search.

    Uses full vocab (50257) -- NOT 1000. ChimeraHaloMini crash taught us
    that mini variants must handle real data tokens.
    """
    def __init__(self):
        super().__init__(
            vocab_size=50257,
            d_model=128,
            embed_rank=32,
            n_shared_layers=4,
            gqa_positions=(1, 3),
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

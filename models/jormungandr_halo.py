"""
JORMUNGANDR-HALO: Hardware-Adapted Looped Architecture for Strix Halo.

Heterogeneous dimensions: d=768 Prelude/Coda, d=512 L2-resident core loop.
Core block (~1.2MB fp16) fits in 1/5 of 6MB L2 cache, making loop iterations
2+ ~3x cheaper. Parcae A-matrix stability, Poisson depth sampling, staged
component activation.

110M unique params, 158M effective at 4 iters, 206M at 8 iters.

Design spec: docs/superpowers/specs/2026-04-16-jormungandr-halo-design.md
Parent plan: mad_llm_scientist/plans/JORMUNGANDR.md

Usage:
    python -m halo_training --model models/jormungandr_halo.py --class-name JormungandrHalo --smoke
    python -m halo_training --model models/jormungandr_halo.py --class-name JormungandrHalo --dataset babylm --compile --optimize-kernels --muon
"""

import math
from typing import List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.amadeus import RMSNorm, SwiGLU, GatedConv
from models.argus import TTTSwiGLU, precompute_freqs_cis, apply_rotary_emb
from models.argus_prime import Attention, ShortConvBlock, GQABlock, MultiStepTTTSwiGLU

try:
    from kernels.hip.hybrid_attention import hybrid_flash_sdpa_attention
    _HAS_HYBRID_ATTN = True
except ImportError:
    _HAS_HYBRID_ATTN = False


# ---------------------------------------------------------------------------
# Parcae Injection (A-matrix stability guarantee + dimension adapter)
# ---------------------------------------------------------------------------

class ParcaeInjection(nn.Module):
    """Stable loop injection: h = A*h + B*input_embed_projected.

    A = -exp(log_A) guarantees eigenvalues in (-1, 0) — contractive.
    B = exp(log_B) controls input re-injection strength.
    proj_down maps d_prelude → d_core for dimension change.
    """

    def __init__(self, d_prelude: int, d_core: int):
        super().__init__()
        self.log_A = nn.Parameter(torch.full((d_core,), -0.7))
        self.log_B = nn.Parameter(torch.full((d_core,), -0.7))
        self.proj_down = nn.Linear(d_prelude, d_core, bias=False)

    def forward(self, h: torch.Tensor, input_embed_projected: torch.Tensor) -> torch.Tensor:
        A = -torch.exp(self.log_A)
        B = torch.exp(self.log_B)
        return A * h + B * input_embed_projected


# ---------------------------------------------------------------------------
# Value Embedding (token-aware bias for attention values)
# ---------------------------------------------------------------------------

class ValueEmbedding(nn.Module):
    """Per-layer vocabulary-aware bias for attention values.

    Projected from d_ve=64 to kv_dim to keep param count reasonable.
    Applied to last Coda GQA layer only. Zero-init starts as no-op.
    """

    def __init__(self, vocab_size: int, d_ve: int, kv_dim: int):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, d_ve)
        self.proj = nn.Linear(d_ve, kv_dim, bias=False)
        nn.init.zeros_(self.embed.weight)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.proj(self.embed(input_ids))


# ---------------------------------------------------------------------------
# CodaAttention (Attention + optional ValueEmbedding bias)
# ---------------------------------------------------------------------------

class CodaAttention(Attention):
    """GQA Attention that optionally adds a ValueEmbedding bias to V."""

    def forward(self, x: torch.Tensor, freqs_cis: torch.Tensor,
                value_bias: Optional[torch.Tensor] = None) -> torch.Tensor:
        B, T, _ = x.shape
        q = self.wq(x).view(B, T, self.n_heads, self.head_dim)
        k = self.wk(x).view(B, T, self.n_kv_heads, self.head_dim)
        v = self.wv(x).view(B, T, self.n_kv_heads, self.head_dim)

        # Inject value embedding bias before RoPE/attention
        if value_bias is not None:
            v = v + value_bias.view(B, T, self.n_kv_heads, self.head_dim)

        q, k = apply_rotary_emb(q, k, freqs_cis)
        q, k, v = q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)

        if self.qk_norm:
            q = F.normalize(q, dim=-1) * self.q_scale
            k = F.normalize(k, dim=-1) * self.k_scale

        if self.n_rep > 1:
            k = k.repeat_interleave(self.n_rep, dim=1)
            v = v.repeat_interleave(self.n_rep, dim=1)

        if _HAS_HYBRID_ATTN and q.dtype == torch.float16:
            y = hybrid_flash_sdpa_attention(
                q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2), causal=True
            ).transpose(1, 2)
        else:
            y = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        return self.wo(y.transpose(1, 2).contiguous().view(B, T, -1))


# ---------------------------------------------------------------------------
# CodaGQABlock (GQA + optional VE + optional TTT)
# ---------------------------------------------------------------------------

class CodaGQABlock(nn.Module):
    """GQA Attention + inlined momentum + SwiGLU/TTT FFN, with optional ValueEmbedding."""

    def __init__(self, d_model: int, ffn_inner: int,
                 n_heads: int = 12, n_kv_heads: int = 4,
                 momentum_beta: float = 0.5,
                 ttt_mode: str = "none",
                 ttt_chunk: int = 512, ttt_lr_init: float = 0.01,
                 value_embedding: Optional[ValueEmbedding] = None):
        super().__init__()
        self.pre_norm = RMSNorm(d_model)
        self.attn = CodaAttention(d_model, n_heads, n_kv_heads, qk_norm=True)
        self.log_beta = nn.Parameter(
            torch.tensor(math.log(momentum_beta / (1 - momentum_beta)))
        )
        self.ffn_norm = RMSNorm(d_model)
        self.ttt_mode = ttt_mode
        self.value_embedding = value_embedding

        if ttt_mode == "single":
            self.ffn = TTTSwiGLU(d_model, ffn_inner, ttt_chunk, ttt_lr_init)
        elif ttt_mode == "multi":
            self.ffn = MultiStepTTTSwiGLU(d_model, ffn_inner, ttt_chunk, ttt_lr_init)
        else:
            self.ffn = SwiGLU(d_model, ffn_inner)

    def forward(self, x, velocity, freqs_cis, ttt_target=None, input_ids=None):
        # Value embedding bias
        value_bias = None
        if self.value_embedding is not None and input_ids is not None:
            value_bias = self.value_embedding(input_ids)

        # Call attention — avoid passing value_bias kwarg when None
        # (autokernel's fused replacement doesn't accept it)
        if value_bias is not None:
            attn_out = self.attn(self.pre_norm(x), freqs_cis, value_bias=value_bias)
        else:
            attn_out = self.attn(self.pre_norm(x), freqs_cis)

        # Inlined momentum
        beta = torch.sigmoid(self.log_beta)
        velocity = beta * velocity + attn_out

        # Inlined residual + RMSNorm
        x = x + velocity
        rms = torch.rsqrt(x.float().pow(2).mean(-1, keepdim=True) + 1e-6)
        normed = (x.float() * rms).to(x.dtype) * self.ffn_norm.weight

        if self.ttt_mode != "none" and ttt_target is not None:
            x = x + self.ffn(normed, ttt_target=ttt_target)
        else:
            x = x + self.ffn(normed)
        return x, velocity


# ---------------------------------------------------------------------------
# Cross-Dimension FiLM Conditioner (d=512 context → d=768 modulation)
# ---------------------------------------------------------------------------

class CrossDimFiLMConditioner(nn.Module):
    """FiLM that bridges d_context (core loop) to d_target (Coda layers).

    context_proj: Linear(d_context, d_film) — compresses loop state to fingerprint
    gamma/beta_projs: Linear(d_film, d_target) — expands to modulation per target
    Zero-init so FiLM starts as identity transform.
    """

    def __init__(self, d_context: int, d_film: int, d_target: int, n_targets: int):
        super().__init__()
        self.context_proj = nn.Linear(d_context, d_film, bias=True)
        self.gamma_projs = nn.ModuleList([
            nn.Linear(d_film, d_target, bias=True) for _ in range(n_targets)
        ])
        self.beta_projs = nn.ModuleList([
            nn.Linear(d_film, d_target, bias=True) for _ in range(n_targets)
        ])
        self._init_identity()

    def _init_identity(self):
        for proj in self.gamma_projs:
            nn.init.zeros_(proj.weight)
            nn.init.zeros_(proj.bias)
        for proj in self.beta_projs:
            nn.init.zeros_(proj.weight)
            nn.init.zeros_(proj.bias)

    def compute_context(self, h: torch.Tensor) -> torch.Tensor:
        return self.context_proj(h.mean(dim=1))

    def apply(self, h: torch.Tensor, context: torch.Tensor, target_idx: int) -> torch.Tensor:
        gamma = self.gamma_projs[target_idx](context) + 1.0
        beta = self.beta_projs[target_idx](context)
        return gamma.unsqueeze(1) * h + beta.unsqueeze(1)


# ---------------------------------------------------------------------------
# JORMUNGANDR-HALO Model
# ---------------------------------------------------------------------------

class JormungandrHaloBase(nn.Module):
    """JORMUNGANDR-HALO: heterogeneous d=768/d=512 looped architecture.

    Prelude (d=768) → Parcae injection + proj_down → Core loop (d=512, L2-resident)
    → proj_up → Coda (d=768) → output.
    """

    def __init__(
        self,
        vocab_size: int = 50257,
        d_prelude: int = 768,
        d_core: int = 512,
        d_conv_prelude: int = 512,
        d_conv_core: int = 512,
        ffn_prelude: int = 2816,
        ffn_core: int = 1792,
        n_heads: int = 12,
        n_kv_heads: int = 4,
        conv_kernel: int = 3,
        n_core_layers: int = 3,
        mean_recurrence: int = 4,
        backprop_depth: int = 3,
        curriculum_steps: int = 5000,
        use_film: bool = True,
        d_film: int = 64,
        n_film_targets: int = 6,
        use_ve: bool = True,
        d_ve: int = 64,
        ttt_mode: str = "single",
        ttt_chunk: int = 512,
        ttt_lr_init: float = 0.01,
        momentum_beta_init: float = 0.5,
        max_seq_len: int = 1024,
        randomize_positions: bool = False,
    ):
        super().__init__()
        self.d_prelude = d_prelude
        self.d_core = d_core
        self.mean_recurrence = mean_recurrence
        self.backprop_depth = backprop_depth
        self.curriculum_steps = curriculum_steps
        self.randomize_positions = randomize_positions
        self.max_seq_len = max_seq_len

        # Embedding + tied output (d=768)
        self.tok_embeddings = nn.Embedding(vocab_size, d_prelude)
        self.norm = RMSNorm(d_prelude)
        self.output = nn.Linear(d_prelude, vocab_size, bias=False)
        self.output.weight = self.tok_embeddings.weight

        # RoPE (for Coda GQA layers)
        head_dim = d_prelude // n_heads
        self.register_buffer(
            "freqs_cis",
            precompute_freqs_cis(head_dim, max_seq_len * 2),
            persistent=False,
        )

        # Step counter for curriculum (persists in checkpoints)
        self.register_buffer("step_counter", torch.tensor(0, dtype=torch.long))

        # === PRELUDE (d=768) ===
        self.prelude_conv = ShortConvBlock(
            d_prelude, d_conv_prelude, ffn_prelude, conv_kernel, momentum_beta_init
        )
        self.prelude_gqa = GQABlock(
            d_prelude, ffn_prelude, n_heads, n_kv_heads,
            momentum_beta_init, ttt_mode="none",
        )

        # === PARCAE INJECTION + DIM ADAPTER ===
        self.injection = ParcaeInjection(d_prelude, d_core)

        # === CORE LOOP (d=512, L2-resident) ===
        self.core_layers = nn.ModuleList([
            ShortConvBlock(d_core, d_conv_core, ffn_core, conv_kernel, momentum_beta_init)
            for _ in range(n_core_layers)
        ])

        # === EXIT (512 → 768) ===
        self.proj_up = nn.Linear(d_core, d_prelude, bias=False)

        # === VALUE EMBEDDING (last Coda GQA only) ===
        kv_dim = n_kv_heads * head_dim
        ve = ValueEmbedding(vocab_size, d_ve, kv_dim) if use_ve else None

        # === CODA (d=768, 4 layers) ===
        self.coda_layers = nn.ModuleList([
            ShortConvBlock(d_prelude, d_conv_prelude, ffn_prelude, conv_kernel, momentum_beta_init),
            CodaGQABlock(d_prelude, ffn_prelude, n_heads, n_kv_heads,
                         momentum_beta_init, ttt_mode="none"),
            ShortConvBlock(d_prelude, d_conv_prelude, ffn_prelude, conv_kernel, momentum_beta_init),
            CodaGQABlock(d_prelude, ffn_prelude, n_heads, n_kv_heads,
                         momentum_beta_init, ttt_mode=ttt_mode,
                         ttt_chunk=ttt_chunk, ttt_lr_init=ttt_lr_init,
                         value_embedding=ve),
        ])

        # Store VE reference for param counting (it's inside coda_layers[-1])
        self.value_embedding = ve

        # === FILM (d=512 context → d=768 modulation) ===
        self.film = CrossDimFiLMConditioner(
            d_core, d_film, d_prelude, n_film_targets
        ) if use_film else None

        self._init_weights()
        n_params = sum(p.numel() for p in self.parameters())
        print(f"{self.__class__.__name__}: {n_params / 1e6:.1f}M parameters")

    def _init_weights(self):
        for name, p in self.named_parameters():
            if p.dim() >= 2:
                nn.init.xavier_uniform_(p)
        # TTT-specific init
        for layer in self.coda_layers:
            if isinstance(layer, CodaGQABlock) and hasattr(layer.ffn, '_init_ttt_weights'):
                layer.ffn._init_ttt_weights()
        # Value embedding zero-init (already done in ValueEmbedding.__init__,
        # but xavier above may have overwritten it)
        if self.value_embedding is not None:
            nn.init.zeros_(self.value_embedding.embed.weight)

    def compile_zones(self):
        """Compile each zone independently for per-layer fusion.

        Call AFTER autokernel.optimize() and BEFORE training.
        The Python loop stays uncompiled; each layer becomes a fused kernel.
        """
        self.prelude_conv = torch.compile(self.prelude_conv, mode="default")
        for i in range(len(self.core_layers)):
            self.core_layers[i] = torch.compile(self.core_layers[i], mode="default")
        for i in range(len(self.coda_layers)):
            if not isinstance(self.coda_layers[i], CodaGQABlock):
                self.coda_layers[i] = torch.compile(self.coda_layers[i], mode="default")
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

    @torch.no_grad()
    def monitor_loop_hsv(self, loop_states: Optional[List[torch.Tensor]] = None) -> dict:
        """CompreSSM-style HSV monitoring on loop state dimensions."""
        if loop_states is None:
            loop_states = getattr(self, '_loop_states', [])
        if len(loop_states) < 2:
            return {}

        H = torch.stack(loop_states, dim=0)
        iter_variance = H.var(dim=0).mean(dim=(0, 1))
        delta = H[-1] - H[0]
        output_contribution = (delta ** 2).mean(dim=(0, 1))
        importance = (iter_variance * output_contribution).sqrt()
        importance_sorted, _ = importance.sort(descending=True)

        total_energy = importance.sum()
        if total_energy < 1e-8:
            return {}
        cumulative = importance_sorted.cumsum(dim=0) / total_energy

        return {
            "hsv/top_10pct_energy": cumulative[int(0.1 * len(cumulative))].item(),
            "hsv/top_50pct_energy": cumulative[int(0.5 * len(cumulative))].item(),
            "hsv/effective_rank": (importance > 0.01 * importance.max()).sum().item(),
            "hsv/energy_concentration": (importance.max() / importance.mean()).item(),
        }

    def forward(self, input_ids: torch.Tensor, targets=None) -> torch.Tensor:
        B, T = input_ids.shape

        # bf16 guardrail
        if torch.is_autocast_enabled() and torch.get_autocast_gpu_dtype() == torch.bfloat16:
            raise RuntimeError(
                "bf16 is 24% slower on gfx1151 and crashes compile with RoPE. Use fp16."
            )

        h = self.tok_embeddings(input_ids)
        freqs_cis = self.freqs_cis[:T]

        # Optional position randomization (disabled by default — RoPE conflict)
        if self.training and self.randomize_positions:
            rand_pos = torch.sort(
                torch.randint(0, self.max_seq_len, (T,), device=h.device)
            )[0]
            freqs_cis = self.freqs_cis[rand_pos]

        # === PRELUDE (d=768) ===
        velocity_768 = torch.zeros_like(h)
        h, velocity_768 = self.prelude_conv(h, velocity_768)
        h, velocity_768 = self.prelude_gqa(h, velocity_768, freqs_cis, ttt_target=None)
        input_embed = h

        # Cache proj_down(input_embed) — reused in every re-injection
        input_embed_down = self.injection.proj_down(input_embed)

        # === FUSED ENTRY (768 → 512) ===
        h_core = self.injection(self.injection.proj_down(h), input_embed_down)

        # === CORE LOOP (d=512, L2-resident) ===
        velocity_512 = torch.zeros(B, T, self.d_core, device=h.device, dtype=h.dtype)

        if self.training:
            step = self.step_counter.item()
            n_detached, n_grad = self.sample_loop_depth(step)
            self.step_counter += 1
        else:
            n_detached = 0
            n_grad = self.mean_recurrence

        loop_states = []
        collect_hsv = not self.training

        # Detached iterations (no grad, no GradScaler overhead)
        for t in range(n_detached):
            with torch.no_grad():
                h_core = self.injection(h_core, input_embed_down)
                for layer in self.core_layers:
                    h_core, velocity_512 = layer(h_core, velocity_512)
                if collect_hsv:
                    loop_states.append(h_core.detach())

        # Gradient-tracked iterations
        context = None
        for t in range(n_grad):
            h_core = self.injection(h_core, input_embed_down)
            for layer in self.core_layers:
                h_core, velocity_512 = layer(h_core, velocity_512)
            if collect_hsv:
                loop_states.append(h_core.detach())

            # FiLM: compute fingerprint on first gradient iteration
            if t == 0 and self.film is not None:
                context = self.film.compute_context(h_core)

            # FiLM: modulate subsequent gradient iterations (targets 0..n_grad-2)
            if t > 0 and context is not None:
                h_core = self.film.apply(h_core, context, t - 1)

        self._loop_states = loop_states

        # === EXIT (512 → 768) ===
        h = self.proj_up(h_core)

        # === CODA (d=768) ===
        velocity_768 = torch.zeros_like(h)
        film_offset = max(n_grad - 1, 0)

        for i, layer in enumerate(self.coda_layers):
            if isinstance(layer, CodaGQABlock):
                ttt_target = h if layer.ttt_mode != "none" else None
                h, velocity_768 = layer(
                    h, velocity_768, freqs_cis,
                    ttt_target=ttt_target, input_ids=input_ids,
                )
            else:
                h, velocity_768 = layer(h, velocity_768)

            # FiLM modulation on each Coda layer
            if context is not None:
                h = self.film.apply(h, context, film_offset + i)

        return self.output(self.norm(h))


# ---------------------------------------------------------------------------
# Variant classes
# ---------------------------------------------------------------------------

class JormungandrHalo(JormungandrHaloBase):
    """Default: 4 iterations, FiLM, VE, single-step TTT."""
    def __init__(self, **kw):
        super().__init__(mean_recurrence=4, **kw)


class JormungandrHaloDeep(JormungandrHaloBase):
    """Deep: 8 iterations for maximum effective capacity."""
    def __init__(self, **kw):
        super().__init__(mean_recurrence=8, **kw)


class JormungandrHaloNoTTT(JormungandrHaloBase):
    """Stage 1-2: No TTT, for bare loop + FiLM testing."""
    def __init__(self, **kw):
        super().__init__(mean_recurrence=4, ttt_mode="none", **kw)


class JormungandrHaloBare(JormungandrHaloBase):
    """Stage 1: Bare loop only — no FiLM, no VE, no TTT."""
    def __init__(self, **kw):
        super().__init__(
            mean_recurrence=4, ttt_mode="none",
            use_film=False, use_ve=False, **kw,
        )


class JormungandrHaloMini(JormungandrHaloBase):
    """Tiny config for smoke testing (~3M params)."""
    def __init__(self):
        super().__init__(
            vocab_size=1000,
            d_prelude=256, d_core=128,
            d_conv_prelude=256, d_conv_core=128,
            ffn_prelude=512, ffn_core=384,
            n_heads=4, n_kv_heads=2,
            n_core_layers=2,
            mean_recurrence=3, backprop_depth=2,
            curriculum_steps=100,
            use_film=False, use_ve=False,
            ttt_mode="none",
            momentum_beta_init=0.5,
            max_seq_len=1024,
        )

"""
HALO-PRIME: JORMUNGANDR-HALO evolved with Mamba-3 SISO in the core loop.

Replaces the local-only ShortConvBlock (kernel=3, ~3 token receptive field)
in the core loop with a parallel Conv+Mamba hybrid block (full sequence
context per iteration). Based on AMADEUS's proven quality-winning mixer.

Structure: S7 hybrid (Prelude d=768 + Core loop d=512 + Coda d=768)
Core mixer: M4+M13 (Mamba-3 SISO + GatedConv, parallel, AMADEUS-style)
Enhancements: XSA, Depth MC, TTT, FiLM, VE, Parcae injection
All proven individually in JORMUNGANDR-HALO ablations.

~107M unique params, ~157M effective at 4 iters.

Usage:
    python -m halo_training --model models/halo_prime.py --class-name HaloPrime --smoke
    python -m halo_training --model models/halo_prime.py --class-name HaloPrime \\
        --compile --optimize-kernels --muon --dataset babylm
"""

import math
from typing import List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

# Reuse all existing components
from models.amadeus import RMSNorm, SwiGLU, GatedConv, Mamba3SISO, _scan_dispatch
from models.argus import TTTSwiGLU, precompute_freqs_cis, apply_rotary_emb
from models.argus_prime import MultiStepTTTSwiGLU
from models.components import Attention, ShortConvBlock, GQABlock
from models.components import CodaAttention, DepthMemoryCache
from models.jormungandr_halo import (
    ParcaeInjection, ValueEmbedding, CodaGQABlock,
    CrossDimFiLMConditioner,
)

_HAS_HYBRID_ATTN = False  # disabled: flash_attn requires aiter


# ---------------------------------------------------------------------------
# Loop-Stable Mamba-3 SISO (tighter dt clamping for iterated use)
# ---------------------------------------------------------------------------

class LoopStableMamba3SISO(nn.Module):
    """Mamba-3 SISO variant hardened for loop iteration.

    Differences from Mamba3SISO:
    - Clamps dt_proj output BEFORE softplus (prevents fp16 gradient poison)
    - Tighter dt max (0.1 vs 0.5) since state accumulates over iterations
    - Identical architecture otherwise
    """

    def __init__(self, d_model: int, d_mamba: int, dstate: int, n_heads: int):
        super().__init__()
        self.d_mamba = d_mamba
        self.dstate = dstate
        self.n_heads = n_heads

        self.x_proj = nn.Linear(d_model, d_mamba, bias=False)
        self.dt_proj = nn.Linear(d_model, d_mamba, bias=True)
        self.B_proj = nn.Linear(d_model, dstate, bias=False)
        self.C_proj = nn.Linear(d_model, dstate, bias=False)

        A_init = torch.log(
            torch.arange(1, dstate + 1, dtype=torch.float32)
        ).unsqueeze(0).repeat(n_heads, 1)
        self.A_log = nn.Parameter(A_init)

        nn.init.constant_(self.dt_proj.bias, -4.0)
        nn.init.normal_(self.dt_proj.weight, std=0.001)

        self.D = nn.Parameter(torch.ones(d_mamba))
        self.out_proj = nn.Linear(d_mamba, d_mamba, bias=False)

    def forward(self, x_norm: torch.Tensor) -> torch.Tensor:
        x_ssm = self.x_proj(x_norm)
        # Clamp BEFORE softplus to prevent fp16 overflow in backward
        dt_raw = self.dt_proj(x_norm).clamp(min=-10.0, max=5.0)
        dt = F.softplus(dt_raw).clamp(min=1e-4, max=0.1)  # tighter max for loop
        B = self.B_proj(x_norm)
        C = self.C_proj(x_norm)
        B = B / (B.norm(dim=-1, keepdim=True).clamp(min=1.0))
        C = C / (C.norm(dim=-1, keepdim=True).clamp(min=1.0))

        y = _scan_dispatch(
            x_ssm, dt, self.A_log.flatten(), B, C, self.D, self.n_heads
        )
        return self.out_proj(y)


# ---------------------------------------------------------------------------
# Core Hybrid Block (Conv || Loop-Stable Mamba-3 SISO at d=512)
# ---------------------------------------------------------------------------

class CoreHybridBlock(nn.Module):
    """Parallel Conv + Mamba-3 SISO for the core loop.

    Same mixer design as AMADEUS's ParallelHybridBlock but at d=512 with
    momentum residual to match the JORMUNGANDR-HALO loop interface.
    Uses LoopStableMamba3SISO with tighter dt clamping for iteration safety.

    Architecture per layer:
        RMSNorm → (GatedConv(d_conv) || Mamba3SISO(d_mamba)) → concat → proj
        → momentum residual → RMSNorm → SwiGLU → residual

    d_conv + d_mamba = d_model (512). Default split: 256/256 (Tensile-friendly).
    """

    def __init__(
        self,
        d_model: int,
        d_conv: int,
        d_mamba: int,
        ffn_inner: int,
        dstate: int = 64,
        n_ssm_heads: int = 4,
        conv_kernel: int = 3,
        momentum_beta_init: float = 0.5,
    ):
        super().__init__()
        assert d_conv + d_mamba == d_model, (
            f"d_conv({d_conv}) + d_mamba({d_mamba}) must equal d_model({d_model})"
        )
        assert n_ssm_heads * dstate == d_mamba, (
            f"n_ssm_heads({n_ssm_heads}) * dstate({dstate}) must equal d_mamba({d_mamba})"
        )

        self.pre_norm = RMSNorm(d_model)
        self.conv = GatedConv(d_model, d_conv, conv_kernel)
        self.ssm = LoopStableMamba3SISO(d_model, d_mamba, dstate, n_ssm_heads)
        self.out_proj = nn.Linear(d_model, d_model, bias=False)

        # Momentum residual (same interface as ShortConvBlock)
        self.log_beta = nn.Parameter(
            torch.tensor(math.log(momentum_beta_init / (1 - momentum_beta_init)))
        )

        self.ffn_norm = RMSNorm(d_model)
        self.ffn = SwiGLU(d_model, ffn_inner)

    def forward(self, x: torch.Tensor, velocity: torch.Tensor):
        x_norm = self.pre_norm(x)

        # Parallel mixer: local (conv) + global (Mamba SISO)
        conv_out = self.conv(x_norm)        # (B, T, d_conv)
        ssm_out = self.ssm(x_norm)          # (B, T, d_mamba)
        mixer_out = self.out_proj(torch.cat([conv_out, ssm_out], dim=-1))

        # Momentum + residual
        beta = torch.sigmoid(self.log_beta)
        velocity = beta * velocity + mixer_out
        x = x + velocity

        # Inlined RMSNorm + SwiGLU (fuses well with torch.compile)
        rms = torch.rsqrt(x.float().pow(2).mean(-1, keepdim=True) + 1e-6)
        normed = (x.float() * rms).to(x.dtype) * self.ffn_norm.weight
        x = x + self.ffn(normed)

        return x, velocity


# ---------------------------------------------------------------------------
# HALO-PRIME Model
# ---------------------------------------------------------------------------

class HaloPrimeBase(nn.Module):
    """HALO-PRIME: JORMUNGANDR-HALO with Mamba-3 SISO core loop.

    Prelude (d=768) → Parcae injection + proj_down → Core hybrid loop (d=512)
    → Depth MC → proj_up → Coda (d=768, FiLM+XSA+TTT+VE) → output.

    Core loop: 3 CoreHybridBlock (Conv+Mamba) × T Poisson-sampled iterations.
    Each iteration has full-sequence receptive field via Mamba selective scan.
    """

    def __init__(
        self,
        vocab_size: int = 50257,
        d_prelude: int = 768,
        d_core: int = 512,
        d_conv_prelude: int = 512,
        d_conv_core: int = 256,
        d_mamba_core: int = 256,
        dstate: int = 64,
        n_ssm_heads: int = 4,
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
        n_film_targets: int = 4,
        use_ve: bool = True,
        d_ve: int = 64,
        ttt_mode: str = "single",
        ttt_chunk: int = 512,
        ttt_lr_init: float = 0.01,
        momentum_beta_init: float = 0.5,
        max_seq_len: int = 1024,
        randomize_positions: bool = False,
        use_xsa: bool = True,
        use_depth_cache: bool = True,
        d_gate: int = 64,
    ):
        super().__init__()
        assert d_conv_core + d_mamba_core == d_core, (
            f"d_conv_core({d_conv_core}) + d_mamba_core({d_mamba_core}) "
            f"must equal d_core({d_core})"
        )

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

        # === CORE LOOP (d=512, Conv+Mamba hybrid) ===
        self.core_layers = nn.ModuleList([
            CoreHybridBlock(
                d_core, d_conv_core, d_mamba_core,
                ffn_core, dstate, n_ssm_heads,
                conv_kernel, momentum_beta_init,
            )
            for _ in range(n_core_layers)
        ])
        # Per-iteration norm: prevents state growth across loop iterations.
        # Mamba's global context adds more to residual than ShortConv; without
        # this, state norm grows ~60% per iteration → NaN by iteration 4.
        self.core_iter_norm = RMSNorm(d_core)

        # === EXIT (512 → 768) ===
        self.proj_up = nn.Linear(d_core, d_prelude, bias=False)

        # === VALUE EMBEDDING (last Coda GQA only) ===
        kv_dim = n_kv_heads * head_dim
        ve = ValueEmbedding(vocab_size, d_ve, kv_dim) if use_ve else None

        # === CODA (d=768, 4 layers) ===
        self.coda_layers = nn.ModuleList([
            ShortConvBlock(
                d_prelude, d_conv_prelude, ffn_prelude,
                conv_kernel, momentum_beta_init,
            ),
            CodaGQABlock(
                d_prelude, ffn_prelude, n_heads, n_kv_heads,
                momentum_beta_init, ttt_mode="none",
                exclusive=use_xsa,
            ),
            ShortConvBlock(
                d_prelude, d_conv_prelude, ffn_prelude,
                conv_kernel, momentum_beta_init,
            ),
            CodaGQABlock(
                d_prelude, ffn_prelude, n_heads, n_kv_heads,
                momentum_beta_init, ttt_mode=ttt_mode,
                ttt_chunk=ttt_chunk, ttt_lr_init=ttt_lr_init,
                value_embedding=ve, exclusive=use_xsa,
            ),
        ])

        self.value_embedding = ve

        # === FILM (d=512 context → d=768 modulation) ===
        self.film = CrossDimFiLMConditioner(
            d_core, d_film, d_prelude, n_film_targets
        ) if use_film else None

        # === DEPTH MEMORY CACHE (GRM over loop iterations) ===
        self.depth_cache = DepthMemoryCache(d_core, d_gate) if use_depth_cache else None

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
        # Value embedding zero-init
        if self.value_embedding is not None:
            nn.init.zeros_(self.value_embedding.embed.weight)

    def compile_zones(self):
        """Compile each zone independently for per-layer fusion.

        The Python loop stays uncompiled; each layer becomes a fused kernel.
        Note: Mamba scan may create a graph break within CoreHybridBlock,
        but conv+ffn are still fused. mamba-ssm runs at native speed anyway.
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

        # === CORE LOOP (d=512, Conv+Mamba hybrid) ===
        velocity_512 = torch.zeros(B, T, self.d_core, device=h.device, dtype=h.dtype)

        if self.training:
            step = self.step_counter.item()
            n_detached, n_grad = self.sample_loop_depth(step)
            self.step_counter += 1
        else:
            n_detached = 0
            n_grad = self.mean_recurrence

        use_dc = self.depth_cache is not None
        cached_states = []

        # Detached iterations (no grad, no GradScaler overhead)
        for t in range(n_detached):
            with torch.no_grad():
                h_core = self.injection(h_core, input_embed_down)
                for layer in self.core_layers:
                    h_core, velocity_512 = layer(h_core, velocity_512)
                h_core = self.core_iter_norm(h_core)
                if use_dc or not self.training:
                    cached_states.append(h_core.detach())

        # Gradient-tracked iterations
        context = None
        for t in range(n_grad):
            h_core = self.injection(h_core, input_embed_down)
            for layer in self.core_layers:
                h_core, velocity_512 = layer(h_core, velocity_512)
            h_core = self.core_iter_norm(h_core)

            # FiLM: compute fingerprint on last gradient iteration
            if t == n_grad - 1 and self.film is not None:
                context = self.film.compute_context(h_core)

            # Cache state for depth memory cache
            if use_dc:
                cached_states.append(h_core)
            elif not self.training:
                cached_states.append(h_core.detach())

        # HSV monitoring (always detached)
        self._loop_states = [s.detach() for s in cached_states]

        # Depth Memory Cache: gated aggregation over iteration states
        if use_dc and len(cached_states) > 1:
            h_core = self.depth_cache(cached_states)

        # === EXIT (512 → 768) ===
        h = self.proj_up(h_core)

        # === CODA (d=768) ===
        velocity_768 = torch.zeros_like(h)

        for i, layer in enumerate(self.coda_layers):
            if isinstance(layer, CodaGQABlock):
                ttt_target = h if layer.ttt_mode != "none" else None
                h, velocity_768 = layer(
                    h, velocity_768, freqs_cis,
                    ttt_target=ttt_target, input_ids=input_ids,
                )
            else:
                h, velocity_768 = layer(h, velocity_768)

            # FiLM modulation on each Coda layer (targets 0..3)
            if context is not None:
                h = self.film.apply(h, context, i)

        return self.output(self.norm(h))


# ---------------------------------------------------------------------------
# Variant classes
# ---------------------------------------------------------------------------

class HaloPrime(HaloPrimeBase):
    """Default: 4 iterations, FiLM, VE, single-step TTT, XSA, Depth Cache."""
    def __init__(self, **kw):
        super().__init__(mean_recurrence=4, **kw)


class HaloPrimeDeep(HaloPrimeBase):
    """Deep: 8 iterations for maximum effective capacity."""
    def __init__(self, **kw):
        super().__init__(mean_recurrence=8, **kw)


class HaloPrimeXSADC(HaloPrimeBase):
    """Lean: XSA + Depth Cache only (no FiLM/VE/TTT)."""
    def __init__(self, **kw):
        super().__init__(
            mean_recurrence=4, ttt_mode="none",
            use_film=False, use_ve=False,
            use_xsa=True, use_depth_cache=True, **kw,
        )


class HaloPrimeBare(HaloPrimeBase):
    """Bare: no enhancements — isolate core mixer quality."""
    def __init__(self, **kw):
        super().__init__(
            mean_recurrence=4, ttt_mode="none",
            use_film=False, use_ve=False,
            use_xsa=False, use_depth_cache=False, **kw,
        )


class HaloPrimeMini(HaloPrimeBase):
    """Tiny config for smoke testing (~4M params)."""
    def __init__(self):
        super().__init__(
            vocab_size=1000,
            d_prelude=256, d_core=128,
            d_conv_prelude=256,
            d_conv_core=64, d_mamba_core=64,
            dstate=32, n_ssm_heads=2,
            ffn_prelude=512, ffn_core=384,
            n_heads=4, n_kv_heads=2,
            n_core_layers=2,
            mean_recurrence=3, backprop_depth=2,
            curriculum_steps=100,
            use_film=False, use_ve=False,
            ttt_mode="none",
            use_xsa=False, use_depth_cache=False,
            momentum_beta_init=0.5,
            max_seq_len=1024,
        )

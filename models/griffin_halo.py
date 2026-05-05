"""
GRIFFIN-HALO: Research-oriented looped architecture for Pareto frontier mapping.

Supports 5 design axes via constructor flags:
  Axis 1 — Core mixer:      "griffin" | "gqa" | "conv"
  Axis 2 — Depth aggregation: "last" | "dmc" | "attnres" | "dmc+attnres"
  Axis 3 — Coda residuals:  standard | AttnRes cross-stage
  Axis 4 — Adaptive depth:  fixed | MoE router (Phase 1 all + Phase 2 hard)
  Axis 5 — Dimensions:      d=768 uniform | d=512 heterogeneous

10 variant classes for the factor sweep + Mini for smoke testing.

Spec: docs/superpowers/specs/2026-04-17-griffin-halo-research-plan.md

Usage:
    python -m halo_training --model models/griffin_halo.py --class-name GriffinHaloMini --smoke
    python -m halo_training --model models/griffin_halo.py --class-name GriffinHaloR1 \\
        --compile --optimize-kernels --muon --dataset babylm
"""

import math
from typing import List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

# Reuse all existing components
from models.amadeus import RMSNorm, SwiGLU, GatedConv
from models.tempest import GriffinRecurrence
from models.argus import TTTSwiGLU, precompute_freqs_cis, apply_rotary_emb
from models.components import Attention, ShortConvBlock, GQABlock
from models.components import CodaAttention, DepthMemoryCache
from models.jormungandr_halo import (
    ParcaeInjection, ValueEmbedding, CodaGQABlock,
)


# ---------------------------------------------------------------------------
# SimpleParcaeInjection (d=768 uniform, no proj_down)
# ---------------------------------------------------------------------------

from models.components.injection import SimpleParcaeInjection


# ---------------------------------------------------------------------------
# Axis 1: Core Loop Mixers
# ---------------------------------------------------------------------------

class GriffinConvBlock(nn.Module):
    """Parallel GatedConv + GriffinRecurrence for the core loop.

    Architecture: RMSNorm → (Conv(d_conv) ∥ Griffin(d_griffin)) → concat → proj
                  → momentum → RMSNorm → SwiGLU → residual
    """

    def __init__(self, d_model: int, d_conv: int, d_griffin: int,
                 ffn_inner: int, conv_kernel: int = 3,
                 momentum_beta_init: float = 0.5):
        super().__init__()
        assert d_conv + d_griffin == d_model

        self.pre_norm = RMSNorm(d_model)
        self.conv = GatedConv(d_model, d_conv, conv_kernel)
        self.griffin = GriffinRecurrence(d_model, d_griffin)
        self.out_proj = nn.Linear(d_model, d_model, bias=False)

        self.log_beta = nn.Parameter(
            torch.tensor(math.log(momentum_beta_init / (1 - momentum_beta_init)))
        )
        self.ffn_norm = RMSNorm(d_model)
        self.ffn = SwiGLU(d_model, ffn_inner)

    def forward(self, x: torch.Tensor, velocity: torch.Tensor):
        x_norm = self.pre_norm(x)
        conv_out = self.conv(x_norm)
        griffin_out = self.griffin(x_norm)
        mixer_out = self.out_proj(torch.cat([conv_out, griffin_out], dim=-1))

        beta = torch.sigmoid(self.log_beta)
        velocity = beta * velocity + mixer_out
        x = x + velocity

        rms = torch.rsqrt(x.float().pow(2).mean(-1, keepdim=True) + 1e-6)
        normed = (x.float() * rms).to(x.dtype) * self.ffn_norm.weight
        x = x + self.ffn(normed)
        return x, velocity


class GQAConvBlock(nn.Module):
    """GatedConv + lightweight GQA for LFM2-inspired loop variant.

    Architecture: RMSNorm → Conv → proj → residual
                  → RMSNorm → GQA(n_kv=2) → momentum → residual
                  → RMSNorm → SwiGLU → residual
    """

    def __init__(self, d_model: int, d_conv: int, n_heads: int = 12,
                 n_kv_heads: int = 2, ffn_inner: int = 2048,
                 conv_kernel: int = 3, momentum_beta_init: float = 0.5):
        super().__init__()
        self.conv_norm = RMSNorm(d_model)
        self.conv = GatedConv(d_model, d_conv, conv_kernel)
        self.conv_proj = nn.Linear(d_conv, d_model, bias=False)

        self.attn_norm = RMSNorm(d_model)
        self.attn = Attention(d_model, n_heads, n_kv_heads, qk_norm=True)

        self.log_beta = nn.Parameter(
            torch.tensor(math.log(momentum_beta_init / (1 - momentum_beta_init)))
        )
        self.ffn_norm = RMSNorm(d_model)
        self.ffn = SwiGLU(d_model, ffn_inner)

    def forward(self, x: torch.Tensor, velocity: torch.Tensor,
                freqs_cis: torch.Tensor):
        # Conv residual
        x = x + self.conv_proj(self.conv(self.conv_norm(x)))

        # GQA + momentum residual
        attn_out = self.attn(self.attn_norm(x), freqs_cis)
        beta = torch.sigmoid(self.log_beta)
        velocity = beta * velocity + attn_out
        x = x + velocity

        # SwiGLU residual
        rms = torch.rsqrt(x.float().pow(2).mean(-1, keepdim=True) + 1e-6)
        normed = (x.float() * rms).to(x.dtype) * self.ffn_norm.weight
        x = x + self.ffn(normed)
        return x, velocity


# ---------------------------------------------------------------------------
# Axis 2: Depth Aggregation — IterationAttnRes
# ---------------------------------------------------------------------------

class IterationAttnRes(nn.Module):
    """Softmax attention over iteration outputs with learned per-consumer queries.

    Each consuming layer gets a learned pseudo-query (zero-init → starts as
    uniform average, learns to specialize). Based on Attention Residuals (Kimi).
    """

    def __init__(self, d_model: int, n_consumers: int):
        super().__init__()
        self.queries = nn.ParameterList([
            nn.Parameter(torch.zeros(d_model)) for _ in range(n_consumers)
        ])
        self.norm = RMSNorm(d_model)

    def forward(self, iteration_states: List[torch.Tensor],
                consumer_idx: int) -> torch.Tensor:
        V = torch.stack(iteration_states)                     # (N, B, T, d)
        K = self.norm(V)
        q = self.queries[consumer_idx]
        logits = torch.einsum('d, n b t d -> n b t', q, K)
        weights = logits.softmax(dim=0)                       # (N, B, T)
        return torch.einsum('n b t, n b t d -> b t d', weights, V)


# ---------------------------------------------------------------------------
# Axis 3: Coda AttnRes — Cross-Stage Residuals
# ---------------------------------------------------------------------------

class CodaAttnRes(nn.Module):
    """Block AttnRes for Coda layers. Sources: [b_0, b_prelude, b_loop, partial].

    Each Coda layer attends to these sources with its own learned query,
    selecting the most useful source for its specific function.
    """

    def __init__(self, d_model: int, n_layers: int):
        super().__init__()
        self.queries = nn.ParameterList([
            nn.Parameter(torch.zeros(d_model)) for _ in range(n_layers)
        ])
        self.norm = RMSNorm(d_model)

    def forward(self, sources: List[torch.Tensor], partial: torch.Tensor,
                layer_idx: int) -> torch.Tensor:
        all_sources = sources + [partial]
        V = torch.stack(all_sources)                          # (N_src+1, B, T, d)
        K = self.norm(V)
        q = self.queries[layer_idx]
        logits = torch.einsum('d, n b t d -> n b t', q, K)
        weights = logits.softmax(dim=0)
        return torch.einsum('n b t, n b t d -> b t d', weights, V)


# ---------------------------------------------------------------------------
# Axis 4: Adaptive Depth — MoE Router + Phase 2 Block
# ---------------------------------------------------------------------------

class DepthRouter(nn.Module):
    """MoE-style top-k token router for adaptive loop depth."""

    def __init__(self, d_model: int, capacity_factor: float = 0.5):
        super().__init__()
        self.gate = nn.Linear(d_model, 1, bias=True)
        self.capacity_factor = capacity_factor
        nn.init.zeros_(self.gate.weight)
        nn.init.zeros_(self.gate.bias)

    def forward(self, h: torch.Tensor):
        scores = self.gate(h).squeeze(-1)                     # (B, T)
        k = max(1, int(h.shape[1] * self.capacity_factor))
        top_vals, top_idx = scores.topk(k, dim=1)            # (B, k)
        return top_idx, torch.sigmoid(top_vals), scores

    def aux_loss(self, scores: torch.Tensor) -> torch.Tensor:
        probs = scores.softmax(dim=1)
        return self.capacity_factor * (probs.float() * scores.float()).sum(dim=1).mean()


class Phase2ConvBlock(nn.Module):
    """Conv-only + optional TTT for Phase 2 hard token refinement.

    No recurrence — global context established in Phase 1.
    """

    def __init__(self, d_model: int, d_conv: int = 512, ffn_inner: int = 2048,
                 conv_kernel: int = 3, momentum_beta_init: float = 0.5,
                 ttt_mode: str = "none", ttt_chunk: int = 512,
                 ttt_lr_init: float = 0.01):
        super().__init__()
        self.pre_norm = RMSNorm(d_model)
        self.conv = GatedConv(d_model, d_conv, conv_kernel)
        self.conv_proj = nn.Linear(d_conv, d_model, bias=False)

        self.log_beta = nn.Parameter(
            torch.tensor(math.log(momentum_beta_init / (1 - momentum_beta_init)))
        )
        self.ffn_norm = RMSNorm(d_model)

        if ttt_mode == "single":
            self.ffn = TTTSwiGLU(d_model, ffn_inner, ttt_chunk, ttt_lr_init)
        else:
            self.ffn = SwiGLU(d_model, ffn_inner)
        self.ttt_mode = ttt_mode

    def forward(self, x: torch.Tensor, velocity: torch.Tensor,
                ttt_target: Optional[torch.Tensor] = None):
        conv_out = self.conv_proj(self.conv(self.pre_norm(x)))

        beta = torch.sigmoid(self.log_beta)
        velocity = beta * velocity + conv_out
        x = x + velocity

        rms = torch.rsqrt(x.float().pow(2).mean(-1, keepdim=True) + 1e-6)
        normed = (x.float() * rms).to(x.dtype) * self.ffn_norm.weight

        if self.ttt_mode != "none" and ttt_target is not None:
            x = x + self.ffn(normed, ttt_target=ttt_target)
        else:
            x = x + self.ffn(normed)
        return x, velocity


# ---------------------------------------------------------------------------
# GriffinHaloBase — Main Model
# ---------------------------------------------------------------------------

class GriffinHaloBase(nn.Module):
    """GRIFFIN-HALO: configurable looped architecture for Pareto frontier research.

    All 5 design axes controlled via constructor parameters.
    """

    def __init__(
        self,
        vocab_size: int = 50257,
        d_model: int = 768,
        d_core: int = 768,
        # Axis 1: Core mixer
        core_mixer: str = "griffin",
        d_conv_core: int = 384,
        d_griffin: int = 384,
        n_core_kv_heads: int = 2,
        ffn_core: int = 2048,
        conv_kernel: int = 3,
        # Loop config
        n_iters: int = 4,
        backprop_depth: int = 3,
        curriculum_steps: int = 5000,
        # Axis 2: Depth aggregation
        depth_agg: str = "last",
        d_gate: int = 64,
        # Axis 3: Coda residuals
        coda_attnres: bool = False,
        # Axis 4: Adaptive depth
        adaptive_depth: bool = False,
        capacity_factor: float = 0.5,
        n_phase1: int = 2,
        n_phase2: int = 2,
        phase2_ttt: str = "none",
        # Coda config
        d_conv_prelude: int = 512,
        ffn_prelude: int = 2816,
        n_heads: int = 12,
        n_kv_heads: int = 4,
        # Fixed components
        use_xsa: bool = True,
        ttt_mode: str = "single",
        ttt_coda_l2: bool = False,
        momentum_beta_init: float = 0.5,
        max_seq_len: int = 1024,
    ):
        super().__init__()
        self.d_model = d_model
        self.d_core = d_core
        self.core_mixer = core_mixer
        self.depth_agg = depth_agg
        self.coda_attnres = coda_attnres
        self.adaptive_depth = adaptive_depth
        self.n_iters = n_iters
        self.n_phase1 = n_phase1 if adaptive_depth else n_iters
        self.n_phase2 = n_phase2 if adaptive_depth else 0
        self.backprop_depth = backprop_depth
        self.curriculum_steps = curriculum_steps
        self.max_seq_len = max_seq_len

        # Embedding + tied output
        self.tok_embeddings = nn.Embedding(vocab_size, d_model)
        self.norm = RMSNorm(d_model)
        self.output = nn.Linear(d_model, vocab_size, bias=False)
        self.output.weight = self.tok_embeddings.weight

        # RoPE
        head_dim = d_model // n_heads
        self.register_buffer(
            "freqs_cis",
            precompute_freqs_cis(head_dim, max_seq_len * 2),
            persistent=False,
        )

        # Step counter
        self.register_buffer("step_counter", torch.tensor(0, dtype=torch.long))

        # === PRELUDE (d=d_model) ===
        self.prelude_conv = ShortConvBlock(
            d_model, d_conv_prelude, ffn_prelude, conv_kernel, momentum_beta_init
        )
        self.prelude_gqa = GQABlock(
            d_model, ffn_prelude, n_heads, n_kv_heads,
            momentum_beta_init, ttt_mode="none",
        )

        # === PARCAE INJECTION ===
        if d_core == d_model:
            self.injection = SimpleParcaeInjection(d_core)
            self.proj_up = None
        else:
            self.injection = ParcaeInjection(d_model, d_core)
            self.proj_up = nn.Linear(d_core, d_model, bias=False)

        # === CORE LOOP BLOCK (Axis 1) ===
        if core_mixer == "griffin":
            self.core_block = GriffinConvBlock(
                d_core, d_conv_core, d_griffin, ffn_core,
                conv_kernel, momentum_beta_init,
            )
        elif core_mixer == "griffin+conv":
            # Fix B: 1 Griffin (global) + 2 ShortConv (local, compile-fusable)
            self.core_block = None
            self.core_layers = nn.ModuleList([
                GriffinConvBlock(d_core, d_conv_core, d_griffin, ffn_core,
                                 conv_kernel, momentum_beta_init),
                ShortConvBlock(d_core, d_conv_core, ffn_core, conv_kernel,
                               momentum_beta_init),
                ShortConvBlock(d_core, d_conv_core, ffn_core, conv_kernel,
                               momentum_beta_init),
            ])
        elif core_mixer == "seq_griffin":
            # Fix C: Sequential Conv → Griffin (single chain, better fusion)
            self.core_block = SeqGriffinConvBlock(
                d_core, d_conv_core, d_griffin, ffn_core,
                conv_kernel, momentum_beta_init,
            )
        elif core_mixer == "gqa":
            self.core_block = GQAConvBlock(
                d_core, d_conv_core, n_heads, n_core_kv_heads,
                ffn_core, conv_kernel, momentum_beta_init,
            )
        elif core_mixer == "conv":
            # 3 ShortConvBlocks like JORMUNGANDR-HALO
            self.core_block = None
            self.core_layers = nn.ModuleList([
                ShortConvBlock(d_core, d_conv_core, ffn_core, conv_kernel, momentum_beta_init)
                for _ in range(3)
            ])

        # Per-iteration norm
        self.iter_norm = RMSNorm(d_core)

        # === PHASE 2 (Axis 4) ===
        if adaptive_depth:
            self.router = DepthRouter(d_core, capacity_factor)
            self.phase2_block = Phase2ConvBlock(
                d_core, d_conv_prelude, ffn_core, conv_kernel,
                momentum_beta_init, phase2_ttt,
            )
            self.iter_norm_p2 = RMSNorm(d_core)
        else:
            self.router = None
            self.phase2_block = None

        # === DEPTH AGGREGATION (Axis 2) ===
        use_dmc = depth_agg in ("dmc", "dmc+attnres")
        self.dmc = DepthMemoryCache(d_core, d_gate) if use_dmc else None

        use_iter_attnres = depth_agg in ("attnres", "dmc+attnres")
        # n_consumers = 4 Coda layers (or 1 if not per-coda)
        self.iter_attnres = IterationAttnRes(d_core, 4) if use_iter_attnres else None

        # === CODA ATTNRES (Axis 3) ===
        self.coda_attn_res = CodaAttnRes(d_model, 4) if coda_attnres else None

        # === CODA (d=d_model) ===
        kv_dim = n_kv_heads * head_dim
        ve = ValueEmbedding(vocab_size, 64, kv_dim)

        ttt_l2 = "single" if ttt_coda_l2 else "none"

        self.coda_layers = nn.ModuleList([
            ShortConvBlock(d_model, d_conv_prelude, ffn_prelude, conv_kernel, momentum_beta_init),
            CodaGQABlock(d_model, ffn_prelude, n_heads, n_kv_heads,
                         momentum_beta_init, ttt_mode=ttt_l2, exclusive=use_xsa),
            ShortConvBlock(d_model, d_conv_prelude, ffn_prelude, conv_kernel, momentum_beta_init),
            CodaGQABlock(d_model, ffn_prelude, n_heads, n_kv_heads,
                         momentum_beta_init, ttt_mode=ttt_mode,
                         value_embedding=ve, exclusive=use_xsa),
        ])

        self.value_embedding = ve

        self._init_weights()
        n_params = sum(p.numel() for p in self.parameters())
        print(f"{self.__class__.__name__}: {n_params / 1e6:.1f}M parameters")

    def _init_weights(self):
        for name, p in self.named_parameters():
            if p.dim() >= 2:
                nn.init.xavier_uniform_(p)
        for layer in self.coda_layers:
            if isinstance(layer, CodaGQABlock) and hasattr(layer.ffn, '_init_ttt_weights'):
                layer.ffn._init_ttt_weights()
        if self.value_embedding is not None:
            nn.init.zeros_(self.value_embedding.embed.weight)

    def compile_zones(self):
        """Compile each zone independently. Griffin and GQA-in-loop both compile."""
        self.prelude_conv = torch.compile(self.prelude_conv, mode="default")
        if self.core_block is not None:
            self.core_block = torch.compile(self.core_block, mode="default")
        else:
            for i in range(len(self.core_layers)):
                self.core_layers[i] = torch.compile(self.core_layers[i], mode="default")
        if self.phase2_block is not None:
            self.phase2_block = torch.compile(self.phase2_block, mode="default")
        for i in range(len(self.coda_layers)):
            if not isinstance(self.coda_layers[i], CodaGQABlock):
                self.coda_layers[i] = torch.compile(self.coda_layers[i], mode="default")
        return self

    def sample_loop_depth(self, step: int) -> Tuple[int, int]:
        """Poisson depth sampling with 1-sqrt curriculum."""
        progress = min(step / max(self.curriculum_steps, 1), 1.0)
        effective_progress = 1 - math.sqrt(1 - progress)
        t_full = max(self.n_phase1 - self.backprop_depth, 0)
        t = max(math.ceil(effective_progress * t_full), 0)
        n_detached = torch.poisson(torch.tensor([float(t)])).long().item()
        n_detached = min(n_detached, 2 * max(t, 1))
        return n_detached, self.backprop_depth

    def forward(self, input_ids: torch.Tensor, targets=None) -> torch.Tensor:
        B, T = input_ids.shape

        if torch.is_autocast_enabled() and torch.get_autocast_gpu_dtype() == torch.bfloat16:
            raise RuntimeError("bf16 is 24% slower on gfx1151. Use fp16.")

        h = self.tok_embeddings(input_ids)
        freqs_cis = self.freqs_cis[:T]

        # Save b_0 for AttnRes
        b_0 = h.detach() if self.coda_attnres else None

        # === PRELUDE ===
        velocity = torch.zeros_like(h)
        h, velocity = self.prelude_conv(h, velocity)
        h, velocity = self.prelude_gqa(h, velocity, freqs_cis, ttt_target=None)
        input_embed = h

        b_prelude = h.detach() if self.coda_attnres else None

        # === PARCAE ENTRY (skip injection — h == input_embed here) ===
        if self.d_core != self.d_model:
            input_embed_core = self.injection.proj_down(input_embed)
            h_core = self.injection.proj_down(h)
        else:
            input_embed_core = input_embed
            h_core = h

        # === CORE LOOP PHASE 1 ===
        vel_core = torch.zeros(B, T, self.d_core, device=h.device, dtype=h.dtype)
        cached_states = []

        if self.training:
            step = self.step_counter.item()
            n_detached, n_grad = self.sample_loop_depth(step)
            self.step_counter += 1
        else:
            n_detached = 0
            n_grad = self.n_phase1

        # First iteration: no re-injection needed
        h_core, vel_core = self._core_forward(h_core, vel_core, freqs_cis)
        h_core = self.iter_norm(h_core)
        if n_detached + n_grad > 1:
            if n_detached > 0:
                cached_states.append(h_core.detach())
                n_detached -= 1
            else:
                cached_states.append(h_core)
                n_grad -= 1

        # Detached re-entry iterations
        for t in range(n_detached):
            with torch.no_grad():
                h_core = self.injection(h_core, input_embed_core)
                h_core, vel_core = self._core_forward(h_core, vel_core, freqs_cis)
                h_core = self.iter_norm(h_core)
                cached_states.append(h_core.detach())

        # Gradient re-entry iterations
        for t in range(n_grad):
            h_core = self.injection(h_core, input_embed_core)
            h_core, vel_core = self._core_forward(h_core, vel_core, freqs_cis)
            h_core = self.iter_norm(h_core)
            cached_states.append(h_core)

        # === ADAPTIVE DEPTH: PHASE 2 (Axis 4) ===
        router_aux_loss = None
        if self.adaptive_depth and self.router is not None:
            top_idx, gate_weights, scores = self.router(h_core)
            router_aux_loss = self.router.aux_loss(scores)

            k = top_idx.shape[1]
            # Gather hard positions
            idx_exp = top_idx.unsqueeze(-1).expand(-1, -1, self.d_core)
            h_hard = h_core.gather(1, idx_exp)
            vel_hard = vel_core.gather(1, idx_exp)

            # Phase 2 iterations on hard positions
            for t in range(self.n_phase2):
                h_hard, vel_hard = self.phase2_block(h_hard, vel_hard)
                h_hard = self.iter_norm_p2(h_hard)
                cached_states.append(
                    # Expand hard states back to full size for caching
                    h_core.scatter(1, idx_exp, h_hard)
                )

            # Scatter hard results back
            h_core = h_core.scatter(1, idx_exp, h_hard)

        # === DEPTH AGGREGATION (Axis 2) ===
        if self.depth_agg == "last":
            b_loop = h_core
        elif self.depth_agg == "dmc":
            b_loop = self.dmc(cached_states) if len(cached_states) > 1 else h_core
        elif self.depth_agg == "attnres":
            # AttnRes deferred to per-coda-layer (consumer_idx)
            b_loop = h_core  # fallback for proj_up
        elif self.depth_agg == "dmc+attnres":
            b_loop = self.dmc(cached_states) if len(cached_states) > 1 else h_core

        # === EXIT ===
        if self.proj_up is not None:
            h = self.proj_up(b_loop)
        else:
            h = b_loop

        # === CODA ===
        velocity = torch.zeros_like(h)
        partial = h.clone()

        # Prepare AttnRes sources
        attnres_sources = None
        if self.coda_attnres and b_0 is not None:
            attnres_sources = [b_0, b_prelude, b_loop if self.depth_agg != "attnres" else h_core]

        for i, layer in enumerate(self.coda_layers):
            # Coda AttnRes (Axis 3)
            if self.coda_attn_res is not None and attnres_sources is not None:
                h = self.coda_attn_res(attnres_sources, partial, i)

            # Iteration AttnRes as depth aggregation (Axis 2, attnres mode)
            if self.iter_attnres is not None and self.depth_agg in ("attnres", "dmc+attnres"):
                iter_contribution = self.iter_attnres(cached_states, i)
                if self.proj_up is not None:
                    iter_contribution = self.proj_up(iter_contribution)
                h = h + iter_contribution

            if isinstance(layer, CodaGQABlock):
                ttt_target = h if layer.ttt_mode != "none" else None
                h, velocity = layer(
                    h, velocity, freqs_cis,
                    ttt_target=ttt_target, input_ids=input_ids,
                )
            else:
                h, velocity = layer(h, velocity)

            partial = h

        logits = self.output(self.norm(h))

        # Add router auxiliary loss if present
        if router_aux_loss is not None and self.training:
            # Store for external access by trainer
            self._router_aux_loss = router_aux_loss

        return logits

    def _core_forward(self, h_core, vel_core, freqs_cis):
        """Single core iteration forward (dispatches by mixer type)."""
        if self.core_mixer in ("conv", "griffin+conv"):
            for layer in self.core_layers:
                if isinstance(layer, GQAConvBlock):
                    h_core, vel_core = layer(h_core, vel_core, freqs_cis)
                else:
                    h_core, vel_core = layer(h_core, vel_core)
        elif self.core_mixer == "gqa":
            h_core, vel_core = self.core_block(h_core, vel_core, freqs_cis)
        else:
            # Griffin or SeqGriffin (single block)
            h_core, vel_core = self.core_block(h_core, vel_core)
        return h_core, vel_core


# ---------------------------------------------------------------------------
# Variant classes for the 10-run factor sweep
# ---------------------------------------------------------------------------

class GriffinHaloR1(GriffinHaloBase):
    """Run 1: Griffin base, last-only, standard, fixed, d=768"""
    def __init__(self, **kw):
        super().__init__(core_mixer="griffin", depth_agg="last",
                         coda_attnres=False, adaptive_depth=False,
                         d_core=768, **kw)


class GriffinHaloR2(GriffinHaloBase):
    """Run 2: GQA-in-loop, last-only, standard, fixed, d=768"""
    def __init__(self, **kw):
        super().__init__(core_mixer="gqa", depth_agg="last",
                         coda_attnres=False, adaptive_depth=False,
                         d_core=768, d_conv_core=512, **kw)


class GriffinHaloR3_Griffin_DMC(GriffinHaloBase):
    """Run 3 (if Griffin wins): Griffin + DMC"""
    def __init__(self, **kw):
        super().__init__(core_mixer="griffin", depth_agg="dmc",
                         coda_attnres=False, adaptive_depth=False,
                         d_core=768, **kw)


class GriffinHaloR3_GQA_DMC(GriffinHaloBase):
    """Run 3 (if GQA wins): GQA + DMC"""
    def __init__(self, **kw):
        super().__init__(core_mixer="gqa", depth_agg="dmc",
                         coda_attnres=False, adaptive_depth=False,
                         d_core=768, d_conv_core=512, **kw)


class GriffinHaloR4_AttnRes(GriffinHaloBase):
    """Run 4: Winner + AttnRes-iter (Griffin default)"""
    def __init__(self, **kw):
        super().__init__(core_mixer="griffin", depth_agg="attnres",
                         coda_attnres=False, adaptive_depth=False,
                         d_core=768, **kw)


class GriffinHaloR5_DMC_AttnRes(GriffinHaloBase):
    """Run 5: Winner + DMC+AttnRes synergy (Griffin default)"""
    def __init__(self, **kw):
        super().__init__(core_mixer="griffin", depth_agg="dmc+attnres",
                         coda_attnres=False, adaptive_depth=False,
                         d_core=768, **kw)


class GriffinHaloR6_CodaAttnRes(GriffinHaloBase):
    """Run 6: Best agg + Coda AttnRes (Griffin+DMC default)"""
    def __init__(self, **kw):
        super().__init__(core_mixer="griffin", depth_agg="dmc",
                         coda_attnres=True, adaptive_depth=False,
                         d_core=768, **kw)


class GriffinHaloR7_Adaptive(GriffinHaloBase):
    """Run 7: Best + Adaptive depth (Griffin+DMC default)"""
    def __init__(self, **kw):
        super().__init__(core_mixer="griffin", depth_agg="dmc",
                         coda_attnres=False, adaptive_depth=True,
                         capacity_factor=0.5, d_core=768, **kw)


class GriffinHaloR8_d512(GriffinHaloBase):
    """Run 8: Best config at d=512 heterogeneous"""
    def __init__(self, **kw):
        super().__init__(core_mixer="griffin", depth_agg="dmc",
                         coda_attnres=False, adaptive_depth=False,
                         d_core=512, d_conv_core=256, d_griffin=256,
                         ffn_core=1792, **kw)


class GriffinHaloR9(GriffinHaloBase):
    """Run 9: Confirmation of best full config (update after sweep)"""
    def __init__(self, **kw):
        super().__init__(core_mixer="griffin", depth_agg="dmc",
                         coda_attnres=False, adaptive_depth=False,
                         d_core=768, **kw)


class SeqGriffinConvBlock(nn.Module):
    """Fix C: Sequential Conv → Griffin (single chain for better compile fusion).

    Instead of parallel (Conv ∥ Griffin) → concat → proj, runs Conv first
    then Griffin on the conv output. Creates one long sequential chain that
    Inductor can trace and fuse more aggressively.
    """

    def __init__(self, d_model: int, d_conv: int, d_griffin: int,
                 ffn_inner: int, conv_kernel: int = 3,
                 momentum_beta_init: float = 0.5):
        super().__init__()
        self.pre_norm = RMSNorm(d_model)
        self.conv = GatedConv(d_model, d_conv, conv_kernel)
        self.conv_proj = nn.Linear(d_conv, d_model, bias=False)
        self.griffin = GriffinRecurrence(d_model, d_griffin)
        self.griffin_proj = nn.Linear(d_griffin, d_model, bias=False)

        self.log_beta = nn.Parameter(
            torch.tensor(math.log(momentum_beta_init / (1 - momentum_beta_init)))
        )
        self.ffn_norm = RMSNorm(d_model)
        self.ffn = SwiGLU(d_model, ffn_inner)

    def forward(self, x: torch.Tensor, velocity: torch.Tensor):
        normed = self.pre_norm(x)

        # Sequential: Conv (local) → residual → Griffin (global) → residual
        conv_out = self.conv_proj(self.conv(normed))
        x = x + conv_out

        griffin_out = self.griffin_proj(self.griffin(x))
        beta = torch.sigmoid(self.log_beta)
        velocity = beta * velocity + griffin_out
        x = x + velocity

        # SwiGLU
        rms = torch.rsqrt(x.float().pow(2).mean(-1, keepdim=True) + 1e-6)
        normed = (x.float() * rms).to(x.dtype) * self.ffn_norm.weight
        x = x + self.ffn(normed)
        return x, velocity


class GriffinHaloOpt_fixA(GriffinHaloBase):
    """Fix A test: Same as R9 but causal_conv1d bypassed under compile (in amadeus.py)."""
    def __init__(self, **kw):
        super().__init__(core_mixer="griffin", depth_agg="dmc",
                         coda_attnres=False, adaptive_depth=False,
                         d_core=768, **kw)


class GriffinHaloOpt_3iter(GriffinHaloBase):
    """Optimization test A: 3 iterations instead of 4 (all else = R9)"""
    def __init__(self, **kw):
        super().__init__(core_mixer="griffin", depth_agg="dmc",
                         coda_attnres=False, adaptive_depth=False,
                         d_core=768, n_iters=3, backprop_depth=2, **kw)


class GriffinHaloOpt_ffn1792(GriffinHaloBase):
    """Optimization test B: FFN 1792 instead of 2048 (all else = R9)"""
    def __init__(self, **kw):
        super().__init__(core_mixer="griffin", depth_agg="dmc",
                         coda_attnres=False, adaptive_depth=False,
                         d_core=768, ffn_core=1792, **kw)


class GriffinHaloOpt_fixB(GriffinHaloBase):
    """Fix B: Griffin + 2 ShortConv per iteration (more compile fusion surface)."""
    def __init__(self, **kw):
        super().__init__(core_mixer="griffin+conv", depth_agg="dmc",
                         coda_attnres=False, adaptive_depth=False,
                         d_core=768, **kw)


class GriffinHaloOpt_fixC(GriffinHaloBase):
    """Fix C: Sequential Conv→Griffin (single chain for better fusion)."""
    def __init__(self, **kw):
        super().__init__(core_mixer="seq_griffin", depth_agg="dmc",
                         coda_attnres=False, adaptive_depth=False,
                         d_core=768, **kw)


# ---------------------------------------------------------------------------
# Idea 1: Lean GRIFFIN-HALO — fewer unique layers, same core width
# ---------------------------------------------------------------------------

class GriffinHaloLean(nn.Module):
    """Lean: 1 Prelude GQA + 3 Griffin iters + 2 Coda layers.

    Cuts 4 unique layers (1 Prelude ShortConv + 2 Coda layers) to reduce
    fixed overhead from ~52% to ~35% of forward time. Targets ~33-37K tok/s
    at d=768 quality.
    """

    def __init__(self, vocab_size=50257, d_model=768, ffn_core=2048,
                 d_conv_core=384, d_griffin=384, ffn_prelude=2816,
                 d_conv_prelude=512, n_heads=12, n_kv_heads=4,
                 conv_kernel=3, n_iters=3, backprop_depth=2,
                 curriculum_steps=5000, momentum_beta_init=0.5,
                 max_seq_len=1024, d_gate=64, **kw):
        super().__init__()
        self.d_model = d_model
        self.d_core = d_model
        self.n_iters = n_iters
        self.backprop_depth = backprop_depth
        self.curriculum_steps = curriculum_steps
        self.max_seq_len = max_seq_len

        # Embedding + tied output
        self.tok_embeddings = nn.Embedding(vocab_size, d_model)
        self.norm = RMSNorm(d_model)
        self.output = nn.Linear(d_model, vocab_size, bias=False)
        self.output.weight = self.tok_embeddings.weight

        head_dim = d_model // n_heads
        self.register_buffer("freqs_cis",
            precompute_freqs_cis(head_dim, max_seq_len * 2), persistent=False)
        self.register_buffer("step_counter", torch.tensor(0, dtype=torch.long))

        # === LEAN PRELUDE: just 1 GQA layer (no ShortConv) ===
        self.prelude_gqa = GQABlock(
            d_model, ffn_prelude, n_heads, n_kv_heads,
            momentum_beta_init, ttt_mode="none",
        )

        # === PARCAE (uniform d, no proj) ===
        self.injection = SimpleParcaeInjection(d_model)

        # === CORE: 1 GriffinConvBlock × n_iters ===
        self.core_block = GriffinConvBlock(
            d_model, d_conv_core, d_griffin, ffn_core,
            conv_kernel, momentum_beta_init,
        )
        self.iter_norm = RMSNorm(d_model)

        # === DMC ===
        self.dmc = DepthMemoryCache(d_model, d_gate)

        # === LEAN CODA: just 2 layers (ShortConv + GQA with XSA+TTT) ===
        kv_dim = n_kv_heads * head_dim
        ve = ValueEmbedding(vocab_size, 64, kv_dim)

        self.coda_layers = nn.ModuleList([
            ShortConvBlock(d_model, d_conv_prelude, ffn_prelude,
                           conv_kernel, momentum_beta_init),
            CodaGQABlock(d_model, ffn_prelude, n_heads, n_kv_heads,
                         momentum_beta_init, ttt_mode="single",
                         value_embedding=ve, exclusive=True),
        ])
        self.value_embedding = ve

        self._init_weights()
        n_params = sum(p.numel() for p in self.parameters())
        print(f"{self.__class__.__name__}: {n_params / 1e6:.1f}M parameters")

    def _init_weights(self):
        for name, p in self.named_parameters():
            if p.dim() >= 2:
                nn.init.xavier_uniform_(p)
        for layer in self.coda_layers:
            if isinstance(layer, CodaGQABlock) and hasattr(layer.ffn, '_init_ttt_weights'):
                layer.ffn._init_ttt_weights()
        if self.value_embedding is not None:
            nn.init.zeros_(self.value_embedding.embed.weight)

    def compile_zones(self):
        self.core_block = torch.compile(self.core_block, mode="default")
        for i in range(len(self.coda_layers)):
            if not isinstance(self.coda_layers[i], CodaGQABlock):
                self.coda_layers[i] = torch.compile(self.coda_layers[i], mode="default")
        return self

    def forward(self, input_ids, targets=None):
        B, T = input_ids.shape
        if torch.is_autocast_enabled() and torch.get_autocast_gpu_dtype() == torch.bfloat16:
            raise RuntimeError("bf16 is 24% slower on gfx1151. Use fp16.")

        h = self.tok_embeddings(input_ids)
        freqs_cis = self.freqs_cis[:T]

        # Lean Prelude: 1 GQA layer
        velocity = torch.zeros_like(h)
        h, velocity = self.prelude_gqa(h, velocity, freqs_cis, ttt_target=None)
        input_embed = h

        # Core loop (skip injection on first iter — h == input_embed)
        h_core = h
        vel_core = torch.zeros_like(h_core)
        cached_states = []

        if self.training:
            step = self.step_counter.item()
            progress = min(step / max(self.curriculum_steps, 1), 1.0)
            eff = 1 - math.sqrt(1 - progress)
            t_full = max(self.n_iters - self.backprop_depth, 0)
            t = max(math.ceil(eff * t_full), 0)
            n_det = min(torch.poisson(torch.tensor([float(t)])).long().item(), 2 * max(t, 1))
            n_grad = self.backprop_depth
            self.step_counter += 1
        else:
            n_det, n_grad = 0, self.n_iters

        # First iteration: no re-injection
        h_core, vel_core = self.core_block(h_core, vel_core)
        h_core = self.iter_norm(h_core)
        if n_det + n_grad > 1:
            if n_det > 0:
                cached_states.append(h_core.detach())
                n_det -= 1
            else:
                cached_states.append(h_core)
                n_grad -= 1

        # Remaining iterations with re-injection
        for t in range(n_det):
            with torch.no_grad():
                h_core = self.injection(h_core, input_embed)
                h_core, vel_core = self.core_block(h_core, vel_core)
                h_core = self.iter_norm(h_core)
                cached_states.append(h_core.detach())

        for t in range(n_grad):
            h_core = self.injection(h_core, input_embed)
            h_core, vel_core = self.core_block(h_core, vel_core)
            h_core = self.iter_norm(h_core)
            cached_states.append(h_core)

        # DMC
        h = self.dmc(cached_states) if len(cached_states) > 1 else h_core

        # Lean Coda: 2 layers
        velocity = torch.zeros_like(h)
        for i, layer in enumerate(self.coda_layers):
            if isinstance(layer, CodaGQABlock):
                ttt_target = h if layer.ttt_mode != "none" else None
                h, velocity = layer(h, velocity, freqs_cis,
                                    ttt_target=ttt_target, input_ids=input_ids)
            else:
                h, velocity = layer(h, velocity)

        return self.output(self.norm(h))


# ---------------------------------------------------------------------------
# Idea 2: Progressive Narrowing — d=768 iter 1, d=512 iters 2-4
# ---------------------------------------------------------------------------

class GriffinHaloProgressive(nn.Module):
    """Progressive: d=768 Griffin iter 1 → d=512 ShortConv iters 2-4.

    First iteration at d=768 establishes rich global representation.
    Remaining iterations at d=512 refine cheaply with compile-friendly
    ShortConvBlocks (3 per iteration, like JORMUNGANDR-HALO).
    Nearly identical core FLOP to JORMUNGANDR-HALO.
    """

    def __init__(self, vocab_size=50257, d_model=768, d_narrow=512,
                 ffn_wide=2048, ffn_narrow=1792,
                 d_conv_wide=384, d_griffin=384,
                 d_conv_narrow=512, d_conv_prelude=512,
                 ffn_prelude=2816, n_heads=12, n_kv_heads=4,
                 conv_kernel=3, n_narrow_iters=3,
                 backprop_depth=3, curriculum_steps=5000,
                 momentum_beta_init=0.5, max_seq_len=1024,
                 d_gate=64, **kw):
        super().__init__()
        self.d_model = d_model
        self.d_narrow = d_narrow
        self.n_narrow_iters = n_narrow_iters
        self.backprop_depth = backprop_depth
        self.curriculum_steps = curriculum_steps
        self.max_seq_len = max_seq_len

        # Embedding + tied output
        self.tok_embeddings = nn.Embedding(vocab_size, d_model)
        self.norm = RMSNorm(d_model)
        self.output = nn.Linear(d_model, vocab_size, bias=False)
        self.output.weight = self.tok_embeddings.weight

        head_dim = d_model // n_heads
        self.register_buffer("freqs_cis",
            precompute_freqs_cis(head_dim, max_seq_len * 2), persistent=False)
        self.register_buffer("step_counter", torch.tensor(0, dtype=torch.long))

        # === PRELUDE (d=768) ===
        self.prelude_conv = ShortConvBlock(
            d_model, d_conv_prelude, ffn_prelude, conv_kernel, momentum_beta_init)
        self.prelude_gqa = GQABlock(
            d_model, ffn_prelude, n_heads, n_kv_heads,
            momentum_beta_init, ttt_mode="none")

        # === WIDE ITERATION (d=768, Griffin, 1 pass) ===
        self.wide_injection = SimpleParcaeInjection(d_model)
        self.wide_block = GriffinConvBlock(
            d_model, d_conv_wide, d_griffin, ffn_wide,
            conv_kernel, momentum_beta_init)
        self.wide_norm = RMSNorm(d_model)

        # === DIMENSION ADAPTER ===
        self.proj_down = nn.Linear(d_model, d_narrow, bias=False)
        self.proj_up = nn.Linear(d_narrow, d_model, bias=False)

        # === NARROW ITERATIONS (d=512, 3 ShortConvBlocks each, compile-friendly) ===
        self.narrow_injection = SimpleParcaeInjection(d_narrow)
        self.narrow_layers = nn.ModuleList([
            ShortConvBlock(d_narrow, d_conv_narrow, ffn_narrow,
                           conv_kernel, momentum_beta_init)
            for _ in range(3)
        ])
        self.narrow_norm = RMSNorm(d_narrow)

        # === DMC at d=512 (aggregates narrow iteration states) ===
        self.dmc = DepthMemoryCache(d_narrow, d_gate)

        # === CODA (d=768, 4 layers) ===
        kv_dim = n_kv_heads * head_dim
        ve = ValueEmbedding(vocab_size, 64, kv_dim)

        self.coda_layers = nn.ModuleList([
            ShortConvBlock(d_model, d_conv_prelude, ffn_prelude,
                           conv_kernel, momentum_beta_init),
            CodaGQABlock(d_model, ffn_prelude, n_heads, n_kv_heads,
                         momentum_beta_init, ttt_mode="none", exclusive=True),
            ShortConvBlock(d_model, d_conv_prelude, ffn_prelude,
                           conv_kernel, momentum_beta_init),
            CodaGQABlock(d_model, ffn_prelude, n_heads, n_kv_heads,
                         momentum_beta_init, ttt_mode="single",
                         value_embedding=ve, exclusive=True),
        ])
        self.value_embedding = ve

        self._init_weights()
        n_params = sum(p.numel() for p in self.parameters())
        print(f"{self.__class__.__name__}: {n_params / 1e6:.1f}M parameters")

    def _init_weights(self):
        for name, p in self.named_parameters():
            if p.dim() >= 2:
                nn.init.xavier_uniform_(p)
        for layer in self.coda_layers:
            if isinstance(layer, CodaGQABlock) and hasattr(layer.ffn, '_init_ttt_weights'):
                layer.ffn._init_ttt_weights()
        if self.value_embedding is not None:
            nn.init.zeros_(self.value_embedding.embed.weight)

    def compile_zones(self):
        self.wide_block = torch.compile(self.wide_block, mode="default")
        for i in range(len(self.narrow_layers)):
            self.narrow_layers[i] = torch.compile(self.narrow_layers[i], mode="default")
        self.prelude_conv = torch.compile(self.prelude_conv, mode="default")
        for i in range(len(self.coda_layers)):
            if not isinstance(self.coda_layers[i], CodaGQABlock):
                self.coda_layers[i] = torch.compile(self.coda_layers[i], mode="default")
        return self

    def forward(self, input_ids, targets=None):
        B, T = input_ids.shape
        if torch.is_autocast_enabled() and torch.get_autocast_gpu_dtype() == torch.bfloat16:
            raise RuntimeError("bf16 is 24% slower on gfx1151. Use fp16.")

        h = self.tok_embeddings(input_ids)
        freqs_cis = self.freqs_cis[:T]

        # Prelude (d=768)
        velocity = torch.zeros_like(h)
        h, velocity = self.prelude_conv(h, velocity)
        h, velocity = self.prelude_gqa(h, velocity, freqs_cis, ttt_target=None)
        input_embed_wide = h

        # === WIDE ITERATION 1 (d=768, Griffin, no injection — h == input_embed) ===
        vel_wide = torch.zeros_like(h)
        h_wide, vel_wide = self.wide_block(h, vel_wide)
        h_wide = self.wide_norm(h_wide)

        # === PROJ DOWN (768 → 512) ===
        h_narrow = self.proj_down(h_wide)
        input_embed_narrow = self.proj_down(input_embed_wide)

        # === NARROW ITERATIONS 2-4 (d=512, 3 ShortConvBlocks each) ===
        vel_narrow = torch.zeros(B, T, self.d_narrow, device=h.device, dtype=h.dtype)
        cached_states = [h_narrow]  # include post-wide state

        if self.training:
            step = self.step_counter.item()
            progress = min(step / max(self.curriculum_steps, 1), 1.0)
            eff = 1 - math.sqrt(1 - progress)
            t_full = max(self.n_narrow_iters - self.backprop_depth, 0)
            t = max(math.ceil(eff * t_full), 0)
            n_det = min(torch.poisson(torch.tensor([float(t)])).long().item(), 2 * max(t, 1))
            n_grad = self.backprop_depth
            self.step_counter += 1
        else:
            n_det, n_grad = 0, self.n_narrow_iters

        for t in range(n_det):
            with torch.no_grad():
                h_narrow = self.narrow_injection(h_narrow, input_embed_narrow)
                for layer in self.narrow_layers:
                    h_narrow, vel_narrow = layer(h_narrow, vel_narrow)
                h_narrow = self.narrow_norm(h_narrow)
                cached_states.append(h_narrow.detach())

        for t in range(n_grad):
            h_narrow = self.narrow_injection(h_narrow, input_embed_narrow)
            for layer in self.narrow_layers:
                h_narrow, vel_narrow = layer(h_narrow, vel_narrow)
            h_narrow = self.narrow_norm(h_narrow)
            cached_states.append(h_narrow)

        # DMC at d=512
        h_agg = self.dmc(cached_states) if len(cached_states) > 1 else h_narrow

        # === PROJ UP (512 → 768) ===
        h = self.proj_up(h_agg)

        # Coda (d=768, 4 layers)
        velocity = torch.zeros_like(h)
        for i, layer in enumerate(self.coda_layers):
            if isinstance(layer, CodaGQABlock):
                ttt_target = h if layer.ttt_mode != "none" else None
                h, velocity = layer(h, velocity, freqs_cis,
                                    ttt_target=ttt_target, input_ids=input_ids)
            else:
                h, velocity = layer(h, velocity)

        return self.output(self.norm(h))


class GriffinHaloMini(GriffinHaloBase):
    """Tiny config for smoke testing (~4M params)."""
    def __init__(self):
        super().__init__(
            vocab_size=1000,
            d_model=256, d_core=256,
            core_mixer="griffin",
            d_conv_core=128, d_griffin=128,
            ffn_core=512,
            d_conv_prelude=256, ffn_prelude=512,
            n_heads=4, n_kv_heads=2,
            n_iters=3, backprop_depth=2,
            curriculum_steps=100,
            depth_agg="last",
            use_xsa=False, ttt_mode="none",
            momentum_beta_init=0.5,
            max_seq_len=512,
        )

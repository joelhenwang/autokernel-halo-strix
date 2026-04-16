---
title: "JORMUNGANDR-HALO: Hardware-Adapted Looped Architecture for Strix Halo"
domain: architectures
type: spec
status: active
related:
  - mad_llm_scientist/plans/JORMUNGANDR.md
  - mad_llm_scientist/plans/OUROBOROS.md
  - mad_llm_scientist/plans/PARCAE.md
  - knowledge/hardware/amd_rdna35_strix_halo.md
  - knowledge/training/argus_prime_results.md
tags: [%jormungandr, %looped, %l2-cache, %strix-halo, %parcae, %heterogeneous-dim, %architecture]
---

# JORMUNGANDR-HALO: Hardware-Adapted Looped Architecture for Strix Halo

## Problem

JORMUNGANDR's core loop (3 ShortConv at d=768, ~2.8MB) fits L2 on paper but competes with activations and gradients for the 6MB L2 cache during training. The throughput estimate (15-19K tok/s) is conservative because L2 residency is uncertain. We want to guarantee L2 residency so that loop iterations 2+ are genuinely cheap.

## Solution

Heterogeneous dimensions: d=768 for Prelude/Coda (capacity), d=512 for the L2-resident core loop (speed). The core block shrinks to ~1.2MB fp16 — 1/5 of L2 — leaving ample room for activations. Entry/exit projections (768↔512) fused with Parcae injection add minimal overhead.

---

## Architecture

```
Tokens → Embedding (d=768, tied LM head, vocab=50257)
  │
  ▼
┌──────────────────────────────────────┐
│ PRELUDE (d=768, 2 unique layers)     │
│   Layer 1: ShortConvBlock(768, 512)  │  ← d_conv=512
│   Layer 2: GQABlock(768, n_kv=4)    │  ← QK-Norm, 3:1 GQA
└───────────┬──────────────────────────┘
            │ input_embed = h (saved, d=768)
            ▼
┌──────────────────────────────────────────────────────┐
│ FUSED ENTRY: Parcae injection + dim adapter           │
│   h_core = A * proj_down(h) + B * proj_down(embed)   │
│   (768→512, fused into one op)                        │
│   proj_down(input_embed) cached for all re-injections │
│                                                       │
│ CORE LOOP (d=512, 3 ShortConv × T iterations)        │
│   ~1.2MB fp16 weights → L2-RESIDENT                   │
│   d_conv=512, ffn_inner=1792 (3.5x)                  │
│                                                       │
│   Each ShortConv: inlined RMSNorm + GatedConv(512)   │
│                   + inlined momentum + SwiGLU(512,1792)|
│                                                       │
│   T ~ Poisson(mean_recurrence), bounded               │
│   Iters 1..T-3: detached (no grad, no GradScaler)    │
│   Iters T-2..T: gradient-tracked                      │
│                                                       │
│   Each layer compiled independently via torch.compile │
│   Re-injection every iteration: A*h + B*cached_embed  │
└───────────┬──────────────────────────────────────────┘
            │ h = proj_up(h_core)  (512→768, once)
            ▼
┌──────────────────────────────────────┐
│ CODA (d=768, 4 unique layers)        │
│   Layer 1: ShortConvBlock + FiLM     │
│   Layer 2: GQABlock + FiLM           │  ← n_kv=4
│   Layer 3: ShortConvBlock + FiLM     │
│   Layer 4: GQABlock + FiLM           │  ← n_kv=4
│            + ValueEmbed(d_ve=64)     │     projected to kv_dim=256
│            + TTT Sniper (1-step)     │
└───────────┬──────────────────────────┘
            ▼
  Final RMSNorm → LM Head → Predictions
```

---

## Parameter Budget

| Component | Params | Memory (fp16) | Notes |
|-----------|--------|---------------|-------|
| Embedding (50257×768, tied) | 38.6M | 77.2MB | Tied with LM head |
| Prelude: ShortConvBlock(768) | ~9.1M | 18.2MB | Unique |
| Prelude: GQABlock(768, n_kv=4) | ~8.8M | 17.6MB | Unique |
| Entry proj_down (768→512) | 0.4M | 0.8MB | Fused with Parcae injection |
| Parcae A, B | 1K | 2KB | 512-dim each |
| **Core: 3× ShortConvBlock(512, 1792)** | **~12.1M** | **~1.2MB** | **SHARED, L2-RESIDENT** |
| Exit proj_up (512→768) | 0.4M | 0.8MB | Once at loop exit |
| Coda: 2× ShortConvBlock(768) | ~18.2M | 36.4MB | Unique |
| Coda: 2× GQABlock(768, n_kv=4) | ~17.6M | 35.2MB | Unique |
| FiLM conditioner (6 targets) | ~0.7M | 1.4MB | d_film=64, projects 512→64→768 |
| Value Embedding (d_ve=64, 1 layer) | ~3.2M | 6.4MB | Last Coda GQA only |
| TTT sniper (1-step) | ~1.2M | 2.4MB | Last Coda GQA |
| **TOTAL UNIQUE** | **~110M** | | |
| **Effective (4× loop)** | **~158M** | | Comparable to ARGUS-PRIME (168M) |
| **Effective (8× loop)** | **~206M** | | Approaching AMADEUS (243M) |

---

## Dimension Alignment (Tensile-Friendly)

All matmul dimensions are multiples of 128:

| Matrix | Dims | Aligned? |
|--------|------|----------|
| Core SwiGLU gate_up | 512 × 3584 | ✓ |
| Core SwiGLU down | 1792 × 512 | ✓ |
| Core GatedConv proj | 512 × 1536 | ✓ |
| Prelude/Coda SwiGLU gate_up | 768 × 5632 | ✓ |
| Prelude/Coda down | 2816 × 768 | ✓ |
| Attention Q | 768 × 768 | ✓ |
| Attention K/V | 768 × 256 | ✓ |
| Loop entry | 768 × 512 | ✓ |
| Loop exit | 512 × 768 | ✓ |

---

## Throughput Estimate

Based on measured baselines (ARGUS-PRIME 18K tok/s, RESONANT-LOOP 15.9K tok/s):

| Component | Est. time | Notes |
|-----------|-----------|-------|
| Prelude (2 layers, d=768) | ~3.0ms | Same as ARGUS-PRIME |
| Entry proj + Parcae injection | ~0.3ms | One matmul + element-wise |
| Core iter 1 (cold, DRAM) | ~2.5ms | 3 ShortConv at d=512 |
| Core iters 2-4 (L2-warm) | 3 × ~0.8ms = ~2.4ms | L2-resident → ~3x speedup |
| Exit proj | ~0.3ms | One matmul |
| Coda (4 layers, d=768) | ~6.0ms | Similar to ARGUS-PRIME |
| LM head + loss | ~1.5ms | Standard |
| **Forward total (4 iters)** | **~16ms** | |
| **Forward + backward** | **~50ms** | 3.1x forward ratio |
| **+ optimizer + overhead** | **~60ms** | 19% optimizer |

| Config | Est. tok/s | Est. MFU | Notes |
|--------|-----------|----------|-------|
| 4 iterations | **17-19K** | 29-32% | Sweet spot — competitive with ARGUS-PRIME |
| 6 iterations | 14-16K | 24-27% | If depth scaling helps |
| 8 iterations | 12-14K | 20-24% | Only if quality demands it |

**Key assumption:** L2 gives ~3x speedup on iterations 2+. Validated by profiling in Stage 1 (rocprof comparison of iter 1 vs iter 2 kernel times).

---

## Hardware-Specific Optimizations (6 baked in)

### 1. Fused Parcae injection + entry projection
The dimension adapter (768→512) and Parcae injection (`A*h + B*input_embed`) fuse into one operation. `proj_down(input_embed)` is computed once and cached for all re-injections across iterations.

### 2. Inlined RMSNorm + momentum in core ShortConvBlock
Same pattern as ARGUS-PRIME — inline the RMSNorm computation and momentum update instead of separate modules. Inductor fuses the element-wise ops. Proven +1-3%.

### 3. Per-layer torch.compile
Each of the 3 core ShortConv layers compiled independently with `mode="default"`. The Python loop connecting them is uncompiled (variable-length Poisson depth + no_grad/grad switching cause graph breaks). Prelude and Coda compiled as separate units.

```python
self.core_layers = nn.ModuleList([
    torch.compile(ShortConvBlock(512, 512, 1792), mode="default")
    for _ in range(3)
])
```

`mode="default"` not `reduce-overhead` — CUDAGraphs require fixed tensor shapes, incompatible with Poisson depth.

### 4. Detached iterations skip GradScaler overhead
Iterations under `torch.no_grad()` bypass GradScaler's inf-checking and unscaling. Small per-iteration saving that compounds over 1-5 detached iterations.

### 5. Pre-computed RoPE frequencies
RoPE only needed for Coda GQA layers (core loop is all ShortConv). Computed once during Prelude, passed through. No recomputation.

### 6. pin_memory=False
Unified memory — pinning is a wasted syscall on Strix Halo. Enshrined in DataLoader config.

---

## Compile Strategy

```
┌─────────────────────────┐
│ torch.compile (default)  │ ← Prelude: 2 layers, one compiled unit
└───────────┬─────────────┘
            │
    Python loop (uncompiled)
            │
    ┌───────┴───────┐
    │ for t in T:   │
    │   injection   │ ← element-wise (tiny, not worth compiling)
    │   ┌─────────────────────────┐
    │   │ torch.compile (default)  │ ← Core layer 0
    │   └─────────────────────────┘
    │   ┌─────────────────────────┐
    │   │ torch.compile (default)  │ ← Core layer 1
    │   └─────────────────────────┘
    │   ┌─────────────────────────┐
    │   │ torch.compile (default)  │ ← Core layer 2
    │   └─────────────────────────┘
    └───────────────┘
            │
┌─────────────────────────┐
│ torch.compile (default)  │ ← Coda: 4 layers, one compiled unit
└─────────────────────────┘
```

Each compiled unit sees a simple, fixed-shape computation. The Python loop handles iteration control, no_grad/grad switching, Parcae re-injection, and FiLM fingerprinting — all control flow that would cause graph breaks.

---

## Staged Activation Protocol

```
STAGE 1: BARE LOOP (Steps 0 - 15%)
    Active:   Prelude + Core Loop (d=512) + Coda
    Loop:     Poisson-sampled, curriculum 2 → 4 iterations
    Goal:     Parcae stability works at d=512. L2 benefit measured.
    Metrics:  val loss decreasing, loop state norms bounded,
              iter 1 vs 2+ timing via rocprof
    Kill:     L2 benefit < 2x or loss not decreasing after 500 steps

STAGE 2: ADD FILM (Steps 15% - 30%)
    Active:   + FiLM conditioner (fingerprint at first gradient iteration)
    Loop:     Curriculum stays at 4 iterations
    Goal:     FiLM gammas/betas diverge from identity
    Monitor:  film/gamma_std, film/beta_std increasing from 0

STAGE 3: ADD TTT (Steps 30% - 45%)
    Active:   + TTT sniper (single-step) at last Coda GQA
    Loop:     Still 4 iterations
    Goal:     TTT delta_norm non-zero, no NaN

STAGE 4: FULL TRAINING (Steps 45% - 100%)
    Active:   All components at 4 iterations
    Goal:     Converge. Compare to ARGUS-PRIME / AMADEUS.

OPTIONAL STAGE 5: DEPTH SCALING
    Test:     val_loss at depth 2, 4, 6, 8
    Action:   Increase mean_recurrence if quality improves with depth
    Fallback: Stay at 4 if no gain (save throughput)

OPTIONAL STAGE 6: UPGRADE TTT (1-step → 3-step)
    Fallback: Revert + reload checkpoint if NaN within 1000 steps

OPTIONAL STAGE 7: ADD MOMENTUM
    Fallback: Disable + reload if loss increases for 500 steps
```

---

## FiLM Cross-Dimension Design

FiLM fingerprint is computed from the core loop output (d=512) but modulates Coda layers (d=768):

```
h_core (B, T, 512) → mean pool → (B, 512) → context_proj → (B, 64)
                                                               │
            ┌──────────────────────────────────────────────────┘
            ▼
gamma_proj: (B, 64) → (B, 768) + 1.0    ← per Coda layer
beta_proj:  (B, 64) → (B, 768)          ← per Coda layer
```

6 targets: 2 gradient-tracked core iterations (modulating the same shared weights) + 4 unique Coda layers. Effective diversity: 4 unique + 2 repeated.

---

## Optimizer Groups

| Pattern | Optimizer | LR | WD | Notes |
|---------|-----------|-----|-----|-------|
| `core_layers.*` (2D) | Muon | 0.01 | decaying | 0.5x base Muon LR (3x grad accumulation) |
| `prelude.*`, `coda.*` (2D) | Muon | 0.02 | decaying | Standard Muon |
| `proj_down.*`, `proj_up.*` | Muon | 0.02 | decaying | Standard — only 2 matmuls |
| `*log_A*`, `*log_B*` | AdamW | 8e-5 | 0 | 0.1x — stability-critical |
| `*film*` | AdamW | 8e-4 | 0.1 | Standard |
| `*ttt*` | AdamW | 8e-4 | 0.1 | Standard |
| `*value_embed*` | AdamW | 8e-4 | 0 | No WD on embeddings |
| `*norm*`, `*bias` | AdamW | 8e-4 | 0 | Standard |
| `wte`, `lm_head` | AdamW | 8e-4 | 0 | Tied embeddings |

Muon weight decay decays linearly to 0 over training (from Parcae's recipe).

---

## CompreSSM HSV Monitoring

Applied to loop state at eval steps. Tracks how many of the 512 core dimensions are actually contributing:

| Metric | Healthy | Concern | Action |
|--------|---------|---------|--------|
| `hsv/effective_rank` | > 128 (25%) | < 64 (12.5%) | Consider CompreSSM truncation |
| `hsv/energy_concentration` | < 50 | > 100 | Representation collapse risk |
| `hsv/top_10pct_energy` | < 0.7 | > 0.9 | Most dims wasted |

If monitoring shows low effective rank after convergence: apply balanced truncation to compress 512→effective_rank, cutting loop compute proportionally.

---

## Guardrails

```python
# In model forward:
assert not (torch.is_autocast_enabled() and
            torch.get_autocast_gpu_dtype() == torch.bfloat16), \
    "bf16 is 24% slower on gfx1151 and crashes compile with RoPE. Use fp16."

# Loss guardrail (from Parcae):
if total_tokens > 5_000_000 and loss > 6.0:
    print(f"Loss guardrail triggered: {loss:.2f} after {total_tokens/1e6:.1f}M tokens")
    break
```

---

## Stage 1 Go/No-Go Metrics

| Metric | Pass | Fail |
|--------|------|------|
| Loop state norm bounded | h.norm() stable across iterations | Growing unboundedly |
| Loss decreasing | Decreasing within 500 steps | Flat or increasing |
| L2 cache benefit | iter 2+ timing < 0.5× iter 1 | < 0.7× (under 1.4x benefit) |
| Throughput | > 14K tok/s at 4 iterations | < 12K tok/s |
| Stability | No NaN in 2000 steps | Any NaN |

---

## Comparison to Baselines

| Model | Unique | Effective | tok/s | Quality (BabyLM) |
|-------|--------|-----------|-------|-------------------|
| ARGUS-PRIME B0 | 168M | 168M | 18K | ~3.00 |
| AMADEUS | 157M | 157M | 13.2K | 2.90 (best) |
| JORMUNGANDR (original) | 124M | 341M (8×) | 15-19K (est) | TBD |
| **JORMUNGANDR-HALO (4 iter)** | **110M** | **158M** | **17-19K (est)** | **TBD** |
| **JORMUNGANDR-HALO (8 iter)** | **110M** | **206M** | **12-14K (est)** | **TBD** |

The 4-iteration config is the primary target: competitive throughput with ARGUS-PRIME, comparable effective params, 34% fewer unique params. If quality matches, the param efficiency is the win.

---

## Files to Create

| File | Description |
|------|-------------|
| `models/jormungandr_halo.py` | Model implementation (~500 lines) |

## Files to Reuse (no modification)

| File | What it provides |
|------|-----------------|
| `models/argus_prime.py` | ShortConvBlock, GQABlock patterns (copy + adapt for d=512 core) |
| `models/amadeus.py` | RMSNorm, SwiGLU, GatedConv, FiLMConditioner base classes |
| `models/argus.py` | TTTSwiGLU, precompute_freqs_cis, apply_rotary_emb |
| `halo_training/muon.py` | MuonAdamW optimizer (unchanged) |
| `halo_training/trainer.py` | Training loop with --resume-from (unchanged) |

## Implementation Dependencies

- All JORMUNGANDR components from `mad_llm_scientist/plans/JORMUNGANDR.md` (staged activation, Poisson depth, Parcae injection, CompreSSM monitoring)
- Autokernel QKV pattern: separate wq/wk/wv/wo in all GQA blocks
- Position randomization: disabled by default (RoPE conflict — test in Stage 1)

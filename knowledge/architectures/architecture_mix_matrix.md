---
title: "Architecture Mix Matrix — Exhaustive Combinatorial Analysis"
domain: architectures
type: analysis
status: active
related:
  - knowledge/architectures/hypothesis_buildout_results.md
  - mad_llm_scientist/plans/
tags: [%mix, %combinatorial, %architecture, %exhaustive]
---

# Architecture Mix Matrix — Every Plausible Combination

**Date:** 2026-04-17
**Method:** Systematic enumeration of all compatible building blocks, filtered by hardware constraints, parameter budget (≤175M), and technical feasibility.

---

## Building Block Inventory

### Token Mixers (M) — The Sequence Processing Core

| ID | Mixer | Cost on Strix Halo | Parallelism | Quality Signal | Proven? |
|----|-------|-------------------|-------------|----------------|---------|
| M1 | Griffin (element-wise recurrence) | FREE | Chunked scan | 2.98 val (Tempest) | YES |
| M2 | EFLA (exact delta rule) | FREE | Chunked scan (FLA) | ~3.1 est (EREBUS) | NO |
| M3 | KDA (channel-wise gated delta) | FREE + tiny matmul | Chunked scan (FLA) | ~2.95 est (BIFROST) | NO |
| M4 | Mamba-3 SISO (selective scan, real) | FREE | mamba-ssm (5.6x) | 2.90 val (AMADEUS) | YES |
| M5 | Mamba-3 Complex MIMO | FREE (4 FMAs) | Custom scan | ~2.88 est (PHENIX) | NO |
| M6 | DeltaProduct Householder (n_h=2) | FREE + 2 matmuls | Custom scan | ~2.82 est (GORGON) | NO |
| M7 | Kalman Linear Attention | FREE | Assoc. scan (FLA) | ~2.88 est (AEGIS) | NO |
| M8 | FFT-LTI + EFLA correction | ~0.2ms FFT | O(N log N) parallel | ~2.95 est (SPECTRUM) | NO |
| M9 | RWKV-7 generalized delta | FREE | FLA RWKV kernel | ~2.85 est (VALKYRIE) | NO |
| M10 | GQA Attention (full/sliding) | ~3.5ms (SDPA) | Fully parallel | 3.00 val (Prometheus) | YES |
| M11 | Gated Attention (SDPA + sigmoid) | ~3.5ms + FREE | Fully parallel | ~2.95 est (Qwen) | NO* |
| M12 | GSA (Gated Slot Attention) | ~0.8ms (2x GLA) | FLA GLA kernel | ~2.90 est (TIAMAT) | NO |
| M13 | Conv (GatedConv / causal-conv1d) | 0.02ms (10x) | Fully parallel | Component only | YES |

*Gated Attention is proven at 15B (Qwen), not at 170M.

### FFN Types (F)

| ID | FFN | Cost | Params/layer | Notes |
|----|-----|------|-------------|-------|
| F1 | SwiGLU (768→1920→768) | ~2ms | 4.42M | Standard, autokernel-optimized |
| F2 | ReLU² (768→1920→768) | ~1.5ms | 2.95M | RWKV-7 style, no gate proj, saves ~1.5M |
| F3 | ScatterMoE (4 experts, top-2) | ~3ms | ~8M | Verified on gfx1151, Triton |
| F4 | SwiGLU + In-Place TTT on W_down | ~2ms + FREE | 4.42M + 0 | TTT adds zero params |

### Structural Patterns (S)

| ID | Structure | Unique Params | Effective Params | L2 Fit? | Throughput Class |
|----|-----------|--------------|-----------------|---------|-----------------|
| S1 | Stacked-16 (16 separate layers) | ~135M | ~135M | No (too large) | 10-15K |
| S2 | Stacked-14 (14 separate layers) | ~118M | ~118M | No | 12-17K |
| S3 | Stacked-12 (12 layers) | ~100M | ~100M | No | 14-19K |
| S4 | Looped-12 (1 block × 12 iter) | ~46M | ~127M | Partial (~3.7MB) | 35-42K |
| S5 | Looped-16 (1 block × 16 iter) | ~46M | ~168M | Partial | 30-38K |
| S6 | Looped-adaptive (variable depth) | ~46M | ~76M avg | Partial | 50-65K |
| S7 | Hybrid (4 unique + 1 shared×8) | ~75M | ~134M | Shared part fits | 18-25K |
| S8 | Multi-resolution (1 block, 4 scales) | ~40M | ~160M | Yes (~2MB) | 30-40K |

### Conditioning/Routing (C)

| ID | Mechanism | Overhead | Params Added | Compatible With |
|----|-----------|----------|-------------|----------------|
| C0 | None | 0 | 0 | All |
| C1 | FiLM (per-iteration modulation) | FREE | ~5M total | Looped only (S4-S6) |
| C2 | Conductor (context summary) | ~0.5ms | ~137K | Any (tested: negligible) |
| C3 | Flux Router (KDA vs Attn per layer) | FREE | ~12K/layer | Stacked with dual mixers |
| C4 | Skip Router (skip mixer) | FREE | ~769/layer | Any |
| C5 | Surprisal Router (depth bands) | ~0.5ms probe | 0 | Looped only (S4-S6) |
| C6 | Uncertainty Router (Kalman) | FREE | 0 | M7 only |
| C7 | Mode Gate (per-iter arch select) | FREE | 48 total | Looped + multi-mixer |
| C8 | Entropy Gate (System 1/2) | FREE | ~2K | Dual-path (d≥512!) |

### Enhancement Modules (E)

| ID | Enhancement | Overhead | Params Added | Constraints |
|----|-------------|----------|-------------|-------------|
| E0 | None | 0 | 0 | — |
| E1 | In-Place TTT (on FFN W_down) | FREE (outer product) | 0 | Any FFN |
| E2 | Engram (hash N-gram tables) | ~1ms (7.4x fused) | ~5M | L2 pressure |
| E3 | PLE Path A (per-layer context) | ~3% | ~2M | Stacked only |
| E4 | MatFormer (nested submodels) | 0% | 0 | Any SwiGLU FFN |
| E5 | Memory Caching (RNN checkpoints) | FREE on APU | ~0.5M | Recurrent mixers |
| E6 | Block Diffusion (dual objective) | ~5% train | ~0.7M | Looped only |
| E7 | CPU pLSTM co-processor | 0% (parallel) | ~5M | APU only, risky |
| E8 | Gated Attention sigmoid | FREE | ~768/head | Attention heads only |

---

## Compatibility Matrix

### Hard Constraints (Cannot Combine)

| Constraint | Reason |
|------------|--------|
| d < 512 anywhere + autokernel | HIP kernels diverge at small dims |
| Looped (S4-S6) + PLE (E3) | PLE is per-layer, meaningless for shared block |
| Non-Kalman mixer + Uncertainty Router (C6) | C6 requires Kalman posterior |
| No recurrence + Memory Caching (E5) | E5 caches recurrent state |
| Block Diffusion (E6) + Stacked (S1-S3) | E6 requires iteration = denoising |
| FiLM (C1) + Stacked (S1-S3) | C1 modulates per iteration, stacked has none |
| CPU track (E7) + Looped (S4-S6) | CPU track needs multi-layer structure |
| Mode Gate (C7) + Stacked (S1-S3) | C7 is per-iteration, stacked has no iterations |

### Soft Constraints (Possible But Risky)

| Combination | Risk | Why |
|-------------|------|-----|
| MoE (F3) + Looped (S4-S6) | HIGH | MoE adds ~4M params/block → block too large for L2 |
| 3+ mixer types in one block | HIGH | Training instability from gradient competition |
| Attention (M10/M11) + Looped | MEDIUM | Attention is expensive; N iterations × 3.5ms |
| Engram (E2) + Looped | MEDIUM | Engram is input-dependent; same table read N times |
| TTT (E1) + Stacked non-looped | LOW | TTT less useful without iteration refinement |

---

## EXHAUSTIVE MIX ENUMERATION

### Category 1: LOOPED BASE (S4/S5/S6) — The Throughput Machines

Looped architectures are the throughput champions. Base block is shared across 10-16 iterations.

#### 1A: Single-Mixer Looped Blocks

| # | Mix Name | Mixer | FFN | Structure | Routing | Enhancement | Est. Params | Est. tok/s | Est. Loss | Notes |
|---|----------|-------|-----|-----------|---------|-------------|-------------|-----------|-----------|-------|
| 1 | EREBUS (exists) | M2 (EFLA) | F1 (SwiGLU) | S4 (×12) | C0 | E0 | 46M | 38-42K | ~3.1 | Baseline looped EFLA |
| 2 | EREBUS+TTT | M2 | F4 (SwiGLU+TTT) | S4 | C0 | E1 | 46M | 37-41K | ~2.90 | = SENTINEL |
| 3 | EREBUS+TTT+Surprisal | M2 | F4 | S6 (adaptive) | C5 | E1 | 46M | 55-62K | ~2.85 | = ORACLE |
| 4 | EREBUS+Diffusion | M2 | F1 | S4 | C0 | E6 | 47M | 35-38K | ~2.95 | = MEDUSA |
| 5 | EREBUS+FiLM | M2 | F1 | S4 | C1 | E0 | 51M | 36-40K | ~3.0 | FiLM conductor per iteration |
| 6 | EREBUS+FiLM+TTT | M2 | F4 | S4 | C1 | E1 | 51M | 35-39K | ~2.85 | FiLM + TTT double refinement |
| 7 | EREBUS+MemCache | M2 | F1 | S4 | C0 | E5 | 47M | 38-42K | ~3.0 | Free on APU, state checkpoints |
| 8 | EREBUS+MemCache+TTT | M2 | F4 | S4 | C0 | E1+E5 | 47M | 37-41K | ~2.88 | State checkpoints + TTT |
| 9 | EREBUS+Skip | M2 | F1 | S4 | C4 | E0 | 46M | 42-48K | ~3.15 | Skip easy layers for speed |
| 10 | EREBUS+MoE | M2 | F3 (MoE) | S4 | C0 | E0 | 52M | 30-35K | ~2.95 | = CHRYSALIS |
| 11 | EREBUS+ReLU² | M2 | F2 (ReLU²) | S4 | C0 | E0 | 43M | 40-45K | ~3.15 | Smaller FFN, faster iterations |
| 12 | EREBUS+Diffusion+TTT | M2 | F4 | S4 | C0 | E1+E6 | 47M | 33-37K | ~2.85 | Triple signal: NTP+denoise+TTT |
| 13 | **PHENIX** (exists) | M5 (Complex MIMO) | F1 | S4 (×14) | C0 | E0 | 50M | 38-42K | ~2.88 | Complex state, half state size |
| 14 | PHENIX+TTT | M5 | F4 | S4 | C0 | E1 | 50M | 37-41K | ~2.83 | Complex MIMO + TTT adaptation |
| 15 | PHENIX+Surprisal | M5 | F4 | S6 | C5 | E1 | 50M | 50-60K | ~2.82 | Complex MIMO + adaptive depth |
| 16 | PHENIX+FiLM | M5 | F1 | S4 | C1 | E0 | 55M | 36-40K | ~2.85 | Complex MIMO + FiLM modulation |
| 17 | PHENIX+Diffusion | M5 | F1 | S4 | C0 | E6 | 51M | 35-38K | ~2.90 | Complex MIMO + block denoise |
| 18 | KDA-Loop | M3 (KDA) | F1 | S4 | C0 | E0 | 48M | 36-40K | ~3.0 | Looped KDA (not in any plan) |
| 19 | KDA-Loop+TTT | M3 | F4 | S4 | C0 | E1 | 48M | 35-39K | ~2.88 | KDA loop + TTT |
| 20 | KDA-Loop+Surprisal | M3 | F4 | S6 | C5 | E1 | 48M | 52-60K | ~2.85 | KDA + adaptive depth |
| 21 | KDA-Loop+FiLM | M3 | F1 | S4 | C1 | E0 | 53M | 34-38K | ~2.95 | KDA + iteration modulation |
| 22 | KDA-Loop+MemCache | M3 | F1 | S4 | C0 | E5 | 49M | 36-40K | ~2.95 | KDA + state checkpoints |
| 23 | Griffin-Loop | M1 (Griffin) | F1 | S5 (×16) | C0 | E0 | 46M | 30-38K | ~3.4 | = RESONANT-LOOP improved |
| 24 | Griffin-Loop+FiLM | M1 | F1 | S5 | C1 | E0 | 51M | 28-36K | ~3.0 | = PROTEUS-LOOP (novelties) |
| 25 | Griffin-Loop+TTT | M1 | F4 | S5 | C0 | E1 | 46M | 29-37K | ~3.1 | Griffin + TTT adaptation |
| 26 | Griffin-Loop+FiLM+TTT | M1 | F4 | S5 | C1 | E1 | 51M | 27-35K | ~2.95 | Griffin + modulation + TTT |
| 27 | Kalman-Loop | M7 (Kalman) | F1 | S4 | C6 | E0 | 46M | 34-38K | ~2.90 | = AEGIS |
| 28 | Kalman-Loop+TTT | M7 | F4 | S4 | C6 | E1 | 46M | 33-37K | ~2.85 | Kalman + TTT on hard tokens |
| 29 | Kalman-Loop+FiLM | M7 | F1 | S4 | C1+C6 | E0 | 51M | 32-36K | ~2.88 | Kalman + FiLM + uncertainty |
| 30 | RWKV7-Loop | M9 (RWKV-7) | F2 (ReLU²) | S4 | C0 | E0 | 44M | 35-40K | ~3.0 | Pure RWKV-7 looped |
| 31 | RWKV7-Loop+KDA-alpha | M9+KDA gating | F2 | S4 | C0 | E0 | 46M | 34-39K | ~2.90 | = VALKYRIE-loop variant |
| 32 | RWKV7-Loop+EFLA | M9+EFLA exact | F2 | S4 | C0 | E0 | 44M | 35-40K | ~2.88 | RWKV-7 + exact ODE |
| 33 | FFT-Loop | M8 (FFT+corr) | F1 | S4 | C0 | E0 | 50M | 32-38K | ~3.0 | = SPECTRUM-loop variant |
| 34 | FFT-Loop+TTT | M8 | F4 | S4 | C0 | E1 | 50M | 31-37K | ~2.92 | FFT backbone + TTT correction |
| 35 | GSA-Loop | M12 (GSA) | F1 | S4 | C0 | E0 | 48M | 34-38K | ~2.95 | Gated Slot Attention looped |
| 36 | GSA-Loop+MemCache | M12 | F1 | S4 | C0 | E5 | 49M | 34-38K | ~2.88 | = TIAMAT-loop variant |
| 37 | GSA-Loop+TTT | M12 | F4 | S4 | C0 | E1 | 48M | 33-37K | ~2.88 | GSA + TTT adaptation |

#### 1B: Multi-Mixer Looped Blocks (2 mixer types sharing one block)

| # | Mix Name | Mixers | FFN | Structure | Routing | Enhancement | Est. Params | Est. tok/s | Est. Loss |
|---|----------|--------|-----|-----------|---------|-------------|-------------|-----------|-----------|
| 38 | EFLA+Mamba3-Loop | M2+M4 parallel | F1 | S4 | C0 | E0 | 52M | 32-38K | ~2.88 | AMADEUS-style in loop |
| 39 | EFLA+Mamba3-Loop+TTT | M2+M4 | F4 | S4 | C0 | E1 | 52M | 31-37K | ~2.83 | Best of AMADEUS in loop + TTT |
| 40 | EFLA+Mamba3-Loop+FiLM | M2+M4 | F1 | S4 | C1 | E0 | 57M | 30-36K | ~2.85 | AMADEUS-loop + modulation |
| 41 | KDA+EFLA-Loop | M3+M2 parallel | F1 | S4 | C0 | E0 | 50M | 33-38K | ~2.90 | Double delta-rule variants |
| 42 | KDA+Complex-Loop | M3+M5 parallel | F1 | S4 | C0 | E0 | 54M | 30-36K | ~2.85 | Channel-wise + rotational |
| 43 | **CHIMERA** (exists) | M2+M5+M11 gated | F1 | S4 | C7 | E0 | 46M | 14-25K | ~2.88 | Per-iter architecture morph |
| 44 | CHIMERA+TTT | M2+M5+M11 | F4 | S4 | C7 | E1 | 46M | 13-24K | ~2.83 | Morphing + TTT on HIGH iters |
| 45 | CHIMERA+Surprisal | M2+M5+M11 | F4 | S6 | C5+C7 | E1 | 46M | 20-35K | ~2.80 | Morphing + adaptive depth + TTT |
| 46 | Griffin+Conv-Loop | M1+M13 parallel | F1 | S4 | C0 | E0 | 48M | 33-38K | ~3.05 | TEMPEST in a loop |
| 47 | EFLA+GatedAttn-Loop | M2+M11 route | F1 | S4 | C7 | E0 | 48M | 25-35K | ~2.90 | EREBUS + gated attention |
| 48 | KDA+GatedAttn-Loop | M3+M11 route | F1 | S4 | C3 | E0 | 50M | 24-33K | ~2.88 | BIFROST in a loop |
| 49 | Complex+GatedAttn-Loop | M5+M11 route | F1 | S4 | C7 | E0 | 52M | 22-32K | ~2.85 | PHENIX + attention correction |
| 50 | Kalman+GatedAttn-Loop | M7+M11 | F1 | S4 | C6 | E0 | 48M | 24-32K | ~2.85 | Uncertainty routes to attention |

#### 1C: Looped + MoE Combos

| # | Mix Name | Mixer | FFN | Structure | Routing | Enhancement | Est. Params | Est. tok/s | Est. Loss |
|---|----------|-------|-----|-----------|---------|-------------|-------------|-----------|-----------|
| 51 | EFLA+MoE-Loop | M2 | F3 | S4 | C0 | E0 | 52M | 30-35K | ~2.95 | = CHRYSALIS |
| 52 | KDA+MoE-Loop | M3 | F3 | S4 | C0 | E0 | 54M | 28-33K | ~2.88 | KDA + sparse experts |
| 53 | Complex+MoE-Loop | M5 | F3 | S4 | C0 | E0 | 56M | 26-32K | ~2.85 | Complex MIMO + MoE |
| 54 | EFLA+MoE+TTT-Loop | M2 | F3+E1 variant | S4 | C0 | E1 | 52M | 29-34K | ~2.88 | MoE experts + TTT |
| 55 | KDA+MoE+Surprisal | M3 | F3 | S6 | C5 | E0 | 54M | 38-48K | ~2.85 | Adaptive depth + MoE |

#### 1D: Looped + Block Diffusion Combos

| # | Mix Name | Mixer | FFN | Structure | Routing | Enhancement | Est. Params | Est. tok/s (train/infer) | Est. Loss |
|---|----------|-------|-----|-----------|---------|-------------|-------------|--------------------------|-----------|
| 56 | EFLA+Diffusion | M2 | F1 | S4 | C0 | E6 | 47M | 35K / 150K | ~2.95 | = MEDUSA |
| 57 | KDA+Diffusion | M3 | F1 | S4 | C0 | E6 | 49M | 33K / 140K | ~2.90 | KDA + block decode |
| 58 | Complex+Diffusion | M5 | F1 | S4 | C0 | E6 | 51M | 33K / 140K | ~2.88 | Complex MIMO + block decode |
| 59 | EFLA+Diffusion+TTT | M2 | F4 | S4 | C0 | E1+E6 | 47M | 33K / 140K | ~2.88 | Triple signal |
| 60 | Kalman+Diffusion | M7 | F1 | S4 | C6 | E6 | 47M | 32K / 130K | ~2.90 | Uncertainty + diffusion |

#### 1E: Looped + Memory Caching (Unique to APU)

| # | Mix Name | Mixer | FFN | Structure | Routing | Enhancement | Est. Params | Est. tok/s | Est. Loss |
|---|----------|-------|-----|-----------|---------|-------------|-------------|-----------|-----------|
| 61 | EFLA+MemCache | M2 | F1 | S4 | C0 | E5 | 47M | 38-42K | ~3.0 | Free state checkpoints |
| 62 | KDA+MemCache | M3 | F1 | S4 | C0 | E5 | 49M | 36-40K | ~2.92 | KDA + long-context memory |
| 63 | GSA+MemCache | M12 | F1 | S4 | C0 | E5 | 49M | 34-38K | ~2.88 | = TIAMAT-loop |
| 64 | EFLA+MemCache+TTT | M2 | F4 | S4 | C0 | E1+E5 | 47M | 37-41K | ~2.88 | All three: exact+cache+adapt |
| 65 | Complex+MemCache | M5 | F1 | S4 | C0 | E5 | 51M | 36-40K | ~2.85 | Complex state checkpoints |

---

### Category 2: STACKED BASE (S1/S2/S3) — The Quality Machines

Stacked architectures have unique layers — more expressive but slower (all weights read once per forward).

#### 2A: Single-Mixer Stacked (16 layers)

| # | Mix Name | Mixer | FFN | Layers | Routing | Enhancement | Est. Params | Est. tok/s | Est. Loss |
|---|----------|-------|-----|--------|---------|-------------|-------------|-----------|-----------|
| 66 | AMADEUS (exists) | M4+M13 parallel | F1 | S1 (16L) | C0 | E0 | 158M | 13.2K | 2.90 | Tested winner |
| 67 | KDA-Stacked | M3 | F1 | S1 | C0 | E0 | 162M | ~14K | ~2.88 | Pure KDA 16 layers |
| 68 | EFLA-Stacked | M2 | F1 | S1 | C0 | E0 | 158M | ~14K | ~2.92 | Pure EFLA 16 layers |
| 69 | DeltaProduct-Stacked | M6 | F1 | S1 | C0 | E0 | 156M | ~12-14K | ~2.82 | = GORGON |
| 70 | Kalman-Stacked | M7 | F1 | S1 | C0 | E0 | 158M | ~13K | ~2.88 | Pure Kalman 16 layers |
| 71 | RWKV7-Stacked | M9 | F2 | S1 | C0 | E0 | 148M | ~15K | ~2.88 | Pure RWKV-7 16 layers |
| 72 | Complex-Stacked | M5 | F1 | S1 | C0 | E0 | 165M | ~12K | ~2.85 | Pure complex MIMO 16 layers |
| 73 | GSA-Stacked | M12 | F1 | S1 | C0 | E0 | 160M | ~13K | ~2.88 | Pure GSA 16 layers |
| 74 | FFT-Stacked | M8 | F1 | S2 (14L) | C0 | E0 | 161M | ~18-21K | ~2.95 | = SPECTRUM |

#### 2B: Hybrid-Mixer Stacked (Interleaved mixer types across layers)

| # | Mix Name | Mixer Pattern | FFN | Layers | Routing | Enhancement | Est. Params | Est. tok/s | Est. Loss |
|---|----------|--------------|-----|--------|---------|-------------|-------------|-----------|-----------|
| 75 | KDA+Attn 3:1 | 12×M3 + 4×M11 | F1 | S1 | C0 | E8 | 164M | ~14K | ~2.85 | = BIFROST (static) |
| 76 | KDA+Attn Flux | 12×M3 + 4×M11 | F1 | S1 | C3 | E8 | 165M | ~15K | ~2.83 | = BIFROST (dynamic) |
| 77 | EFLA+Attn 14:2 | 14×M2 + 2×M11 | F1 | S1 | C0 | E8 | 160M | ~14K | ~2.88 | EREBUS-stacked + 2 attn |
| 78 | KDA+EFLA 8:8 | 8×M3 + 8×M2 | F1 | S1 | C0 | E0 | 162M | ~14K | ~2.88 | Two delta-rule variants |
| 79 | RWKV7+Attn 14:2 | 14×M9 + 2×M11 | F2 | S1 | C0 | E8 | 152M | ~14K | ~2.85 | VALKYRIE-stacked |
| 80 | Griffin+Attn 14:2 | 14×M1 + 2×M10 | F1 | S1 | C0 | E0 | 174M | ~13K | 3.00 | = PROMETHEUS (tested) |
| 81 | Complex+Attn 12:4 | 12×M5 + 4×M11 | F1 | S1 | C0 | E8 | 168M | ~12K | ~2.83 | Complex + gated attn |
| 82 | DeltaProd+Attn 14:2 | 14×M6 + 2×M11 | F1 | S1 | C0 | E8 | 160M | ~12K | ~2.80 | Strongest quality mix |
| 83 | Kalman+Attn 14:2 | 14×M7 + 2×M11 | F1 | S1 | C6 | E8 | 160M | ~13K | ~2.83 | Kalman + correction attn |
| 84 | GSA+Attn 14:2 | 14×M12 + 2×M11 | F1 | S1 | C0 | E8 | 162M | ~13K | ~2.85 | GSA + gated attn |
| 85 | FFT+Attn 12:2 | 12×M8 + 2×M11 | F1 | S2 (14L) | C0 | E8 | 163M | ~16K | ~2.90 | FFT backbone + attn |
| 86 | EFLA+Complex 8:8 | 8×M2 + 8×M5 | F1 | S1 | C0 | E0 | 165M | ~13K | ~2.85 | Exact + rotational |
| 87 | KDA+Complex 8:8 | 8×M3 + 8×M5 | F1 | S1 | C0 | E0 | 168M | ~12K | ~2.83 | Channel-gate + complex |
| 88 | Griffin+KDA 8:8 | 8×M1 + 8×M3 | F1 | S1 | C0 | E0 | 164M | ~14K | ~2.90 | Proven + novel |
| 89 | Griffin+EFLA 8:8 | 8×M1 + 8×M2 | F1 | S1 | C0 | E0 | 160M | ~14K | ~2.92 | Proven + exact |
| 90 | RWKV7+KDA 8:8 | 8×M9 + 8×M3 | F2 | S1 | C0 | E0 | 156M | ~14K | ~2.85 | Two delta variants |

#### 2C: Parallel Hybrid Stacked (Two mixers per layer, AMADEUS-style)

| # | Mix Name | Parallel Mixers | FFN | Layers | Routing | Enhancement | Est. Params | Est. tok/s | Est. Loss |
|---|----------|----------------|-----|--------|---------|-------------|-------------|-----------|-----------|
| 91 | Conv+Mamba3 | M13+M4 | F1 | S1 | C0 | E0 | 158M | 13.2K | 2.90 | = AMADEUS |
| 92 | Conv+KDA | M13+M3 | F1 | S1 | C0 | E0 | 162M | ~13K | ~2.88 | Conv local + KDA global |
| 93 | Conv+EFLA | M13+M2 | F1 | S1 | C0 | E0 | 160M | ~13K | ~2.90 | Conv local + EFLA global |
| 94 | Conv+Complex | M13+M5 | F1 | S1 | C0 | E0 | 165M | ~12K | ~2.85 | Conv + rotational SSM |
| 95 | Conv+DeltaProd | M13+M6 | F1 | S1 | C0 | E0 | 170M | ~11K | ~2.82 | Conv + Householder |
| 96 | Conv+Kalman | M13+M7 | F1 | S1 | C0 | E0 | 160M | ~13K | ~2.88 | Conv + Bayesian filter |
| 97 | Conv+RWKV7 | M13+M9 | F2 | S1 | C0 | E0 | 152M | ~14K | ~2.88 | Conv + generalized delta |
| 98 | Conv+GSA | M13+M12 | F1 | S1 | C0 | E0 | 162M | ~12K | ~2.88 | Conv + slot attention |
| 99 | EFLA+Mamba3 | M2+M4 | F1 | S1 | C0 | E0 | 165M | ~12K | ~2.85 | Exact delta + selective scan |
| 100 | KDA+Mamba3 | M3+M4 | F1 | S1 | C0 | E0 | 168M | ~12K | ~2.83 | Channel-gate + selective |
| 101 | KDA+Complex | M3+M5 | F1 | S1 | C0 | E0 | 170M | ~11K | ~2.82 | Channel-gate + rotational |
| 102 | EFLA+Complex | M2+M5 | F1 | S1 | C0 | E0 | 168M | ~12K | ~2.85 | Exact + rotational parallel |
| 103 | DeltaProd+Mamba3 | M6+M4 | F1 | S1 | C0 | E0 | 172M | ~10K | ~2.80 | Householder + selective |
| 104 | Kalman+Mamba3 | M7+M4 | F1 | S1 | C0 | E0 | 165M | ~12K | ~2.83 | Bayesian + selective scan |
| 105 | RWKV7+Mamba3 | M9+M4 | F1 | S1 | C0 | E0 | 160M | ~12K | ~2.85 | Gen. delta + selective |

#### 2D: Triple-Mixer Hybrid-Head Stacked (BASILISK-style)

| # | Mix Name | Head Types | FFN | Layers | Routing | Enhancement | Est. Params | Est. tok/s | Est. Loss |
|---|----------|-----------|-----|--------|---------|-------------|-------------|-----------|-----------|
| 106 | KDA+Complex+GatedAttn | M3+M5+M11 | F1 | S1 | per-head | E8 | 151M | 10-14K | ~2.85 | = BASILISK |
| 107 | EFLA+Complex+GatedAttn | M2+M5+M11 | F1 | S1 | per-head | E8 | 149M | 10-14K | ~2.88 | BASILISK with EFLA |
| 108 | KDA+Mamba3+GatedAttn | M3+M4+M11 | F1 | S1 | per-head | E8 | 155M | 10-14K | ~2.83 | KDA + proven SSM + attn |
| 109 | EFLA+Kalman+GatedAttn | M2+M7+M11 | F1 | S1 | per-head | E8 | 150M | 10-14K | ~2.85 | Exact + Bayesian + attn |
| 110 | DeltaProd+KDA+GatedAttn | M6+M3+M11 | F1 | S1 | per-head | E8 | 158M | 9-13K | ~2.80 | Strongest triple |
| 111 | RWKV7+Complex+GatedAttn | M9+M5+M11 | F1 | S1 | per-head | E8 | 152M | 10-14K | ~2.85 | Gen. delta + rotational + attn |
| 112 | Griffin+KDA+GatedAttn | M1+M3+M11 | F1 | S1 | per-head | E8 | 155M | 11-14K | ~2.88 | Proven + novel + attn |
| 113 | GSA+KDA+GatedAttn | M12+M3+M11 | F1 | S1 | per-head | E8 | 155M | 10-13K | ~2.83 | Slot + channel + attn |

#### 2E: Stacked + Enhancement Modules

| # | Mix Name | Base | Enhancement Combo | Est. Params | Est. tok/s | Est. Loss | Notes |
|---|----------|------|-------------------|-------------|-----------|-----------|-------|
| 114 | AMADEUS+TTT | #91 | E1 | 158M | ~12K | ~2.85 | Proven base + TTT adaptation |
| 115 | AMADEUS+Engram | #91 | E2 | 163M | ~12K | ~2.85 | SSM hybrid + N-gram knowledge |
| 116 | AMADEUS+PLE-A | #91 | E3 | 160M | ~12K | ~2.87 | Per-layer context embedding |
| 117 | AMADEUS+MatFormer | #91 | E4 | 158M | ~13K | ~2.90 | Free nested submodels |
| 118 | AMADEUS+TTT+MatFormer | #91 | E1+E4 | 158M | ~12K | ~2.85 | TTT + elastic inference |
| 119 | AMADEUS+Engram+MatFormer | #91 | E2+E4 | 163M | ~12K | ~2.83 | Knowledge + elastic |
| 120 | KDA-Stack+PLE-A | #67 | E3 | 164M | ~13K | ~2.85 | KDA + per-layer conditioning |
| 121 | KDA-Stack+TTT | #67 | E1 | 162M | ~13K | ~2.83 | KDA + In-Place TTT |
| 122 | KDA-Stack+MatFormer | #67 | E4 | 162M | ~14K | ~2.88 | KDA + elastic |
| 123 | KDA-Stack+Engram | #67 | E2 | 167M | ~13K | ~2.83 | KDA + N-gram tables |
| 124 | DeltaProd+TTT | #69 | E1 | 156M | ~12K | ~2.78 | Strongest quality + TTT |
| 125 | DeltaProd+MatFormer | #69 | E4 | 156M | ~13K | ~2.82 | Householder + elastic |
| 126 | RWKV7+Engram | #71 | E2 | 153M | ~14K | ~2.85 | Gen. delta + knowledge |
| 127 | Complex+TTT | #72 | E1 | 165M | ~11K | ~2.82 | Rotational + adaptation |
| 128 | Complex+MatFormer | #72 | E4 | 165M | ~12K | ~2.85 | Complex + elastic |
| 129 | Tempest+Engram | existing | E2 | ~172M | ~12K | ~2.93 | Proven Griffin + N-gram |
| 130 | Tempest+TTT | existing | E1 | ~177M | ~12K | ~2.92 | Proven + adaptation |
| 131 | Prometheus+GatedAttn | existing | E8 | ~174M | ~13K | ~2.95 | Add sigmoid gate (free) |

#### 2F: Stacked + MoE FFN

| # | Mix Name | Mixer | FFN | Layers | Routing | Enhancement | Est. Params | Est. tok/s | Est. Loss |
|---|----------|-------|-----|--------|---------|-------------|-------------|-----------|-----------|
| 132 | KDA+MoE-Stack | M3 | F3 (4 exp, top-2) | S2 (14L) | C0 | E0 | 168M | ~12K | ~2.85 | KDA + sparse experts |
| 133 | EFLA+MoE-Stack | M2 | F3 | S2 | C0 | E0 | 165M | ~12K | ~2.88 | EFLA + sparse experts |
| 134 | Complex+MoE-Stack | M5 | F3 | S2 | C0 | E0 | 172M | ~11K | ~2.83 | Complex + sparse |
| 135 | AMADEUS+MoE | M4+M13 | F3 | S2 | C0 | E0 | 170M | ~11K | ~2.83 | Proven hybrid + MoE |
| 136 | KDA+Attn+MoE | 12×M3+2×M11 | F3 | S2 | C3 | E8 | 172M | ~11K | ~2.80 | BIFROST + MoE |

#### 2G: Stacked + Flux/Dynamic Routing (Dual Mixer with Router)

| # | Mix Name | Mixer A | Mixer B | FFN | Layers | Router | Enhancement | Est. Params | Est. tok/s | Est. Loss |
|---|----------|---------|---------|-----|--------|--------|-------------|-------------|-----------|-----------|
| 137 | KDA↔Attn Flux | M3 | M11 | F1 | S1 | C3 | E8 | 165M | ~15K | ~2.83 | = BIFROST |
| 138 | EFLA↔Attn Flux | M2 | M11 | F1 | S1 | C3 | E8 | 161M | ~15K | ~2.86 | EREBUS mixer + dynamic routing |
| 139 | KDA↔Complex Flux | M3 | M5 | F1 | S1 | C3 | E0 | 170M | ~13K | ~2.83 | Channel vs rotational routing |
| 140 | Complex↔Attn Flux | M5 | M11 | F1 | S1 | C3 | E8 | 168M | ~14K | ~2.83 | Complex + dynamic attn |
| 141 | RWKV7↔Attn Flux | M9 | M11 | F2 | S1 | C3 | E8 | 155M | ~15K | ~2.85 | RWKV-7 + dynamic attn |
| 142 | DeltaProd↔Attn Flux | M6 | M11 | F1 | S1 | C3 | E8 | 163M | ~13K | ~2.80 | Householder + dynamic attn |
| 143 | Kalman↔Attn Flux | M7 | M11 | F1 | S1 | C3+C6 | E8 | 162M | ~14K | ~2.83 | Uncertainty-routed attn |
| 144 | GSA↔Attn Flux | M12 | M11 | F1 | S1 | C3 | E8 | 164M | ~13K | ~2.85 | Slot + dynamic attn |

---

### Category 3: HYBRID STACKED+LOOPED (S7) — Best of Both Worlds

Some unique layers for diversity + a shared looped core for throughput.

| # | Mix Name | Unique Layers | Shared Loop | FFN | Routing | Enhancement | Est. Params | Est. tok/s | Est. Loss |
|---|----------|--------------|-------------|-----|---------|-------------|-------------|-----------|-----------|
| 145 | 4-Attn + 8-EFLA-Loop | 4×M11 unique | 1×M2 shared×8 | F1 | C0 | E8 | ~85M | 22-28K | ~2.88 | Attn endpoints + EFLA core |
| 146 | 2-Attn + 1-EFLA-Loop×10 | 2×M11 unique | 1×M2 shared×10 | F1 | C0 | E8 | ~65M | 28-35K | ~2.90 | Minimal attn + deep loop |
| 147 | 2-Attn + 1-KDA-Loop×10 | 2×M11 unique | 1×M3 shared×10 | F1 | C3 | E8 | ~67M | 26-33K | ~2.88 | Attn bookends + KDA core |
| 148 | 4-KDA + 1-EFLA-Loop×8 | 4×M3 unique | 1×M2 shared×8 | F1 | C0 | E0 | ~80M | 24-30K | ~2.88 | KDA diversity + EFLA loop |
| 149 | 2-Complex + 1-EFLA-Loop×10 | 2×M5 unique | 1×M2 shared×10 | F1 | C0 | E0 | ~70M | 27-34K | ~2.88 | Complex endpoints + EFLA core |
| 150 | 2-Attn + 1-Complex-Loop×10 | 2×M11 unique | 1×M5 shared×10 | F1 | C0 | E8 | ~70M | 25-32K | ~2.85 | Attn + complex loop |
| 151 | 4-DeltaProd + 1-EFLA-Loop×8 | 4×M6 unique | 1×M2 shared×8 | F1 | C0 | E0 | ~90M | 20-26K | ~2.82 | Householder diversity + loop |
| 152 | Hybrid-TTT: 2-Attn + EFLA-Loop×10+TTT | 2×M11 | 1×M2×10 | F4 | C0 | E1+E8 | ~65M | 27-34K | ~2.85 | Attn + TTT-enhanced loop |
| 153 | Hybrid-FiLM: 4-unique + EFLA×8+FiLM | 4×M3 | 1×M2×8 | F1 | C1 | E0 | ~85M | 22-28K | ~2.85 | Diverse layers + modulated core |
| 154 | Hybrid-Surprisal: 2-Attn + EFLA adaptive | 2×M11 | 1×M2 adaptive | F4 | C5 | E1+E8 | ~65M | 35-45K | ~2.83 | Attn + adaptive-depth loop |

---

### Category 4: CPU-GPU COOPERATIVE (APU-Native, E7)

| # | Mix Name | GPU Track | CPU Track | Fusion | Est. Params | Est. tok/s | Est. Loss |
|---|----------|-----------|-----------|--------|-------------|-----------|-----------|
| 155 | HELIX (exists) | 12×M3 (KDA) | 4×pLSTM | FiLM | 129M | 15-17K | ~2.88 |
| 156 | HELIX-EFLA | 12×M2 (EFLA) | 4×pLSTM | FiLM | 127M | 15-17K | ~2.90 |
| 157 | HELIX-Complex | 12×M5 (Complex) | 4×pLSTM | FiLM | 133M | 14-16K | ~2.85 |
| 158 | HELIX-Looped | 1×M2-Loop×12 GPU | 4×pLSTM CPU | FiLM | 52M | 35-40K | ~2.95 |
| 159 | HELIX-RWKV7 | 12×M9 GPU | 4×pLSTM CPU | FiLM | 123M | 15-17K | ~2.88 |
| 160 | HELIX-Mamba | 12×(M13+M4) GPU | 4×pLSTM CPU | FiLM | 135M | 13-15K | ~2.85 |
| 161 | HELIX-Hybrid | 10×M3+2×M11 GPU | 4×pLSTM CPU | FiLM | 132M | 14-16K | ~2.83 |
| 162 | HELIX-Loop-TTT | 1×M2-Loop×12+TTT GPU | 4×pLSTM CPU | FiLM | 52M | 34-39K | ~2.88 |
| 163 | HELIX-DeltaProd | 12×M6 GPU | 4×pLSTM CPU | FiLM | 140M | 12-14K | ~2.80 |

---

### Category 5: MULTI-RESOLUTION (S8) — KALEIDOSCOPE-Style

| # | Mix Name | Block Mixer | Scales | FFN | Enhancement | Est. Params | Est. tok/s | Est. Loss |
|---|----------|-------------|--------|-----|-------------|-------------|-----------|-----------|
| 164 | EFLA-Kaleidoscope | M2 | L, L/2, L/4, L/8 | F1 | E0 | ~42M | 30-38K | ~2.95 |
| 165 | KDA-Kaleidoscope | M3 | L, L/2, L/4, L/8 | F1 | E0 | ~44M | 28-36K | ~2.90 |
| 166 | Complex-Kaleidoscope | M5 | L, L/2, L/4, L/8 | F1 | E0 | ~46M | 28-35K | ~2.88 |
| 167 | Griffin-Kaleidoscope | M1 | L, L/2, L/4, L/8 | F1 | E0 | ~42M | 32-40K | ~3.0 |
| 168 | Kaleidoscope+TTT | M2 | 4 scales | F4 | E1 | ~42M | 29-37K | ~2.90 |
| 169 | Kaleidoscope+FiLM | M2 | 4 scales, iterated | F1 | C1 | ~47M | 27-35K | ~2.88 |

---

### Category 6: CROSS-CUTTING MEGA-MIXES (3+ Innovations Combined)

These combine innovations from multiple categories. Higher risk but potentially highest reward.

| # | Mix Name | Structure | Mixers | FFN | Routing | Enhancements | Est. Params | Est. tok/s | Est. Loss | Risk |
|---|----------|-----------|--------|-----|---------|-------------|-------------|-----------|-----------|------|
| 170 | **PANTHEON** | S4 (loop×12) | M5 (Complex) | F4 (TTT) | C5 (surprisal) | E1+E6 (TTT+Diffusion) | 51M | 45-55K / 200K infer | ~2.80 | HIGH |
| 171 | **RAGNAROK** | S4 (loop×12) | M3+M11 (KDA+GAttn) | F4 (TTT) | C5+C7 (surprisal+morph) | E1+E8 | 50M | 35-48K | ~2.78 | HIGH |
| 172 | **PROMETHEUS-II** | S7 (2 attn + loop×10) | M2+M11 | F4 (TTT) | C5 (surprisal) | E1+E5+E8 (TTT+MemCache+Gate) | 67M | 30-42K | ~2.80 | MED |
| 173 | **VALKYRIE-PRIME** | S1 (stacked-16) | M9+M3 gated (RWKV7+KDA) | F1 | C4 (skip) | E0+E4 (MatFormer) | 158M | 15-18K | ~2.82 | LOW |
| 174 | **BIFROST-LOOP** | S4 (loop×12) | M3↔M11 (KDA↔GAttn) | F1 | C3+C7 (Flux+morph) | E8 | 50M | 25-35K | ~2.85 | MED |
| 175 | **AMADEUS-LOOP** | S4 (loop×12) | M4+M13 (Mamba3+Conv) | F1 | C1 (FiLM) | E0 | 55M | 30-38K | ~2.90 | LOW |
| 176 | **AMADEUS-TTT** | S1 (stacked-16) | M4+M13 | F4 (TTT) | C0 | E1 | 158M | ~12K | ~2.85 | LOW |
| 177 | **GORGON-PRIME** | S1 (stacked-16) | M6+M13 (DeltaProd+Conv) | F1 | C0 | E1+E4 (TTT+MatFormer) | 170M | ~11K | ~2.78 | MED |
| 178 | **AEGIS-PRIME** | S4 (loop×adaptive) | M7 (Kalman) | F4 (TTT) | C6 (uncertainty) | E1+E5 (TTT+MemCache) | 47M | 40-58K | ~2.82 | MED |
| 179 | **PHENIX-ORACLE** | S6 (loop adaptive) | M5 (Complex) | F4 (TTT) | C5 (surprisal) | E1 | 50M | 50-60K | ~2.80 | MED |
| 180 | **HELIX-ORACLE** | S6 (loop adaptive) + CPU | M2 GPU + pLSTM CPU | F4 (TTT) | C5 | E1+E7 | 55M | 48-58K | ~2.82 | HIGH |
| 181 | **TIAMAT-LOOP** | S4 (loop×12) | M12 (GSA) | F1 | C0 | E5 (MemCache) | 49M | 34-38K | ~2.88 | MED |
| 182 | **BASILISK-LOOP** | S4 (loop×12) | M3+M5+M11 heads | F1 | C7 (per-iter) | E8 | 50M | 14-25K | ~2.85 | HIGH |
| 183 | **SPECTRUM-TTT** | S2 (stacked-14) | M8 (FFT+corr) | F4 (TTT) | C0 | E1 | 162M | ~17K | ~2.88 | MED |
| 184 | **MEDUSA-KDA** | S4 (loop×12) | M3 (KDA) | F1 | C0 | E6 (Diffusion) | 49M | 33K / 140K inf | ~2.88 | HIGH |
| 185 | **MEDUSA-COMPLEX** | S4 (loop×12) | M5 (Complex) | F1 | C0 | E6 | 51M | 33K / 140K inf | ~2.85 | HIGH |
| 186 | **CHRYSALIS-TTT** | S4 (loop×10) | M2 (EFLA) | F3+E1 (MoE+TTT) | C0 | E1 | 53M | 28-33K | ~2.85 | MED |
| 187 | **CHRYSALIS-KDA** | S4 (loop×10) | M3 (KDA) | F3 (MoE) | C0 | E0 | 55M | 26-32K | ~2.85 | MED |
| 188 | **EREBUS-ENGRAM** | S4 (loop×12) | M2 (EFLA) | F1 | C0 | E2 (Engram) | 52M | 35-40K | ~2.95 | MED |
| 189 | **KDA-FULL-STACK** | S1 (stacked-16) | M3 (KDA) | F1 | C3+C4 (Flux+Skip) | E1+E4+E8 (TTT+MF+Gate) | 165M | ~14K | ~2.80 | MED |
| 190 | **COMPLEX-FULL-STACK** | S1 (stacked-16) | M5 (Complex) | F1 | C0 | E1+E4 (TTT+MatFormer) | 165M | ~11K | ~2.80 | MED |
| 191 | **DELTAPROD-LOOP** | S4 (loop×12) | M6 (DeltaProduct) | F1 | C0 | E0 | 52M | 28-35K | ~2.90 | MED |
| 192 | **DELTAPROD-LOOP-TTT** | S4 (loop×12) | M6 | F4 (TTT) | C0 | E1 | 52M | 27-34K | ~2.85 | MED |
| 193 | **DELTAPROD-ADAPTIVE** | S6 (adaptive) | M6 | F4 (TTT) | C5 (surprisal) | E1 | 52M | 38-50K | ~2.82 | MED |
| 194 | **KALMAN-DIFFUSION** | S4 (loop×12) | M7 (Kalman) | F1 | C6 (uncertainty) | E6 (Diffusion) | 48M | 32K / 130K inf | ~2.88 | HIGH |
| 195 | **RWKV7-LOOP-FULL** | S4 (loop×12) | M9+KDA gating+EFLA | F2 (ReLU²) | C4 (skip) | E0 | 46M | 35-40K | ~2.85 | MED |

---

### Category 7: ENGRAM-ENHANCED MIXES

Engram adds hash-based N-gram knowledge with 7.4x fused kernel. Limited by L2 pressure (~5M params).

| # | Mix Name | Base | With Engram | Est. Params | Est. tok/s | Est. Loss | Notes |
|---|----------|------|-------------|-------------|-----------|-----------|-------|
| 196 | Griffin+Engram-Stack | Tempest | +E2 | 172M | ~12K | ~2.93 | Proven recurrence + knowledge |
| 197 | KDA+Engram-Stack | KDA-16L | +E2 | 167M | ~13K | ~2.83 | Channel-gate + N-gram |
| 198 | EFLA+Engram-Stack | EFLA-16L | +E2 | 163M | ~13K | ~2.87 | Exact + N-gram |
| 199 | AMADEUS+Engram | AMADEUS | +E2 | 163M | ~12K | ~2.85 | SSM hybrid + N-gram |
| 200 | Complex+Engram-Stack | Complex-16L | +E2 | 170M | ~11K | ~2.83 | Rotational + knowledge |
| 201 | KDA+Attn+Engram | BIFROST | +E2 | 170M | ~14K | ~2.80 | Dynamic routing + knowledge |
| 202 | EFLA-Loop+Engram | EREBUS | +E2 | 52M | 35-40K | ~2.95 | Looped EFLA + knowledge |

---

## SUMMARY STATISTICS

**Total unique plausible mixes enumerated: 202**

### By Structure
- Looped (S4-S6): 65 mixes
- Stacked (S1-S3): 82 mixes
- Hybrid (S7): 10 mixes
- Multi-resolution (S8): 6 mixes
- CPU-GPU Cooperative: 9 mixes
- Mega-mixes (cross-category): 26 mixes
- Engram-enhanced: 7 mixes

### By Risk Level
- LOW risk (proven components, simple combination): ~40 mixes
- MEDIUM risk (unproven components, moderate complexity): ~120 mixes
- HIGH risk (novel combinations, untested interactions): ~42 mixes

### Top 10 Highest Expected Value (Quality × Throughput × Feasibility)

| Rank | # | Mix Name | Est. Loss | Est. tok/s | Risk | Why |
|------|---|----------|-----------|-----------|------|-----|
| 1 | 3 | EREBUS+TTT+Surprisal (=ORACLE) | 2.85 | 55-62K | MED | Adaptive depth is the #1 throughput lever |
| 2 | 15 | PHENIX+Surprisal | 2.82 | 50-60K | MED | Complex MIMO + adaptive = quality+speed |
| 3 | 179 | PHENIX-ORACLE | 2.80 | 50-60K | MED | Best of both + TTT |
| 4 | 171 | RAGNAROK | 2.78 | 35-48K | HIGH | Everything + kitchen sink, but grounded |
| 5 | 14 | PHENIX+TTT | 2.83 | 37-41K | LOW | Simple: complex loop + TTT |
| 6 | 39 | EFLA+Mamba3-Loop+TTT | 2.83 | 31-37K | MED | AMADEUS architecture in a loop + TTT |
| 7 | 45 | CHIMERA+Surprisal | 2.80 | 20-35K | HIGH | Architecture morph + adaptive depth |
| 8 | 82 | DeltaProd+Attn 14:2 | 2.80 | ~12K | MED | Strongest quality stacked |
| 9 | 124 | DeltaProd+TTT | 2.78 | ~12K | MED | Absolute quality champion |
| 10 | 177 | GORGON-PRIME | 2.78 | ~11K | MED | DeltaProd + Conv + TTT + MatFormer |

### Top 5 Throughput Champions

| Rank | # | Mix Name | Est. tok/s | Est. Loss | Notes |
|------|---|----------|-----------|-----------|-------|
| 1 | 3 | ORACLE | 55-62K | 2.85 | Adaptive depth with EFLA |
| 2 | 15 | PHENIX+Surprisal | 50-60K | 2.82 | Complex + adaptive |
| 3 | 20 | KDA-Loop+Surprisal | 52-60K | 2.85 | KDA + adaptive |
| 4 | 9 | EREBUS+Skip | 42-48K | 3.15 | Simple skip routing |
| 5 | 55 | KDA+MoE+Surprisal | 38-48K | 2.85 | MoE + adaptive depth |

### Top 5 Quality Champions

| Rank | # | Mix Name | Est. Loss | Est. tok/s | Notes |
|------|---|----------|-----------|-----------|-------|
| 1 | 124 | DeltaProd+TTT | 2.78 | ~12K | Householder rank-2 + adaptation |
| 2 | 171 | RAGNAROK | 2.78 | 35-48K | Multi-mechanism mega-mix |
| 3 | 177 | GORGON-PRIME | 2.78 | ~11K | DeltaProd + Conv + TTT + MF |
| 4 | 110 | DeltaProd+KDA+GatedAttn heads | 2.80 | 9-13K | Triple-head Householder |
| 5 | 82 | DeltaProd+Attn 14:2 | 2.80 | ~12K | Householder + correction attn |

---

## FEASIBILITY NOTES

### Mixes Requiring New Kernels
- All DeltaProduct mixes (#69, 82, 95, 103, 110, 124, etc.): Need deltaproduct_chunkwise scan
- All Kalman mixes (#70, 83, etc.): Need kalman_chunkwise associative scan
- All Complex MIMO mixes (#72, 81, etc.): Need complex_mimo_chunkwise scan
- All FFT mixes (#74, 85, etc.): Use rocFFT (system library, no custom kernel)
- All RWKV-7 mixes (#71, 79, etc.): Use FLA RWKV kernel (verified)

### Mixes Requiring No New Kernels (Fastest to Implement)
- All Griffin-based mixes: Existing vectorized chunked scan
- All EFLA mixes with FLA DeltaNet: Pre-compute alpha, drop into FLA
- All KDA mixes with FLA DeltaNet: Pre-compute alpha, drop into FLA
- AMADEUS variants: Existing mamba-ssm + causal-conv1d
- All Engram mixes: Existing fused_engram_gate_conv kernel
- All TTT mixes: Element-wise outer product, no kernel needed
- All MatFormer mixes: Random prefix sampling, no kernel needed
- All routing mixes: Element-wise gates, no kernel needed

### Implementation Priority (by new kernel requirement)
1. **Zero new kernels:** Mixes #2-12, 23-26, 38-40, 46, 91-98, 114-131, 175-176, 188, 196-202
2. **One new kernel (EFLA→FLA):** Mixes #1, 18-22, 27-29, 41, 47-48, 61-65, 67-68, 77-78, 86, 89, 93, 99, 102, 133, 138, 145-154
3. **Custom scan needed:** Mixes #13-17, 30-35, 50, 53, 57-58, 60, 65, 69-73, 81-84, 87, 94-98, 100-105, 106-113, 127-128, 134, 139-143, 155-163, 164-169, 191-195

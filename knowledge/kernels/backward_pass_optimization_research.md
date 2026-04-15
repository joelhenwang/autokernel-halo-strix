---
title: "Backward Pass Optimization Research"
domain: kernels
type: reference
status: active
related:
  - knowledge/kernels/backward_pass_optimization_results.md
  - knowledge/hardware/amd_rdna35_strix_halo.md
tags: [%backward, %optimization, %research, %fp8, %sampled-softmax]
---

# Backward Pass Optimization Research

**Date:** 2026-04-10
**Context:** Backward pass is 53% of training step time. Research into reducing backward matmul cost.

## Why Backward Has Matmuls (It's the Math)

For any linear layer Y = X @ W.T, the chain rule gives exactly:
- dX = dY @ W (gradient to input) — one GEMM
- dW = dY.T @ X (weight gradient) — one GEMM

These are unavoidable for exact gradients. The LM head is the worst case (V=50257).

## What Works Now (gfx1151)

| Approach | Status | Impact |
|----------|--------|--------|
| IO-aware fused kernels | **ACTIVE** | Best practical approach (our fused kernels: mHC 28.5x, Engram 7.4x) |
| Chunked CE (Approach D) | **TODO** | Save grad_logits during forward, eliminate recompute GEMM in backward |
| Chunked CE (Approach C) | **TODO** | Stream overlap — hide recompute behind gradient GEMMs |
| Sampled softmax (LM head) | Available | 16x smaller LM head GEMMs for early training warmup |

## What's Blocked

| Approach | Why Blocked | When Available |
|----------|-------------|----------------|
| FP8 backward GEMMs | No FP8 tensor support on RDNA 3.5 | RDNA 4 or CDNA (MI300) |
| Hardware sparse tensor | No 2:4 sparsity on RDNA 3.5 | RDNA 4+ |

## Promising Research (Not Yet Implemented)

| Method | Paper | What It Does | Applicability |
|--------|-------|-------------|---------------|
| **INSTANT** | OpenReview | Low-rank projection of activations+gradients for compressed backward | Could reduce LM head weight gradient cost |
| **PAMM** | OpenReview | Approximate Q/K/V projection backward | Useful for PROMETHEUS attention layers |
| **GaLore/Fira** | arXiv:2403.03507 | Low-rank gradient projection for memory savings | Memory opt, not compute |
| **Monarch matrices** | ICML 2022 | Structured matrix parameterization, O(N log N) fwd+bwd | Architecture change for FFN |
| **MatMul-Free LM** | arXiv:2406.02528 | Ternary forward, full backward | Forward savings only, aligns with TERNARY-REFLEX |
| **BitNet b1.58** | arXiv:2504.12285 | 1.58-bit weights, bf16 backward | Validates ternary forward at 2B scale |

## Priority for Implementation

1. **Approach D** — save grad_logits during forward (eliminates recompute GEMM, ~25% backward speedup for chunked CE)
2. **Approach C** — stream overlap (hides remaining latency)
3. **Sampled softmax** — for early training warmup on large vocab
4. **INSTANT** — research prototype for LM head weight gradient compression

## Key Insight

Nobody has eliminated matmuls from the backward pass at scale. Even the most aggressive papers (MatMul-Free LM, BitNet b1.58) keep full-precision backward matmuls. The best practical path remains IO-aware kernel fusion.

See also: `docs/possible_techniques_bwd_improv.md` for the full survey with paper references.

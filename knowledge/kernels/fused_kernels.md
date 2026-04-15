---
title: "Fused Kernels"
domain: kernels
type: results
status: active
related:
  - kernels/hip/fused_mhc_sinkhorn.py
  - kernels/hip/fused_engram_gate_conv.py
  - kernels/hip/fused_ple_gate.py
  - kernels/hip/chunked_linear_cross_entropy.py
  - knowledge/kernels/kernel_benchmarks.md
tags: [%kernels, %fused, %hip, %mhc-sinkhorn, %engram, %ple, %chunked-ce, %gfx1151]
---

# Fused Kernels (2026-04-10)

| Kernel | Speedup | File | Notes |
|--------|---------|------|-------|
| **fused_mhc_sinkhorn** | **28.5x** | `kernels/hip/fused_mhc_sinkhorn.py` | 3 projections + 20-iter Sinkhorn 4x4 in registers. Exact correctness. |
| **fused_engram_gate_conv** | **7.4x** | `kernels/hip/fused_engram_gate_conv.py` | Gate + gated value + depthwise conv1d. Wired into `models/engram.py`. |
| **fused_ple_gate** | ~3-5x (est.) | `kernels/hip/fused_ple_gate.py` | Linear->GELU->Linear->RMSNorm. Wired into `models/ple.py` (mode "a" only). |
| **chunked_linear_cross_entropy** | Memory opt | `kernels/hip/chunked_linear_cross_entropy.py` | Saves 2-12 GB by chunking LM head+CE. 25% slower backward (recomputes logits). |

## Anti-pattern: Matmuls in HIP Kernels

Engram Variants A (hash+gather+gate) and C (full fusion) were 5-7x SLOWER than PyTorch because they put matmuls inside HIP kernels. On gfx1151 without MFMA, never put matmuls in HIP kernels — let rocBLAS handle them.

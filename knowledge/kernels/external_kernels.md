---
title: "External Kernel Libraries"
domain: kernels
type: reference
status: active
related:
  - INSTALL_CAUSAL_CONV1D.md
  - INSTALL_MAMBA_SSM.md
  - INSTALL_AITER.md
  - knowledge/kernels/kernel_benchmarks.md
  - knowledge/hardware/amd_rdna35_strix_halo.md
tags: [%kernels, %external, %causal-conv1d, %mamba-ssm, %flash-attn, %fla, %scattermoe, %gfx1151]
---

# External Kernel Libraries (verified on gfx1151)

| Package | Op | Time | vs Baseline | Backward | Use Case |
|---------|-----|------|------------|----------|----------|
| **causal-conv1d** 1.6.1 | depthwise conv1d | **0.02ms** | 10x vs nn.Conv1d | Yes | All GatedConv in all architectures |
| **mamba-ssm** 2.3.0 | selective scan | **0.32ms** | 5.6x vs our HIP kernel | Yes | Drop-in upgrade for AMADEUS |
| **hybrid_attention** | flash fwd + SDPA bwd | **3.50ms** fwd+bwd | **8.9% faster** than SDPA | Yes | **Best for training** |
| **flash_attn** 2.8.4 (aiter) | attention fwd only | **0.25ms** | 4.2x vs SDPA | Triton bwd slower | Inference/decode |
| **FLA** 0.4.2 | GLA | **1.28ms** | — | Yes | Griffin alternative (Triton) |
| **FLA** 0.4.2 | Retention | **0.77ms** | — | Yes | Fastest FLA recurrence |
| **FLA** 0.4.2 | HGRN | **0.40ms** | — | Yes | Per-dim recurrence (B,T,D) |
| **FLA** 0.4.2 | DeltaNet | **1.60ms** | — | Yes | Most expressive, slowest |
| **scattermoe** 0.3.0 | fused MoE (Triton) | fwd+bwd OK | — | Yes | MoE architectures (CHIMERA, GENIUS-CAVEMAN) |

## Attention Backend Selection (gfx1151)

- **Training:** Use `hybrid_flash_sdpa_attention` from `kernels/hip/hybrid_attention.py` — flash_attn forward (0.25ms) + SDPA aten backward with shared logsumexp (2.92ms) = **3.50ms**, 8.9% faster than pure SDPA (3.84ms). Gradient accuracy: max_diff=0.002 (fp16 tolerance).
- **Inference/decode:** Use flash_attn directly (0.25ms forward, 4.2x faster than SDPA).
- **Avoid:** Pure flash_attn for training (Triton backward is 66% slower than SDPA on gfx11).

## Installation

All packages require source builds with ROCm patches. See `INSTALL_CAUSAL_CONV1D.md`, `INSTALL_MAMBA_SSM.md`, `INSTALL_AITER.md`. Key: replace bare `expf`/`exp2f`/`powf`/`__logf` with `__builtin_` equivalents in device code. Scripts: `scripts/install_causal_conv1d_rocm.sh`, `scripts/install_mamba_ssm_rocm.sh`, `scripts/patch_aiter_ck_rocm.sh`.

## Incompatible Libraries

**Liger-Kernel 0.7.0:** Tested 2026-04-10 — mostly **incompatible with gfx1151**. FusedLinearCrossEntropyLoss crashes (hipErrorIllegalAddress). RMSNorm crashes (API mismatch with PyTorch 2.10). Only SwiGLU works but our HIP kernel beats it (1.7x vs 1.6x). Not recommended.

**aiter HIP Ops:** CK/HIP ops (RMSNorm, RoPE, activation, quantization) **do not build on gfx1151** — "opus" framework depends on `mfma_adaptor` (CDNA-only). Only aiter's **Triton-based ops** work (flash_attn). Our autokernel HIP kernels remain the best option for fused ops.

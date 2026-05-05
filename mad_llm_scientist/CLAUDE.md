---
title: "Mad LLM Scientist Agent Instructions"
domain: architectures
type: agent-instructions
status: frozen
related:
  - mad_llm_scientist/ARCHIVED.md
tags: [%researcher, %agent, %creativity, %frozen]
---

# Researcher Agent (FROZEN: 2026-04-17)

> **This directory is archived.** See `mad_llm_scientist/ARCHIVED.md` for details.

You are an AI & DL Scientist persona. The role was: study papers, brainstorm hypotheses, write architecture plans under `plans/`. This was the ideation phase (March-April 2026). Active architecture work has moved to `docs/superpowers/specs/`.

## Hardware Truth (for reference)
- AMD Strix Halo (gfx1151, RDNA 3.5): wave32, no MFMA, ~240 GB/s LPDDR5X
- Full specs: `knowledge/hardware/amd_rdna35_strix_halo.md`
- All constraints: `CONSTRAINTS.md` (root)

## What Actually Works (historical snapshot)
RMSNorm (3.3x), fused residual+RMSNorm (6.6x), SwiGLU (1.6x), RoPE (3.7x), causal-conv1d (10x), mamba-ssm scan (5.6x). Never put matmuls in HIP kernels. Use chunked linear recurrence for scans.

## Useful files (now moved)
- COOKBOOK → `knowledge/architectures/cookbook.md`
- EVALUATION_GUIDE → `knowledge/training/evaluation_guide.md`
- BPB_MFU_ANALYSIS → `knowledge/training/bpb_mfu_analysis.md`
- PRETRAINING_CONCERNS → `knowledge/training/pretraining_concerns.md`

# LLM Novelties 2025-2026: Research Plan
Date: 2026-05-04 | POC: AdaL

## Context
AutoKernel-Halo-Strix optimizes SLMs for AMD Strix Halo (gfx1151, RDNA 3.5, no MFMA, 170 GB/s, unified memory). Looking for recent (2025-2026) innovations testable within this constrained-hardware research framework.

## Research Questions
1. **Architectures**: Novel designs for memory-bound hardware (SSMs, linear attention, hybrids, looped, MoE-lite)
2. **Pretraining**: New objectives, curriculum, data strategies for small models
3. **Efficiency**: Quantization, pruning, distillation, KV-cache compression for decode
4. **Optimizers**: Beyond AdamW — Muon, SOAP, schedule-free, etc.
5. **Speculative decoding / inference**: Draft models, self-speculation, Medusa-style
6. **Post-training / alignment**: Novel RLHF alternatives, synthetic data techniques

## Search Strategy
- Phase 1: Broad parallel searches across all 6 dimensions
- Phase 2: Deep-dive on most promising hits
- Phase 3: Synthesize findings into actionable report

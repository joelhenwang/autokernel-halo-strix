# LLM Novelties 2025–2026 for AutoKernel-Halo-Strix — Outline
Date: 2026-05-04 | POC: AdaL
Estimated report length: 2200–3000 words

## Goal
Identify recent LLM novelties (2025 → 2026-05-04) that are realistically testable in this repository, especially under Strix Halo constraints: memory-bandwidth-limited decode, strong benefits from kernel fusion, interest in hybrid/looped/SSM architectures, tokenizer efficiency, and low-cost experimentation.

## Sections
1. Executive summary
   - Top recommendations ranked by expected ROI
   - Which are easiest vs highest upside

2. Most promising directions
   - Hybrid distillation via HALO/HypeNet
     - Claims: transformer→hybrid conversion in 2.3B tokens, HyPE, attention-layer selection, Lightning Attention fit
     - Sources: d8c6b467
   - Curriculum pretraining warm-start
     - Claims: 18–45% fewer steps to baseline, up to 3.5% sustained gains, best signals
     - Sources: b2c30756
   - Training-free self-speculative decoding cascades
     - Claims: 1.1x–2.3x speedup over AR, DyTC gains vs static cascade/tree baselines
     - Sources: c16a4fcd, 6d31004e

3. Ranked experiment backlog for this repo
   - 8–10 ideas, each with:
     - what it is
     - why it fits this project
     - expected upside
     - implementation difficulty
     - risks / caveats

4. Poor-fit / lower-priority ideas
   - Where evidence is weaker or hardware fit is poor

5. Concrete next 3 experiments
   - Smallest realistic tests for this repository

## Intended citations
- d8c6b467: HALO / HypeNet / HyPE / hybrid distillation / Lightning Attention
- b2c30756: curriculum pretraining
- c16a4fcd and 6d31004e: CAS-Spec / DyTC / training-free speculative decoding

# Broader Deep Dive Plan for AutoKernel-Halo-Strix
Date: 2026-05-04 | POC: AdaL
TL;DR: Expand the previous novelty scan into a broader, project-grounded research program covering architecture, pretraining, inference, systems, data, evaluation, and experimental prioritization for Strix Halo.

## Context
This project is not just “another LLM repo.” It is a hardware-aware research environment centered on:
- AMD Strix Halo / gfx1151 constraints
- bandwidth-limited decode and strong fusion wins
- custom HIP kernels
- hybrid / looped / recurrent architectures
- small-to-mid scale experimental validation
- tokenizer and data-efficiency work
- pretraining and inference co-design

The next report should therefore go broader than “recent LLM novelties” and answer:
**What research directions are actually worth building here, across the full stack?**

## Research Questions
1. **Architectures**
   - Hybrid attention/RNN/SSM designs
   - Looped / iterative / recurrent depth reuse
   - long-context designs
   - MoE-lite / conditional compute for small systems
   - test-time compute or adaptive-depth mechanisms

2. **Pretraining**
   - curriculum learning
   - tokenization improvements
   - data selection / mixture optimization
   - distillation-first vs scratch training
   - schedule design for small models

3. **Inference / Decode**
   - speculative decoding
   - self-speculation
   - quantization-aware inference
   - KV-cache reduction / elimination
   - adaptive compute during generation

4. **Systems / Hardware Fit**
   - which ideas map well to Strix Halo’s memory hierarchy
   - which ideas are likely defeated by lack of MFMA
   - where kernel fusion can amplify model-level ideas
   - implications for unified memory and local deployment

5. **Optimization / Training Mechanics**
   - optimizer innovations
   - gradient scaling / stability
   - selective fp32 islands
   - compile/autograd/custom-op co-design
   - parameter-efficient stabilization tricks

6. **Evaluation / Methodology**
   - what benchmarks matter for this repo
   - what “good” looks like for small models on this hardware
   - what experiments can falsify an idea quickly
   - what ablations are worth standardizing

7. **Portfolio / Roadmap**
   - quick wins
   - medium-term bets
   - moonshots
   - poor-fit ideas to explicitly avoid

## Expected Deliverables
1. **Broader research report**
   - 3000–4500 words
   - ranked by fit, upside, difficulty, and hardware alignment

2. **Experiment backlog**
   - 12–20 candidate experiments
   - each with scope, expected upside, complexity, dependencies, and kill criteria

3. **Decision matrix**
   - research area × hardware fit × implementation cost × upside

4. **Recommended quarter roadmap**
   - immediate (1–2 weeks)
   - medium (1–2 months)
   - ambitious (quarter-scale)

## Search Strategy
### Phase 1: Expand the landscape
- recent papers/projects (2025 → 2026-05-04)
- architecture, pretraining, inference, systems, optimizers, evaluation

### Phase 2: Project-specific filtering
- keep only ideas that fit:
  - small-team validation
  - custom-kernel environment
  - Strix Halo hardware limits
  - current repo direction

### Phase 3: Synthesis
- connect paper-level ideas to repo-specific experiment designs
- identify interactions across layers:
  - model design × kernel opportunities
  - tokenizer/data × throughput
  - inference methods × bandwidth bottlenecks

## Known Unknowns
- Which optimizer / schedule innovations are strongest for this project class?
- Which post-training or reasoning-time methods matter for small local models?
- Which ideas look exciting on paper but are bad fits for this hardware?
- Which architecture ideas are only viable with larger-scale distillation teachers?

## Sections for Final Report
1. Executive summary
2. What kinds of research this project is uniquely suited for
3. Architecture opportunities
4. Pretraining and data opportunities
5. Inference and systems opportunities
6. Optimization and training opportunities
7. Evaluation methodology upgrades
8. Ranked experiment backlog
9. Recommended roadmap
10. Explicit anti-roadmap: what not to chase

## Success Criteria
A successful report should make it easy to answer:
- “What should we test next?”
- “What should we stop chasing?”
- “Which ideas are actually amplified by this hardware?”
- “Where can AutoKernel-Halo-Strix produce original research rather than just reproductions?”

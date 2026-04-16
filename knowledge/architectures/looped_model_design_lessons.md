---
title: "Looped Model Design Lessons (OUROBOROS -> JORMUNGANDR)"
domain: architectures
type: reference
status: active
tags: [looped-models, parcae, jormungandr, ouroboros, design-lessons, stability]
related:
  - parcae_stable_looped_models.md
  - hypothesis_buildout_results.md
  - ../training/argus_prime_results.md
  - ../training/muon_optimizer_results.md
  - ../training/compressm_in_training_ssm_compression.md
---

# Looped Model Design Lessons

Lessons from designing OUROBOROS and its matured successor JORMUNGANDR, cross-referenced against Parcae's actual code/training recipe and our lab's prior results.

## Critical Findings

### 1. Use Parcae's Actual Optimizer (MuonAdamW), Not AdamW

Parcae's published code uses MuonAdamW with:
- Muon momentum: 0.85 -> 0.95 warmup over 300 steps
- Muon weight decay: linearly decays to 0 over training
- Muon LR ~0.02 for 2D matmul weights, AdamW LR ~8e-4 for everything else

Our own Muon results showed 2x token efficiency. On small token budgets (BabyLM ~16.5M), this is the single highest-impact training recipe change.

### 2. Don't Add Momentum to Parcae's Loop

OUROBOROS proposed adding `velocity = beta * velocity + block_output` across loop iterations. Analysis showed this risks breaking Parcae's stability guarantee:

```
Parcae guarantees: rho(A) < 1 -> h converges
Momentum adds:     velocity accumulates without A-matrix bound
Combined:          velocity can grow unboundedly if CoreBlock is directionally consistent
```

Parcae itself doesn't use momentum and validated stability at 770M-1.3B scale without it. Recommendation: **disable by default, enable only as ablation if loss plateaus.**

### 3. Single-Step TTT, Not Multi-Step (From Scratch)

ARGUS-PRIME B0 showed that multi-step TTT (3 inner gradient steps) caused NaN when trained from scratch. The OUROBOROS plan specified multi-step TTT from the start. JORMUNGANDR corrects this:
- Default: single-step TTT at last Coda GQA layer
- Upgrade path: switch to multi-step after training stability is confirmed (60%+ through training)

### 4. Poisson-Sampled Loop Depth Is Key Regularization

Parcae's code samples loop depth from a Poisson distribution, not a fixed curriculum. This prevents the model from overfitting to a specific iteration count. During training:
- Easy batches might get 4 iterations
- Hard batches might get 12 iterations
- Mean stays at target (e.g., 8)

At inference, use the mean. But the model is robust to variation.

### 5. FiLM Fingerprint Must Be in Gradient-Tracked Iteration

OUROBOROS placed FiLM fingerprint at iteration 4 (detached). With 5 detached iterations (t=1..5), the fingerprint at t=4 never receives direct gradients. JORMUNGANDR moves it to iteration 6 (first gradient iteration), ensuring the fingerprint computation can be directly optimized.

### 6. Shared Core Block Needs Reduced Learning Rate

The core block receives gradients from all gradient-tracked iterations (3 per step). This is effectively 3x gradient accumulation vs Prelude/Coda layers. Without compensation:
- Core block learns 3x faster than surrounding layers
- Creates optimization imbalance
- Fix: 0.5x LR for core block params

### 7. Value Embeddings Are Free Quality

Parcae's code includes per-layer learned embedding tables (50257 x kv_dim) added to attention values. Zero-initialized. Cost: ~3.2M params for 3 GQA layers. This was missing from OUROBOROS.

### 8. L2 Cache Benefit Is Real But Overestimated

OUROBOROS estimated 10x effective bandwidth from L2 for loop iterations 2-8. Reality:
- 240 GB/s LPDDR5X bandwidth is shared CPU+GPU
- During training, optimizer runs on CPU and competes for memory bandwidth
- L2 must hold core block weights + activations + gradient buffers
- Conservative estimate: 3-5x benefit, not 10x
- Throughput: 15-19K tok/s (not 19-23K)

Still faster than AMADEUS (13.2K tok/s). The param efficiency is the real win.

### 9. BabyLM May Be Too Small for Loop Scaling

Parcae's key finding: "compute-optimal training requires increasing BOTH loop depth AND data together." BabyLM's 16.5M tokens may not show benefit from 8 iterations. Expected: quality gains manifest at 100M+ tokens (GPT-training-small or larger).

Dataset funnel for looped models: BabyLM smoke -> GPT-training-small full -> WikiText-103 -> Dolma subset.

### 10. CompreSSM Monitoring Applies to Loop State

Hankel singular value analysis can quantify which loop state dimensions are actually used:
- High energy concentration (top 10% of dims hold 90%+ energy) = most dimensions wasted
- Low effective rank (< d_model/4) = loop underutilized
- If effective rank stabilizes early, CompreSSM balanced truncation can compress mid-training

### 11. Stage Components, Don't Launch Everything At Once

OUROBOROS activated 6 novel components from step 0 (relying on zero-init). JORMUNGANDR uses staged activation:
1. Bare loop (steps 0-15%) — prove Parcae stability
2. + FiLM (15-30%) — prove mid-loop introspection
3. + TTT (30-45%) — prove adaptive Coda
4. Full training (45-100%) — all components
5. (Optional) Upgrade TTT to multi-step
6. (Optional) Enable momentum if plateaued

Each stage has clear success criteria and rollback plan.

### 12. fp16 Not bf16 on gfx1151

Parcae's code defaults to bf16. On our hardware (gfx1151, RDNA 3.5), bf16 is 24% slower than fp16 and torch.compile crashes with bf16. Always use fp16 + GradScaler. This is a known constraint documented in training_antipatterns.md but worth repeating for looped models where the cost multiplies with iterations.

### 13. Randomized Position IDs Prevent Loop Counting

Parcae randomly samples position IDs during training instead of sequential 0..T-1. For looped models, the same RoPE frequencies are applied at every iteration. Sequential positions could let the model learn to "count" iterations via positional patterns — a degenerate shortcut. Randomization prevents this.

## Architecture Comparison

| Model | Unique Params | Effective | Val Loss | tok/s | Stability |
|-------|--------------|-----------|----------|-------|-----------|
| RESONANT-LOOP | 50.7M | ~150M | 3.42 | 15.9K | SCORE (failed) |
| AMADEUS | 157M | 157M | 2.90 | 13.2K | N/A (not looped) |
| TEMPEST | 177M | 177M | 2.98 | 22.3K | N/A (not looped) |
| ARGUS-PRIME | 156M | 156M | ~4.01* | 16.7K | N/A (not looped) |
| JORMUNGANDR (target) | 124M | 341M | < 2.90 | 15-19K | Parcae A-matrix |
| JORMUNGANDR-MINI (target) | 68M | 165M | < 3.10 | 20-26K | Parcae A-matrix |

*ARGUS-PRIME val loss is on GPT-training-small (different dataset), not directly comparable.

## Plans Reference

- `mad_llm_scientist/plans/PARCAE.md` — Research assessment of Parcae
- `mad_llm_scientist/plans/PRE_OUROBOROS.md` — Design decision log (8 forks)
- `mad_llm_scientist/plans/OUROBOROS.md` — Original design
- `mad_llm_scientist/plans/JORMUNGANDR.md` — Matured design with all fixes applied

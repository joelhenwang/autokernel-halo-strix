---
title: "LOTTERY-FORGE"
domain: architectures
type: plan
status: stale
related:
  - mad_llm_scientist/COOKBOOK.md
  - knowledge/architectures/hypothesis_buildout_results.md
tags: [%hypothesis, %plan, %lottery-forge]
---

# LOTTERY FORGE

**Overparameterized Training + Structured Pruning as Architecture Search**

## The Blind Spot

Every architecture plan says "d=1024, 16 layers, ffn=2240." But WHY those dimensions? Because someone hand-picked them. The DATA never got a vote.

**Lottery Forge** trains a 330M model (30% over budget) for 5 minutes, measures which neurons actually matter, then prunes to 250M. The pruning pattern IS the architecture search — each layer keeps its most important neurons. The resulting 250M model is better than hand-designed because every surviving neuron EARNED its place.

**Works with ANY of our architectures.** It's a training strategy, not a model change.

**Evidence:**
- Lottery Ticket Hypothesis (Frankle & Carlin 2019): sparse subnetworks match full network quality
- "Train Large, Then Compress" (Li et al. 2020): pruned large models outperform same-size trained from scratch
- Our twist: prune from LARGER-than-target, not from target-to-smaller

---

## Three Phases (Metallurgy)

### Forge (0-5 min, ~8M tokens)

Train the overparameterized 330M model. Track per-neuron importance.

```python
class ImportanceTracker:
    def __init__(self, model, beta=0.99):
        self.scores = {name: torch.zeros_like(p) for name, p in model.named_parameters()}
        self.beta = beta

    def update(self, model):
        for name, param in model.named_parameters():
            if param.grad is not None:
                instant = param.data.abs() * param.grad.abs()
                self.scores[name] = self.beta * self.scores[name] + (1 - self.beta) * instant
```

- Importance = |weight| x |gradient|, smoothed with EMA (beta=0.99)
- Computed every step (negligible overhead — element-wise on existing tensors)
- After 5 min (~4000 steps), importance scores are stable enough for pruning

### Quench (5-6 min, ~1.5M tokens)

Gradual structured pruning from 330M to 250M over 1000 steps.

```python
def quench(model, importance, step, quench_start, quench_steps=1000):
    progress = min(1.0, (step - quench_start) / quench_steps)

    # Smooth mask: pruned neurons fade from 1 → 0
    for name, param in model.named_parameters():
        if name in prune_targets:
            threshold = importance[name].quantile(progress * prune_fraction)
            mask = (importance[name] >= threshold).float()
            soft_mask = 1.0 - progress * (1.0 - mask)  # smooth transition
            param.data *= soft_mask

    # At progress=1.0: physically remove dead neurons (free memory)
    if progress >= 1.0:
        model = physically_prune(model, importance)
    return model
```

**Structured pruning targets:**

| Dimension | Forge (330M) | After Quench (250M) | How Pruned |
|-----------|-------------|---------------------|------------|
| d_model | 1280 | 1024 | Global: remove 256 least important dims across ALL layers |
| ffn_inner | 2816 | 2240 | **Per-layer:** each layer keeps its own top-2240 FFN neurons |
| d_conv | 800 | 640 | Proportional to d_model reduction |
| d_rec/d_dho | 480 | 384 | Proportional to d_model reduction |
| Embedding | 50257×1280 | 50257×1024 | Slice to keep_dims |

**Key insight:** FFN pruning is NON-UNIFORM per layer. Layer 3 may keep different neurons than layer 12. This is data-driven architecture search — each layer's effective width is determined by the data.

### Temper (6-15 min, ~18M tokens)

Fine-tune the pruned 250M model.

- **Reset optimizer states** — Adam momentum/variance are stale for pruned dimensions
- **LR warmup:** 50 steps from 0 → 8e-4, then cosine decay to 8e-5
- **Continue training** — the pruned model retains learned features from 330M phase
- **Compatible with Self-Curriculum** — Temper phase can use self-curriculum sampling

---

## Forge Configuration (330M)

| Parameter | Forge Value | After Pruning |
|-----------|------------|---------------|
| d_model | 1280 | 1024 |
| n_layers | 16 | 16 (preserved) |
| ffn_inner | 2816 | 2240 (varies per layer) |
| d_conv | 800 | 640 |
| d_rec/d_dho | 480 | 384 |
| Embedding | 50257×1280 = 64.3M | 50257×1024 = 51.5M |
| Engram tables | Same size | Same size (NOT pruned) |
| MTP adapters | Scaled to d=1280 | Sliced to d=1024 |
| **Total params** | **~330M** | **~250M** |

---

## Token Budget

| Phase | Duration | Model Size | Throughput | Tokens |
|-------|----------|-----------|------------|--------|
| Forge | 5 min | 330M | ~1.6M tok/min | ~8M |
| Quench | 1 min | 330M→250M | ~1.6M tok/min | ~1.5M |
| Temper | 9 min | 250M | ~2M tok/min | ~18M |
| **Total** | **15 min** | | | **~27.5M** |

Standard 250M training: ~29M tokens. Lottery Forge: ~27.5M tokens. Small token loss (~5%) in exchange for data-driven architecture selection.

---

## Compatibility

| Strategy | Compatible | Integration |
|----------|-----------|-------------|
| Self-Curriculum | **Yes** | Forge uses uniform sampling; Temper uses self-curriculum |
| Knowledge Distillation | **Yes** | KD during Temper phase (pruned model from teacher) |
| Any architecture | **Yes** | Width pruning works on conv, recurrence, FFN, any mixer |
| Engram | **Yes** | Hash tables NOT pruned (no "neurons" to remove) |
| MTP | **Yes** | MTP adapters sliced along with d_model |
| Phase training | **Yes** | Forge IS a phase; architecture's phases run during Temper |

---

## Combined Schedule: Lottery Forge + Self-Curriculum

| Time | Phase | Model | Data Sampling |
|------|-------|-------|--------------|
| 0-2 min | Forge + Knowledge Prime | 330M | Wikipedia/textbook (factual) |
| 2-5 min | Forge + Warm-up | 330M | Uniform, difficulty scoring begins |
| 5-6 min | Quench | 330M→250M | Uniform (pruning needs stable gradients) |
| 6-15 min | Temper + Self-Curriculum | 250M | Alpha 0→1.5, hard sequences 3× exposure |

---

## Risks & Mitigations

| Risk | Severity | Mitigation |
|------|----------|------------|
| 5 min at 330M not enough for meaningful importance | HIGH | EMA β=0.99 stabilizes fast; importance is relative ranking, not absolute |
| Optimizer reset at Temper causes instability | MEDIUM | 50-step LR warmup; smooth Quench transition prepares model |
| Non-uniform FFN widths complicate kernel fusion | MEDIUM | Pad pruned layers to nearest multiple of 64; or enforce uniform pruning |
| d_model pruning requires slicing ALL weight matrices | MEDIUM | One-time cost at Quench end; pre-compute slice indices |
| 330M training is 30% slower per step | LOW | Only 5 min at 330M; total token loss is ~5% (27.5M vs 29M) |

## Success Criteria

1. Pruned 250M outperforms from-scratch 250M at equal total training time
2. Non-uniform FFN widths show meaningful variation (std > 100 neurons across layers)
3. Training stable through Quench (loss spike < 2× baseline)
4. Total wall-clock stays within 15-minute budget
5. At least 3 layers have meaningfully different FFN widths (architecture was "searched")

## Implementation Roadmap

1. Implement ImportanceTracker (EMA of |w|×|grad|)
2. Implement structured_prune() — global d_model + per-layer FFN
3. Implement smooth Quench mask (linear fade over 1000 steps)
4. Implement physical_prune() — actually remove neurons and free memory
5. Implement optimizer state reset + LR warmup for Temper phase
6. Integrate with training loop (any architecture)
7. Add importance visualization to eval framework (which neurons survive?)
8. Test: compare pruned-from-330M vs trained-from-scratch-250M on GPT-2, measure loss difference

---

## Hardware Optimization Notes (Strix Halo gfx1151)

> Added 2026-04-09 based on AMADEUS implementation findings.

### Overparameterized Training (330M)
330M model in training: ~14-15 GB memory (based on AMADEUS 243M = 12.7 GB, linear scaling). Fits easily in 116 GB.

### Pruning Phase
Structured pruning (330M → 250M) is a one-time CPU operation — no GPU optimization needed. But verify that pruned model still matches autokernel patterns (RMSNorm, SwiGLU structure must survive pruning).

### Post-Pruning Fine-Tuning
The 250M pruned model can use all autokernel optimizations:
- `autokernel.optimize(pruned_model, training=True)` for HIP kernels
- Chunked linear recurrence for any SSM/recurrence components

### Throughput
- **330M training:** ~5-6K tok/s (slightly larger than AMADEUS 243M)
- **250M fine-tuning:** ~6-8K tok/s (same as AMADEUS range)
- **Token budget:** 330M phase needs more time — budget 60-90 min, not 15

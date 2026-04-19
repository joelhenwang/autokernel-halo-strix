---
title: "Parcae: Stable Looped Language Models"
domain: architectures
type: reference
status: active
tags: [parcae, looped-models, parameter-efficiency, stability, ssm, together-ai]
paper: "Parcae: Stable Looped Language Models (Together AI, arXiv:2604.12946, 2026)"
code: "https://github.com/sandyresearch/parcae"
related:
  - hypothesis_buildout_results.md
  - ../training/compressm_in_training_ssm_compression.md
  - ../training/muon_optimizer_results.md
---

# Parcae: Stable Looped Language Models

## What It Is

Parcae is the first stable looped language model. It reuses a small "core block" of transformer layers multiple times (8-16 iterations), achieving 2x parameter efficiency: a 770M Parcae matches a 1.3B standard transformer in quality.

The key innovation is applying SSM-style stability constraints (spectral norm ρ(A) < 1) to prevent the feedback amplification that killed all prior looped models (including our RESONANT-LOOP).

## Architecture: Prelude / Core Loop / Coda

```
Tokens -> Prelude (unique layers, run once)
       -> Core Block (shared layers, run T times with input re-injection)
       -> Coda (unique layers, run once)
       -> Predictions
```

**Recurrence update (iterations t ≥ 1 only):**
```
h_{t+1} = A * h_t + B * input_embed + CoreBlock(h_t, input_embed)
A = diag(-exp(log_A))    -> eigenvalues in (-1, 0) -> guaranteed stable
B = diag(exp(log_B))     -> controls input injection strength
```

**First iteration (t=0):** Pass `h` directly to CoreBlock — skip the A/B injection. When `h == input_embed` (always true at loop entry), `A*h + B*h = (A+B)*h = 0` because `A = -exp(c)` and `B = exp(c)` are initialized symmetrically. Gradient symmetry prevents divergence during training. The injection only has semantic meaning on re-entry where `h_t ≠ input_embed`.

## Key Results

| Scale | Transformer PPL | Parcae PPL | Benchmark (Core) |
|-------|----------------|------------|------------------|
| 140M | 21.48 | 19.06 | 14.04 vs 13.00 |
| 370M | 15.79 | 14.49 | 20.00 vs 17.46 |
| 770M | 13.08 | 12.49 | 25.07 vs 22.42 |
| 1.3B | 11.95 | 11.42 | 28.44 vs 25.45 |

- 770M Parcae approaches 1.3B transformer quality (~2x param efficiency)
- 6.3% lower perplexity vs RDM (prior looped baseline)
- First scaling laws for looping: optimal compute requires increasing depth AND data together

## Training Recipe (from actual code)

| Parameter | Value |
|-----------|-------|
| Optimizer | **MuonAdamW** (not plain AdamW) |
| Muon momentum | 0.85 -> 0.95 (warmup 300 steps) |
| Muon weight decay | Linear decay from base to 0 |
| LR schedule | Cosine/trapezoid with warmup + cooldown |
| Precision | bf16 mixed (CAUTION: bf16 is 24% slower on gfx1151, use fp16) |
| Distributed | FSDP (Full/Hybrid/GradOp sharding) |
| Loss head | Fused cross-entropy (CCE, HHE, or full-Triton variants) |
| Batch size | Ramps linearly from small to target |
| Data | FineWeb-Edu with sequence packing |
| Loss guardrail | After 10B tokens, loss > 6 -> auto-terminate |

### Detached Iterations (Memory Saving)

```
First N iterations: torch.no_grad() — cheap forward only, no gradient
Last K iterations:  with gradients — expensive, learning happens

Example: 12 total = 8 detached + 4 gradient
Memory scales with K, not N+K. Backward cost is only K iterations.
```

### Loop Depth Sampling

Not fixed — sampled from Poisson distribution each batch:
```python
n_detached ~ Poisson(mean_detached)           # varies per batch
n_grad     ~ Uniform(1, 2 * mean_grad + 1)   # or fixed
```

Variants: poisson-bounded, poisson-full, poisson-truncated-full, fixed.
Curriculum: `1 - sqrt(1 - progress)` ramp (fast early, slows later).
Granularity: per-batch, per-sequence, or per-token.

### Value Embeddings

Per-layer learned embedding tables added to attention values:
```python
# Each GQA layer can have its own vocab -> kv_dim embedding
ve = ValueEmbedding(vocab_size, kv_dim)  # zero-initialized
# In attention: V = V + ve(input_ids)
```

Sparse (not every layer gets them). Near-zero cost. Gives each attention layer a vocabulary-aware bias.

### Randomized Position IDs

During training, position IDs are randomly sampled (not sequential):
```python
position_ids = torch.sort(torch.randint(0, max_pos, (seq_len,)))[0]
```

Prevents the model from memorizing absolute positions or counting loop iterations via positional patterns.

## Relevance to Our Lab

### Connection to RESONANT-LOOP

RESONANT-LOOP (our shared-block architecture) achieved 15.9K tok/s but only val 3.42. It used SCORE damping `h = (1-d)*h + d*F(h)` — a heuristic stability mechanism that was insufficient. Parcae is what RESONANT-LOOP was trying to be, with stability mathematically solved via the SSM A-matrix constraint.

### Hardware Fit (Strix Halo)

Looping is ideal for our hardware:
- **L2 Cache (6MB):** A small core block (~2-3MB fp16) fits in L2. Iterations 2-8 read from cache.
- **No MFMA:** ShortConv (element-wise + conv1d) doesn't need matrix cores. Loop iterations are bandwidth-bound, benefiting from L2.
- **Unified memory:** No GPU-CPU transfer overhead for loop state.

### Connection to CompreSSM

CompreSSM's Hankel singular value analysis applies to loop state monitoring. If HSV monitoring shows low effective rank in the loop state, CompreSSM balanced truncation could compress the loop state dimension mid-training.

### Key Differences from Our Prior Work

| | RESONANT-LOOP | LAZARUS (TTT) | Parcae |
|--|--------------|---------------|--------|
| What loops? | Layers rerun | Weights change per chunk | Layers rerun |
| What's shared? | All block layers | Nothing | Core block layers |
| Stability | SCORE damping (heuristic) | N/A | SSM A-matrix (proven) |
| Param savings | High (shared block) | None | High (2x efficiency) |
| Scale tested | 50.7M | 170M | 140M - 1.3B |

## Code Notes

- Official implementation: **JAX** (not PyTorch)
- Repository: `github.com/sandyresearch/parcae`
- Models on HuggingFace: `together-ai/Parcae` (140M, 370M, 770M, 1.3B)
- AMD ROCm: Training script detects `torch.version.hip` for device counting — ROCm-aware
- Compile: Selective pre-DDP compilation of individual blocks (prelude, core, coda separately)
- FSDP wraps at the Block level, same as standard transformer training

## Scaling Law Finding

Parcae establishes the first scaling laws for looped models: compute-optimal training requires increasing BOTH loop depth AND training data together. This means:
- Small datasets (e.g., BabyLM ~16.5M tokens) may not benefit from 8+ iterations
- The scaling benefit manifests at 100M+ token scale
- Optimal mean recurrence increases with compute budget

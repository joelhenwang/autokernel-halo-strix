# PARCAE — Research Assessment

**Paper:** "Parcae: Stable Looped Language Models" (Together AI, 2026)
**Blog:** https://www.together.ai/blog/parcae
**Code:** https://github.com/sandyresearch/parcae

---

## What is Parcae? (Explained for Beginners)

### The Problem

Imagine building a text-prediction machine. The standard approach (a Transformer) is like building a **skyscraper** — you stack 32 floors, each with its own furniture, its own plumbing, its own electrical wiring. Every floor is unique. More floors = smarter model, but also = more materials (parameters = memory = cost).

**The dream:** What if you could build ONE really good floor and just... USE IT MULTIPLE TIMES? Like an elevator that goes through the same floor over and over, but each pass refines the result. You'd need way fewer materials but could still build a "tall" building.

**The problem:** Every time someone tried this, the building COLLAPSED. When you loop through the same layer repeatedly, small errors compound — like a microphone pointed at a speaker creates FEEDBACK SCREECH. The numbers blow up. Training fails.

Parcae solves the feedback screech.

---

## The Architecture

Parcae has 3 parts:

```
INPUT TOKENS
    │
    ▼
┌─────────┐
│ PRELUDE │  ← Regular transformer layers (run ONCE)
│ (entry) │     Converts tokens into a "thinking space"
└────┬────┘
     │
     ▼
┌─────────────────────────┐
│     CORE BLOCK          │ ◄──── THIS PART LOOPS!
│  (a few transformer     │       Same layers, run 8-16 times
│   layers, reused)       │       Each loop: inject the original input again
│                         │       h_new = A·h + B·input + CoreBlock(h + input)
└────────┬────────────────┘
         │  ↻ loop T times
         ▼
┌─────────┐
│  CODA   │  ← Regular transformer layers (run ONCE)
│ (exit)  │     Converts "thinking space" into predictions
└────┬────┘
     │
     ▼
PREDICTED NEXT TOKEN
```

**In plain English:**
1. **Prelude** = "Look at the words and understand them" (runs once)
2. **Core Block** = "Think about it... think harder... think HARDER..." (runs 8-16 times)
3. **Coda** = "Okay, I've thought enough, here's my answer" (runs once)

---

## Why Doesn't It Explode? (The Key Innovation)

Every time the core block loops, it updates the hidden state:

```
h_new = A × h_old + B × input + CoreBlock(h_old, input)
```

The `A` matrix controls how much the old state carries forward:
- If A is too big → signal AMPLIFIES each loop → explosion
- If A is too small → model forgets everything → useless

**Parcae's fix:** Force A to be a negative diagonal matrix:

```python
A = diag(-exp(log_A))
```

This GUARANTEES that the eigenvalues (think of them as "amplification factors") are always less than 1 in magnitude. Mathematically provable stability. No more screech.

This is borrowed from **SSM theory** (state space models, like Mamba) — the same math that makes linear recurrences stable. Parcae applies it to LOOPED transformers.

**Key math:**
- For discrete LTI systems, stability requires the spectral radius ρ(A) < 1
- A = diag(-exp(θ)) where θ is learned → all eigenvalues are in (-1, 0)
- The B matrix uses similar parameterization for the input injection
- Discretization follows ZOH (Zero-Order Hold) or Euler scheme from SSM literature

---

## The Training Trick: Detached Loops

The memory-saving genius. During training:

- **First N loops**: Run WITHOUT tracking gradients (cheap, just forward pass)
- **Last K loops**: Run WITH gradient tracking (expensive, but needed for learning)

Example with 12 total loops: 8 loops detached (fast) + 4 loops with gradients (learning). The model "thinks" for 12 steps but only "learns" from the last 4. This means:

1. Memory usage scales with K (gradient iterations), not N+K (total iterations)
2. Forward pass cost is N+K, but backward pass cost is only K
3. The detached iterations still contribute to the FORWARD result — they just don't generate gradients

**Implementation from the code:**

```python
# Phase 1: No-gradient iterations (cheap forward only)
with torch.no_grad():
    for step in range(num_steps_no_grad):
        x = update_recurrent_state(x, input_embeds, ...)

# Phase 2: Gradient-tracked iterations (expensive, learning happens)
for step in range(num_steps_with_grad):
    x = update_recurrent_state(x, input_embeds, ...)
```

They also use **curriculum learning**: start with fewer loops early in training, gradually increase:

```
Training progress:  0%  → 20%  → 40%  → 100%
Loop count:         2   →  4   →  6   →   8
```

The model learns to walk before it learns to run.

---

## Training Recipe (from the code)

| Parameter | Value |
|-----------|-------|
| **Optimizer** | MuonAdamW (hybrid optimizer with momentum warmup) |
| **Muon momentum** | 0.85 → 0.95 warmup over 300 steps |
| **LR schedule** | Configurable: linear, cosine, constant, trapezoid |
| **Warmup** | Configurable (fraction or absolute steps) |
| **Cooldown** | Configurable |
| **Precision** | bf16 mixed precision |
| **Gradient clipping** | Configurable max norm |
| **Distributed** | FSDP (Full Shard, Grad Shard, Hybrid Shard) |
| **Loss heads** | Fused cross-entropy: CCE, HHE, or full-Triton |
| **Batch size** | Ramps linearly from small to target |
| **Data** | FineWeb-Edu, with sequence packing |
| **Checkpointing** | Stateful — saves exact data position for perfect resumption |
| **Monitoring** | WandB: loss, PPL, throughput, MFU, memory, gradients |
| **Safety** | After 10B tokens, loss > 6 → training terminates |
| **Non-finite handling** | Skip or halt on NaN/Inf gradients |

### Curriculum for Recurrence Depth

```python
# Schedules: "constant", "linear", "1-sqrt"
# "1-sqrt" follows 1 - sqrt(1 - progress) — fast ramp early, slows later
# Can target forward-only depth, backprop depth, or both
```

### Iteration Depth Sampling

The number of loops isn't fixed — it's SAMPLED each batch:

```python
# Poisson sampling: n ~ Poisson(mean_recurrence)
# Bounded or unbounded variants
# Per-batch, per-sequence, or per-token granularity
```

This regularization prevents the model from overfitting to a specific loop count.

---

## Results

| Scale | Model | Val. Perplexity | Core Benchmark | Core-Extended |
|-------|-------|----------------|----------------|---------------|
| 140M | Transformer | 21.48 | 13.00±0.15 | 8.80±0.21 |
| **140M** | **Parcae** | **19.06** | **14.04±0.20** | **9.67±0.28** |
| 370M | Transformer | 15.79 | 17.46±0.03 | 11.71±0.22 |
| **370M** | **Parcae** | **14.49** | **20.00±0.06** | **12.75±0.31** |
| 770M | Transformer | 13.08 | 22.42±0.20 | 14.20±0.63 |
| **770M** | **Parcae** | **12.49** | **25.07±0.33** | **15.19±0.43** |

**Key finding:** 770M Parcae ≈ 1.3B Transformer in quality. Roughly **2× parameter efficiency.**

### Additional Claims

- **6.3% lower validation perplexity** vs RDM (prior looped baseline — "Recurrence over Depth Models")
- **First scaling laws for looping**: compute-optimal training requires increasing BOTH looping depth AND data together
- Robust across hyperparameters where RDMs diverged (training instability was the #1 blocker)

---

## Interesting Technical Details

### State Initialization Options

The code supports multiple ways to start the hidden state before looping:

| Method | Description |
|--------|-------------|
| `"zero"` | Start from zeros — loop must build everything from scratch |
| `"embed"` | Scaled random based on embedding dimension |
| `"like-init"` | Truncated normal matching the embedding init distribution |
| `"normal"` | Standard random normal |
| `"unit"` | Unit-normalized random (all states start same magnitude) |

### Value Embeddings

An interesting per-layer technique: each layer can have its own ADDITIONAL embedding table that's added to the value projection in attention. These are sparse (not every layer gets them) and selected by a `has_ve()` predicate.

### Monitoring

The code tracks fascinating metrics about the recurrence:
- **Token correlation**: How similar are hidden states across token positions (high = collapsed, bad)
- **Recurrent state norm**: How big does h get over iterations
- **Recurrent residual**: How much does each iteration change h
- **Relative residual**: Residual / state norm — convergence indicator

### Logit Softcap

```python
logits = logit_softcap * tanh(logits / logit_softcap)
```

Prevents extreme logit values — a technique from Gemma/PaLM that helps stability, especially important when the recurrent state might produce unusual magnitude outputs.

---

## What This Means For Our Lab

### Parcae is NOT the same as In-Place TTT (LAZARUS)

| | LAZARUS / In-Place TTT | Parcae (Looped) |
|--|----------------------|----------------|
| **What loops?** | The WEIGHTS change (ΔW updates) | The LAYERS rerun (same weights) |
| **What's shared?** | Nothing — each layer is unique | Core block — same weights, multiple passes |
| **Param savings** | Minimal (~2-5M TTT overhead) | Massive (core block reused 8-16×) |
| **Innovation** | Adaptive FFN per document | Stable looped computation |
| **Memory savings** | None (extra ΔW state) | Huge (fewer unique params) |
| **Closest to us** | LAZARUS, ARGUS-PRIME plans | **RESONANT-LOOP** (our shared-block architecture) |

### Connection to RESONANT-LOOP

RESONANT-LOOP was our 50.7M model that hit 15.9K tok/s but only val 3.42 (quality-limited). It used:
- SCORE damping: `h = (1-d)·h + d·F(h)` — heuristic stability
- ACT halting: adaptive computation time

Parcae is what RESONANT-LOOP was TRYING to be, with the stability math SOLVED:
- SSM-style A-matrix: `A = -exp(log_A)` — mathematically guaranteed stability
- Proven at 770M scale to match 1.3B transformers

### Strix Halo Hardware Fit

The looping idea is PERFECT for our hardware:
- **L2 Cache (6MB)**: A small core block (~3-5MB) fits in L2. Iterations 2-8 read from cache at 10× effective bandwidth.
- **No MFMA**: ShortConv (element-wise + conv1d) doesn't need matrix cores. Loop iterations are pure memory-bandwidth workloads that benefit from L2.
- **Unified memory**: No GPU-CPU transfer overhead for the loop state.

### Potential New Architecture Direction

Combine:
1. LFM2.5's ShortConv/GQA block structure (proven, simple)
2. Parcae's stable looping (2× param efficiency)
3. Our additions: TTT sniper, FiLM, momentum

= A tiny model (~80-120M unique params) with massive effective capacity (~175-340M equivalent) that runs at high throughput because the core block is L2-cached.

---

## References

- Parcae blog: https://www.together.ai/blog/parcae
- Parcae code: https://github.com/sandyresearch/parcae
- Models on HuggingFace: together-ai/Parcae (140M, 370M, 770M, 1.3B)
- Prior work: RDM (Recurrence over Depth Models), Universal Transformers
- SSM stability theory: Gu et al. (S4, Mamba) — eigenvalue constraint on recurrence

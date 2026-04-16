# PRE-OUROBOROS — Design Decision Log

**The journey from "Parcae looks cool" to OUROBOROS: every fork in the road, every option considered, and what we left on the table.**

This document records the brainstorming process that led to OUROBOROS. Each section is a design decision, with the options considered, the choice made, and the reasoning. Future architects can revisit any of these forks.

---

## Decision 1: The Core Innovation Axis

**Context:** We had just studied Parcae (stable looped transformers, 2× param efficiency) and noticed the connection to our own RESONANT-LOOP and the LFM2.5 architecture.

### Options Considered

| Option | Description | Pros | Cons |
|--------|------------|------|------|
| **(A) Griffin + TTT ("Compile-friendly LAZARUS")** | Take compile-optimized Tempest (22K tok/s), add 2-4 TTT layers. | Fast baseline (22K), proven Griffin. | Griffin scan still fragments compile. TTT is incremental, not transformative. |
| **(B) Distill AMADEUS into Griffin** | Train AMADEUS as teacher, distill into fast Griffin student. | Best of both quality + speed. | Need to train TWO models. Distillation adds complexity. |
| **(C) Better training recipe only** | Don't change architecture. Longer sequences, curriculum, better data. | Zero architecture risk. | Might not be enough. Architecture ceiling exists. |
| **(D) Parcae-style looping on LFM2.5 skeleton** | Loop ShortConv blocks with Parcae stability. Tiny params, huge effective depth. L2-cached. | 2-3× param efficiency. Hardware-perfect. Novel. | Unproven combination. Looping + ShortConv never tested. |

**Choice: (D)** — after user pushed for LFM2.5 inspiration and we discovered Parcae.

**What could still be tried:** (B) remains a strong option for later — if OUROBOROS produces a good model, it could serve as a distillation TARGET for an even smaller model. (C) should be tried on whatever architecture wins — training recipe improvements are orthogonal.

---

## Decision 2: What Goes Inside the Loop?

**Context:** The core block runs 8× — every inefficiency is multiplied 8×. LFM2.5's cost chart showed ShortConv (1.0×) is much cheaper than GQA (1.9×).

### Options Considered

| Option | Description | Pros | Cons |
|--------|------------|------|------|
| **(A) Pure ShortConv loop** | Only ShortConv + SwiGLU inside the loop. All attention in Prelude/Coda. | Cheapest operator (1.0×). L2-cacheable. Compile-uniform. | No global context inside the loop. 8 iterations of LOCAL-only mixing. |
| **(B) Mixed loop (ShortConv + 1 GQA)** | 2-3 ShortConv + 1 GQA per loop iteration. Global + local every iteration. | Global context every iteration. More expressive. | GQA inside loop = 8× attention cost (~28ms). Attention weights may not fit L2. |
| **(C) Alternating loop** | GQA fires only on even iterations (2,4,6,8). Odd = pure ShortConv. | Half the attention cost of (B). Still 4 effective attention passes. | Branching complexity hurts compile. Non-uniform iterations. |

**Choice: (A)** — pure ShortConv inside the loop.

**Reasoning:**
1. ShortConv is 1.9× cheaper than GQA per LFM2 profiling — amplified 8× inside the loop
2. ShortConv weights (~2.8MB for 3 layers) fit in L2 cache. Attention KV projections might not.
3. Compile loves uniformity — same block × 8 iterations = perfect fusion target
4. Global context is handled by GQA in Prelude (1 layer) and Coda (2 layers)

**What could still be tried:**
- **(B) with hybrid_flash_sdpa** — if OUROBOROS quality suffers from lack of in-loop global context, adding 1 GQA per iteration is the fix. The 28ms cost is significant but not fatal.
- **(C) with torch.library custom op** — register the iteration as an opaque op, let compile handle each variant independently. Eliminates the branching problem.
- **FLA HGRN inside the loop** — 0.40ms per-dimension recurrence, Triton-based. Could add long-range within the loop at minimal cost. Not attention, but not purely local either.

---

## Decision 3: Overall Structure (Prelude / Loop / Coda Sizing)

**Context:** How many layers in each zone? The loop is the workhorse, but Prelude and Coda need enough capacity to convert in/out of the loop's representation space.

### Options Considered

| Option | Prelude | Core Loop | Coda | Total Unique | Effective Depth |
|--------|---------|-----------|------|-------------|----------------|
| **(A) Thin-Fat-Thin** | 2 (1 Conv + 1 GQA) | 3 Conv × 8 loops | 4 (2 Conv + 2 GQA + TTT) | 9 layers | 30 |
| **(B) Balanced** | 3 (2 Conv + 1 GQA) | 2 Conv × 8 loops | 5 (2 Conv + 3 GQA + TTT) | 10 layers | 26 |
| **(C) Heavy Prelude** | 4 (2 Conv + 2 GQA) | 2 Conv × 8 loops | 2 (1 Conv + 1 GQA + TTT) | 8 layers | 24 |

**Choice: (A)** — Thin Prelude, Fat Loop, Thin-but-Armed Coda.

**Reasoning:**
1. The loop IS the model. 3 × 8 = 24 effective layers of iterative refinement. Make it fat.
2. Prelude just converts tokens to latent space. 1 ShortConv + 1 GQA is sufficient for initial representation with one global pass.
3. Coda is the "sniper nest" — 2 GQA layers for final global context, TTT for adaptive prediction, FiLM modulation. All precision weapons concentrated at the point of maximum impact.
4. FiLM fingerprint at loop iteration 4 is novel — the model introspects its own thinking process.

**What could still be tried:**
- **(B) with 3 Coda GQA layers** — if OUROBOROS needs more global context for quality. The cost is ~3.5ms × 1 extra GQA = minor.
- **(C) Heavy Prelude** — if the loop struggles to learn without a strong initial representation. The bet is that MORE initial processing helps the loop converge faster.
- **Asymmetric core block** — instead of 3 identical ShortConv layers, use 2 ShortConv + 1 wider/different layer. The loop is still shared but each iteration has an internal hierarchy.
- **Progressive widening** — Prelude at d=768, loop at d=512 (fits L2 better), Coda at d=768. Requires projection layers but maximizes L2 utilization.

---

## Decision 4: Stability Mechanism

**Context:** This is what killed RESONANT-LOOP (val 3.42, SCORE damping was too heuristic). Parcae's A-matrix constraint is the key innovation. But we have additional tools.

### Options Considered

| Option | Description | Pros | Cons |
|--------|------------|------|------|
| **(A) Pure Parcae: A·h + B·e** | `h = diag(-exp(log_A))·h + diag(exp(log_B))·e + CoreBlock(h+e)`. A,B learned per-dimension. | Mathematically proven stable. Parcae validated at 770M. | Purely linear injection. No interaction between h and e before CoreBlock. |
| **(B) SCORE + Parcae A** | `h = (1-d)·h + d·CoreBlock(h+e)` with A-constraint inside CoreBlock. | Double safety net. | SCORE already failed quality-wise for us. Adding A inside CoreBlock is redundant. |
| **(C) Parcae + Momentum across loops** | Parcae injection `h = A·h + B·e` plus `velocity = β·velocity + block_output; h = h + velocity`. Momentum gives loop iterations INERTIA. | Stability (Parcae) + coherence (momentum). Novel combination. The loop CONVERGES smoothly. | More complex. Momentum adds state. Untested combination. |

**Choice: (C)** — Parcae injection + momentum across loop iterations.

**Reasoning:**
1. Parcae provides mathematical stability (eigenvalues bounded < 1)
2. Momentum provides directional coherence (iterations that agree amplify, oscillations dampen)
3. Together: a DAMPED HARMONIC OSCILLATOR in the loop dimension
   - Spring constant = (1 - |A|) → restoring force
   - Mass = 1/(1-β) → inertia
4. The physics analogy isn't just poetic — it predicts faster convergence with fewer iterations

**What could still be tried:**
- **(A) Pure Parcae** as a simpler ablation baseline — if momentum doesn't help, drop it
- **Learned injection method** — instead of additive A·h + B·e, use a small MLP: `h = MLP(h, e)` with spectral normalization. More expressive but harder to guarantee stability.
- **Per-iteration A** — instead of one global A, learn different decay rates per loop iteration. Early iterations could have higher A (more memory), later iterations lower A (more input). Parcae's code already supports this via the adapter mechanism.
- **DeltaNet-style injection** — `h = h + ΔW·e` where ΔW is a running outer-product (like TTT but for the injection, not the FFN). Combines looping stability with TTT-style adaptation.

---

## Decision 5: Where Does FiLM Go?

**Context:** FiLM conditioning (from AMADEUS, val 2.90 recipe) needs a fingerprint point and modulation targets. In a looped architecture, the fingerprint could happen at various points.

### Options Considered (discussed implicitly)

| Option | Fingerprint Point | Modulates | Notes |
|--------|-------------------|-----------|-------|
| **Before the loop** | End of Prelude | All 8 loop iterations + Coda | Early context, but from only 2 layers of processing |
| **Mid-loop (iteration 4)** | After 4th loop iteration | Iterations 5-8 + Coda (7 targets) | Novel: introspects the loop's own refinement process |
| **After the loop** | Start of Coda | Coda layers only (4 targets) | Rich context but doesn't help the loop at all |
| **Multiple points** | Iterations 2, 4, 6 | Subsequent iterations | More fingerprints = richer but more compute |

**Choice: Mid-loop (iteration 4)** — fingerprint halfway through the loop.

**Reasoning:** The model sees 2 Prelude layers + 4 loop iterations (= 14 effective layers) before fingerprinting. This is rich context. The fingerprint then modulates the remaining 4 iterations + 4 Coda layers. The model literally watches its own thinking process evolve and adjusts the second half accordingly.

**What could still be tried:**
- **Dual FiLM** — fingerprint at iteration 2 (early) AND iteration 6 (late). Two levels of introspection.
- **Adaptive FiLM** — fingerprint every iteration, but with a learned gate on whether to update the fingerprint. The model decides WHEN to take stock of its thinking.
- **FiLM on the injection** — instead of modulating the ShortConv layers, modulate the Parcae A and B parameters. The context changes HOW MUCH to remember vs inject, not HOW to process.

---

## Decision 6: TTT Configuration

**Context:** TTT (test-time training / fast weights) was proven in LAZARUS and ARGUS. The question is how much TTT and where.

### Options Considered (from ARGUS-PRIME brainstorming)

| Option | TTT Layers | Style | Cost |
|--------|-----------|-------|------|
| 0 TTT | None | — | Zero overhead |
| **1 TTT, single-step** | Last Coda GQA only | Standard outer-product | ~1% throughput |
| **1 TTT, multi-step (3 steps)** | Last Coda GQA only | 3 inner gradient steps | ~2-3% throughput |
| 2 TTT | Coda GQA layers 2 and 4 | Standard | ~2% throughput |
| 4 TTT | All GQA layers (Prelude + Coda) | Standard | ~4% throughput |

**Choice: 1 TTT at the last Coda GQA layer, multi-step (3 steps)** — "The Sniper."

**Reasoning:**
1. Layer 16 (last GQA in Coda) is the final transform before LM head — maximum leverage
2. Multi-step (3 inner gradient steps) is cheaper than 3 separate TTT layers but gives deeper adaptation at one point
3. 3 steps × 1 layer ≈ cost of 1.5 single-step layers — well within budget
4. The Parcae paper showed multi-step helps at longer contexts

**What could still be tried:**
- **TTT INSIDE the loop** — adaptive FFN within the looped block. The weights reshape EVERY iteration. Extreme: the model's processing function adapts mid-loop. Risk: TTT overhead × 8 iterations.
- **TTT on the injection** — instead of adapting the FFN, adapt the A and B parameters per-chunk. The LOOP DYNAMICS adapt to the document, not just the final projection.
- **LAZARUS-style damped TTT** — `ΔW = γ·ΔW + η·V̂^T·Z` with decay. Could be applied to either the Coda or the injection.
- **Zero TTT** — if the looping + FiLM + momentum is enough, TTT may be unnecessary overhead. ARGUS-PRIME ablation B0 tests this.

---

## Decision 7: Parameter Dimension (d_model)

**Context:** Target ~175M params. LFM2-350M uses d=1024, which scales to d=768 at half size.

### Options Considered

| Option | d_model | FFN | Layers | Unique Params | Notes |
|--------|---------|-----|--------|--------------|-------|
| d=768, 16 layers | 768 | 2816 | 16 | ~175M (no loop) | LFM2-faithful, our ARGUS-PRIME config |
| d=768, looped | 768 | 2816 | 9 unique | ~121M unique | OUROBOROS standard: 3× param efficiency |
| **d=512, looped (MINI)** | 512 | 1792 | 9 unique | ~55M unique | Core block fits L2! Ultra-edge model |
| d=1024, looped | 1024 | 2560 | 9 unique | ~180M unique | Better Tensile but defeats param savings |

**Choice:** Both d=768 (standard OUROBOROS) and d=512 (OUROBOROS-MINI) as two variants.

**Reasoning:**
- d=768 gives ~121M unique params with ~338M effective → serious model, comparable to lab baselines
- d=512 gives ~55M unique params with ~155M effective → the ULTIMATE edge model where core block fits entirely in L2

**What could still be tried:**
- **d=640** — a middle ground where the core block (~3.8MB) JUST fits L2. Better quality than d=512, still L2-resident.
- **Heterogeneous d** — d=768 for Prelude/Coda, d=512 for the core loop. Projection layers between zones. Maximizes L2 utilization for the looped portion while keeping higher capacity at the edges.
- **d=1024 with 2 core layers** — fewer layers in the core block to stay under L2. Loop 8× with 2 layers instead of 3. More unique-param heavy (Prelude/Coda dominate), less loop-dependent.

---

## Decision 8: Number of Loop Iterations

**Context:** Parcae tested up to 16 iterations. More iterations = more effective depth but more compute.

### Considered (implicit)

| Loops | Effective Depth | Training (detached + grad) | Notes |
|-------|----------------|---------------------------|-------|
| 4 | 2+4×3+4 = 18 | 1+3 or 2+2 | Minimal loop, fast |
| **8** | 2+8×3+4 = 30 | 5+3 | Sweet spot: deep enough, manageable gradient |
| 12 | 2+12×3+4 = 42 | 9+3 | Very deep, more detached overhead |
| 16 | 2+16×3+4 = 54 | 13+3 | Maximum depth, Parcae tested this |

**Choice: 8 iterations** (5 detached + 3 with gradients)

**Reasoning:** 30 effective layers is comparable to GPT-2 Large (36 layers) at a fraction of the parameters. Curriculum starts at 2 loops and ramps to 8. The 5+3 split means backward pass only covers 3 iterations × 3 layers = 9 layer-equivalents of gradient computation — very cheap.

**What could still be tried:**
- **Dynamic loop count at inference** — more iterations for harder inputs, fewer for easy ones (like Parcae's per-token sampling). Adaptive compute time for the loop.
- **12+ iterations for long-context** — more iterations might help with longer sequences where the model needs more "thinking time."
- **Poisson-sampled iterations** — Parcae samples from Poisson distribution during training. This regularizes and prevents overfitting to a fixed depth. We should consider this.

---

## Summary: The Road Not Taken

These are the most promising unexplored directions from this design process:

1. **GQA inside the loop (alternating)** — if quality needs global context mid-loop
2. **FLA HGRN inside the loop** — cheap recurrence (0.40ms) for in-loop long-range mixing
3. **TTT on the Parcae injection** — adaptive loop dynamics per document
4. **Dual FiLM** — multiple introspection points during the loop
5. **Heterogeneous d (768 edges + 512 core)** — maximize L2 for the hot path
6. **Poisson-sampled loop depth** — Parcae's regularization technique
7. **DeltaNet-style injection** — outer-product injection (TTT meets Parcae)
8. **Progressive widening** — different d_model in different zones

Each of these is a valid ablation or extension of OUROBOROS. The base design was chosen for SIMPLICITY and TESTABILITY — prove the core idea works, then add complexity.

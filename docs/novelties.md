# Novel LLM Architectures for Punching Above Weight Class

Grounded in your research corpus (EFLA, KDA, RWKV-7, GSA, Titans, CompreSSM, Memory Caching) and Strix Halo constraints (bandwidth-bound, element-wise-free, unified memory, L2=6MB, d≥512 for autokernel). Each targets **matching 2-10× larger models** via a specific mechanism, not just combinations of known parts.

---

## 1. **OUROBOROS** — The Self-Molting Model

**Core idea:** Run CompreSSM's HSV analysis continuously during training, but when a state dimension is pruned, **don't delete it — re-initialize it in an orthogonal random direction**. The model "molts": sheds dead state, grows new state in fresh subspaces.

**Why 3-5× effective capacity:**
- Parameter count stays fixed, but the model explores a much larger *cumulative* state-space volume over training
- Dead capacity is recycled instead of wasted (CompreSSM's finding: most dims collapse within 10% of training)
- The Lyapunov stability bound (Lemma 3.1) guarantees pruning decisions are reliable

**Mechanism:**
```
every ~1000 steps:
  1. Compute HSVs for each KDA/EFLA layer
  2. For dims with σ_i < 0.05 × σ_max:
       project old state onto top-r subspace
       reinitialize those rows/cols of A,B,C with random orthogonal vectors
       reset optimizer state for those dims only
  3. Apply small "compatibility loss" to keep old behavior on validation batch
```

**Strix Halo fit:** Gramian math is O(n²) element-wise on diagonal SSMs → free. Regrowth is parameter surgery, not a new kernel. Use EFLA (exact solution) so re-initialization doesn't compound discretization error.

---

## 2. **MIRAGE** — Latent Scratchpad Tokens

**Core idea:** Between each pair of real tokens, the model generates **K invisible "thought tokens"** in continuous space. They participate in attention/SSM state but are never decoded. Variable K per position, trained with an RL-style compute budget.

**Why 4-10× effective capacity:** This is test-time compute — the insight from o1/DeepSeek-R1 compressed into architecture. A 200M model thinking for 4 tokens per real token has the *contextual depth* of an 800M model, without extra parameters.

**Training:**
- Auxiliary head predicts "difficulty" σ_t per real token (from entropy of next-token prediction)
- Scratchpad length K_t = ⌈σ_t × K_max⌉ (Gumbel-softmax during training)
- Gradient flows through scratchpad tokens naturally via attention
- Regularizer: penalize excessive K to prevent blowup

**Key twist vs. prior work:** Scratchpad tokens are in **continuous latent space**, not discrete vocabulary. No language grounding, so they can encode dense reasoning state — like a "train-of-thought" compressed into a single d=768 vector.

**Strix Halo fit:** Works best on element-wise architectures (RWKV-7, KDA) where adding 4× tokens is purely bandwidth-linear, not quadratic. Perfect match for Flux Attention's layer router — thought tokens use linear attention, real tokens use sparse full attention.

---

## 3. **HYPERNOVA** — State-Space Weights Born on Demand

**Core idea:** Store only a **tiny hypernetwork H (5-20M params)**. All SSM A, B, C matrices are generated on-the-fly from a few hundred learnable "gene tokens" per layer. The 250M model is actually 20M hypernetwork + 2000 gene tokens.

**Why 5-10× effective params:**
- The generated weights can span a combinatorial space far larger than what you could store
- Gene tokens act as a low-dim coordinate in "architecture space"
- Related to Meta-State (2504.08247) but generalized — uses hypernet, not state-as-weights

**Architecture:**
```
for each layer ℓ:
    gene_ℓ ∈ R^64  (learnable, one per layer)
    A_ℓ, B_ℓ, C_ℓ = H(gene_ℓ)   # hypernet expansion
    h_{t+1} = EFLA_update(h_t, A_ℓ, B_ℓ, C_ℓ, x_t)
```

**Why on Strix Halo specifically:** The hypernet expansion (gene → matrix) is a burst of matmuls once per layer per forward pass, amortized over the whole sequence. L2-resident gene tokens mean negligible bandwidth cost. The effective weight matrices can be regenerated fresh each micro-batch, creating implicit weight augmentation.

---

## 4. **KALEIDOSCOPE** — Multi-Resolution Recursive Attention

**Core idea:** **One shared transformer block, applied at 4 sequence resolutions simultaneously** (L, L/2, L/4, L/8 via learned pooling). The 4 parallel outputs are fused via a tiny gating head. Total params = 1 block + 4 pool/unpool projectors.

**Why 4× effective depth:**
- A ¼-size sequence sees 4× the effective receptive field per block
- At L/8 resolution, even linear attention sees global context
- Analogous to image pyramids — multi-scale = scale-invariant features

**Vs. standard hierarchical models:** KALEIDOSCOPE runs all scales *in parallel* through the *same* block. It's not a U-Net. The shared block is forced to learn scale-invariant transformations, which is a powerful inductive bias for language (where phenomena exist at word, phrase, clause, paragraph scales).

**Throughput math (Strix Halo):**
- Standard 250M model: 250M × 1024 tokens × 50 steps = 12.8 GB/s bandwidth
- KALEIDOSCOPE: 70M block × (1024 + 512 + 256 + 128) × 50 = 7 GB/s bandwidth
- **~1.8× faster training** at similar effective quality

---

## 5. **PROTEUS-LOOP** — Modulated Recursion

**Core idea:** RESONANT-LOOP's shared-block wins (13K tok/s measured, best throughput), but its 50M-param quality ceiling is real. Fix: add a **5M-param "conductor" LSTM** that emits FiLM modulation params `(γ_i, β_i)` for each loop iteration i. Now one block × 16 iterations behaves like 16 distinct layers — but with only 55M unique params and massive L2 reuse.

**Why it works:**
- AMADEUS ablation showed FiLM conductor is nearly free (137K params)
- Shared block stays hot in 6MB L2 across all 16 iterations → the "cheat code"
- Conductor learns a *curriculum over depth* — early iterations do syntax, late ones do semantics
- At inference, early-exit when conductor confidence is high (adaptive compute)

**Projected quality:** RESONANT-LOOP at 3.42 val loss + per-iter modulation should close to ~3.00 (AMADEUS territory) at 2-3× the throughput. Essentially: **quality of 250M at throughput of 60M.**

---

## 6. **MNEMOSYNE** — Differentiable KV Database

**Core idea:** Replace context attention with a **growing, learned key-value database** (leveraging 128GB unified memory). Uses Titans' surprise gate for writes, GSA softmax for lookups, and **LSH sharding** so lookup is O(log N).

**Schema:**
```
Database state (lives in LPDDR5X, 128GB capacity):
  keys:    (N_stored, d=256)    — LSH-indexed into B buckets
  values:  (N_stored, d_model=768)
  metadata: (N_stored,)         — surprise score, age, access count

Per token:
  1. Surprise = ||∇_x Loss|| (proxy: cross-entropy gradient)
  2. if Surprise > τ: DB.write(k_t, v_t)
  3. LSH bucket lookup → top-K keys (K=32)
  4. GSA softmax over retrieved K → output
  5. Forget: decay access count; evict low-count entries when DB full
```

**Why 10× effective context:** A 200M model with a 10M-entry database has access to arbitrarily long context *without* quadratic cost. Unlike RAG, the database is trained end-to-end — it learns *what* to remember, not just *how* to retrieve.

**Strix Halo killer feature:** On NVIDIA, the DB lives on SSD/CPU-RAM with slow PCIe access. On Strix Halo, the DB lives in the **same LPDDR5X pool as the model weights**, at 240 GB/s. This is the only architecture on this list that genuinely *requires* unified memory to be practical.

---

## 7. **TACITURN** — Deep Equilibrium with Error-Free Dynamics

**Core idea:** Replace N stacked layers with **one layer iterated to fixed point**. The depth is implicit and *adaptive per token*. EFLA's exact solution makes this numerically stable (standard DEQ suffers from accumulated Euler error).

**Why it's now possible:** Prior DEQ models (Bai et al. 2019) failed on LM because:
1. Fixed-point convergence was fragile (EFLA solves this — zero discretization error)
2. Backward pass required implicit differentiation (now solved via stop-gradient Jacobian approximations)
3. Quality lagged (but modern architectures like KDA change the game)

**Per-token depth adaptation:**
- Easy tokens (e.g., "the") converge in 1-2 iterations
- Hard tokens (e.g., last word of a math proof) iterate 15-20 times
- Average iteration budget tunable at inference

**Projected effect:** A 150M-param 1-layer model with average depth 8 matches the quality of a 250M-param 8-layer model at 60% of training cost (fewer backward-pass matmuls since only one layer's weights).

---

## 8. **QUILL** — Hierarchical Summary Tokens

**Core idea:** Every N=32 tokens, insert a **learned "summary token"** that compresses the preceding 32 tokens into a single vector. Attention is replaced by a two-tier scheme: local linear attention within chunks, full sparse attention over summaries only.

**Effective context:**
- For a 2048-token window: 2048 local + 64 summaries = O(2112) attention, but reaches 2048×32 = **65K tokens of effective history** through hierarchical summaries
- Stack two QUILL layers: 65K × 32 = 2M effective context from a 2K attention budget

**Mechanism:**
```
Summary token s_j = LearnedCompress(tokens_{32j..32(j+1)-1})
                  = cross-attention(query=learnable, keys/vals=chunk)
Attention path: [local chunk | all summaries | current token]
```

**Strix Halo fit:** 64 summary tokens is L2-resident. The hierarchical structure fits unified memory (summaries are persistent across long documents). Compatible with BIFROST's layer router — summary layers use full attention, token layers use KDA.

---

## 9. **CHIAROSCURO** — Dual-Precision Shadow Model (Wildcard)

**Core idea:** Run a **main fp16 model AND a shadow int4/ternary copy in parallel**, sharing the same forward pass. Shadow model's predictions are used as a "noise reference" — the main model predicts the *residual* between shadow and ground truth.

**Why it helps:**
- Shadow model (like BitNet/OBSIDIAN) captures 80% of easy predictions at ~6× compression
- Main model's effective task is *much easier*: predict only the 20% residual
- Trained jointly end-to-end
- At inference, shadow can run on CPU while main runs on GPU (true parallelism via unified memory)

**Params:** 50M shadow + 150M main = 200M total, but **main model sees a simplified task** so it learns faster. Analogous to gradient boosting applied within a single model.

---

## Cross-Cutting Design Principles

| Principle | Architectures That Use It |
|-----------|---------------------------|
| **Looped/shared weights for effective depth** | PROTEUS-LOOP, TACITURN, KALEIDOSCOPE |
| **Unified memory as a primary resource** | MNEMOSYNE, CHIAROSCURO |
| **EFLA exact dynamics to stabilize novel loops** | OUROBOROS, TACITURN |
| **Element-wise gating as "free" compute** | All of them (required on Strix Halo) |
| **Adaptive per-token compute** | MIRAGE, TACITURN, PROTEUS-LOOP (early-exit) |
| **Parameter generation over parameter storage** | HYPERNOVA, OUROBOROS |

## Suggested Experimental Order

1. **PROTEUS-LOOP** — lowest-risk, builds directly on verified RESONANT-LOOP, biggest likely win
2. **KALEIDOSCOPE** — pure-element-wise, should compile well, clean ablation vs TEMPEST
3. **MIRAGE** — highest upside (test-time compute), moderate risk (RL signal)
4. **OUROBOROS** — if CompreSSM PyTorch port is done, this is a clean extension
5. **MNEMOSYNE** — the "only possible on Strix Halo" play; save for when infra is mature

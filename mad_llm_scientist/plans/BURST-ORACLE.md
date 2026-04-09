# BURST ORACLE

**Multi-Token Prediction with Engram-Verified Burst Decoding on a Caveman LFM Backbone**

## Hypothesis

On memory-bound hardware, reading weights costs the same whether you predict 1 token or 4. A model trained with **multi-token prediction (MTP)** heads that generates tokens in **adaptive bursts** — with **Engram tables validating burst predictions in O(1)** instead of expensive verification forward passes — can achieve **2-3× effective throughput** over standard autoregressive decoding while simultaneously improving model quality.

**The magician's hat trick:** Engram tables serve DUAL PURPOSE:
1. **During forward pass:** Inject N-gram knowledge (standard Engram)
2. **During burst decoding:** VALIDATE predicted token sequences via O(1) hash lookup — if the predicted burst matches a known N-gram, accept without verification

**Fairy dust from past creations:**
- Caveman LFM backbone (Hypothesis #6, LFM2-validated)
- Engram knowledge tables (Chimera Engram #3, now dual-purpose)
- Caveman glue/content routing (Genius Caveman #5, now controls burst SIZE)
- Griffin recurrence with decay bias spectrum (Spectral Hydra #1 + Griffin)

**Grounded in:** Meta MTP (2404.19737) — 3× speedup + 12% quality improvement. Medusa (2401.10774) — 2.2-3.6× self-speculative speedup.

**Novel contribution:** Engram-verified burst decoding (O(1) N-gram validation replacing tree attention verification) + caveman confidence burst sizing + MTP on a non-transformer backbone. Not in any existing paper.

---

## Why Burst Decoding is Perfect for Memory-Bound Hardware

Standard autoregressive decode on Strix Halo (int4, 248M model):
- Read ~124 MB of weights from DRAM: **0.73 ms**
- Compute logits (one token): **~0.01 ms**
- **Total: 0.74 ms per token → 1351 tok/s**

The compute is 1.4% of total time. The model is **98.6% memory-bound.** Predicting 4 tokens instead of 1 adds ~0.03 ms of compute (4× the logit computation) but saves 3× the weight reads:

Burst-4 decode:
- Read weights once: **0.73 ms**
- Compute 4 sets of logits: **~0.04 ms**
- **Total: 0.77 ms for 4 tokens → 5195 tok/s effective**

But we can't always trust all 4 predictions. Hence: adaptive bursts with Engram verification.

---

## Architecture

### Backbone: Caveman LFM (~248M)

The full Caveman LFM architecture as designed in hypothesis #6:
- 16 layers (10 gated conv + 6 Griffin recurrence), d=1024
- Engram tables at layers 2 and 9
- Caveman routing: glue tokens bypass recurrence
- SwiGLU FFN (2240 inner), RMSNorm, RoPE, weight tying

### Multi-Token Prediction Heads (+3M)

K=4 prediction heads sharing the LM head (weight-tied embedding):

```python
# Head 1 (standard next-token): uses backbone output directly
logits_1 = lm_head(rmsnorm(h_t))                      # predicts t+1

# Head 2: small MLP adapter + shared LM head
h2 = h_t + mlp_2(concat(h_t, embed(sample(logits_1)))) # condition on t+1 prediction
logits_2 = lm_head(rmsnorm(h2))                        # predicts t+2

# Head 3: condition on t+1 and t+2
h3 = h_t + mlp_3(concat(h_t, embed(sample(logits_2))))
logits_3 = lm_head(rmsnorm(h3))                        # predicts t+3

# Head 4: condition on t+1, t+2, t+3
h4 = h_t + mlp_4(concat(h_t, embed(sample(logits_3))))
logits_4 = lm_head(rmsnorm(h4))                        # predicts t+4
```

Each adapter MLP: `concat(1024, 1024) → 1024` = ~2M params. 3 adapters = ~6M.

But we can make them cheaper: `concat(1024, 1024) → 512 → 1024` = ~1.6M each. 3 adapters = ~4.8M.

**Total model: 248M + 4.8M = ~253M** (slightly over 250M; reduce Engram tables by 3M to fit).

### Engram-Verified Burst Decoding (The New Trick)

At inference, after generating K=4 token predictions:

```python
predicted_burst = [token_1, token_2, token_3, token_4]

# Step 1: Check if the burst matches known N-grams in Engram tables
for k in range(1, K):
    ngram = tuple(predicted_burst[k-2:k+1])  # trigram ending at position k
    engram_match = check_engram(ngram)         # O(1) hash lookup
    
    if engram_match and confidence[k] > theta:
        accept(predicted_burst[k])             # trusted: known N-gram + high confidence
    else:
        # Reject remaining burst, regenerate from position k
        break

# Step 2: If Engram can't verify, fall back to standard verification
# Run forward pass with accepted prefix to verify remaining predictions
```

**Why this works:** Glue word sequences ("of the", "in the morning", "it is") are stored in the Engram hash tables during training. At inference, if the predicted burst forms a known N-gram, the Engram table confirms it in O(1) — no need for an expensive verification forward pass.

**When Engram verification fires (~65% of bursts):** Zero-cost verification → full burst accepted.
**When Engram can't verify (~35%):** Fall back to standard self-speculative verification (1 additional forward pass for content token bursts).

### Caveman Confidence Burst Sizing

Before generating the burst, estimate how many tokens to predict:

```python
# After backbone forward pass:
is_glue_context = all(GLUE_DICT[token_ids[t-2:t+1]])  # recent context is all glue?
state_delta = norm(h_t - h_{t-1})                       # how much did state change?

if is_glue_context and state_delta < theta_easy:
    burst_size = 4    # very predictable context → big burst
elif is_glue_context or state_delta < theta_medium:
    burst_size = 2    # somewhat predictable → medium burst
else:
    burst_size = 1    # hard content → standard decode
```

This AVOIDS generating predictions that will be rejected — saving compute on the adapter MLPs.

---

## Training

### MTP Training Loss

```python
# Standard training with K=4 heads
loss = 0
for k in range(K):
    logits_k = mtp_head_k(h, targets[t+1:t+k])  # teacher-forced
    loss += weight[k] * CE(logits_k, targets[t+k+1])

# Weights: head 1 = 1.0, head 2 = 0.5, head 3 = 0.3, head 4 = 0.2
# (decreasing weight for further predictions — they're harder and noisier)
```

**Key insight from Meta:** MTP training IMPROVES the backbone representations. The backbone learns to encode information useful for predicting multiple future tokens, making it better even for standard next-token prediction. At 13B: +12% HumanEval, +17% MBPP.

### Phase Training (adapted from Caveman LFM)

1. **Phase 1 (40%):** Standard next-token only (head 1). Train backbone.
2. **Phase 2 (30%):** Enable all 4 MTP heads. Backbone improves from multi-token signal.
3. **Phase 3 (20%):** Enable Engram + caveman routing. Tables learn N-gram patterns.
4. **Phase 4 (10%):** Fine-tune burst sizing thresholds on validation set.

### Hyperparameters

| Parameter | Value |
|-----------|-------|
| Dataset | OpenWebText ~100M tokens |
| Batch | 48×512, accum=2 (48K effective) |
| Optimizer | AdamW (β₁=0.9, β₂=0.95) |
| LR | 8e-4 cosine → 8e-5 |
| Engram LR | 5× base, Adam, zero weight decay |
| MTP head LR | 1× base (standard) |
| MTP loss weights | [1.0, 0.5, 0.3, 0.2] for heads 1-4 |
| Grad clip | 1.0 |
| Precision | fp16 mixed + fp32 scan |
| Est. throughput | ~25M tokens in 15 min |

---

## Inference Modes

### Mode 1: Standard Decode (Baseline)
- 1 token per forward pass
- int4: ~1429 tok/s

### Mode 2: Fixed Burst-4 (Maximum Throughput, Lower Quality)
- Always predict 4 tokens, accept all without verification
- Good for: bulk text generation where minor errors are acceptable
- int4: ~5195 tok/s effective (theoretical max)

### Mode 3: Engram-Verified Adaptive Burst (Recommended)
- Burst size adapted by caveman confidence (1-4 tokens)
- Glue bursts verified by Engram O(1) lookup
- Content predictions verified by standard forward pass
- Average burst: ~2.3 tokens
- int4: **~3287 tok/s effective**

### Mode 4: Conservative Burst-2 (High Quality)
- Maximum burst size 2
- All bursts verified (Engram or forward pass)
- int4: ~2500 tok/s effective

---

## Decode Speed (Strix Halo)

### Burst Calculation

| Context Type | Fraction | Avg Burst | Verification | Cost per Burst |
|-------------|----------|-----------|--------------|----------------|
| Glue (easy) | 65% | 3.5 | Engram O(1) | 0.74 ms (1 fwd) |
| Content (medium) | 25% | 1.5 | Forward pass | 1.48 ms (2 fwd) |
| Hard content | 10% | 1.0 | N/A | 0.74 ms (1 fwd) |

Effective tokens per weighted ms:
- Glue: 65% × 3.5 / 0.74 = 3.07 tok/ms
- Content: 25% × 1.5 / 1.48 = 0.25 tok/ms
- Hard: 10% × 1.0 / 0.74 = 0.14 tok/ms
- **Total: 3.46 effective tok/ms → wait, let me redo this properly.**

Average time per effective token:
- Total passes per 100 tokens: 65/3.5 + 25/1.5 × 2 + 10/1.0 = 18.6 + 33.3 + 10 = 61.9 passes
- Time: 61.9 × 0.74 ms = 45.8 ms for 100 tokens
- **~0.46 ms per effective token = ~2174 tok/s int4**

### Comparison Table

| Model | Params | Standard int4 tok/s | With Burst Oracle |
|-------|--------|--------------------|--------------------|
| GPT-2 124M | 124M | — | — |
| Spectral Hydra | 244M | ~714 | N/A |
| Resonant Loop | 59M | ~1613 | N/A |
| Chimera Engram B | 231M | ~2000 | N/A |
| Genius Caveman | 234M | ~2439 | N/A |
| Caveman LFM | 248M | ~1429 | N/A |
| **Burst Oracle** | **~251M** | ~1429 (mode 1) | **~2174 (mode 3)** |

### fp16

| Mode | Effective tok/s |
|------|----------------|
| Standard | ~360 |
| Adaptive burst (mode 3) | **~548** |
| Fixed burst-4 (mode 2) | ~1300 |

---

## Parameter Count

| Component | Params |
|-----------|--------|
| Caveman LFM backbone | ~245M (reduced Engram to fit) |
| MTP adapter MLPs (3 × ~1.6M) | ~4.8M |
| **TOTAL** | **~250M** |

Engram table reduction: bigram 50K×64 (from 65K), trigram 25K×64 (from 32K). Saves ~3M for the MTP heads.

---

## HIP Kernels

**Reuse everything from Caveman LFM** (5 existing kernels).

**New kernels (in addition to Caveman LFM's 4):**

5. **Fused MTP Head Chain** (inference) — Run 3 adapter MLPs sequentially, each conditioned on previous head's argmax. Single kernel launch for the chain. The adapters are tiny (1.6M each), fitting in L2 in int4.
6. **Fused Engram Burst Verification** (inference) — Hash the predicted N-gram, check against Engram table, return accept/reject. Pure integer ops + 1 table lookup. Negligible cost.

---

## What Makes This New

| Existing Work | What It Does | What Burst Oracle Adds |
|--------------|-------------|----------------------|
| Meta MTP (2404.19737) | K prediction heads, speculative decode | Engram-verified bursts (O(1) verification), adaptive sizing |
| Medusa (2401.10774) | Tree attention verification | NO tree attention (too expensive on our hardware). Engram + confidence instead. |
| Speculative decoding | Draft model + verify | Draft IS the model (self-speculative), Engram IS the verifier |
| Engram (2601.07372) | Knowledge injection during forward | DUAL PURPOSE: injection + burst verification |
| Caveman routing | Skip compute for glue tokens | Controls BURST SIZE, not just compute routing |

**The genuinely new idea:** Using hash-indexed N-gram tables as a zero-cost verification oracle for multi-token prediction bursts, combined with information-weighted adaptive burst sizing. This eliminates the verification overhead that limits standard speculative decoding on memory-bound hardware.

---

## Risks & Mitigations

| Risk | Severity | Mitigation |
|------|----------|------------|
| MTP heads degrade quality at 250M scale | MEDIUM | Meta showed improvement at 13B; may be neutral/positive at 250M. Head weights decrease for further predictions. MTP is optional — disable and fall back to standard decode. |
| Engram verification has false positives | LOW | Hash collision = wrong N-gram matches → accepts bad burst. But: 4 hash heads make all-collide probability ~0. Plus context-aware gating already filters noise. |
| Burst sizing is miscalibrated | MEDIUM | Three modes available (standard, adaptive, aggressive). Tune theta on validation set. Conservative mode 4 (burst-2) is always safe. |
| MTP adapters add latency | LOW | Each adapter is 1.6M params = 3.2KB int4. Fits in L2. Sequential chain of 3 adds ~0.01 ms. |
| Training with 4 heads is harder | MEDIUM | Phase 1-2 trains backbone first, then enables MTP heads. MTP is additive loss — doesn't hurt standard next-token training. |

## Success Criteria

1. All Caveman LFM criteria met (loss < 4.5, routing works, Engram activates)
2. MTP heads predict valid tokens (head 2 accuracy > 40%, head 3 > 25%, head 4 > 15%)
3. Engram verification accepts > 60% of glue bursts correctly
4. Adaptive burst mode achieves > 2× effective throughput over standard decode
5. **No quality degradation** from MTP training (loss ≤ standard training loss)
6. Effective decode > 2000 tok/s int4 on Strix Halo

## Implementation Roadmap

1. Implement full Caveman LFM backbone (steps 1-9 from Caveman LFM plan)
2. Add MTP adapter MLPs (3 small networks, concat input)
3. Modify training loop for MTP loss (weighted sum across K=4 heads)
4. Phase 1-2 training: backbone + MTP heads
5. Phase 3-4: Engram + routing + burst threshold tuning
6. Implement Engram burst verification (hash predicted N-grams, check table)
7. Implement adaptive burst sizing logic (glue context + state delta)
8. Fused MTP head chain kernel (tiny, L2-resident)
9. Fused Engram verification kernel (integer hash + lookup)
10. Benchmark: standard vs burst-2 vs adaptive vs burst-4 modes
11. Quality eval: compare MTP-trained vs standard-trained perplexity
12. Decode benchmark on Strix Halo across all 4 modes

---

*The oracle doesn't predict the future one moment at a time. It sees whole phrases forming in the mist — and when the Engram stones confirm the vision, it speaks them all at once.*

---

## Hardware Optimization Notes (Strix Halo gfx1151)

> Added 2026-04-09 based on AMADEUS implementation findings.

### Existing HIP Kernels to Reuse
- **fused_residual_add_rmsnorm** (6.6x), **silu_gate_mul** (1.6x), **cross_entropy** (1.8x)
- Backbone (Caveman LFM) kernels all apply
- Apply via `autokernel.optimize(model, training=True)`

### MTP Heads: Shared LM Head is Key
Multi-token prediction adds K adapter MLPs but shares the LM head. The shared LM head (103 MB fp16) dominates decode time. MTP adapters are small — negligible overhead during training.

### Throughput Correction
Previous estimate "~25M tokens in 15 min" is optimistic. Based on AMADEUS baseline:
- **Eager:** ~6-8K tok/s → 5.4-7.2M in 15 min
- MTP overhead during training: ~10-15% (K forward passes through adapter + shared head)
- **Effective:** ~5-7K tok/s with MTP active

### Scan: Chunked Linear Recurrence
Backbone uses Griffin recurrence → same chunked scan. Reference: `models/amadeus.py:selective_scan_chunked`.

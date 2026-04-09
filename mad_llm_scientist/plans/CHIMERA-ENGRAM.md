# CHIMERA ENGRAM

**mHC Multi-Branch Residual + Mamba-3 Complex SSM + Hash-Indexed Knowledge Tables + Sparse MoE**

## Hypothesis

Fuse four mechanisms: (1) **mHC** 4-branch residual with doubly stochastic mixing, (2) **Mamba-3 SISO** complex-valued SSM for sequence mixing, (3) **Engram** hash-indexed N-gram tables for O(1) knowledge retrieval, (4) **sparse MoE** FFN with top-2 routing. Three variants explore the Engram–Compute tradeoff.

**Papers:** mHC (2512.24880), Engram (2601.07372), Mamba-3 (2603.15569).

**Novel contribution:** First architecture combining all four. Variant C adds writable Engram tables (inference-time learning via exact sparse gradients).

---

## Variant Summary

| | A: Conservative | B: Aggressive | C: Unhinged |
|---|---|---|---|
| d_model | 896 | 512 | 512 |
| Layers | 12 (6 Mamba + 6 MoE) | 8 (4+4) | 8 (4+4) |
| Engram params | ~18M (7%) | ~151M (65%) | ~151M writable |
| Total params | ~243M | ~231M | ~231M |
| Active/token | ~155M | ~63M | ~63M |
| fp16 tok/s | ~505 | ~794 | ~746 |
| int4 tok/s | ~1299 | ~2000 | ~1887 |

---

## Core Architecture (All Variants)

### Layer Structure

Alternating Mamba-3 and MoE-FFN sublayers, each wrapped in mHC:

```
[4-branch mHC stream: 4 × d_model]
  → Layer 0: mHC_pre → RMSNorm → Mamba3-SISO → mHC_post/res
  → Layer 1: mHC_pre → RMSNorm → MoE-FFN(8E,top2) → mHC_post/res
  → ... (alternating, + Engram at designated layers)
  → mHC readout → Final RMSNorm → LM Head (tied embedding)
```

### Component 1: mHC Residual Highway (n=4 branches)

Per sublayer:
```
x̄ = RMSNorm(flatten(stream))           # normalize expanded stream
H̃_pre  = α_pre  · (x̄ @ φ_pre)  + b_pre    # α_pre init=0.01
H̃_post = α_post · (x̄ @ φ_post) + b_post   # α_post init=0.01
H̃_res  = α_res  · mat(x̄ @ φ_res) + b_res  # α_res init=0.01

H_pre = sigmoid(H̃_pre)                 # ∈ (0,1), readout weights
H_post = 2·sigmoid(H̃_post)             # ∈ (0,2), write-in weights
H_res = Sinkhorn(exp(H̃_res), iters=20) # doubly stochastic 4×4

x = H_pre @ stream                      # weighted sum: 4 branches → d_model
y = sublayer(RMSNorm(x))                # Mamba-3 or MoE-FFN
stream = H_res @ stream + H_post ⊗ y    # cross-branch mixing + write-in
```

**Critical from paper:** H_res provides the **majority of the gain** (ablation). Sinkhorn starts with **exp()** of logits, then alternates row/col normalization. α params init to **0.01** (dynamic nearly off at start). Without Sinkhorn, HC **diverges at ~12K steps**.

~25×d_model params per sublayer (~0.3% overhead).

### Component 2: Mamba-3 SISO Sequence Mixer

**3-term exponential-trapezoidal recurrence (complex-valued):**

```
α_t = exp(a_t + i·θ_t)           # complex decay (data-dependent magnitude + rotation)
a_t = -softplus(W_a @ x_t)       # clamped negative → |α| ≤ 1
θ_t = W_θ @ x_t                  # data-dependent rotation angle
B_t = RMSNorm(W_B @ x_t) + b_B   # complex input projection + learnable bias
C_t = RMSNorm(W_C @ x_t) + b_C   # complex output projection + learnable bias
z_t = B_t · x_t

h_t = α_t · h_{t-1} + β_t · z_{t-1} + γ_t · z_t    # 3-term recurrence
y_t = Re(C_t · h_t)                                   # readout
```

**Key properties:**
- **Data-dependent RoPE is NON-NEGOTIABLE:** standard RoPE → 1.56% parity. Data-dependent → 100%. Without RoPE → 2.27%.
- **NO short conv** — trapezoidal 2-band structure + B,C biases replace it. Conv is slightly WORSE with Mamba-3 (15.85 vs 15.72 ppl in ablation).
- **dstate=64 matches Mamba-2 dstate=128** — half the state for equal quality.
- SISO only (no MIMO — no matrix cores on Strix Halo)
- ~6 × d_model × d_inner params per layer

**Parallel scan for training:** Precompute z_t, then standard associative scan:
```
(α₂, u₂) ∘ (α₁, u₁) = (α₂·α₁, α₂·u₁ + u₂)
```
Adapt `kernels/hip/prefix_scan.py` for complex float32. Warp-level `__shfl_up` on float4.

### Component 3: Engram Hash-Indexed N-gram Tables

```python
# Per token, per hash head k, per N-gram order n:
e = table_n[hash_k(compress(token_ids[-n:]))]     # O(1) lookup
alpha = sigmoid(RMSNorm(h) @ RMSNorm(W_K @ e) / sqrt(d))  # context gate
output += alpha * W_V @ e                           # gated contribution
```

- K=8 hash heads with distinct primes (Bloom-filter redundancy)
- Bigram (n=2) and trigram (n=3) orders minimum; Variant B adds 4-grams
- Context gate suppresses hash collisions: alpha→0 when h disagrees with retrieved embedding
- **No conv needed** (paper ablation: marginal contribution; Mamba-3 trapezoidal already provides local context). Zero-initialize if included.
- **mHC integration is the MOST IMPORTANT feature** (largest regression without it in ablation). Shared table + W_V across branches; per-branch W_K for specialized gating. Fused into single FP8 matmul.
- Placed at layers 2 and L/2 (Variant A) or every Mamba-3 layer (Variants B/C)
- **Mechanistic insight:** Engram doubles effective depth (Layer 5 aligns with Layer 12 of baseline per CKA analysis). Removing Engram at inference destroys 56-71% of factual knowledge but keeps 81-93% of comprehension.

### Component 4: MoE FFN (Sparse Experts)

```python
scores = softmax(W_router @ x)              # (d_model,) → (8,)
top2_idx, top2_w = topk(scores, k=2)
y = Σ(w_i * SwiGLU_expert_i(x))            # only 2 of 8 experts fire
```

Uses existing `moe_gating.py` (3.5×) and `silu_gate_mul.py` (1.6×).

Load balancing: `L_aux = 0.01 · 8 · Σ(f_e · P_e)` + expert dropout (mask 1/batch).

---

## Variant A: Conservative

| Parameter | Value |
|-----------|-------|
| d_model | 896 |
| d_inner (Mamba-3, expand=1.5) | 1344 |
| d_state | 64 (= Mamba-2 dstate=128 quality) |
| n_layers | 12 (6 Mamba-3 + 6 MoE-FFN) |
| Experts | 8, top-2, d_ffn=896 |
| Engram | 2×65K×128 tables, layers 2 & 6 |
| **Total: ~243M, Active: ~155M** | |

Engram is 7% of params — supplementary. Degrades gracefully if Engram adds nothing.

## Variant B: Aggressive

| Parameter | Value |
|-----------|-------|
| d_model | 512 |
| d_inner (Mamba-3, expand=1.5) | 768 |
| d_state | 64 (= Mamba-2 dstate=128 quality) |
| n_layers | 8 (4 Mamba-3 + 4 MoE-FFN) |
| Experts | 8, top-2, d_ffn=512 |
| Engram | bigram 786K×128, trigram 262K×128, 4gram 131K×128 |
| Engram at | every Mamba-3 layer (0, 2, 4, 6) |
| **Total: ~231M, Active: ~63M, Engram: 151M (65%)** | |

**The bet:** Most factual knowledge is tabular. Lookups read ~6 KB/token (negligible). Compute engine just composes.

**Risk:** d=512 with 8 layers may lack reasoning capacity. Fallback: increase d_model to 640.

## Variant C: Writable Engram

Same as Variant B, but Engram tables update at inference:

```python
# After each token:
logits = forward(token)                          # normal forward pass
loss = CE(logits, next_token)
grad = backward(loss, through_engram_only=True)  # freeze Mamba-3, MoE, embeddings
for row in retrieved_rows:
    row -= lr * grad_row                          # SGD, lr=0.01
    row = (1-ε)*row + ε*pretrained_row            # decay toward pretrained, ε=0.001
```

**What gets updated:** 24 retrieved table rows + W_V per Engram layer. NOT Mamba-3/MoE/embeddings.

**Compute cost:** ~3M FLOPs write vs ~50M forward = **6% overhead**.

**Safety mechanisms:**
- Decay toward pretrained values (ε=0.001)
- Gradient clipping per row (max 0.1)
- Threshold-gated writes (only if loss > 3.0)
- Write budget: max 1024 unique row updates per session (LRU eviction)

**Session modes:** Stateless (reset), Persistent (keep updates), Domain adaptation (process doc first).

---

## Training (All Variants)

### Phase Training (Decouple Components)

1. **Phase 1 (50% of steps):** Mamba-3 + dense FFN + mHC. No Engram, no MoE.
2. **Phase 2 (25%):** Enable Engram at designated layers.
3. **Phase 3 (25%):** Replace dense FFN with MoE.

### Hyperparameters

| Parameter | Variant A | Variants B/C |
|-----------|-----------|--------------|
| Batch | 32×512, accum=4 (64K eff) | 48×512, accum=2 (48K eff) |
| LR | 6e-4 cosine → 6e-5 | 8e-4 cosine → 8e-5 |
| Engram table LR | **5× base, Adam** (per paper) | **5× base, Adam** (per paper) |
| Engram table weight decay | **0** (per paper) | **0** (per paper) |
| Engram conv init | **zero** (identity) | **zero** (identity) |
| mHC α init | **0.01** (per paper) | **0.01** (per paper) |
| Warmup | 100 steps | 100 steps |
| Backbone weight decay | 0.1 | 0.1 |
| Grad clip | 1.0 | 1.0 |
| MoE aux loss α | 0.01 | 0.01 |
| Precision | fp16 mixed + fp32 scan | fp16 mixed + fp32 scan |
| Sinkhorn iters | 20 | 20 |
| Est. tokens/15min | ~16M | ~25M |

## HIP Kernels

**Reuse:** `fused_residual_add_rmsnorm` (6.6×), `silu_gate_mul` (1.6×), `moe_gating` (3.5×), `rotary_embedding` (3.7×), `prefix_scan` (8.4×), `cross_entropy` (1.8×), `dequantize_int4` (16.3×).

**New (priority order):**
1. **Fused Mamba-3 Decode Step** — all element-wise ops: softplus, exp_complex, rmsnorm, complex multiply, 3-term update, readout. Adapt `silu_gate_mul.py`. Est. 8–12×.
2. **Complex Associative Scan** — extend `prefix_scan.py` for complex `(α·a, α·b+u)`. Float4 in LDS.
3. **Fused mHC Read-Write** — sigmoid readout + Sinkhorn 4×4 (in registers) + write-in.
4. **Fused Engram Lookup+Gate** — hash (integer) + gather + RMSNorm + sigmoid + weighted sum.

## Risks & Mitigations

| Risk | Severity | Mitigation |
|------|----------|------------|
| Too many moving parts | HIGH | Phase training; each component validated independently at scale |
| Complex scan instability | HIGH | fp32 scan; clamp decay via -softplus; log-space magnitude fallback |
| Engram hash collisions | MEDIUM | K=8 heads w/ distinct primes (Bloom-filter redundancy); gating suppresses noise |
| MoE expert collapse | MEDIUM | Load-balancing loss; expert dropout; monitor utilization |
| Variant B: d=512 too small | MEDIUM | Engram offloads recall; MoE gives 8× FFN capacity; fallback: d=640 |
| Variant C: noisy writes | HIGH | Decay toward pretrained; grad clip; threshold gates; write budget |

## Success Criteria

1. Loss < 4.5 in 15 min
2. mHC branches develop distinct H_pre patterns
3. Engram gates activate selectively (entities/phrases → high α)
4. All 8 MoE experts receive >5% of tokens
5. Per-token efficiency ≥ GPT-2 124M
6. Variant C: writable Engram improves on repeated domain text

## Implementation Roadmap

1. Implement mHC module: 4-branch residual, Sinkhorn H_res
2. Implement Mamba-3 SISO: complex recurrence + parallel scan
3. Implement Engram: hash tables, lookup, gating, conv
4. Implement MoE-FFN with moe_gating kernel
5. Assemble Variant A, verify params, test fwd/bwd
6. Train Variant A (15 min), log loss + branch specialization
7. Assemble Variant B (large Engram), train, compare
8. Implement Variant C writable Engram
9. Custom HIP kernels (decode → scan → mHC → Engram)
10. Int4 quantization + decode benchmark
11. Compare all variants vs GPT-2 vs Spectral Hydra vs Resonant Loop

---

## Hardware Optimization Notes (Strix Halo gfx1151)

> Added 2026-04-09 based on AMADEUS implementation findings.

### Existing HIP Kernels to Reuse
- **fused_residual_add_rmsnorm** (6.6x), **silu_gate_mul** (1.6x), **cross_entropy** (1.8x)
- Apply via `autokernel.optimize(model, training=True)`

### Mamba-3 Complex SSM: Use Chunked Linear Recurrence
Mamba-3 with complex state needs **fp32 complex accumulation** (not fp16). Use chunked linear recurrence with complex dtype. Reference: `models/amadeus.py:selective_scan_chunked` — adapt for complex multiplication in the decay step.

### mHC Sinkhorn: fp32 Required
Sinkhorn normalization (20 iterations) must run in fp32. This is a known precision requirement from COOKBOOK.md. Budget ~0.5ms/layer for Sinkhorn overhead.

### Sparse MoE Warning
If using sparse MoE routing, expert selection on GPU requires careful implementation. PyTorch's `torch.topk` for expert routing is adequate but the expert dispatch (gathering tokens per expert) can be slow if experts have unbalanced loads.

### Throughput: ~5-7K tok/s (complex SSM + MoE overhead), MFU: 60-70%

# DUAL CORTEX

**System-1/System-2 Architecture with Entropy-Gated Routing**

## Hypothesis

A dual-path architecture where a tiny **fast path** (System 1, d=320, 8M params, fits in L2) processes all tokens, and a large **slow path** (System 2, d=1280, 160M params) engages only when the fast path is uncertain. Entropy-gated routing (zero learned parameters) decides which tokens need deep processing. ~70% of tokens use only the fast path → near-zero DRAM reads → extreme decode speed.

**This is not from a paper.** Dual-process theory is from cognitive science (Kahneman). No ML architecture implements it as two genuinely different processing paths with entropy-gated routing on memory-bound hardware.

**Novel contribution:** Hardware-native dual-process architecture where System 1 is L2-cache-resident and System 2 engages sparsely, with a parameter-free entropy router.

---

## First-Principles Argument

1. **In natural language, ~70% of tokens have low conditional entropy** (function words, punctuation, predictable continuations). They don't need 500MB of weight reads.
2. **Strix Halo's L2 cache is 6 MB.** A tiny model (≤3 MB int4) lives there permanently after first use.
3. **Element-wise ops are fast.** Gated recurrence at d=320 is pure element-wise — no matmul bottleneck.
4. **DRAM reads dominate decode latency.** Skipping DRAM reads for 70% of tokens is the single biggest speed win possible.

**The key insight:** Don't make ONE model faster. Make TWO models — one so small it's free, one large but rarely used.

---

## Architecture

```
Token → Embedding (d=1280, tied LM head)
  → Project down: 1280 → 320                    # fast embedding
  │
  ├─ FAST PATH (System 1): 8 layers, d=320      # ALWAYS runs
  │    Gated Recurrence + Small SwiGLU per layer
  │    → Entropy Estimator (||h_new - h_old||)
  │
  ├─ if entropy > θ:                              # ~30% of tokens
  │    Project up: 320 → 1280
  │    SLOW PATH (System 2): 10 layers, d=1280
  │    → Project down: 1280 → 320
  │    → Add to fast path state (residual)
  │
  ├─ Project up: 320 → 1280                      # for LM head
  → Final RMSNorm → LM Head (1280 → 50257)
```

## Configuration

| Parameter | Fast Path (System 1) | Slow Path (System 2) |
|-----------|---------------------|---------------------|
| d_model | 320 | 1280 |
| n_layers | 8 | 10 |
| n_heads | 8 | 16 |
| d_head | 40 | 80 |
| ffn_inner | 640 (SwiGLU, 2×) | 3200 (SwiGLU, 2.5×) |
| Sequence mixer | Griffin-style gated recurrence | Griffin-style gated recurrence |
| conv_kernel | 4 (depthwise causal) | 4 (depthwise causal) |
| Activation rate | 100% of tokens | ~30% of tokens |

| Global | Value |
|--------|-------|
| vocab_size | 50257 |
| block_size | 1024 |
| d_embedding | 1280 (shared) |
| entropy_threshold | learned scalar (init = 1.0, tuned during training) |

## Parameter Count

| Component | Params |
|-----------|--------|
| **Token embedding (50257 × 1280, tied)** | **64,328,960** |
| Embed → fast proj (1280 → 320) | 410,240 |
| **Fast path (8 layers):** | |
| — Per layer: gated recurrence proj (320→960) + (960→320) | 614,400 |
| — Per layer: conv1d (320, k=4) | 1,600 |
| — Per layer: SwiGLU (320→640→320) | 614,720 |
| — Per layer: RMSNorm × 2 | 640 |
| — Per layer total | ~1,231,360 |
| — 8 layers | **9,850,880 (~9.9M)** |
| Fast → slow proj (320 → 1280) | 410,240 |
| Slow → fast proj (1280 → 320) | 410,240 |
| **Slow path (10 layers):** | |
| — Per layer: gated recurrence proj (1280→3840) + (3840→1280) | 9,830,400 |
| — Per layer: conv1d (1280, k=4) | 6,400 |
| — Per layer: SwiGLU (1280→3200→1280) | 12,290,560 |
| — Per layer: RMSNorm × 2 | 2,560 |
| — Per layer total | ~22,129,920 |
| — 10 layers | **~155M** (note: only active for ~30% of tokens) |
| Fast → embed proj (320 → 1280) | 410,240 |
| Entropy estimator MLP (320→64→1) | 20,545 |
| Final RMSNorm | 1,280 |
| **TOTAL** | **~231M** |
| **Active per easy token** | **~10M** (fast path only) |
| **Active per hard token** | **~165M** (fast + slow) |

### Weight Sizes

| Component | fp16 | int4 |
|-----------|------|------|
| Fast path | 19.8 MB | **5.0 MB** |
| Slow path | 310 MB | 77.5 MB |
| Embedding + LM head | 128.7 MB | 32.2 MB |
| **Fast path int4** | — | **Fits in L2 (5.0 MB < 6 MB)** |

## Entropy-Gated Routing (Zero Learned Parameters)

The router is NOT a learned gate. It's a **physics-based signal:**

```python
# After fast path processes token t:
delta = torch.norm(h_fast_new - h_fast_prev)   # L2 norm of state change
engage_slow = (delta > theta)                    # scalar comparison

# theta is a single learnable parameter (or fixed threshold)
# Large delta = the fast path was "surprised" = high entropy = needs slow path
# Small delta = predictable token = fast path sufficient
```

**Why this works:** When the fast path processes a predictable token ("the" after "in"), the hidden state barely changes (small delta). When it processes something surprising ("quantum" after "the"), the state changes dramatically (large delta). The L2 norm of the state delta is a direct proxy for prediction uncertainty.

**Why not use actual entropy:** Computing entropy requires the full LM head (50257-dim softmax), which is expensive. The state delta is free (already computed during the recurrence update).

**Training the threshold:** θ is a single scalar, trained with a combined loss:

```python
loss = CE_loss + λ_compute * fraction_slow_tokens
```

Where λ_compute penalizes engaging the slow path too often. Start with λ=0, let the model learn to use both paths, then gradually increase λ to push easy tokens toward System 1 only.

## Sequence Mixer: Griffin-Style Gated Recurrence

Both paths use the same type of mixer (different dimensions):

```
a_t = σ(W_a · x_t + b_a)                        # decay gate
i_t = σ(W_i · x_t)                               # input gate
v_t = W_v · x_t                                   # value

h_t = a_t ⊙ h_{t-1} + √(1 - a_t²) ⊙ (i_t ⊙ v_t)  # Griffin coupling
```

**Why Griffin coupling:** Ensures bounded state norm. √(1-a²) automatically scales input when a is large. Proven stable in the Griffin paper (2402.19427).

**Parallel scan for training:** Standard associative operator `(a₂·a₁, a₂·b₁+b₂)` on pairs `(a_t, √(1-a²)·i_t·v_t)`. Adapt `kernels/hip/prefix_scan.py`.

## Slow Path Integration

When the slow path engages, it operates on the fast path's representation:

```python
if engage_slow:
    h_up = proj_up(h_fast)              # 320 → 1280
    for layer in slow_layers:
        h_up = gated_recurrence(h_up)    # d=1280 recurrence
        h_up = swiglu_ffn(rmsnorm(h_up))
    h_correction = proj_down(h_up)       # 1280 → 320
    h_fast = h_fast + h_correction       # residual connection
```

The slow path produces a **correction** to the fast path's state, not a replacement. This means:
- If the slow path is skipped, the fast path's state is still valid
- The slow path learns to ADD what the fast path missed
- During training, both paths always run (slow path gated by soft sigmoid of delta, not hard threshold)

## Training

### Soft Routing During Training

Hard if/else breaks gradient flow. During training, use soft gating:

```python
gate = sigmoid((delta - theta) / temperature)    # soft, differentiable
h_correction = gate * proj_down(slow_path(proj_up(h_fast)))
h_fast = h_fast + h_correction
```

Temperature anneals from 1.0 → 0.1 during training (sharper gating over time).

At inference: hard threshold (actual skip).

### Phase Training

1. **Phase 1 (50%):** Train fast + slow together, gate always 1.0 (all tokens use slow path). Both paths learn.
2. **Phase 2 (30%):** Enable soft gating. Model learns which tokens need slow processing.
3. **Phase 3 (20%):** Anneal temperature, add λ_compute penalty. Push easy tokens to fast-only.

### Hyperparameters

| Parameter | Value |
|-----------|-------|
| Dataset | OpenWebText ~100M tokens |
| Batch | 48 × 512 = 24K tok, grad_accum=2 (48K effective) |
| Optimizer | AdamW (β₁=0.9, β₂=0.95) |
| LR | 8e-4 cosine → 8e-5, warmup 100 steps |
| Weight decay | 0.1 |
| Grad clip | 1.0 |
| λ_compute | 0 → 0.1 (ramped in Phase 3) |
| Temperature | 1.0 → 0.1 (annealed in Phase 2–3) |
| Precision | fp16 mixed + fp32 scan |
| Est. throughput | ~40M tokens in 15 min |

## Decode Speed (Strix Halo)

### fp16

| Token type | Fraction | DRAM reads | Time |
|-----------|----------|-----------|------|
| Easy | ~70% | Fast path from L2 (~0 MB) + LM head (128 MB) | ~0.95 ms |
| Hard | ~30% | Fast (L2) + Slow (310 MB) + LM head (128 MB) | ~2.98 ms |
| **Weighted** | | | **~1.56 ms = ~641 tok/s** |

Note: LM head (128 MB) dominates for easy tokens. Optimization: quantize LM head to int4 (32 MB → 0.19 ms).

### int4

| Token type | Fraction | DRAM reads | Time |
|-----------|----------|-----------|------|
| Easy | ~70% | Fast path (**L2!**) + LM head (32 MB) | ~0.34 ms |
| Hard | ~30% | Fast (L2) + Slow (77.5 MB) + LM head (32 MB) | ~0.89 ms |
| **Weighted** | | | **~0.51 ms = ~1961 tok/s** |

### Comparison

| Model | Params | fp16 tok/s | int4 tok/s |
|-------|--------|-----------|-----------|
| GPT-2 124M | 124M | ~198 | — |
| Spectral Hydra | 244M | ~285 | ~714 |
| Resonant Loop | 59M unique | ~694 | ~1613 |
| **Dual Cortex** | **231M (10M active easy)** | **~641** | **~1961** |

Dual Cortex is competitive on speed AND has 231M total params (2× GPT-2's capacity). The fast path is essentially free. The speed comes from 70% of tokens skipping DRAM reads entirely.

## HIP Kernels

**Reuse:** `fused_residual_add_rmsnorm` (6.6×), `silu_gate_mul` (1.6×), `cross_entropy` (1.8×), `dequantize_int4` (16.3×).

**New (priority order):**
1. **Fused Griffin Recurrence** — sigmoid(a+bias), sigmoid(i), sqrt(1-a²), mul, mul, add → 1 kernel. Adapt `silu_gate_mul.py`. Est. 6–10×.
2. **Associative Scan** — adapt `prefix_scan.py` for `(a₂·a₁, a₂·b₁+b₂)`. Float32 in LDS.
3. **Fused Entropy Check + Skip** (inference) — compute state delta norm + threshold comparison. If skip, jump directly to LM head. Saves kernel launches for 70% of tokens.

## Risks & Mitigations

| Risk | Severity | Mitigation |
|------|----------|------------|
| Fast path too weak (d=320, 8 layers) | HIGH | 70% of tokens ARE genuinely easy; fast path only needs to predict them, not represent all knowledge. If too weak: increase to d=384 or d=448 |
| Entropy threshold miscalibrated | MEDIUM | Learned θ with λ_compute penalty; monitor fast/slow token ratio during training; target 65-75% fast |
| Slow path gradient starvation (only 30% of tokens) | HIGH | Phase 1 trains with gate=1.0 (all tokens); slow path fully trained before gating kicks in |
| Embedding projection bottleneck (1280→320) | MEDIUM | Use two projections with SiLU: 1280→512→320. Or use separate smaller embedding for fast path |
| Fast→slow transition disrupts recurrent state | MEDIUM | Slow path correction is ADDITIVE (residual), doesn't replace fast state. Fast state remains valid regardless |
| Training is harder (two paths, gating, annealing) | MEDIUM | Phase training decouples. Phase 1 is just a standard model with d=320 + d=1280 layers in sequence |

## Success Criteria

1. Loss < 4.5 in 15 min
2. 65-75% of tokens routed to fast path at convergence
3. Easy tokens (articles, prepositions, commas) consistently route fast
4. Hard tokens (rare words, ambiguous constructions) consistently route slow
5. Decode > 600 tok/s fp16, > 1500 tok/s int4 on Strix Halo
6. Per-token efficiency ≥ GPT-2 124M at equal tokens seen

## Implementation Roadmap

1. Implement FastPath: d=320 gated recurrence + SwiGLU (8 layers)
2. Implement SlowPath: d=1280 gated recurrence + SwiGLU (10 layers)
3. Implement soft entropy gating + projection bridges
4. Verify param count, forward/backward, gradient flow through gate
5. Phase 1 training: gate=1.0 (all tokens use slow), 50% of budget
6. Phase 2: enable soft gating, anneal temperature, 30% of budget
7. Phase 3: add λ_compute, push fast ratio to 70%, 20% of budget
8. Decode benchmark: measure fast/slow ratio, latency distribution
9. Fused Griffin recurrence HIP kernel
10. Int4 quantization + L2 cache validation for fast path
11. Compare to GPT-2, Spectral Hydra, Resonant Loop, Chimera Engram

---

## Hardware Optimization Notes (Strix Halo gfx1151)

> Added 2026-04-09 based on AMADEUS implementation findings.

### Existing HIP Kernels to Reuse
- **fused_residual_add_rmsnorm** (6.6x), **silu_gate_mul** (1.6x), **cross_entropy** (1.8x)
- Apply via `autokernel.optimize(model, training=True)`

### Attention Warning (NO MFMA)
If System-2 path uses attention heads: **attention is 0.05x without MFMA on gfx1151.** Use flash_attn (ROCm build available) or replace with linear attention / SSM. If attention is required, minimize to 1-2 layers and accept the MFU penalty.

### Entropy-Gated Routing
The entropy gate (`norm(h_new - h_old) > θ`) is a cheap element-wise op — negligible overhead. The routing decision itself doesn't hurt throughput.

### Scan: Use Chunked Linear Recurrence
For any recurrence in System-1 path. Reference: `models/amadeus.py:selective_scan_chunked`.

### Throughput: ~5-7K tok/s (depends on System-2 activation rate), MFU: 60-75%
If System-2 rarely activates (>80% System-1), throughput approaches Caveman LFM levels. If System-2 activates frequently with attention, throughput drops significantly.

### External Kernel Integration (verified 2026-04-10)

- **Slow path attention:** hybrid_flash_sdpa_attention (8.9% faster than SDPA) — from kernels/hip/hybrid_attention.py
- **GatedConv (if used):** causal-conv1d (10x vs nn.Conv1d) — auto-used if installed

---

## Possible Optimizations & Throughput Estimate

**Baseline (estimated):** ~6,000 tok/s eager (14% MFU)

| Optimization | Expected Impact | Status |
|-------------|----------------|--------|
| `torch.compile(mode="default")` | +100% MFU — two independent paths compile separately | Not tested |
| `autokernel.optimize(model, training=True)` | RMSNorm 6.6x, SwiGLU 1.6x, cross_entropy 1.8x | Available |
| Fast path (d=320, 8L) L2 partial fit | Smaller hidden dim → some weights approach L2 size | By design |
| FLA HGRN for Griffin recurrence | 0.40ms Triton kernel for both paths | Available |
| Entropy gating fusion | Fuse entropy compute + routing into single kernel | Possible |
| Batch=16, seq=256 | L2 sweet spot | Expected |

**Estimated optimized throughput (50 steps):** ~12,000 tok/s (28% MFU)
**Tokens in 45 min:** ~32.4M (2.0 BabyLM epochs)
**Ranking:** #11 of 22 architectures

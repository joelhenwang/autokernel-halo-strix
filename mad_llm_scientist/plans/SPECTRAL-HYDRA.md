# SPECTRAL HYDRA

**Multi-Scale Gated Recurrence with Heterogeneous Temporal Decay**

## Hypothesis

Replace self-attention entirely with **Multi-Scale Gated State Recurrence (MSGR)**: 16 recurrent heads per layer, each initialized to a structurally different temporal decay rate (fast/medium/slow). A cross-head linear mixer combines their outputs. All ops are element-wise — no attention, no O(T²).

**Inspired by:** Mamba (selective gating), minGRU (h=α·h+β·v), Griffin (gated recurrence), LFM2 (hardware-aware conv+attention), Comba (340M recurrence).

**Novel contribution:** Structurally heterogeneous decay initialization across heads as the sole sequence mixer — no attention layers, no routing, no homogeneous SSM.

---

## Architecture

```
Tokens → Embedding (tied LM head)
  → [SpectralHydraBlock × 16]:
      RMSNorm → Conv1d(k=4) → Input Proj → Multi-Scale Gated Recurrence → Cross-Head Mixer → +Residual
      RMSNorm → SwiGLU FFN → +Residual
  → Final RMSNorm → LM Head
```

## Configuration

| Parameter | Value |
|-----------|-------|
| d_model | 1024 |
| n_heads | 16 (5 fast + 6 medium + 5 slow) |
| d_head / d_state | 64 |
| n_layers | 16 |
| ffn_inner | 2560 (SwiGLU) |
| conv_kernel | 4 (depthwise causal) |
| vocab_size | 50257 (tiktoken GPT-2) |
| block_size | 1024 |
| **Total params** | **~244.6M** |

## Core Equations

### State Update (per head i, per timestep t)

```
a_i[t] = σ(W_a · x[t] + b_a_i)     # decay gate: input-dependent + per-head bias
i_i[t] = σ(W_i · x[t])              # input gate
v_i[t] = W_v · x[t]                  # value projection

h_i[t] = a_i[t] ⊙ h_i[t-1] + √(1 - a_i[t]²) ⊙ (i_i[t] ⊙ v_i[t])
```

**Griffin-style coupling** (per 2402.19427): the input coefficient √(1-a²) automatically shrinks when a is large (slow heads retain state, absorb less) and grows when a is small (fast heads forget, absorb more). This guarantees bounded state norm by construction.

**NOT minGRU's β=1-α** (too restrictive) or **independent α,β** (stability bug: α+β can exceed 1 → state overflow). Griffin's formulation keeps two independent gate signals while ensuring stability.

The separate input gate `i_i[t]` provides fine-grained control over WHAT to absorb (via i), while √(1-a²) controls HOW MUCH (coupled to retention a).

### Decay Bias Initialization (The Spectrum)

| Heads | Band | decay_bias | σ(decay_bias) | Role |
|-------|------|--------|-----------|------|
| 0–4 | Fast | -2.2 | ~0.10 | N-grams, local syntax |
| 5–10 | Medium | 0.0 | ~0.50 | Clause structure, agreement |
| 11–15 | Slow | +4.6 | ~0.99 | Topic, entity tracking |

At init, projection weights are near zero (Xavier), so α ≈ σ(b_α_i) — heads start in their intended regime.

### Cross-Head Mixer

```
output = W_mix · concat(h_0..h_15) + b_mix    # (1024×1024) fixed-size, seq-length independent
```

## Parameter Breakdown

| Component (per layer) | Params |
|-----------------------|--------|
| Input projection (→ α, β, v) | 3,148,800 |
| Conv1d (depthwise, k=4) | 5,120 |
| Per-head decay_bias (b_a_i) | 1,024 |
| Cross-head mixer | 1,049,600 |
| SwiGLU (gate + up + down) | 7,865,344 |
| RMSNorm × 2 | 2,048 |
| **Per layer total** | **~12.1M** |

| Global | Params |
|--------|--------|
| Embedding (50257 × 1024, tied) | 51,463,168 |
| Final RMSNorm | 1,024 |
| **Grand total** | **~244.6M** |

## Training

### Parallel Scan

The Griffin-style recurrence `h = a·h + √(1-a²)·(i·v)` is still a first-order linear recurrence, parallelizable via associative scan:

```
(a₂, b₂) ∘ (a₁, b₁) = (a₂·a₁, a₂·b₁ + b₂)
```

Scan over `{(a[t], √(1-a[t]²)·i[t]·v[t])}` produces all states in O(T log T). Use `torch.associative_scan` (PyTorch 2.5+) or adapt `kernels/hip/prefix_scan.py`.

Run scan in **fp32** (inputs/outputs fp16). Log-space fallback if needed.

### Pseudocode

```python
x = embed(tokens)                           # (B, T, 1024)
for layer in layers:
    residual = x
    x = rmsnorm(x)
    x = causal_conv1d(x)
    α_logit, β_logit, v = input_proj(x).chunk(3, dim=-1)
    a = sigmoid(a_logit.view(B,T,H,d) + decay_bias)   # per-head bias
    i_gate = sigmoid(i_logit.view(B,T,H,d))
    v = v.view(B,T,H,d)
    input_val = torch.sqrt(1 - a**2) * i_gate * v     # Griffin coupling
    h = associative_scan(combine_fn, (a, input_val), dim=1)
    x = residual + cross_head_mixer(h.reshape(B,T,1024))
    residual = x
    x = residual + swiglu_ffn(rmsnorm(x))
logits = lm_head(rmsnorm(x))
```

### Hyperparameters

| Parameter | Value |
|-----------|-------|
| Dataset | OpenWebText ~100M tokens |
| Batch | 32 × 512 = 16K tok, grad_accum=4 (64K effective) |
| Optimizer | AdamW (β₁=0.9, β₂=0.95) |
| LR | 6e-4 cosine → 6e-5, warmup 100 steps |
| decay_bias LR | **0.1× base** (preserve spectrum) |
| Weight decay | 0.1 |
| Grad clip | 1.0 |
| Precision | fp16 mixed + fp32 scan |
| Est. throughput | ~20–30M tokens in 15 min |

## Inference (Recurrent Mode)

Sequential state update per token — O(1) per step:

```python
for l, layer in enumerate(layers):
    x = fused_residual_rmsnorm(x, residual, weight)
    x = conv1d_step(x, conv_buffer[l])
    a, i_gate, v = compute_gates(x)
    state[l] = a * state[l] + sqrt(1-a**2) * i_gate * v   # Griffin coupling
    x = residual + cross_head_mixer(state[l].flatten())
    x = residual + swiglu_ffn(fused_rmsnorm(x))
```

### Inference State

| Component | Size |
|-----------|------|
| Recurrent state (16L × 16H × 64d × 2B) | 32 KB |
| Conv ring buffers (16L × 3 × 1024 × 2B) | 96 KB |
| **Total** | **128 KB** (always in L2) |

### Decode Speed (Strix Halo)

| Mode | Estimate |
|------|----------|
| fp16 | ~285 tok/s (~3.5 ms/tok) |
| int4 | ~714 tok/s (~1.4 ms/tok) |

## HIP Kernels

**Reuse:** `fused_residual_add_rmsnorm` (6.6×), `silu_gate_mul` (1.6×), `cross_entropy` (1.8×), `dequantize_int4` (16.3×).

**New (priority order):**
1. **Fused Griffin Recurrence** — sigmoid(logit+bias), sigmoid(logit), sqrt(1-a²), mul, mul, add → 1 kernel. Adapt `silu_gate_mul.py`. Est. 6–10×.
2. **Associative Scan** — adapt `prefix_scan.py` for `(a₂·a₁, a₂·b₁+b₂)` operator. Float32 pairs in LDS.
3. **Fused Conv1d+Gate** (optional) — conv lookback + sigmoid. Saves 2 intermediates.

**Phased:** Start with `torch.compile(mode="reduce-overhead")`, then replace with HIP kernels.

## Risks & Mitigations

| Risk | Severity | Mitigation |
|------|----------|------------|
| No attention = weak retrieval | HIGH | Griffin's Hawk confirms: pure recurrence fails copy/retrieval. **Plan for 1–2 local attention layers (window=128) at L/2 and L as likely required, not just fallback.** |
| fp16 scan instability | MEDIUM | fp32 scan accumulation; log-space fallback |
| Head specialization collapse | MEDIUM | 0.1× LR on decay_bias; diversity regularization L_div = -λ·Var(decay_bias_means) |
| 15-min training insufficient | HIGH | Focus on per-token efficiency trend, not absolute quality |

## Success Criteria

1. Loss < 5.0 in 15 min (random ~10.8)
2. Per-token efficiency ≥ GPT-2 124M
3. decay_bias histogram shows distinct clusters (not collapsed)
4. Decode > 250 tok/s fp16 on Strix Halo

## Implementation Roadmap

1. PyTorch model class (sequential recurrence for correctness)
2. Parallel scan integration
3. Verify param count, gradient flow, shapes
4. Training run (15 min), log loss + α distributions
5. Fused gated recurrence HIP kernel
6. Associative scan HIP kernel
7. Int4 quantization + decode benchmark
8. Compare to GPT-2 baseline

---

## Hardware Optimization Notes (Strix Halo gfx1151)

> Added 2026-04-09 based on AMADEUS implementation findings.

### Existing HIP Kernels to Reuse
- **fused_residual_add_rmsnorm** (6.6x), **silu_gate_mul** (1.6x), **cross_entropy** (1.8x)
- **rotary_emb** (3.7x) — if RoPE is used on recurrence projections
- Apply via `autokernel.optimize(model, training=True)`

### Scan Implementation
16 heterogeneous recurrence heads × 64 dims = 1024 — same shape as AMADEUS's Mamba-3 SSM. **Use chunked linear recurrence** (chunk_size=64), NOT sequential or `torch.associative_scan`. Reference: `models/amadeus.py:selective_scan_chunked`.

### MFU Potential: 85-90% (HIGHEST of all plans)
- All-element-wise recurrence → ~95% MFU (L2-cache bound, not compute-bound)
- Only SwiGLU FFN has matmuls → ~80% MFU (weight-bound)
- **Overall: 85-90% training MFU** — best theoretical hardware utilization

### Critical Decision: Pure Recurrence vs Hybrid
**Pure recurrence has known retrieval weakness** (acknowledged in risk section). Two paths:
1. **Pure recurrence (recommended for MFU):** Accept retrieval gap. Best tok/s and MFU. Quality limited on factual/lookup tasks.
2. **Hybrid with 1-2 attention layers:** MFU drops to ~60% average because attention is 0.05x without MFMA. Throughput drops ~30%.

**Recommendation:** Start pure recurrence. If eval shows >10% retrieval gap vs Caveman, add attention layers as a separate experiment — don't design them in from the start.

### Throughput Estimates (CORRECTED)
- Previous "~20-30M tokens in 15 min" is too optimistic
- **Eager mode:** ~6-8K tok/s, 16-20% MFU → 5.4-7.2M in 15 min
- **With autokernel:** ~7-9K tok/s → 6.3-8.1M in 15 min
- MFU advantage vs Caveman only materializes with HIP-fused scan kernel

### Decay Bias Verification
After training, plot histogram of α = sigmoid(linear + decay_bias) per dimension band. If fast/medium/slow bands don't separate, increase decay_bias LR from 0.1x to 0.2x base.

### External Kernel Integration (verified 2026-04-10)

- **GatedConv:** causal-conv1d (10x vs nn.Conv1d) — auto-used if installed
- **Element-wise recurrence:** FLA HGRN (0.40ms) as alternative to custom per-dim recurrence
- **Griffin scan:** Chunked linear recurrence or FLA chunk_hgrn

---

## Possible Optimizations & Throughput Estimate

**Baseline (estimated):** ~7,500 tok/s eager (19% MFU)

| Optimization | Expected Impact | Status |
|-------------|----------------|--------|
| `torch.compile(mode="default")` | +113% MFU — all element-wise recurrence, highly fusable | Not tested |
| `autokernel.optimize(model, training=True)` | RMSNorm 6.6x, SwiGLU 1.6x, cross_entropy 1.8x | Available |
| `causal-conv1d` for Conv1d(k=4) | 10x conv speedup (16 layers x 1024 channels) | Available |
| FLA HGRN for per-dim recurrence | 0.40ms Triton kernel matches Griffin shape | Available |
| Cross-head mixer fusion | Fuse 16-head concat + linear into single GEMM | With compile |
| Batch=16, seq=256 | L2 sweet spot | Expected |

**Estimated optimized throughput (50 steps):** ~16,000 tok/s (40% MFU)
**Tokens in 45 min:** ~43.2M (2.7 BabyLM epochs)
**Ranking:** #5 of 22 architectures

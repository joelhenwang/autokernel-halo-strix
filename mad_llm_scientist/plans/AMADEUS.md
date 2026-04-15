---
title: "AMADEUS"
domain: architectures
type: plan
status: active
related:
  - mad_llm_scientist/COOKBOOK.md
  - knowledge/architectures/hypothesis_buildout_results.md
tags: [%hypothesis, %plan, %amadeus]
---

# AMADEUS

**The Simple Masterpiece: Parallel Hybrid + FiLM Conditioning — 4 Components, 1 Innovation**

*"Salieri envied Mozart's elaborate symphonies. Then he wrote one perfect piece."*
*We are Salieri. Liquid AI is Mozart. LFM2.5 is the symphony. AMADEUS is our answer.*

## Hypothesis

The simplest competitive architecture — gated conv (local) + Mamba-3 SISO (global) + SwiGLU FFN (transformation) — plus ONE novel seasoning nobody has tried in language modeling: **FiLM conditioning from the midpoint.** The first half UNDERSTANDS. The second half GENERATES conditioned on that understanding. 4 components. 2 training phases. ~400 lines of code.

---

## Architecture

```
Tokens → Embedding (d=1024, tied LM head, vocab=50257)
  │
  → 16 Parallel Hybrid Blocks:
  │     RMSNorm
  │     ┌─────────────────────────────────────────┐
  │     │ Gated Conv (10ch)  ║  Mamba-3 SISO      │
  │     │ B⊙h̃ → Conv₃ → C⊙z ║  (6 heads, dstate=64) │
  │     │ d_conv=640         ║  d_mamba=384        │
  │     │ LOCAL patterns     ║  GLOBAL context     │
  │     └─────────────────────────────────────────┘
  │     Concat (640+384=1024) → OutProj → +Residual
  │     RMSNorm → SwiGLU FFN (1024→2560→1024) → +Residual
  │
  │     ★ LAYER 8: Context Fingerprint ★
  │     ★ LAYERS 9-16: FiLM Conditioned ★
  │
  → Final RMSNorm → LM Head
```

---

## Component 1: Gated Short Conv (LOCAL — 3 tokens)

```python
B, C, h_tilde = linear(x_norm, 1024 → 3×640).chunk(3)
y = B * h_tilde                    # element-wise gate
z = causal_conv1d(y, k=3)          # depthwise, 640 channels
conv_out = C * z                    # (B, T, 640)
```

LFM2 hardware-search validated. Element-wise ops. Fast on Strix Halo.

## Component 2: Mamba-3 SISO (GLOBAL — linear complexity)

```python
# Data-dependent SSM (6 heads × 64 dims = 384)
x_proj = linear(x_norm, 1024 → 384)           # input
dt = softplus(linear(x_norm, 1024 → 384))     # discretization step
B = linear(x_norm, 1024 → 64)                 # data-dependent B
C = linear(x_norm, 1024 → 64)                 # data-dependent C

# Data-dependent RoPE on B, C (NON-NEGOTIABLE: 100% vs 1.56% parity)
theta = learned_theta_proj(x_norm)
B = apply_rope(B, theta)
C = apply_rope(C, theta)

# SSM: A = diag(-exp(A_log)), discretized with dt
# Training: parallel associative scan (fp32). Inference: sequential.
```

**Why Mamba-3 solves the context problem:** SSD framework proves SSMs and attention are DUAL. Data-dependent A,B,C + data-dependent RoPE = attention-quality context at O(T) complexity. No MFMA needed.

## Component 3: SwiGLU FFN (TRANSFORMATION)

```python
gate, up = linear(x_norm, 1024 → 2×2560).chunk(2)
out = linear(silu(gate) * up, 2560 → 1024)
```

## Component 4: FiLM Conditioning (THE NOVEL SEASONING)

**FiLM** (Feature-wise Linear Modulation) conditions generation on a context vector. Used in image diffusion. **Never tried in autoregressive language modeling.**

```python
# === AFTER LAYER 8 ===
context = context_proj(h.mean(dim=1))         # (B, d) → (B, 64) — "what is this text about?"

# === LAYERS 9-16: before each block ===
gamma = gamma_proj[i](context) + 1.0          # (B, d) — scale (init: identity)
beta = beta_proj[i](context)                  # (B, d) — shift (init: zero)
h = gamma.unsqueeze(1) * h + beta.unsqueeze(1)  # FiLM: channel-wise affine
```

**Cost:** 1.1M params (0.5% of model). Context proj: d→64 shared. Per-layer: 64→d × 2. That's it.

**What it does:** Layers 1-8 build understanding. Layer 8 compresses it to a 64-dim fingerprint. Layers 9-16 generate CONDITIONED on that fingerprint. Every token in the second half knows the "genre" of the input.

**Inference cost:** Near-zero. The fingerprint is computed once at prefill. FiLM modulation is an element-wise multiply+add per layer.

---

## Configuration

| Parameter | Value |
|-----------|-------|
| d_model | 1024 |
| d_conv | 640 (10 × 64) |
| d_mamba | 384 (6 × 64) |
| dstate | 64 |
| n_layers | 16 |
| ffn_inner | 2560 (SwiGLU, 2.5×) |
| d_film | 64 |
| film_layers | 9-16 |
| conv_kernel | 3 |
| vocab_size | 50257 |
| block_size | 1024 |
| weight_tying | yes |

## Parameter Count

| Component | Params |
|-----------|--------|
| Embedding (50257×1024, tied) | 51.5M |
| Per layer: gated conv | ~2.0M |
| Per layer: Mamba-3 SISO | ~0.9M |
| Per layer: output proj | ~1.05M |
| Per layer: SwiGLU FFN | ~7.86M |
| Per layer: RMSNorm ×2 | ~2K |
| **Per layer total** | **~11.81M** |
| **16 layers** | **189.0M** |
| FiLM (1 context proj + 8 gamma/beta) | 1.1M |
| Final RMSNorm | 1K |
| **GRAND TOTAL** | **~241.6M** |

## Decode Speed (Strix Halo)

| Mode | Weight Reads | Overhead | Total | Throughput |
|------|-------------|----------|-------|------------|
| fp16 | 481 MB / 240 GB/s = 2.0ms | 0.48ms | ~2.48ms | **~403 tok/s** |
| int4 | 223 MB / 240 GB/s = 0.93ms | 0.48ms | ~1.41ms | **~709 tok/s** |

---

## Training

### 2 Phases (the simplest in our lab)

| Phase | Budget | Active | Purpose |
|-------|--------|--------|---------|
| 1 (60%) | 9 min | Full backbone. FiLM OFF. | Learn language |
| 2 (40%) | 6 min | + FiLM conditioning | Learn context-conditioned generation |

### Hyperparameters

| Parameter | Value |
|-----------|-------|
| Optimizer | AdamW (single group) |
| LR | 8e-4 cosine → 8e-5, 100-step warmup |
| Weight decay | 0.1 |
| Batch | 48×512, accum=2 (48K effective) |
| Precision | fp16 mixed + fp32 Mamba-3 scan |
| Grad clip | 1.0 |
| Gradient checkpointing | Every 4 layers |

**ONE optimizer group.** No special LR for Engram (none). No special LR for decay bias (none). No special LR for meta tokens (none). Just one LR for everything. Simple.

---

## HIP Kernels

**Reuse (4):** fused_residual_add_rmsnorm (6.6×), silu_gate_mul (1.6×), cross_entropy (1.8×), dequantize_int4 (16.3×)

**New (2):**
1. **Mamba-3 Associative Scan** — Adapt prefix_scan.py for Mamba-3 (a,b) operator with complex state. fp32 in LDS.
2. **Fused Gated Conv Block** — B⊙h̃ + conv_step(k=3) + C⊙z in one kernel.

**FiLM needs NO custom kernel** — it's an element-wise multiply+add. PyTorch handles it.

---

## Why AMADEUS Wins

| | ARCHON | Caveman LFM | **AMADEUS** |
|---|---|---|---|
| Components | 9 | 5 | **4** |
| Training phases | 4 | 3 | **2** |
| Optimizer groups | 7 | 4 | **1** |
| Novel modules | 5+ | 0 | **1** |
| Est. code | ~2000 | ~800 | **~400** |
| Context mechanism | Griffin (decays) | Griffin (decays) | **Mamba-3 (linear attention)** |
| Innovation | Combination | LFM2 adapted | **FiLM (new to LM)** |

---

## Risks & Mitigations

| Risk | Severity | Mitigation |
|------|----------|------------|
| Mamba-3 complex to implement | MEDIUM | Follow paper exactly. Data-dependent RoPE is non-negotiable. |
| FiLM doesn't help at 250M | LOW | 1.1M params. Disable = still competitive base. Zero risk. |
| No Engram = weaker factual | MEDIUM | Mamba-3 state carries context. FFN stores knowledge. Accept trade-off. |
| Too simple to compete | LOW | LFM2 proved conv+FFN is competitive. We ADD Mamba-3 (global) + FiLM (novel). |

## Success Criteria

1. Loss < 4.5 in 15 min
2. FiLM improves loss > 0.5% vs no-FiLM ablation
3. Mamba-3 context: copying/counting tasks > 50%
4. Decode > 300 tok/s fp16
5. Implementation < 500 lines of PyTorch

## Implementation Roadmap

1. Implement RMSNorm + SwiGLU FFN (from COOKBOOK)
2. Implement GatedConv (from COOKBOOK)
3. Implement Mamba-3 SISO (data-dependent A,B,C + RoPE, parallel scan)
4. Implement ParallelHybridBlock (conv || Mamba-3 → concat → outproj)
5. Assemble 16 blocks, verify ~242M params
6. Phase 1: train backbone (9 min)
7. Implement FiLM conditioning (context proj + gamma/beta per layer)
8. Phase 2: train with FiLM (6 min)
9. Ablation: with/without FiLM
10. Mamba-3 scan kernel (adapt prefix_scan.py)

---

## Actual Results (implemented in `models/amadeus.py`)

**Model:** 243.8M params (confirmed). Implementation: ~300 lines.

### SSM Scan Optimization

| Implementation | tok/s | MFU | Notes |
|---------------|-------|-----|-------|
| Sequential loop | 1,300 | 4% | Python for-loop, 512 serial steps/layer |
| `torch.associative_scan` | 1,300 | 4% | Equally slow on gfx1151 |
| **Chunked linear recurrence** | **6,400** | **16%** | chunk_size=64, 8 serial steps/layer |

### Training on BabyLM (eager mode, no compile/autokernel)

| Step | Loss | BPB | tok/s | MFU | Memory |
|------|------|-----|-------|-----|--------|
| 10 | 42.2 | 16.9 | 6,233 | 15.4% | 12.7 GB |
| 100 | 19.8 | 7.95 | 6,463 | 15.9% | 12.7 GB |
| 300 | 16.6 | 6.67 | 6,463 | 15.9% | 12.7 GB |
| 560 | 15.5 | 6.19 | 6,462 | 15.9% | 12.7 GB |

### Key Implementation Lessons

- **Chunked scan is critical** — 5x faster than sequential. Uses cumprod+cumsum within chunks.
- **SSM init:** A_log=log(arange(1,N+1)), dt_proj bias=-4.0, dt clamped [1e-4, 0.5], B/C normalized by max(norm,1)
- **FiLM identity init:** zero weights/biases so initial transform is h×1+0=h
- **GradScaler inf during warmup is normal** — scaler adjusts scale down, training continues
- **StateNormMonitor gives false positives** for data-dependent SSMs (max_ratio 3-5 is normal)

### External Kernel Integration (verified 2026-04-10)

- **GatedConv:** causal-conv1d (10x vs nn.Conv1d) — `try/except` import in `models/amadeus.py`, auto-used if installed
- **SSM scan:** mamba-ssm `selective_scan_fn` (5.6x vs HIP kernel, 0.32ms) — highest priority in `_scan_dispatch()`, replaces HIP and chunked fallbacks
- **Expected throughput with external kernels:** 12-15K tok/s (vs 10.4K baseline)

---

## Possible Optimizations & Throughput Estimate

**Baseline (verified):** 6,400 tok/s eager (16% MFU), 10,400 tok/s with autokernel+compile+HIP scan (26% MFU)

| Optimization | Expected Impact | Status |
|-------------|----------------|--------|
| `torch.compile(mode="default")` | +60% MFU (16% → 26%) | **Verified** |
| `autokernel.optimize(model, training=True)` | RMSNorm 6.6x, SwiGLU 1.6x, cross_entropy 1.8x | **Verified** |
| `causal-conv1d` in GatedConv | 10x conv speedup, saves ~0.2ms/layer | **Wired in** |
| `mamba-ssm` selective_scan_fn | 5.6x vs HIP scan (0.32ms), saves ~1ms/layer | **Wired in** |
| Batch=16, seq=256 | L2 sweet spot (vs batch=4: +40% tok/s) | **Verified** |
| FiLM removal (layers 9-16) | Eliminates fingerprint compute; ~2% speedup | Not tested |

**Estimated optimized throughput (50 steps):** ~13,000 tok/s (32% MFU)
**Tokens in 45 min:** ~35.1M (2.2 BabyLM epochs)
**Ranking:** #8 of 22 architectures

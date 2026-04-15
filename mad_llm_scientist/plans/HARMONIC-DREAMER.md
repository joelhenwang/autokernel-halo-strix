---
title: "HARMONIC-DREAMER"
domain: architectures
type: plan
status: stale
related:
  - mad_llm_scientist/COOKBOOK.md
  - knowledge/architectures/hypothesis_buildout_results.md
tags: [%hypothesis, %plan, %harmonic-dreamer]
---

# HARMONIC DREAMER

**Damped Harmonic Oscillator Recurrence + Dynamic Scratchpad on Parallel Hybrid Backbone**

## Hypothesis

Every recurrence in our architectures is first-order: `h = a*h + input`. First-order recurrence can only DECAY — it's a lowpass filter. But language has periodic patterns: subject-verb-object repetition, claim-evidence-claim discourse, syntactic rhythm. A first-order recurrence literally cannot capture periodicity.

**Damped Harmonic Oscillator (DHO)** is a second-order recurrence that OSCILLATES while decaying. Combined with a **Dynamic Scratchpad** (input-dependent meta tokens computed via tiny cross-attention), this creates a model with temporal rhythm AND a per-input thinking buffer.

**Evidence:**
- Griffin/Mamba recurrence = first-order = lowpass only (no oscillation)
- Mamba-3 complex SSM has rotation but not true damped oscillation with learnable frequency/damping
- Hymba meta tokens are static (same for every input); our scratchpad is dynamic (input-dependent)
- DHO worst case: heads learn overdamped regime (gamma >> omega) = degrades to Griffin. No quality risk.

---

## Architecture

```
Input → Embedding (d=1024, tied LM head, vocab=50257)
  │
  ├─ Scratchpad Encoder: 32 learned queries × cross-attn on input
  │
  [32 Scratchpad Tokens] + [Input Tokens] → (B, 32+T, 1024)
  │
  → 16 Parallel Hybrid Blocks:
  │     RMSNorm
  │     ┌──────────────────────────────────────┐
  │     │ Gated Conv (10ch)  ║  DHO Recurrence │
  │     │ B⊙h̃ → Conv₃ → C⊙z ║  (6 heads)      │
  │     │ d_conv=640         ║  d_dho=384       │
  │     │                    ║  2 fast  (π/4)   │
  │     │                    ║  2 medium(π/16)  │
  │     │                    ║  2 slow  (π/64)  │
  │     └──────────────────────────────────────┘
  │     Concat (640+384=1024) → OutProj (1024→1024) → +Residual
  │     RMSNorm → SwiGLU FFN (1024→2240→1024) → +Residual
  │     (+ Engram injection at layers 2, 9)
  │
  → Discard scratchpad positions
  → Final RMSNorm → 2 MTP Heads → Shared LM Head
```

---

## Component Details

### 1. DHO Recurrence (Novel Primitive)

Complex-valued first-order recurrence = real-valued second-order (damped oscillator):

```python
# Complex recurrence per DHO head (64 dims each, 6 heads = 384 total)
x_gate = sigmoid(linear(x_norm, 1024 → 384))       # input gate
x_val  = linear(x_norm, 1024 → 384)                 # value projection
x_complex = (x_gate * x_val) + 0j                   # lift to complex

# Learned oscillator parameters (per-dim, 384 each)
r = exp(-softplus(gamma_param))                      # decay magnitude ∈ (0, 1)
omega = softplus(omega_param)                        # angular frequency ∈ (0, ∞)
pole = r * exp(1j * omega)                           # complex pole

# Inference: sequential complex multiply
z_t = pole * z_{t-1} + x_complex_t
h_t = z_t.real                                       # real-valued output

# Training: complex associative scan (adapt prefix_scan.py)
# Operator: (a₂, b₂) ⊕ (a₁, b₁) = (a₂·a₁, a₂·b₁ + b₂) with complex multiply
z = complex_parallel_scan(pole.expand(T), x_complex)  # O(T log T)
h = z.real                                             # (B, T, 384)
```

### DHO Head Initialization

| Heads | ω_init | γ_init | r = e^(-γ) | Period | Captures |
|-------|--------|--------|------------|--------|----------|
| 0-1 (fast) | π/4 | 0.3 | 0.74 | ~8 tokens | Syntax oscillation |
| 2-3 (medium) | π/16 | 0.05 | 0.95 | ~32 tokens | Clause structure |
| 4-5 (slow) | π/64 | 0.01 | 0.99 | ~128 tokens | Topic/discourse |

**Key difference from Spectral Hydra:** Spectral Hydra has multi-scale DECAY (lowpass). DHO has multi-scale OSCILLATION (bandpass). DHO captures periodicity; Spectral Hydra cannot.

### 2. Dynamic Scratchpad

```python
class DynamicScratchpad(nn.Module):
    def __init__(self, n_scratch=32, d=1024):
        self.queries = nn.Parameter(torch.randn(n_scratch, d) * 0.02)
        self.key_proj = nn.Linear(d, d)
        self.val_proj = nn.Linear(d, d)

    def forward(self, x):  # x: (B, T, d) — embedded input
        K = self.key_proj(x)                                # (B, T, d)
        V = self.val_proj(x)                                # (B, T, d)
        Q = self.queries.unsqueeze(0).expand(x.size(0), -1, -1)  # (B, 32, d)
        attn = F.softmax(Q @ K.transpose(-1, -2) / 32.0, dim=-1) # (B, 32, T)
        return attn @ V                                      # (B, 32, d)
```

- **Prefill:** 33M FLOPs at T=1024. Negligible (0.017ms on Strix Halo).
- **Decode:** Scratchpad is FIXED after prefill. Zero per-token overhead.
- **vs Hymba meta tokens:** Hymba's 128 tokens are static (same for every input). Our 32 tokens are input-dependent — they summarize THIS specific input.

### 3. Engram (Layers 2, 9)

Standard Engram: bigram + trigram + 4-gram hash tables with context-aware gating. Shared tables, per-layer W_K/W_V. Adam optimizer, 5× LR, zero weight decay.

---

## Configuration

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| d_model | 1024 | Standard for ~250M |
| d_conv | 640 (10 × 64) | LFM2 gated conv proven |
| d_dho | 384 (6 × 64) | Novel DHO heads |
| n_layers | 16 | Standard depth |
| ffn_inner | 2240 (SwiGLU) | Fits param budget |
| n_scratch | 32 | Input-dependent buffer |
| conv_kernel | 3 | LFM2 validated |
| Engram layers | 2, 9 | Factual knowledge |
| MTP heads | 2 | Modest burst decoding |
| vocab_size | 50257 | tiktoken GPT-2 |
| block_size | 1024 | |

## Parameter Count

| Component | Params |
|-----------|--------|
| Embedding (50257×1024, tied) | 51.5M |
| Scratchpad encoder (queries + K,V proj) | 2.1M |
| Per layer: gated conv (1024→3×640 + conv₃) | ~2.0M |
| Per layer: DHO (gate + value + ω,γ) | ~0.8M |
| Per layer: output proj (1024→1024) | ~1.05M |
| Per layer: SwiGLU FFN (1024→2240→1024) | ~6.88M |
| Per layer: RMSNorm ×2 | ~2K |
| **Per layer total** | **~10.73M** |
| **16 layers** | **171.7M** |
| Engram tables (65K×96 + 32K×96 + 16K×96) | 10.9M |
| Engram projections (2 layers) | 1.6M |
| MTP adapters (2) | 3.1M |
| Final RMSNorm | 1K |
| **GRAND TOTAL** | **~241M** |

## Decode Speed (Strix Halo)

| Mode | Weight Reads | Kernel Overhead (16L×6×5μs) | Total | Throughput |
|------|-------------|----------------------------|-------|------------|
| fp16 | 446 MB / 240 GB/s = 2.62ms | 0.48ms | ~3.1ms | **~323 tok/s** |
| int4 | 215 MB / 240 GB/s = 1.26ms | 0.48ms | ~1.74ms | **~575 tok/s** |
| int4 + MTP burst (1.5×) | ~1.74ms for 1.5 tok | | | **~862 tok/s** |

Scratchpad: zero decode overhead (computed once at prefill). Engram: negligible (hash lookup).

---

## Training

### Optimizer Groups

| Group | Optimizer | LR | Weight Decay |
|-------|-----------|-----|-------------|
| Backbone (conv, FFN, norms, outproj) | AdamW | 8e-4 cosine → 8e-5 | 0.1 |
| DHO projections (gate, value) | AdamW | 8e-4 | 0.1 |
| DHO ω (frequency) | AdamW | **1e-4 (0.125×)** | **0** |
| DHO γ (damping) | AdamW | **1e-4 (0.125×)** | **0** |
| Scratchpad queries | AdamW | 8e-4 | 0.01 |
| Scratchpad K/V proj | AdamW | 8e-4 | 0.1 |
| Engram tables | **Adam** | **4e-3 (5×)** | **0** |
| Engram projections | AdamW | 8e-4 | 0.1 |
| MTP adapters | AdamW | 8e-4 | 0.1 |

**Critical:** ω and γ use 0.125× base LR and zero WD to preserve the initialized frequency spectrum.

### Phase Training (15-minute budget)

| Phase | Duration | Tokens | Components Active | Purpose |
|-------|----------|--------|-------------------|---------|
| 1 (35%) | 5.25 min | ~10M | Backbone (conv+DHO+FFN). No scratch, no Engram, no MTP. | Establish oscillation spectrum |
| 2 (30%) | 4.5 min | ~9M | + Scratchpad + Engram | Scratchpad learns to summarize; Engram absorbs facts |
| 3 (35%) | 5.25 min | ~10M | + MTP heads. Full model. | Multi-token signal improves backbone |

### Precision

| Component | Precision | Reason |
|-----------|-----------|--------|
| DHO scan + ω,γ params | **fp32 complex** | Accumulated complex products need precision |
| Conv channels | fp16 | Standard |
| FFN, norms, Engram | fp16 | Standard |
| Scratchpad cross-attn | fp16 | Small (32×T) |

### Other Hyperparams

| Parameter | Value |
|-----------|-------|
| Dataset | OpenWebText ~100M tokens |
| Batch | 32 × 512 = 16K, accum=4 (64K effective) |
| Warmup | 100 steps |
| Grad clip | 1.0 |
| Gradient checkpointing | Every 4 layers |

---

## HIP Kernels

**Reuse (5):** fused_residual_add_rmsnorm (6.6×), silu_gate_mul (1.6×), prefix_scan (8.4×), cross_entropy (1.8×), dequantize_int4 (16.3×)

**New (3):**
1. **Complex Associative Scan** — Adapt prefix_scan.py for complex (a,b) operator. fp32 complex in LDS. 2× real scan cost but d_dho=384 (smaller than typical d_rec).
2. **Fused DHO Gating** — sigmoid gate + value proj + complex lift + output real-part extraction. One kernel per layer.
3. **Fused Scratchpad Cross-Attention** — 32 queries × T keys/values. Small enough for single wavefront per query. Run once at prefill.

---

## Risks & Mitigations

| Risk | Severity | Mitigation |
|------|----------|------------|
| DHO frequencies collapse (all heads learn same ω) | HIGH | 0.125× LR for ω; diverse init; monitor ω spread |
| Complex scan slower than real scan | MEDIUM | d_dho=384 < typical d_rec=1024; net cost manageable |
| Scratchpad cross-attn = matmul on Strix Halo | LOW | 32×T only, 33M FLOPs = 0.017ms. Negligible. |
| DHO oscillation useless for language | MEDIUM | Worst case: heads learn overdamped (γ>>ω) = Griffin. No quality risk. |
| 32 scratch tokens insufficient | LOW | Increase to 64 costs only +2.1M params, +0.03ms |

## Success Criteria

1. DHO ω values remain spread after training (std > 0.5)
2. At least 2 DHO heads underdamped (γ < ω) — oscillation is used
3. Scratchpad attention weights are non-uniform across queries
4. Loss < 4.5 in 15 min
5. Removing scratchpad degrades loss > 0.5%
6. Decode > 300 tok/s fp16 on Strix Halo

## Implementation Roadmap

1. Implement DHO recurrence (complex-valued, sequential + parallel scan)
2. Implement DynamicScratchpad (learned queries, cross-attention encoder)
3. Implement ParallelHybridBlock (gated conv 10ch || DHO 6 heads)
4. Assemble HarmonicDreamerModel, verify ~241M params
5. Phase 1 training: backbone only (conv+DHO+FFN)
6. Phase 2: + scratchpad + Engram
7. Phase 3: + MTP heads
8. Complex associative scan kernel (adapt prefix_scan.py)
9. Fused DHO gating kernel
10. Int4 quantization + decode benchmark

---

## Hardware Optimization Notes (Strix Halo gfx1151)

> Added 2026-04-09 based on AMADEUS implementation findings.

### Existing HIP Kernels to Reuse
- **fused_residual_add_rmsnorm** (6.6x), **silu_gate_mul** (1.6x), **cross_entropy** (1.8x)
- Apply via `autokernel.optimize(model, training=True)`

### DHO Recurrence: fp32 Complex Scan Required
Damped Harmonic Oscillator uses complex poles `r * exp(1j * omega)`. The scan must use **fp32 complex** dtype — fp16 complex loses precision on accumulated products. Use chunked linear recurrence adapted for complex multiplication:
- Decay: `dA = r * exp(1j * omega * dt)` (complex scalar per dimension)
- State: `state = dA * state + dBx` (complex state vector)
- Reference: `models/amadeus.py:selective_scan_chunked` — replace real exp with complex exp

### Multi-Scale Oscillation
The 3 frequency bands (8/32/128 token periods) are set via omega init. Verify after training that bands remain separated — plot omega histogram. If bands merge, increase omega LR from 0.125x to 0.25x.

### Throughput: ~5-7K tok/s (complex ops are 2x slower than real), MFU: 55-65%
Complex arithmetic doubles the FLOPs vs real. Budget accordingly.

### External Kernel Integration (verified 2026-04-10)

- **DHO recurrence:** FLA DeltaNet (1.60ms) as alternative to custom DHO implementation
- **GatedConv (if used):** causal-conv1d (10x vs nn.Conv1d) — auto-used if installed

---

## Possible Optimizations & Throughput Estimate

**Baseline (estimated):** ~4,500 tok/s eager (11% MFU)

| Optimization | Expected Impact | Status |
|-------------|----------------|--------|
| `torch.compile(mode="default")` | +100% MFU — DHO is element-wise but 2nd-order adds complexity | Not tested |
| `autokernel.optimize(model, training=True)` | RMSNorm 6.6x, SwiGLU 1.6x, cross_entropy 1.8x | Available |
| Complex DHO scan | Needs complex-valued chunked scan; no off-the-shelf kernel | Needs custom |
| Dynamic scratchpad cross-attention | 32 queries × d_model adds ~3ms/layer; heavy if many layers | By design |
| MTP (2 heads) overhead | ~3% additional training compute | By design |
| Batch=16, seq=256 | L2 sweet spot | Expected |

**Estimated optimized throughput (50 steps):** ~9,000 tok/s (22% MFU)
**Tokens in 45 min:** ~24.3M (1.5 BabyLM epochs)
**Ranking:** #19 of 22 architectures

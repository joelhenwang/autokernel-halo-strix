---
title: "Backward Pass Optimization Design"
domain: design-specs
type: spec
status: active
related:
  - docs/superpowers/specs/2026-04-09-adaptive-lm-head-design.md
  - docs/superpowers/specs/2026-04-10-training-pipeline-optimization-design.md
  - docs/possible_techniques_bwd_improv.md
tags: [%backward, %optimization, %chunked-ce, %fusion]
---

# Backward Pass Optimization Design

**Date:** 2026-04-10
**Status:** Approved
**Target:** gfx1151 (RDNA 3.5, 40 CUs, no MFMA, ~240 GB/s)

## Problem

Backward pass is 53% of the training step. All custom op backward implementations are currently pure PyTorch — no HIP kernels. The same fusion strategy that achieved 6-28x forward speedups has not been applied to backward.

### Current Backward Profile (AMADEUS 243.8M, eager)

| Component | % of Step | Backward Ops | Custom Kernel? |
|-----------|-----------|-------------|----------------|
| Backward total | 53% | — | No |
| Forward total | 20% | — | Yes (HIP) |
| Optimizer | 19% | — | AdamW fused |
| Grad clip | 7% | — | PyTorch |
| Loss | 1.3% | — | Chunked CE |

### Current Backward Op Breakdown (per layer, pure PyTorch)

| Op | Backward Implementation | Bottleneck | Calls/Step (16L) |
|----|------------------------|-----------|-----------------|
| RMSNorm | 5 ops: fp32 cast + recompute rms + reduction + chain rule | Per-position reduction not fused | 32 |
| Rotary Embedding | 3 ops: chunk + rotate_half (view+cat) + multiply | Permutation via 3 separate ops | 16 |
| SiLU Gate Mul | 3 ops: sigmoid + derivative + multiply | Element-wise not fused | 16 |
| Fused Res+RMSNorm | 8 ops: dual output merge + full RMSNorm backward | Dual path not merged | 16 |
| Selective Scan (AMADEUS) | 2 sequential Python loops over T=1024 | **Serial: 1024 steps** | 16 |
| PLE Gate | 4 GEMMs + GELU via autograd context | Autograd overhead | 16 |
| Hybrid Attention | HIP-native aten backward | Already optimized | 2-16 |
| Chunked CE | 2 GEMMs per chunk + fp16→fp32 | Conversion overhead | 1 |

## Approach

Three workstreams executed in order, each tested in isolation then combined.

### Workstream 1: Fused Backward HIP Kernels (Low Risk, High Impact)

Write HIP backward kernels mirroring forward fusion wins. Register via `torch.library` custom op backward.

#### 1a. Fused RMSNorm Backward

**Current:** 5 PyTorch ops per call (fp32 cast, recompute rms_inv, normed * grad, sum reduction, chain rule).

**Fused kernel:** Single HIP kernel computing `grad_x` and `grad_weight` in one pass.
```
Phase 1: Load x, weight, grad_output. Recompute rms_inv in shared memory.
Phase 2: Compute normed = x * rms_inv.
Phase 3: Warp reduction for inner_sum = (grad_output * weight * normed).sum(-1).
Phase 4: grad_x = rms_inv * (grad_output * weight - normed * inner_sum / D).
Phase 5: Atomically accumulate grad_weight += grad_output * normed.
```
- **Expected speedup:** 1.5-2x per call
- **Calls per step:** 32 (2 per layer × 16 layers)
- **File:** `kernels/hip/rmsnorm_backward.py`

#### 1b. Fused Rotary Embedding Backward

**Current:** chunk → rotate_half (view+neg+cat) → multiply+add. Three separate tensor ops.

**Fused kernel:** Single HIP kernel applying inverse rotation.
```
For each (batch, seq, head, d_pair):
  grad_x[d]   = grad_out[d] * cos + grad_out[d+1] * sin
  grad_x[d+1] = grad_out[d+1] * cos - grad_out[d] * sin
```
- **Expected speedup:** 1.3x per call
- **Calls per step:** 16
- **File:** `kernels/hip/rotary_embedding_backward.py`

#### 1c. Fused SiLU Gate Mul Backward

**Current:** 3 ops: sigmoid(gate), derivative = sig*(1+gate*(1-sig)), grad_gate = grad*up*derivative.

**Fused kernel:** One pass computing both grad_gate and grad_up.
```
For each element (vectorized half2):
  sig = 1 / (1 + exp(-gate))
  d_silu = sig * (1 + gate * (1 - sig))
  grad_gate = grad_output * up * d_silu
  grad_up = grad_output * gate * sig  // = grad_output * silu(gate)
```
- **Expected speedup:** 1.2x per call
- **Calls per step:** 16
- **File:** `kernels/hip/silu_gate_mul_backward.py`

#### 1d. Fused Residual+RMSNorm Backward

**Current:** 8 ops: merge dual gradients, then full RMSNorm backward (5 ops), then split to residual.

**Fused kernel:** Combines dual-output merge with RMSNorm backward.
```
Phase 1: total_grad = grad_hidden + grad_norm_path
Phase 2: Recompute rms_inv (shared memory)
Phase 3: Chain rule for grad through norm
Phase 4: Output grad_x = grad_residual = total_grad_through_norm
Phase 5: Accumulate grad_weight
```
- **Expected speedup:** 1.5x per call
- **Calls per step:** 16
- **File:** `kernels/hip/fused_residual_rmsnorm_backward.py`

#### 1e. Parallel Selective Scan Backward (AMADEUS only)

**Current:** Two sequential Python loops, each iterating T=1024 times:
- Forward recompute loop: `states[t+1] = dA[t] * states[t] + dBx[t]`
- Reverse gradient loop: accumulates grad_dA, grad_dBx, grad_C, grad_D serially

**Fused kernel:** Parallel reverse prefix scan in HIP (mirrors forward kernel architecture).
```
Stage 1: Per-thread local reverse scan within chunks of 64
Stage 2: Inter-chunk reduction via shared memory
Stage 3: Final propagation of gradient state
```
- **Expected speedup:** 8-16x on scan backward
- **Calls per step:** 16
- **File:** `kernels/hip/selective_scan_backward.py`
- **Correctness:** Exact (parallel prefix scan is algebraically equivalent to serial)

#### 1f. Fix PLE Gate Backward

**Current:** Creates nested autograd context for GELU backward (lines 378-381 in _torch_ops.py).

**Fix:** Replace with direct GELU derivative formula:
```python
# GELU'(x) = 0.5 * (1 + erf(x/sqrt(2))) + x * (1/sqrt(2*pi)) * exp(-x^2/2)
grad_bp = grad_bottleneck * gelu_derivative(bp)
```
- **Expected speedup:** 1.3x per call
- **Calls per step:** 16 (if PLE enabled)
- **Change in:** `kernels/hip/_torch_ops.py` (no new kernel file needed, pure PyTorch fix)

### Workstream 2: LM Head Backward Optimization (Low Risk)

#### 2a. Chunked CE Approach D Integration

**Current:** Already coded in `kernels/hip/chunked_linear_cross_entropy.py` but may not be default in trainer.

**Action:** Ensure Approach D is wired into `halo_training/trainer.py` as the default loss function when using optimized kernels. Verify backward saves 1 GEMM.

- **Expected speedup:** 25% on LM head backward
- **File changes:** `halo_training/trainer.py` (integration)

#### 2b. Sampled Softmax Warmup

**New module:** During epochs 1-3, sample 8192 of 50257 vocab tokens for loss computation. Reduces LM head backward GEMM from (B*T, 50257) to (B*T, 8192).

**Implementation:**
```python
class SampledSoftmaxLoss:
    def __init__(self, full_vocab, sample_size=8192, warmup_steps=3000):
        self.freq_sorted_indices = sort_by_frequency(vocab)
        # Always include target tokens + top-K frequent + random sample
    
    def forward(self, hidden, weight, targets):
        if self.step < self.warmup_steps:
            sampled_weight = weight[self.sample_indices]
            logits = hidden @ sampled_weight.T  # (B*T, 8192) vs (B*T, 50257)
            # Remap targets to sampled indices
        else:
            # Full vocabulary
```
- **Expected speedup:** 3-4x on LM head backward during warmup
- **File:** `halo_training/sampled_softmax.py`
- **Risk:** Medium — must verify convergence matches full-vocab baseline

#### 2c. fp32 grad_logits Retention

**Current:** Approach D saves grad_logits in fp16 during forward, converts to fp32 in backward.

**Change:** Option to keep grad_logits in fp32, skipping conversion. Costs ~2x memory for grad_logits tensor but eliminates per-chunk fp16→fp32 cast.

- **Expected speedup:** 5-10% on chunked CE backward
- **Change in:** `kernels/hip/chunked_linear_cross_entropy.py` (add `store_fp32` flag)

### Workstream 3: Experimental Techniques (High Risk)

#### 3a. INSTANT Low-Rank Backward Projection

**Concept:** For weight gradient computation `dW = X^T @ dY`, project X and dY to rank-r:
```python
P = random_projection(d_model, rank)  # Fixed random matrix
X_proj = X @ P        # (B*T, rank) instead of (B*T, d_model)
dY_proj = dY @ P      # (B*T, rank)
dW_approx = X_proj.T @ dY_proj  # (rank, rank) << (d_model, d_model)
```

**Implementation:**
- Apply to Linear layers (FFN up/down/gate projections)
- Rank: 256 for layers with d=1024
- Fixed random projection matrix (no learnable params)
- **File:** `halo_training/lowrank_backward.py`
- **Expected speedup:** 20-30% on weight gradient GEMMs
- **Risk:** High — approximate gradients may slow convergence

#### 3b. HIP Stream Overlap

**Concept:** Overlap forward computation of chunk N+1 with backward of chunk N using separate HIP streams.

**Implementation:**
- Requires layer-by-layer streaming (Mode B compatible)
- Forward stream and backward stream run concurrently
- Synchronize at gradient accumulation points
- **File:** `halo_training/stream_overlap.py`
- **Expected speedup:** 15-20% hiding
- **Risk:** High — stream scheduling non-deterministic, debugging difficult

#### 3c. Activation Quantization for Backward

**Concept:** Store activations in int8 during forward, dequantize during backward.
```python
# Forward: save quantized
scale = x.abs().max() / 127
x_int8 = (x / scale).round().to(torch.int8)
ctx.save_for_backward(x_int8, scale)

# Backward: dequantize
x_approx = ctx.saved_tensors[0].float() * ctx.saved_tensors[1]
```
- **File:** `halo_training/activation_quant.py`
- **Expected speedup:** Memory savings → enables larger batch → amortizes backward overhead
- **Risk:** Medium — quantization noise in gradients

## Testing Protocol

### Phase 1: Baseline Profiling (~30 min)

```bash
# Profile both models, capture per-op backward timing
python scripts/profile_backward_breakdown.py --model models/llama_7b.py --class-name LlamaModel
python scripts/profile_backward_breakdown.py --model models/amadeus.py --class-name AMADEUS
```

Output: `workspace/backward_profile_llama.json`, `workspace/backward_profile_amadeus.json`

### Phase 2: Isolated Testing (~4-5 hours)

For each technique in Workstreams 1-3:
1. Apply single optimization
2. **Correctness gate:** `torch.autograd.gradcheck` on the modified op
3. **Smoke test:** 200 steps, verify loss decreasing + no NaN + grad norms < 10
4. **Profile:** Measure per-op backward time delta
5. **Record:** tok/s, backward_ms, peak_memory_gb, gradient max_diff

### Phase 3: Combined Testing (~2-3 hours)

1. Stack all passing Workstream 1 techniques
2. Add passing Workstream 2 techniques
3. Add passing Workstream 3 techniques (if any)
4. **10-minute bakeoff** on both models
5. **Loss curve overlay** vs baseline

### Phase 4: Hypothesis Reranking (~1 hour)

For each of the 22 hypothesis plans:
1. Identify which backward ops the architecture uses (attention, Griffin, SSM, SwiGLU, RMSNorm, etc.)
2. Apply measured per-op backward speedups to estimate overall backward improvement
3. Compute new tok/s = old_tok/s × (1 / (1 - backward_fraction × (1 - 1/backward_speedup)))
4. Rerank by estimated optimized throughput
5. Update `knowledge/architectures/Estimation_Hypothesis_Ranking.md`

## File Plan

### New Files
| File | Purpose | LOC Est. |
|------|---------|----------|
| `kernels/hip/rmsnorm_backward.py` | Fused RMSNorm backward HIP kernel | 150 |
| `kernels/hip/rotary_embedding_backward.py` | Fused rotary backward HIP kernel | 120 |
| `kernels/hip/silu_gate_mul_backward.py` | Fused SiLU gate mul backward HIP kernel | 100 |
| `kernels/hip/fused_residual_rmsnorm_backward.py` | Fused res+RMSNorm backward HIP kernel | 180 |
| `kernels/hip/selective_scan_backward.py` | Parallel reverse prefix scan backward | 250 |
| `halo_training/sampled_softmax.py` | Sampled softmax warmup loss | 150 |
| `halo_training/lowrank_backward.py` | INSTANT low-rank backward projection | 200 |
| `halo_training/stream_overlap.py` | HIP stream overlap for layer pipeline | 250 |
| `halo_training/activation_quant.py` | Int8 activation quantization for backward | 150 |
| `scripts/profile_backward_breakdown.py` | Per-op backward profiler | 200 |
| `scripts/bench_backward_optimizations.py` | Benchmark all backward techniques | 300 |
| `knowledge/kernels/backward_pass_optimization_results.md` | Results documentation | — |

### Modified Files
| File | Change |
|------|--------|
| `kernels/hip/_torch_ops.py` | Wire fused backward kernels into custom op `.register_autograd` |
| `halo_training/trainer.py` | Integrate chunked CE Approach D as default, add sampled softmax option |
| `knowledge/architectures/Estimation_Hypothesis_Ranking.md` | Reranked throughput estimates with backward gains |

## Expected Results

### Per-Model Backward Speedup Estimates

**LlamaModel 124.7M:**
- Workstream 1 (fused norms + rotary + silu): ~1.4x backward
- Workstream 2 (chunked CE + sampled softmax): ~1.2x on LM head portion
- Combined: ~1.5x backward → ~1.25x overall training throughput
- 43K tok/s → ~54K tok/s (estimated)

**AMADEUS 243.8M:**
- Workstream 1 (all above + parallel scan backward): ~2x backward (dominated by scan fix)
- Workstream 2 (chunked CE): ~1.15x on LM head portion
- Combined: ~2.2x backward → ~1.5x overall training throughput
- 10.4K tok/s → ~15.6K tok/s (estimated)

### Hypothesis Reranking Impact

Architectures with heavy recurrence (Griffin, SSM) benefit most from Workstream 1e (parallel scan backward). Architectures with large FFN benefit from fused norm/activation backward. All architectures benefit from LM head optimization.

Estimated reranking shifts:
- **TEMPEST, PROMETHEUS** (Griffin-heavy): +15-25% tok/s from fused backward
- **AMADEUS** (SSM-heavy): +40-50% tok/s from parallel scan backward
- **LlamaModel** (attention-heavy): +20-25% tok/s from fused norm/rotary backward
- **RESONANT-LOOP** (shared block): +30% tok/s (backward recompute dominates with 16 iterations)
- **ARCHON, CHIMERA-ENGRAM** (complex hybrid): +20-30% from multiple backward fusions

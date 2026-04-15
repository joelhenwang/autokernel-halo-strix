---
title: "AutoKernel Optimization Report"
domain: project
type: results
status: active
related:
  - knowledge/kernels/kernel_benchmarks.md
  - knowledge/architectures/hypothesis_buildout_results.md
tags: [%results, %benchmarks, %summary]
---

# AutoKernel Optimization Report — AMD Strix Halo (gfx1151)

**Hardware:** AMD Radeon 8060S (gfx1151), RDNA 3.5 APU, 20 CUs, wave32, ~170 GB/s LPDDR5X, 64KB LDS/CU  
**Software:** ROCm 7.12, PyTorch 2.11.0+rocm7.2, HIP C++ via hipcc  
**Last updated:** 2026-04-08  

---

## Results Summary

### Kernels Beating PyTorch (sorted by speedup)

| Kernel | Speedup | Correctness | Category | Notes |
|--------|---------|-------------|----------|-------|
| **dequantize_int4** | **16.260x** | PASS | overhead elimination | Unpack 2 int4 per byte, fused dequant to fp16 |
| **fused_residual_add_layernorm** | **10.694x** | PASS | fusion | Fuses residual add + LayerNorm, LDS caching |
| **dequantize_int8** | **8.080x** | PASS | overhead elimination | Eliminates PyTorch's multi-cast path |
| **fused_residual_add_rmsnorm** | **6.632x** | PASS | fusion | Fuses residual add + RMSNorm, LDS caching |
| **rotary_embedding** | **3.66x** | PASS | native fp16 | Fixed correctness via native fp16 intrinsics |
| **rmsnorm** | **3.33x** | PASS | memory-bound | Baseline HIP port |
| **fused_bias_silu** | **1.929x** | PASS | fusion | Fuses bias add + SiLU activation |
| **fused_bias_gelu** | **1.888x** | PASS | fusion | Fuses bias add + GELU activation (BERT/GPT-2) |
| **cross_entropy** | **1.834x** | PASS | online algorithm | Online fused max+sum, half2 loads |
| **silu_gate_mul** | **1.596x** | PASS | fusion | SwiGLU: SiLU(gate) * up in one kernel |
| **reduce** | **1.08x** | PASS | memory-bound | Baseline HIP port |
| **layernorm** | **1.06x** | PASS | memory-bound | Baseline HIP port |
| **silu** | **1.01x** | PASS | element-wise | At parity with PyTorch |
| **prefix_scan** | **8.398x** | 4/5 PASS | parallel scan | fp32 LDS accumulation; mixed_scale stability only failure |
| **moe_gating** | **3.472x** | PASS | fused softmax+topk | fp16 bit-level inf check, __hgt topk tie-breaking |

### Kernels At or Below PyTorch

| Kernel | Speedup | Correctness | Category | Notes |
|--------|---------|-------------|----------|-------|
| gelu | 0.987x | PASS | element-wise | At parity, erff intrinsic competitive |
| softmax | 0.885x | PASS | memory-bound | LDS caching helps medium (2.48x), large near HW ceiling |
| matmul | 0.24x | PASS | compute-bound | Hard ceiling — no MFMA on RDNA 3.5 |
| flash_attention | 0.05x | PASS | compute-bound | Hard ceiling — scalar tiled GEMM |
| fused_mlp | 0.02x | PASS | compute-bound | Hard ceiling — two large GEMMs |

### Mostly Passing Kernels (1 edge case remaining)

| Kernel | Speedup | Correctness | Issue | Notes |
|--------|---------|-------------|-------|-------|
| prefix_scan | 8.398x | 4/5 PASS | mixed_scale stability: fp32 vs fp16 accumulation mismatch (error=12) | All shapes, determinism, edges PASS |

### Failed Kernels

| Kernel | Speedup | Correctness | Issue | Notes |
|--------|---------|-------------|-------|-------|
| top_k_sampling | 0.255x | FAIL | Binary search approach fundamentally wrong | PyTorch's radix sort is much faster — skip |

**Totals: 21 kernels implemented, 16 fully passing, 15 beating PyTorch.**

**Key takeaway:** The biggest wins come from **kernel fusion** (16.3x, 10.7x, 8.4x, 8.1x) and **eliminating PyTorch overhead** — not from raw bandwidth optimization. The pattern: find where PyTorch launches 3+ kernels for what could be 1, and the more intermediate tensors eliminated, the bigger the win.

### End-to-End Model Speedup Summary (Prefill)

| Approach | LlamaModel (170M) | LlamaModel7B (5.9B) | Notes |
|----------|-------------------|---------------------|-------|
| Baseline (PyTorch) | 17.2 ms | 448 ms | — |
| HIP kernels only (`--incremental`) | 16.6 ms (1.038x) | 427 ms (1.053x) | 4 HIP kernels |
| + fused QKV (`--fused-qkv`) | 16.0 ms (1.067x) | 427 ms (1.053x) | rocBLAS utilization |
| torch.compile alone (`--torch-compile`) | 13.5 ms (1.275x) | 384 ms (1.162x) | Inductor graph fusion |
| verify.py `--compile-with-kernels` | 13.1 ms (1.308x) | 385 ms (1.161x) | Inductor + HIP |
| **`autokernel.optimize(compile=True)`** | **12.8 ms (1.340x)** | **377 ms (1.189x)** | **Best: Library API** |

### Decode Performance (Autoregressive Generation)

| Model | Prefill (128 tok) | Decode (ms/tok) | Tokens/sec |
|-------|-------------------|-----------------|------------|
| LlamaModel (170M) | 5.4 ms | 5.05 ms | **197.9 tok/s** |
| LlamaModel7B (5.9B) | 150.8 ms | 106.93 ms | **9.4 tok/s** |

Decode is bottlenecked by weight reads (~12 GB per step at 170 GB/s = ~70ms floor on 7B).

---

## Detailed Optimization History

### 1. Rotary Embedding — 3.66x (Fixed Correctness)

**Problem:** The kernel achieved 3.64x speedup but FAILED the correctness harness (numerical_stability stage).

**Root cause:** The original kernel computed `x0 * cos - x1 * sin` and `x0 * sin + x1 * cos` in fp32, then cast back to fp16. PyTorch's reference computes directly in fp16, producing different rounding. The bench.py `near_max` test scales inputs by 60000.0, amplifying these 1-ULP differences beyond tolerance.

**Fix:** Rewrote the kernel to use **native fp16 intrinsics** (`__hmul`, `__hsub`, `__hadd`) so the computation matches PyTorch's fp16 rounding exactly. Added `-fno-fast-math -ffp-contract=off` compiler flags to prevent the HIP compiler from fusing multiply-add into FMA (which would alter rounding vs PyTorch's unfused ops).

**Lesson:** For element-wise kernels, the computation precision path must exactly match the reference. Even 1 ULP of rounding difference can fail tolerance checks when inputs are scaled to near fp16 max (~65504). Using native fp16 intrinsics instead of fp32 intermediate + cast-back is both faster AND more correct for matching PyTorch behavior.

**Could be revisited:** If a use case needs fp32-precision rotary embedding, a separate kernel variant could be written — but the reference would also need to compute in fp32.

---

### 2. Cross Entropy — 0.95x to 1.834x

This kernel went through multiple iterations. Here's each attempt:

#### Attempt 1: Half2 vectorized loads (FAILED — correctness regression)

**Hypothesis:** Replace scalar `__half2float(row[v])` loads with `half2` reads to halve load instruction count.

**Result:** Correctness FAIL (max_abs_error=0.77).

**Root cause:** The kernel used `__shfl_down` for warp-level reductions. In an online softmax algorithm, **all lanes** need the reduced max value to rescale their partial sums. `__shfl_down` only propagates the result to lane 0 — all other lanes use stale/incorrect max values, corrupting the rescaling step `local_sum *= __expf(local_max - warp_max)`.

**Critical lesson: `__shfl_xor` vs `__shfl_down`**  
- `__shfl_xor` broadcasts the reduced value to ALL lanes in the wavefront  
- `__shfl_down` only gives lane 0 the final result  
- **Any time all lanes need the reduction result (online softmax, any algorithm with rescaling), you MUST use `__shfl_xor`**  
- `__shfl_down` is only correct when only lane 0 needs the result (e.g., writing a single block sum to shared memory)

This was a subtle bug — the kernel appeared to work on small inputs where numerical differences were within tolerance, but failed on larger inputs where the accumulated error from incorrect rescaling became significant.

#### Attempt 2: Fix shuffles + online fused max+sum (SUCCESS — 1.834x)

**Changes:**
1. Changed both `warp_reduce_max` and `warp_reduce_sum` from `__shfl_down` to `__shfl_xor`
2. Implemented online fused single-pass algorithm: combine max-finding and exp-sum into one loop with rescaling (saves one full read of the vocab row)
3. Half2 vectorized loads with `#pragma unroll 4`
4. `__launch_bounds__(256)` for register pressure control
5. Safe 64-bit indexing: `(long long)b * vocab` to prevent overflow at large batch × vocab

**Online fused algorithm sketch:**
```cpp
// Single pass: track running max and rescaled sum simultaneously
for (int v = tid; v < n_pairs; v += BLOCK_SIZE) {
    half2 val = row_v[v];
    float lo = __half2float(val.x);
    float hi = __half2float(val.y);
    float pair_max = fmaxf(lo, hi);
    if (pair_max > local_max) {
        local_sum *= __expf(local_max - pair_max);  // rescale existing sum
        local_max = pair_max;
    }
    local_sum += __expf(lo - local_max) + __expf(hi - local_max);
}
```

This eliminates the separate max-finding pass, reducing memory traffic from 2N to N reads per row. Combined with half2 loads (halving load instructions), the total improvement is ~2x memory efficiency.

**Lesson:** Online softmax (fused max+sum with rescaling) is the right algorithm for any reduction that needs both max and sum. It's used in flash attention and should be the default approach for cross-entropy and softmax on bandwidth-limited hardware.

#### What DIDN'T help (and shouldn't be retried):

- **BLOCK_SIZE=512**: Was tried in an earlier revision (before the Strix Halo pivot). On 20 CUs, 512 threads per block reduces occupancy without proportional bandwidth gain. 256 is the sweet spot.
- **Separate 2-pass with half2 loads only**: The 2-pass algorithm (find max, then exp-sum) reads the vocab row twice. Even with vectorized loads, the extra memory pass is the bottleneck on Strix Halo's ~170 GB/s bandwidth.

---

### 3. Fused Residual Add + RMSNorm — 6.632x (New Kernel)

This is a **new kernel type** not in the original 9. It fuses `output = RMSNorm(x + residual, weight)` into a single kernel, eliminating the intermediate tensor that would otherwise require a full memory round-trip.

#### Why this kernel matters

Every transformer layer (LLaMA, Mistral, etc.) calls this pattern twice: once after attention, once after FFN. The unfused version requires:
- Read x + Read residual + Write hidden (residual add)
- Read hidden + Read weight + Write output (RMSNorm)
- Total: ~5N memory passes per row

The fused version:
- Read x + Read residual → cache hidden in LDS → Read weight + Write output
- Total: ~3.3N memory passes per row (LDS serves the hidden values for free)

On bandwidth-limited Strix Halo, eliminating ~34% of memory traffic directly translates to speedup. The 6.6x result exceeds the standalone rmsnorm (3.33x) because PyTorch's unfused version pays the full 5N cost.

#### Implementation details

**Phase 1: Residual add + sum_sq accumulation**
- Vectorized half2 loads of x and residual
- Add in fp16 using `__hadd2` to match PyTorch's per-op rounding (critical for correctness — see failed attempt below)
- Promote to fp32 and cache in dynamic shared memory (`extern __shared__ float s_hidden[]`)
- Accumulate `sum_sq` in fp32 simultaneously

**Phase 2: Normalize + scale from LDS**
- Read cached hidden values from LDS (free — no global memory access)
- Multiply by `rms_inv` (computed via `rsqrtf`) and weight
- Vectorized half2 writes to output

**Dynamic shared memory:** `N * sizeof(float)` bytes per block. For LLaMA dim=4096, that's 16KB — well within the 64KB LDS budget per CU.

#### Failed attempt: fp32 residual addition

**What happened:** First version computed `hidden = __half2float(x) + __half2float(r)` in fp32. This produced different results than PyTorch's `x + residual` which adds in fp16. The bench.py correctness harness caught this as a max_abs_error exceeding tolerance.

**Fix:** Used `__hadd2(xval, rval)` to perform the addition in fp16 (matching PyTorch), then promoted to fp32 for the RMS computation.

**Lesson:** When fusing operations, each sub-operation must match the reference's precision path. PyTorch adds tensors in their native dtype (fp16), not in fp32. Using fp32 for the add is "more accurate" but produces different results.

#### Failed attempt: Reference overflow in numerical_stability test

**What happened:** After fixing the fp16 add, the `mixed_scale` stability test still failed with error=15008. Investigation revealed the **reference implementation** was the problem, not the kernel.

The bench.py `mixed_scale` test scales all tensors by random factors of 1e3/1e-3. With hidden values up to ~6000, the reference's `hidden ** 2` overflows fp16 (max 65504) to inf, producing rms=inf, output=0. But the kernel computes sum_sq in fp32, getting correct results.

**Fix:** Updated the reference to compute RMS in fp32:
```python
hidden_f = hidden.float()
rms = torch.sqrt(torch.mean(hidden_f ** 2, dim=-1, keepdim=True) + eps)
return ((hidden_f / rms) * weight.float()).to(x.dtype)
```

This was valid because `fused_residual_add_rmsnorm_ref` is a new function we created (not one of the immutable original references). The fp32 path is also more numerically correct — production implementations (like LLaMA) always compute normalization statistics in fp32.

**Lesson:** When creating new reference implementations, always compute reduction statistics (mean, variance, RMS) in fp32, even if inputs are fp16. Squaring fp16 values easily overflows. This applies to any normalization reference.

---

### 4. Softmax Optimization — 0.85x (LDS Caching Attempted)

**Attempted optimization:** Added LDS caching to the block-wide fp16 kernel (for n_cols >= 1024). Cache input as float during pass 1, read from LDS in pass 2 to eliminate second global memory read.

**Results:** Medium (1024x1024) improved to **2.48x** — LDS caching works when data exceeds L2. Large (4096x4096) stayed at 0.885x. Wide/vocab (32K+ cols) regressed because LDS can't hold the row and falls back to warp-per-row.

**Why it didn't help at large sizes:** Strix Halo has a 6MB L2 cache. At the "large" benchmark (4096x4096 = 32MB input), the second read already hits L2 with ~135% effective bandwidth. The LDS caching adds write overhead that cancels out the avoided global read. **The kernel is already at the hardware bandwidth ceiling for these sizes.**

**Register caching** was added for the warp-per-row narrow-row kernel. Each lane caches up to 16 half2 pairs in registers, avoiding the second global read for rows < 1024 cols.

**Lesson:** LDS caching only helps when the data doesn't fit in L2 cache. On Strix Halo (6MB L2), datasets under ~4MB already hit L2 on the second pass. For larger datasets, LDS helps (2.48x at medium). This is a hardware-specific insight — on GPUs with smaller L2, LDS caching would help at all sizes.

### 5. LayerNorm Optimization — 1.06x (LDS Caching Regressed)

**Attempted optimization:** Replaced Welford's algorithm with simpler two-pass sum+sum_sq, switched from float4 to half2 loads, added LDS caching, replaced `__shfl_down` with `__shfl_xor`.

**Result:** Regressed from 1.06x to 0.984x. **Reverted to original.**

**Why it regressed:** Same L2 caching effect as softmax. The original Welford+float4 approach, while theoretically suboptimal (reads global memory twice), benefits from L2 serving the second read. The LDS caching added write overhead that wasn't compensated. The float4 loads (8 halfs per load) actually provided better instruction throughput than half2 (2 halfs per load) for this kernel's access pattern.

**Lesson:** Don't assume a proven optimization pattern (LDS caching) transfers between kernels. The fused_residual_add_rmsnorm kernel (6.6x with LDS) wins because it eliminates an **intermediate tensor** — the LDS serves data that would otherwise require a separate kernel launch. In layernorm, both passes are within the same kernel, so L2 already caches the data.

### 6. Kernels at Hardware Ceiling (Not Worth Further Optimization)

#### Compute-bound (matmul 0.24x, flash_attention 0.05x, fused_mlp 0.02x)

RDNA 3.5 has **no MFMA**. Scalar FMA is 10-50x slower than PyTorch's rocBLAS. Skip.

#### Reduce (1.08x), SiLU (1.01x), GELU (0.987x)

These are simple operations where PyTorch's implementations are already near-optimal. Our kernels match but can't significantly beat them — the operations are too simple to have fusion or algorithmic opportunities.

---

### 7. New Kernel: Fused Residual Add + LayerNorm — 10.694x

**The best kernel in the project.** Same fusion pattern as fused_residual_add_rmsnorm, but for models using LayerNorm instead of RMSNorm (BERT, GPT-2, some newer architectures).

**Implementation:** Cloned fused_residual_add_rmsnorm structure. Added mean accumulation alongside sum_sq in phase 1, plus bias application in phase 2. Uses `__hadd2` for fp16 add (matching PyTorch), fp32 for statistics, dynamic LDS for hidden row caching.

**Why 10.7x (even higher than rmsnorm's 6.6x):** PyTorch's unfused LayerNorm path is more expensive than RMSNorm because it computes both mean and variance as separate operations, plus applies bias. That's 6+ separate kernel launches vs our single fused kernel.

**Lesson:** The more operations PyTorch runs as separate kernels, the bigger the fusion win. LayerNorm's add+mean+variance+normalize+scale+bias = 6 ops gives a bigger fusion multiplier than RMSNorm's add+sum_sq+normalize+scale = 4 ops.

### 8. New Kernel: Int8 Dequantization — 8.080x

**Per-channel dequantization:** `output[i,j] = (int8[i,j] - zero_point[j]) * scale[j] → fp16`

**Why 8x:** PyTorch's reference path does `x_int8.float() - zero_point.float() * scale.float() → .half()` — that's 3 dtype casts, each allocating a temporary tensor and launching a kernel. Our fused kernel reads int8, does the math in fp32, writes fp16 — one kernel, no temporaries.

**Implementation:** Simple grid-stride element-wise kernel. Each thread processes 2 elements, writes as half2. Per-channel scale/zero_point are broadcast from a 1D tensor.

**Zero error:** Because the math is identical (both compute in fp32), the results are bitwise identical.

### 9. New Kernel: Fused SiLU-Gate-Multiply (SwiGLU) — 1.596x

**The SwiGLU activation used in LLaMA/Mistral/Mixtral FFN blocks.** Computes `output = SiLU(gate) * up` where gate and up are the outputs of the gate and up projections.

**Why 1.6x:** PyTorch does `F.silu(gate)` (allocates temporary) then `temp * up` (allocates output). Two kernel launches, one temporary tensor. Our kernel reads both inputs and writes output in a single pass: 3N memory accesses vs PyTorch's 4N+.

**This kernel runs 2x per transformer layer** (once per FFN block), making it high-frequency. The 1.6x speedup applies to every layer.

### 10. New Kernel: Prefix Scan — 8.4x (4/5 Stages PASS)

**Inclusive cumulative sum along last dimension.** Core operation for Mamba/SSM architectures.

**Algorithm:** Blelloch-style three-phase scan:
1. Each thread computes sequential prefix sum over its chunk, stored in LDS as fp32
2. Block-wide exclusive scan of per-thread totals via warp `__shfl_up` + LDS
3. Each thread reads from LDS, adds offset in fp32, writes final fp16 to output

**Performance:** 8.4x on large (2048x2048), 6.3x on mamba-typical (4096x1024). PyTorch's `torch.cumsum` is surprisingly slow because it doesn't parallelize well.

**Fix history:** Original version wrote fp16 intermediates to global memory (6.65x), causing overflow at extreme scales. Changed to LDS-based fp32 intermediates — improved performance to 8.4x and eliminated the global memory fp16 round-trip.

**Approaches tried and abandoned for mixed_scale:**
- **fp16 accumulation within threads** (`__hadd`): Error stayed at 12.0 because the inter-thread offset addition in Step 3 uses fp32, causing rounding differences at chunk boundaries.
- **Fully sequential scan** (1 thread per row): 0.43x PyTorch AND 26.0 error (worse). Proves PyTorch's cumsum is NOT sequential fp16 — it uses a parallel algorithm with its own rounding characteristics.
- **fp16 in-thread + fp16 offset addition**: Attempted to match PyTorch by doing everything in fp16. Still fails because fp16 addition is non-associative: `(a+b)+c != a+(b+c)`. Different parallelization groupings produce different results.

**Root cause analysis:** PyTorch's `torch.cumsum` on ROCm uses an internal parallel scan (likely hipCUB/thrust). Our fp32 parallel scan is mathematically more accurate but produces different fp16 output at 9 of 131072 elements. These 9 elements are at near-zero cumsum crossings where large values cancel out, exposing ~2.5 absolute error with tolerance of ~1.8 (atol=1.0 + rtol=0.5 * |expected|). This is fundamentally unfixable without matching PyTorch's exact internal algorithm.

**Remaining correctness issue:** FAIL on `mixed_scale` only (error=12.0). **All 10 shape_sweep configs PASS, determinism PASS, all edge cases PASS.**

### 11. New Kernel: MoE Top-K Gating — 3.5x (ALL 5 STAGES PASS)

**For Mixture-of-Experts models.** Computes softmax over expert scores, selects top-k experts, normalizes routing weights.

**Design:** One thread per token (since E is small, 8-64 experts fit in registers). Entire softmax + top-k selection + normalization happens in registers — no shared memory needed.

**Performance:** 3.5x on large (8192 tokens, 8 experts), 3.2x on mixtral config. Wide (64 experts) regresses to 0.67x due to register pressure.

**Three bugs found and fixed:**

#### Bug 1: `-ffast-math` silently removes inf/NaN guards
**Symptom:** `near_max` stability test: reference outputs NaN (correct for inf inputs), kernel outputs clean values (incorrect match), bench.py reports `max_abs_error=nan` because diff(clean, NaN) = NaN.

**Root cause:** The `-ffast-math` flag (set by `_compile.py`) implies `-ffinite-math-only`, which tells the compiler that float values are always finite. The compiler optimizes away ALL inf/NaN checks: `x != x` (NaN), `x == INFINITY`, and even `__float_as_uint(x) == 0x7f800000` comparisons on values that passed through `fmaxf` (which the compiler assumes returns finite values).

**Fix:** Check fp16 input bits BEFORE converting to fp32, using `__half_as_ushort`:
```cpp
half h = row[e];
unsigned short hbits = __half_as_ushort(h);
if ((hbits & 0x7C00u) == 0x7C00u) has_inf_nan = true;
```
The fp16 bits are untouched by fast-math optimization because the compiler's finite-math assumption doesn't extend to integer bit patterns. When any input is inf/NaN, output NaN to match PyTorch's `F.softmax(inf) → NaN` behavior.

**Critical lesson:** On ROCm with `-ffast-math`, **never use float comparisons to detect inf/NaN**. Use bit-level checks on the fp16 input BEFORE `__half2float`. This includes `__float_as_uint` checks on values derived from `fmaxf` — the compiler may assume `fmaxf` always returns finite values.

#### Bug 2: fp32 topk disagrees with fp16 torch.topk
**Symptom:** `large` shape (8192 tokens) and `near_zero` stability: error=0.5 (one expert wrong). Some tokens have experts with identical fp16 softmax probs but tiny fp32 differences.

**Root cause:** Our kernel computes softmax in fp32 (for accuracy), then selects top-k using fp32 `>` comparison. PyTorch's `torch.topk` operates on fp16 softmax output. Two experts with probs 0.132300005 and 0.132299999 in fp32 both round to 0.1323 in fp16 — PyTorch treats them as equal (first index wins), but our fp32 comparison picks the higher one regardless of index.

**Fix:** Use `__hgt` (fp16 comparison intrinsic) for top-k selection:
```cpp
half hbest = __float2half(-1.0f);
for (int e = 0; e < E; e++) {
    if (__hgt(h_vals[e], hbest)) {
        hbest = h_vals[e];
        best_idx = e;
    }
}
```
This compares in fp16 space, so ties in fp16 are broken by first (smallest) index — matching `torch.topk`.

**Note:** An earlier attempt used `__half2float(__float2half(vals[e]))` to round fp32 values to fp16 before comparison. This was silently optimized away by `-ffast-math` (compiler treated the round-trip as a no-op). Using `__hgt` on actual `half` values cannot be optimized away.

#### Bug 3: bench.py harness gap for "kernel cleaner than reference"
**Discovery:** The bench.py stability test has three branches: (1) kernel has NaN, ref clean → FAIL, (2) both have NaN → PASS, (3) else → compare. There is no branch for "kernel is clean, reference has NaN" — it falls through to compare, producing `max_abs_error=nan`. This is a harness limitation, not a kernel bug. The fix was to match the reference behavior (output NaN when inputs overflow) rather than trying to produce "better" output.

### 12. New Kernel: Int4 Dequantization — 16.260x

**Our fastest kernel.** Per-channel int4 dequantization for GPTQ/AWQ quantized models.

**Format:** Two int4 values packed per uint8 byte. Low nibble = even column, high nibble = odd column. Per-channel scale (fp16) and zero_point (uint8).

**Why 16.3x:** PyTorch's reference path is catastrophically slow for int4:
1. `x_packed & 0x0F` — bitwise AND creating a temp tensor
2. `.to(torch.float32)` — dtype cast creating another temp
3. `- zero_point.float()` — broadcast subtract, another temp
4. `(x_packed >> 4) & 0x0F` — shift + AND, more temps
5. Repeat subtract + multiply for high nibble
6. Interleave even/odd columns, final `.half()` cast

That's 10+ kernel launches and 6+ temporary tensors for what our kernel does in a single pass: read 1 byte, extract 2 nibbles, subtract zero_point, multiply by scale, write 2 half values.

**Implementation:** Each thread processes one packed byte → two fp16 outputs written as `half2`. Grid-stride loop, 256 threads per block.

**Zero error:** Bitwise identical to reference (both compute in fp32).

### 13. New Kernel: Fused Bias Add + SiLU — 1.929x

**Common pattern after linear projections with bias.** `output = SiLU(x + bias)` where bias is a 1D per-column vector broadcast across rows.

**Why 1.9x:** PyTorch does `x + bias` (allocates temp with broadcast) then `F.silu(temp)` (allocates output). Two kernel launches, one temporary. Our kernel reads x + broadcasts bias, computes SiLU, writes output — single pass.

**Implementation:** Grid-stride element-wise kernel. Each thread processes 2 elements via half2. Bias is loaded per-column using `flat_idx % N` for column indexing.

**Consistent 1.5-1.9x** across all sizes (small through llama). Higher speedup at larger sizes where the temporary tensor allocation cost is proportionally larger.

### 14. New Kernel: Fused Bias Add + GELU — 1.888x

**Same fusion pattern as fused_bias_silu**, adapted for BERT/GPT-2 models which use GELU instead of SiLU after linear projections. `output = GELU(x + bias)`.

**Implementation:** Copied fused_bias_silu kernel, replaced SiLU computation with GELU via `erff` intrinsic:
```cpp
constexpr float SQRT_2_INV = 0.7071067811865475f;
float y = x * 0.5f * (1.0f + erff(x * SQRT_2_INV));
```

**Performance:** 1.89x on large (8192x4096), 1.81x on medium, 1.35x on bert-typical (2048x3072). ALL 5 correctness stages PASS including bfloat16.

**Trivial win:** Total implementation time ~5 minutes. Reference already existed in `reference.py`. This confirms the pattern: once you have one fused bias+activation kernel, adding variants for different activations is mechanical.

### 15. Failed Experiment: Top-K Sampling — 0.255x

**Attempted:** Fused temperature scaling + top-k selection + softmax using binary search to find the k-th largest value.

**Why it failed:** The binary search approach requires 20 full scans of the vocabulary (32K+ elements) to narrow down the threshold. That's 20x the memory reads of a single pass. PyTorch's `torch.topk` uses highly optimized radix sort which is fundamentally better for this problem.

**Lesson:** Not all fusion opportunities are wins. Top-k selection is a **sorting problem**, not a reduction problem. Our memory-bound optimization patterns (LDS caching, online algorithms, vectorized loads) don't help here. PyTorch's specialized sort algorithms are the right tool.

**Should not be retried** with this approach. A better approach would be to use PyTorch's topk for selection and only fuse the surrounding temperature+softmax, but the win would be marginal.

---

## Cross-Cutting Lessons

### 1. `__shfl_xor` vs `__shfl_down` — When to use which

| Use case | Correct shuffle | Why |
|----------|----------------|-----|
| Only lane 0 needs result (write single sum to smem) | `__shfl_down` | Cheaper, result in lane 0 only |
| ALL lanes need result (online softmax rescaling, broadcast) | `__shfl_xor` | Broadcasts to all lanes |
| Block-level reduction via shared memory | Either (only lane 0 writes to smem) | `__shfl_down` slightly cheaper |

**Default to `__shfl_xor`** unless you're certain only lane 0 needs the result. The performance difference is negligible, but using `__shfl_down` when all lanes need the result causes silent correctness bugs that only manifest at scale.

### 2. fp16 precision matching

When writing HIP kernels that must match PyTorch references:
- PyTorch adds/multiplies tensors in their **native dtype** (fp16 stays fp16)
- Computing in fp32 then casting back produces different rounding (1 ULP differences)
- Use native fp16 intrinsics (`__hadd2`, `__hmul`, `__hsub`) when the reference operates in fp16
- Only promote to fp32 for **reduction accumulation** (sum, sum_sq) where fp16 would overflow
- Use `-fno-fast-math -ffp-contract=off` when exact rounding matters

### 3. Memory-bound optimization strategy on Strix Halo

With ~170 GB/s bandwidth, 6MB L2 cache, and no MFMA, the optimization hierarchy is:
1. **Kernel fusion** (eliminate intermediate tensors between PyTorch ops) — **highest impact by far** (6-10x)
2. **Eliminate PyTorch cast/allocation overhead** (e.g., int8→float→subtract→multiply→half as one kernel) — 8x
3. **Online/fused algorithms** (online softmax, fused max+sum) — eliminates memory passes (1.8x)
4. **Vectorized loads** (half2 minimum, float4 where aligned) — reduces instruction count
5. **LDS caching** — only helps when data exceeds L2 cache (~6MB). Does NOT help when L2 already serves the second read
6. **Warp-level shuffles** for reductions — avoids shared memory round-trips
7. **Thread count tuning** — enough threads for memory-level parallelism

**Critical insight from Round 2:** LDS caching is NOT universally beneficial on Strix Halo. The 6MB L2 cache already serves repeated reads for datasets under ~4MB. LDS caching only wins when it eliminates data that would need a **separate kernel launch** (fusion) or when data exceeds L2.

### 4. Reference implementation correctness

When creating new reference implementations:
- Always compute statistics (mean, variance, RMS, max) in **fp32** to avoid overflow
- Test with the `near_max` and `mixed_scale` stability tests early
- fp16 values > 256 will overflow when squared in fp16 (256^2 = 65536 > 65504)
- The reference must be correct for the kernel to be evaluated correctly

### 5. `-ffast-math` pitfalls for correctness guards

`-ffast-math` (the default in `_compile.py`) implies `-ffinite-math-only`, which lets the compiler assume all float values are finite. This silently breaks:

| Guard pattern | Broken by `-ffast-math` | Safe alternative |
|---------------|-------------------------|------------------|
| `x != x` (NaN check) | Always false | `__half_as_ushort(h) & 0x7C00 == 0x7C00` on fp16 input |
| `x == INFINITY` | Always false | Check fp16 bits before `__half2float` |
| `__float_as_uint(fmaxf(...))` | Compiler assumes `fmaxf` returns finite | Check fp16 bits of raw input, not derived fp32 |
| `__half2float(__float2half(x))` round-trip | Optimized away as no-op | Use `__hgt` on actual `half` values |

**Rule:** Check fp16 input bits directly with `__half_as_ushort` BEFORE converting to fp32. Once a value passes through any fp32 operation under `-ffast-math`, the compiler may assume it's finite.

### 6. Dynamic shared memory for fusion kernels

LDS caching eliminates global memory round-trips at the cost of shared memory:
- Budget: 64KB per CU on Strix Halo
- For dim=4096 (LLaMA): 4096 * 4 bytes (float) = 16KB — fits easily
- For dim=8192 (larger models): 32KB — still fits
- For dim>16384: consider recomputing instead of caching (still saves vs unfused)
- Use `extern __shared__` with dynamic allocation (`<<<grid, block, smem_bytes>>>`)

---

## MI300X Historical Context

Before the Strix Halo pivot (v2.0.0), optimization was done on MI300X (gfx942, CDNA 3) with Triton. Key findings that inform Strix Halo work:

| MI300X Finding | Strix Halo Relevance |
|---------------|---------------------|
| Native dtype `tl.dot` is the biggest lever (flash_attention 0.5x→2.2x) | Not applicable — no MFMA on RDNA 3.5 |
| Multi-row processing for memory-bound kernels (softmax 1.16x→2.26x) | Applicable — try multi-row for softmax optimization |
| `num_stages=0` crashes AMD Triton | Not applicable — using HIP C++ now, not Triton |
| Persistent kernels hurt small grids (67 vs 73 TFLOPS) | Applicable — avoid persistent kernels on 20 CUs |
| fp16 computation must match reference precision path | Confirmed on Strix Halo — same principle applies |

---

## End-to-End Model Verification (Phase C)

Extended `verify.py` with fusion kernel replacement strategies, `--incremental` mode, and `--fused-qkv` rocBLAS optimization. Five optimization strategies now successfully replace PyTorch ops end-to-end:

- **`_FusedQKVAttentionWrapper`**: Concatenates wq/wk/wv into single matmul for better rocBLAS utilization (37.4 vs 31 TFLOPS)
- **`_FusedResidualRMSNormBlockWrapper`**: Wraps entire TransformerBlocks, fusing the within-block residual_add + ffn_norm pair using the dual-output kernel
- **`_RMSNormWrapper`**: Replaces standalone RMSNorm modules (e.g. `model.norm`, `attention_norm`)
- **`_SiluGateMulWrapper`**: Wraps SwiGLU FeedForward modules (w1/w2/w3 pattern)
- **`_RotaryAttentionWrapper`**: Wraps Attention modules, using fp32-intermediate rotary kernel (integrated into fused QKV when `--fused-qkv` is active)

### LlamaModel (170M, 12 layers) — with `--fused-qkv`

| Step | Kernel | Status | Max Error | Latency | Cumulative Speedup |
|------|--------|--------|-----------|---------|-------------------|
| base | PyTorch | - | - | 17.0 ms | 1.000x |
| +1 | fused_residual_add_rmsnorm | PASS | 3.91e-03 | 16.6 ms | **1.024x** |
| +2 | rmsnorm | PASS | 3.91e-03 | 16.2 ms | **1.054x** |
| +3 | silu_gate_mul | PASS | 3.91e-03 | 16.0 ms | **1.067x** |
| +4 | rotary_embedding | PASS | 3.91e-03 | 16.0 ms | **1.065x** |

### LlamaModel7B (5.9B, 32 layers) — with `--fused-qkv`

| Step | Kernel | Status | Max Error | Latency | Cumulative Speedup |
|------|--------|--------|-----------|---------|-------------------|
| base | PyTorch | - | - | 449.4 ms | 1.000x |
| +1 | fused_residual_add_rmsnorm | PASS | 5.74e-03 | 441.3 ms | **1.018x** |
| +2 | rmsnorm | PASS | 6.35e-03 | 435.2 ms | **1.033x** |
| +3 | silu_gate_mul | PASS | 5.86e-03 | 427.1 ms | **1.052x** |
| +4 | rotary_embedding | PASS | 5.86e-03 | 426.7 ms | **1.053x** |

**Key observations:**
- **All 5 optimizations PASS** correctness at atol=0.01 on both model sizes
- **5.3% end-to-end speedup on 7B** (449ms → 427ms) with `--fused-qkv` — up from 4.6% without
- **Fused QKV accounts for ~0.7% additional speedup** by improving rocBLAS utilization on larger GEMM shapes
- **silu_gate_mul shows clearest per-kernel impact at 7B** (2.1% alone) — 32 SwiGLU layers
- **LlamaModel7B runs without OOM** on Strix Halo's 108 GB unified LPDDR5X
- **Max errors well within tolerance** (6.35e-03 worst case, vs 0.01 threshold)

### rocBLAS GEMM Profiling (LlamaModel7B, B=1, T=512)

Benchmarked all GEMM shapes to understand time distribution:

| Op | Shape | Time | TFLOPS | % Peak | Count |
|----|-------|------|--------|--------|-------|
| wq | [512,4096]@[4096,4096] | 0.546ms | 31.4 | 62.9% | x32 |
| wk | [512,4096]@[4096,1024] | 0.142ms | 30.3 | 60.6% | x32 |
| wv | [512,4096]@[4096,1024] | 0.141ms | 30.6 | 61.1% | x32 |
| wo | [512,4096]@[4096,4096] | 0.548ms | 31.4 | 62.7% | x32 |
| w1 | [512,4096]@[4096,11008] | 1.493ms | 30.9 | 61.8% | x32 |
| w3 | [512,4096]@[4096,11008] | 1.487ms | 31.1 | 62.1% | x32 |
| w2 | [512,11008]@[11008,4096] | 1.528ms | 30.2 | 60.4% | x32 |
| output | [512,4096]@[4096,32000] | 4.587ms | 29.3 | 58.5% | x1 |
| **Total** | | **192.9ms** | | | **43.1% of forward** |

**Fused QKV finding:** Combined [512,4096]@[4096,6144] achieves 37.4 TFLOPS (74.9% peak) vs ~31 TFLOPS for separate calls — 1.3x speedup per-layer. hipBLASLt tested but slightly slower than default rocBLAS.

### Bugs Fixed During Phase C

**1. `model.to(dtype=float16)` destroys complex buffers (critical bug)**
- `freqs_cis` is a complex64 buffer (`cos + j*sin`) used for RoPE
- `model.to(dtype=float16)` silently casts it to real fp16, **discarding the sine component**
- This meant the REFERENCE model was computing with `(cos + 0j)` — fundamentally wrong rotation
- Fix: verify.py now saves complex buffers before dtype cast and restores them after
- This bug caused all prior rotary_embedding failures (max_err=0.13 was comparing correct vs incorrect output)

**2. rotary_embedding fp16 vs fp32 precision mismatch**
- LLaMA's `apply_rotary_emb` computes in fp32 (`.float()` promotion) then casts back to fp16
- Our bench.py kernel uses native fp16 intrinsics (3.7x speedup, passes bench.py)
- For verify.py, added `kernel_fn_fp32`: loads fp16 x, accepts fp32 cos/sin, computes in fp32, stores fp16
- Isolated per-element error: 2.4e-07 (vs 0.13 from the buffer bug above)

**3. fused_residual_add_rmsnorm dual-output kernel**
- Original kernel returns only `rmsnorm(x + residual)`, but TransformerBlock needs both `hidden = x + residual` (as next residual) and `rmsnorm(hidden)`
- Added `kernel_fn_dual` returning `(hidden, normalized)` tuple via second output buffer in HIP kernel
- The kernel already computed `hidden` in Phase 1 for LDS caching — just added a global memory write

**verify.py capabilities:**
- `--incremental`: kernel-by-kernel cumulative measurement with automatic fail-and-remove
- `--kernel-order fused_residual_add_rmsnorm,rmsnorm,silu_gate_mul,rotary_embedding`: control application order
- `--fused-qkv`: fuse wq/wk/wv into single matmul (rocBLAS utilization optimization)
- `--chart`: generate matplotlib progress chart from historical log
- Append-only JSON log at `workspace/incremental_log.json`
- Complex buffer preservation across dtype casts
- fp32-intermediate rotary kernel for model-level precision matching

### torch.compile Results (Inductor backend on ROCm)

Added `--torch-compile` flag. Results show significant speedup from Inductor's graph-level fusion:

| Model | Baseline | Compiled | Speedup |
|-------|----------|----------|---------|
| LlamaModel (170M) | 17.2 ms | 13.5 ms | **1.275x** |
| LlamaModel7B (5.9B) | 446 ms | 384 ms | **1.162x** |

**Key findings:**
- **torch.compile and HIP kernel replacements are mutually exclusive** — compiled models can't integrate dynamically-loaded C++ extensions
- **Inductor fuses operations our manual kernels can't**: embedding lookups, attention softmax internals, tensor transposes, residual adds across the full graph
- **16.2% speedup on 7B is larger than our 5 manual optimizations combined** (5.3%)
- **Limitation**: project's `profile.py` shadows stdlib `profile` module; verify.py works around this by temporarily removing CWD from sys.path
- **Not composable with `--incremental`/`--fused-qkv`**: use either torch.compile OR manual kernel replacements, not both

### compile-with-kernels: Custom Ops + Inductor Fusion

Registered HIP kernels as `torch.library.custom_op` so Inductor can fuse PyTorch ops AROUND them. New file: `kernels/hip/_torch_ops.py`.

| Model | Mode | Ref (ms) | Compiled (ms) | Speedup |
|-------|------|----------|---------------|---------|
| LlamaModel 170M | default | 17.2 | 13.1 | **1.308x** |
| LlamaModel 170M | reduce-overhead | 17.2 | 13.4 | 1.281x |
| LlamaModel7B 5.9B | default | 446.9 | 385.0 | **1.161x** |
| LlamaModel7B 5.9B | reduce-overhead | 448.4 | 384.7 | **1.166x** |

- **1.308x on 170M** exceeds torch.compile alone (1.275x) — custom ops add measurable benefit
- On 7B, matmuls dominate so custom ops have less relative impact (~same as torch.compile alone)
- `reduce-overhead` (CUDA graphs) is neutral — ROCm's graph support doesn't add pipelining benefit on gfx1151
- All correctness PASS (max_err < 0.01)

### AutoKernel Library API (`autokernel/` package)

Productionized the optimization pipeline as a library with one-liner API:

```python
import autokernel
model = autokernel.optimize(model, compile=True)  # auto-detect & apply everything
print(autokernel.report(model))                     # inspect applied patterns
autokernel.restore(model)                           # revert to original
```

| Model | Patterns Applied | Compiled (ms) | Speedup |
|-------|-----------------|---------------|---------|
| LlamaModel (170M) | 4 patterns, 49 modules | 12.8 ms | **1.340x** |
| LlamaModel7B (5.9B) | 4 patterns, 129 modules | 376.9 ms | **1.189x** |

The library auto-detects:
- **fused_residual_rmsnorm** (TransformerBlock pattern: attention + attention_norm + feed_forward + ffn_norm)
- **fused_qkv** (Attention with Q/K/V projections: wq/wk/wv or q_proj/k_proj/v_proj)
- **rmsnorm** (RMSNorm class names or heuristic: has weight+eps, no bias)
- **silu_gate_mul** (SwiGLU FFN: w1/w2/w3 or gate_proj/down_proj/up_proj)

HuggingFace naming aliases supported (q_proj, gate_proj, input_layernorm, etc.).

### Decode Benchmark (autoregressive generation with KV-cache)

Added `--decode-benchmark` mode that implements prefill + autoregressive decode with KV-cache:

| Model | Prefill (128 tok) | Decode (ms/tok) | Tokens/sec |
|-------|-------------------|-----------------|------------|
| LlamaModel (170M) | 5.4 ms | 5.05 ms | **197.9 tok/s** |
| LlamaModel7B (5.9B) | 150.8 ms | 106.93 ms | **9.4 tok/s** |

**Decode bottleneck analysis (LlamaModel7B):**
- 106.93 ms/token = 32 layers × ~3.3 ms/layer
- Each layer runs: wq+wk+wv+wo (4 matmuls) + w1+w3+w2 (3 matmuls FFN) + attention + norms
- At B=1, T=1: matmuls dominate (~95% of decode time) because input is [1, dim] — tiny compute but full weight reads
- Attention with KV-cache: reads ~2 MB/layer (K+V cache) at 128 context — only ~0.6 ms total across 32 layers
- **Custom decode attention kernel would save <1% of decode time** — not worth implementing

**Key insight:** Decode performance on bandwidth-limited APU (170 GB/s) is bottlenecked by **weight reads**, not KV-cache reads. Each decode step reads ALL model weights (~12 GB at fp16 for 7B) through every layer. At 170 GB/s, this is the fundamental floor: `12 GB / 170 GB/s ≈ 70ms` minimum per token.

**Remaining limitations:**
- **matmul (0.24x)**: Skipped — no MFMA, would regress
- **Cross-block fused_residual_add_rmsnorm**: Marginal benefit (~0.16%)

---

## Future Opportunities

Ranked by expected impact based on what we've learned:

### High Impact
1. **Int4/Int8 quantized inference** — Our `dequantize_int4` kernel is 16.3x. Quantizing model weights to int4 transforms matmuls from compute-bound to memory-bound, cutting weight traffic by 4x. Could yield 2-4x end-to-end decode speedup. **Highest remaining opportunity.**
2. **Medusa heads** (speculative decoding) — Add lightweight prediction heads to generate 3-4 tokens per forward pass. Published results: 2.2-3.6x decode speedup. Directly addresses the decode bottleneck (weight reads dominate, so amortizing them over more tokens helps).
3. **AdaInfer / Early exit** — Skip 17-43% of transformer layers when intermediate output is "good enough" for easy tokens. Reduces weight reads proportionally. Would save 20-46ms/token on 7B.

### Completed
- ~~**Fused bias + GELU**~~ — **DONE** (1.89x)
- ~~**Extend verify.py replacement strategies**~~ — **DONE**: 4 HIP kernels + fused QKV, all PASS on both models
- ~~**Fix rotary_embedding verify.py integration**~~ — **DONE**: fp32 variant + complex buffer fix
- ~~**Modify fused_residual_add_rmsnorm to return tuple**~~ — **DONE**: `kernel_fn_dual`
- ~~**torch.compile integration**~~ — **DONE**: 1.16x on 7B via Inductor
- ~~**Custom ops + Inductor fusion**~~ — **DONE**: `--compile-with-kernels` (1.31x on 170M, 1.16x on 7B)
- ~~**Fused QKV projection**~~ — **DONE**: `--fused-qkv` (rocBLAS utilization 31→37 TFLOPS)
- ~~**Decode benchmark**~~ — **DONE**: `--decode-benchmark` with KV-cache
- ~~**rocBLAS GEMM profiling**~~ — **DONE**: mapped all 8 GEMM shapes, identified 43% of forward time in matmuls

### Won't Help (learned from experiments)
- **Custom decode attention kernel** — Attention is <1% of decode time on 7B. Weight reads dominate, not KV-cache reads.
- **Cross-block fused_residual_add_rmsnorm** — Benchmarked at ~0.16% improvement. Not worth the architectural complexity.
- **Fused proj+RoPE** — L2 cache already absorbs intermediate tensor I/O. Saves only kernel launch overhead (~0.2ms).
- **hipBLASLt** — Tested, slightly slower than default rocBLAS on gfx1151.
- **Combined w1+w3 (FFN gate+up)** — Individual GEMMs already large enough for good utilization. No speedup.
- **`reduce-overhead` compile mode** — CUDA graphs are neutral on gfx1151. ROCm's graph support doesn't add pipelining.
- **LDS caching for standalone normalization kernels** — L2 already caches the second read.
- **Simple element-wise kernels** — SiLU/GELU already at parity with PyTorch.
- **Top-k selection via custom algorithms** — PyTorch's radix sort is better (0.25x).
- **Compute-bound kernels without MFMA** — Can't beat rocBLAS on scalar FMA.

### Key Patterns Discovered

**For op-level speedups (HIP kernels):**
The winning formula: **find where PyTorch launches 3+ separate kernels for what could be 1.** Each intermediate tensor eliminated saves 2 memory passes. Top results: dequantize_int4 (16.3x), fused_residual_add_layernorm (10.7x), dequantize_int8 (8.1x).

**For model-level speedups (end-to-end):**
torch.compile's Inductor backend provides the **largest single improvement** (1.16x on 7B) by fusing at the graph level — operations between our kernels that we couldn't manually fuse. Registering HIP kernels as `torch.library.custom_op` lets Inductor work alongside them.

**Fundamental bottleneck on Strix Halo:**
- **Prefill:** Matmuls are 43% of time, running at 60-63% of peak TFLOPS. rocBLAS is already near-optimal.
- **Decode:** Weight reads are ~70ms minimum per token at 170 GB/s for 7B. Only weight reduction (quantization, pruning) or amortization (speculative decoding) can break this floor.

---

## Appendix: Benchmark Commands

```bash
# SSH to test machine
ssh joelwang-ai-2@192.168.1.140

# Run a specific kernel benchmark
cd ~/Desktop/autokernel-halo-strix
source .venv/bin/activate
cp kernels/hip/<kernel>.py kernel.py
python bench.py --kernel <kernel_type>

# Example: benchmark cross_entropy
cp kernels/hip/cross_entropy.py kernel.py
python bench.py --kernel cross_entropy
```

Note: Use `python bench.py` directly, not `uv run bench.py` — the `uv` binary may not be in PATH on the remote machine.

---

## Training Stack Results (halo_training/)

After kernel optimization, the `halo_training/` package applies optimized kernels to pretraining. This section documents verified training performance on AMD Ryzen AI MAX+ 395 (Radeon 8060S, gfx1151).

### Phase-by-Phase Results

| Phase | Configuration | Throughput | MFU | Memory | Key Finding |
|-------|---------------|-----------|-----|--------|-------------|
| 1 (Mode A) | Eager baseline, 124.7M | 14.5K tok/s | 17% | ~17 GB | Basic loop works, loss converges |
| 1.5 (Fixes) | + cudagraph fix, compile | 13.8K tok/s | — | ~17 GB | Stable 126 steps, grad norms healthy |
| 2 (Metrics) | + BPB/MFU tracking | — | 17% | — | Smoke test passes all 6 criteria |
| 3 (Mode B) | Layer-streaming, 2.09B | 853 tok/s | — | 34.5 GB | Per-layer checkpoint every 2 layers |
| 4 (Autokernel) | + autograd backward, compile | **43K tok/s** | **54%** | ~17 GB | **3.05x speedup** over Phase 1 |
| 5 (CLI/Eval) | Decode benchmark, 7B | 103 tok/s | — | — | Prefill 10ms, KV-cache decode |

### Mode A vs Mode B

- **Mode A** (<2B params): Whole-model `torch.compile`. Best throughput (43K tok/s with autokernel). Auto-selected when model + optimizer + activations fit in ~60% of GPU memory.
- **Mode B** (>2B params): Per-layer activation checkpointing via `LayerStreamingTrainer`. Trades ~20-30% throughput for dramatically lower memory. Enables 2B+ training on 116 GB unified memory.
- Mode selection is automatic via `suggest_mode()` — uses 60% of GPU-visible memory as the threshold.

### Autokernel Training Integration

The key result: **3.05x training speedup** by adding autograd backward to 4 HIP custom ops:

| Op | Forward | Backward | Speedup Contribution |
|----|---------|----------|---------------------|
| `fused_res_rmsnorm` | HIP kernel (6.6x inference) | PyTorch (fp32 for stability) | Largest — fused residual + norm |
| `rmsnorm` | HIP kernel (3.3x inference) | PyTorch | Per-layer norm |
| `silu_gate_mul` | HIP kernel (1.6x inference) | PyTorch (mixed precision) | SwiGLU activation |
| `rotary_emb_fp32` | HIP kernel (3.7x inference) | PyTorch (orthogonal transform) | RoPE embeddings |

All 4 ops register via `torch.library` and compose with `torch.compile`. The backward pass uses pure PyTorch ops for correctness — HIP forward provides the speedup, PyTorch backward provides the gradients.

### Smoke Test Criteria (from llm_engineer playbook)

The `run_smoke_test()` validates training stability with 6 criteria:
1. Loss decreases over first 100 steps
2. No NaN/Inf for 200 steps
3. Gradient norms < 10 (no spikes)
4. Peak memory < 6 GB (for 250M model)
5. Throughput > 10K tok/s
6. State-norm ratio < 1.05 (for recurrent models)

---

## AMADEUS: Non-Transformer Architecture Experiment

**Hypothesis:** Parallel hybrid (gated conv + Mamba-3 SISO SSM) + FiLM conditioning can match transformer performance with better decode speed. From `mad_llm_scientist/plans/AMADEUS.md`.

### Architecture

AMADEUS (243.8M params): 16 parallel hybrid blocks, each with gated conv (640-dim, local patterns) running in parallel with Mamba-3 SISO SSM (384-dim, global context). Outputs concatenated → projection → SwiGLU FFN. FiLM conditioning applied on layers 9-16 using a 64-dim fingerprint from layer 8.

### SSM Scan Optimization Journey

| Implementation | tok/s | MFU | Speedup | Notes |
|---------------|-------|-----|---------|-------|
| Sequential loop | 1,300 | 4% | 1.0x | Python for-loop, 512 serial steps per layer |
| `torch.associative_scan` | 1,300 | 4% | 1.0x | PyTorch higher_order_ops, equally slow |
| **Chunked linear recurrence** | **6,400** | **15.9%** | **4.9x** | chunk_size=64, 8 serial inter-chunk steps |

The chunked approach uses cumprod+cumsum within chunks (fully vectorized) and only 8 serial steps for cross-chunk state propagation (T=512/64=8). This was the key optimization — neither the sequential loop nor PyTorch's built-in associative scan were viable.

### Training Results (BabyLM, 2 epochs, eager mode)

| Step | Loss | BPB | tok/s | MFU | Memory |
|------|------|-----|-------|-----|--------|
| 10 | 42.2 | 16.9 | 6,233 | 15.4% | 12.7 GB |
| 100 | 19.8 | 7.95 | 6,463 | 15.9% | 12.7 GB |
| 500 | 15.6 | 6.27 | 6,462 | 15.9% | 12.7 GB |
| 1000 | 13.6 | 5.46 | 6,461 | 15.9% | 12.7 GB |
| 1500 | 13.1 | 5.24 | 6,462 | 15.9% | 12.7 GB |
| **1880** | **12.18** | **4.88** | **6,462** | **15.9%** | **12.7 GB** |

**Training complete:** 2 epochs on BabyLM (30.8M tokens), 1880 optimizer steps, 79.4 min wall time.

### SSM Implementation Lessons

- **A_log init:** `log(arange(1, dstate+1))` per head — gives exponentially spaced decay rates (fast to slow)
- **dt_proj:** Bias initialized to -4.0 (softplus(-4)≈0.018), weights scaled small (std=0.001) — starts with gentle state updates
- **dt clamping:** [1e-4, 0.5] prevents extreme discretization steps
- **B/C normalization:** `B / max(B.norm(), 1.0)` prevents unbounded state growth
- **FiLM identity init:** gamma_proj and beta_proj zeroed so initial transform is h×1+0=h
- **GradScaler warmup inf:** Normal behavior — scaler starts at 65536, first fp16 overflow triggers scale reduction. Not a stability issue.
- **StateNormMonitor false positives:** Designed for Griffin-style fixed recurrence. Data-dependent SSMs (Mamba) naturally produce variable output norms between batches. Ratios of 3-5 are normal.

### Component Profiling (per forward pass, batch=8, seq=512)

| Component | Time (eager) | Time (optimized) | Optimization Applied |
|-----------|-------------|-----------------|---------------------|
| Mamba3SISO scan | 10.6ms | **1.8ms** | HIP selective_scan kernel (5.9x) |
| SwiGLU FFN | 3.7ms | ~2.3ms | silu_gate_mul HIP kernel (1.6x) |
| RMSNorm ×2 | 1.4ms | ~0.4ms | rmsnorm HIP kernel (3.3x) |
| GatedConv | 1.2ms | ~1.2ms | (no optimization) |
| LM Head | 13.1ms | ~13.1ms | (rocBLAS matmul, can't beat) |

### Throughput Optimization Journey

| Config | tok/s | MFU | vs Baseline | What Changed |
|--------|-------|-----|-------------|-------------|
| Eager baseline | 5,940 | 14.6% | 1.0x | Raw PyTorch, chunked scan |
| + autokernel patterns | 7,638 | 18.8% | 1.29x | RMSNorm (3.3x) + SwiGLU (1.6x) HIP kernels |
| + torch.compile(default) | 10,100 | 24.9% | 1.70x | Inductor graph fusion |
| **+ HIP scan kernel** | **10,406** | **25.6%** | **1.75x** | Fused parallel scan (5.9x on scan) |

Key finding: `torch.compile(mode="reduce-overhead")` gives identical throughput to `mode="default"` on SSM models (8,278 vs 8,258 tok/s). CUDAGraphs don't help when chunked scan dominates. Use `mode="default"` to avoid CUDAGraph conflicts with autokernel patterns.

### Adaptive LM Head Experiment (did NOT help training)

Tested a 3-tier adaptive softmax (8K full-rank + 16K low-rank(256) + 26K low-rank(128)) with chunked cross-entropy to reduce the LM head matmul cost:

| Config | tok/s | Memory | Notes |
|--------|-------|--------|-------|
| Standard head + autokernel + compile | **10,421** | 12.7 GB | One large matmul (4096, 1024) @ (1024, 50257) |
| Adaptive head + autokernel + compile | 9,983 | 9.7 GB | Three smaller matmuls + tier routing overhead |

**Result: 4% slower.** The overhead of tier routing, masked gathers, and 3 separate matmuls outweighs the savings from smaller individual matmuls. On this memory-bound hardware, rocBLAS handles one large matmul more efficiently than three smaller ones. Memory savings are real (9.7 vs 12.7 GB) but throughput didn't improve.

**Lesson:** On gfx1151 without MFMA, the LM head matmul is memory-bound, not compute-bound. Reducing FLOPS via low-rank tiers doesn't help because bandwidth is the bottleneck — and splitting into 3 matmuls adds kernel launch overhead. The adaptive head may still benefit decode/inference (early exit skips weight reads) but not training.

---

## Prior Training Baselines (from mad_llm_scientist)

Training experiments conducted before halo_training/ was built, using `~/Desktop/ai_lab/mad_llm_scientist/` on the remote machine. These provide reference points for evaluating new architectures.

### META-ENGRAM (234.5M)

| Metric | Value | Notes |
|--------|-------|-------|
| Throughput | ~4K tok/s | Before torch.compile |
| 200-step loss | 1008 → 72 | 92.8% reduction |
| Stability | 200 steps, no NaN | With tuned config |
| Peak memory | <6 GB | 250M model |
| Winning LR | 2e-5 | With grad_clip=0.5 |
| Key fix | d_rec=256→64, LRU_init_A=0.5→3.0 | Extended stability 2.4x |

### Key Stability Findings (apply to all recurrent architectures)

1. **Recurrence dimension matters more than depth:** d_rec=64 much more stable than d_rec=256
2. **LRU init A=3.0** (sigmoid decay ≈ 0.95): slow initial decay prevents early state explosion
3. **State-norm ratio monitoring** (`||h_t||/||h_{t-1}||`): detects instability ~10 steps before gradient explosion
4. **FP32 required for scans/recurrence:** Instability is architectural, not from AMP
5. **Separate optimizer groups:** 5x LR for Engram tables, 0.1x for decay_bias, standard for backbone

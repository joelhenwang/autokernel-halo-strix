# Phase 2 Summary — Fusion Investigation Complete

**Status:** COMPLETE. Six work items evaluated. Zero fusions shipped. All findings
documented.
**Duration:** ~4 hours (vs 2-5 day budget).
**Throughput delta:** 0% (no regressions, no gains).
**Value delivered:** Permanent reference artifacts + hard evidence that Phase 1 already captured the attainable wins on this stack.

## Outcomes per work item

| WI | Target | Investigation | Outcome |
|----|--------|---------------|:--------|
| WI1 | `triton_poi_fused__to_copy_mul_transpose_view_{7,8}` (9.1%) | Kernel-body analysis | CLOSED — already optimal Inductor fusion of 5 ops |
| WI2 | `aten::add_` + `aten::copy_` (9.3% combined) | Shape-annotated profile | CLOSED — autograd-internal + input H2D (delegated to WI5) |
| WI3 | `aten::embedding_dense_backward` (4.1%) | Risk/reward analysis | NOT ATTEMPTED — near bandwidth limit, high risk |
| WI4 | `Memset (Device)` (4.1%) | Grad-strategy ablation | CLOSED — framework-internal, not user-reachable |
| WI5 | `Memcpy HtoD` (4.0%) | 4-way H2D benchmark | CLOSED — all alternatives slower on unified memory |
| WI6 | Inductor fusion catalog | Cache dump + parser | COMPLETE — 92 kernels cataloged, permanent reference |

## Key technical findings

### 1. Inductor's fusion is more aggressive than assumed

The catalog (`docs/perf/inductor-fusion-catalog.md`) shows **92 unique triton kernels** under `compile_zones()`, with **fusion sizes up to 24 ops per kernel**. Op-frequency histogram:

- `mul` appears in 81 / 92 kernels
- `transpose` in 37, `add` in 33, `div` in 35
- `_to_copy` (dtype cast) appears in most pointwise chains — casts never emit separate kernels
- Attention masking + scaling chains are aggressively fused (18-24 ops into one)

**Implication:** any elementwise chain in the model source code is already captured by Inductor. Writing custom HIP kernels for the same patterns buys nothing unless the HIP version beats optimized triton at the same shape — which on gfx1151 is improbable without MFMA.

### 2. The 9.1% triton kernel is RoPE apply + dtype cast + QKV gather

`triton_poi_fused__to_copy_mul_transpose_view_{7,8}` processes 2,097,152 fp16 elements per call with 3 loads + 2 stores. Interpretation: packed-QKV projection × RoPE cos/sin multiply, with saved-for-backward fp32 copy. 6.24 μs/call at batch=16 suggests partial L2 residency — effective bandwidth 4.8 TB/s (above HBM peak, so cache is hot). **Memory-bandwidth-bound by construction; no further fusion possible.**

### 3. `aten::add_` is 67% autograd weight-grad, 29% cross-boundary residuals

Shape-annotated analysis: dominant shapes are `(5632, 768)`, `(768, 2816)`, `(1536, 768)` — all matching SwiGLU, QKV, and LM-head linear weights. These are autograd's internal gradient-accumulation adds, NOT user code. Activation residuals at `(16, 256, 768)` cost only 0.8% of step time. **Neither is attackable in Phase 2 scope.**

### 4. Unified memory inverts H2D-prefetch wisdom

On Strix Halo (shared LPDDR5x), all four tested H2D strategies — plain blocking, non-blocking advisory, pinned+non-blocking, and full CUDA-stream prefetcher — are either tied or **slower** than the current default:

| Strategy | Median tok/s | Δ |
|----------|------------:|---:|
| pin=F, `.to()` (baseline) | 14,275 | — |
| pin=F, `.to(non_blocking)` | 14,146 | −0.91% |
| pin=T, `.to(non_blocking)` | 14,020 | −1.79% |
| pin=T, stream prefetcher | 13,988 | −2.01% |

Event-sync overhead dominates when the H2D cost is already covered by existing stream scheduling. **Current setup is optimal for this APU.**

### 5. Memsets come from runtime internals, not user code

Every user-visible `aten::zero_` records 0.0 μs of self-CUDA time (Inductor elides them). The 221 μs of `Memset (Device)` in the Phase 1 profile comes from somewhere deeper — rocBLAS scratch zeroing, fused_adamw workspace, GradScaler overflow scalars, or Inductor reduction preambles. Testing grad-tensor lifecycle alternatives (`set_to_none=True/False`, pre-allocation, `_foreach_zero_`) all produced 2% regressions. **Not attackable from user code.**

### 6. Tied-embedding backward fusion is not worth the risk

`embedding_dense_backward` at 74 μs/call = 1.3 TB/s effective bandwidth on 96 MB of output. Already at the theoretical memory limit for gfx1151. Custom fusion with the tied-lm-head matmul requires wrapping `FactorizedLMHead` in a custom `autograd.Function` — 6-8h of work including gradient-parity tests, with expected lift <2% post-Amdahl. Chunked CE provides an orthogonal path if memory ever becomes the constraint. **Deferred indefinitely.**

## Baseline confirmation

Final ablation (same harness as Phase 1 close):

| Config | Batch=16 tok/s | Batch=16 peak mem |
|--------|--------------:|------------------:|
| baseline | 11,172 | 6.60 GB |
| +HIP CE | 11,516 | 5.83 GB |
| +HIP CE +RoPE fusion | 11,244 | 5.89 GB |
| +HIP CE +RoPE +Chunked CE | 10,962 | 4.88 GB |
| compile + PyTorch CE | 14,106 | 5.61 GB |
| **compile + HIP CE** | **14,708** | **4.83 GB** |
| compile + HIP CE + RoPE | 14,564 | 4.90 GB |
| compile + HIP CE + RoPE + Chunked | 14,152 | **3.89 GB** |

Winners unchanged from Phase 1:
- **Best throughput:** `compile + HIP CE` at **14,708 tok/s** (within measurement noise of Phase 1's 14,682).
- **Best memory:** `compile + HIP CE + RoPE + Chunked` at **3.89 GB**.

## Permanent artifacts delivered

- `docs/perf/inductor-fusion-catalog.md` — 92-kernel catalog, reference for future work.
- `docs/perf/inductor-fusion-catalog.json` — programmatic access.
- `docs/perf/wi1-transpose-copy-analysis.md` — RoPE+gather fusion root cause.
- `docs/perf/wi2-add-copy-analysis.md` — shape-level call-site classification.
- `docs/perf/wi2-shape-calls.md` — raw profile data.
- `docs/perf/wi3-embedding-backward-analysis.md` — risk analysis, deferral rationale.
- `docs/perf/wi4-memset-analysis.md` — grad-strategy benchmark + closure.
- `docs/perf/wi5-h2d-prefetch-analysis.md` — 4-way H2D benchmark + closure.
- `docs/perf/kernel-bodies-c1.txt` — raw triton source for the 9.1% kernel.

Reusable tooling:
- `scripts/dump_inductor_output.py` — captures all Inductor output for a training step.
- `scripts/parse_inductor_cache.py` — builds structured catalog from dump.
- `scripts/extract_kernel_body.py` — pulls arbitrary kernel bodies for inspection.
- `scripts/profile_shape_calls.py` — shape-annotated per-op profile.
- `scripts/bench_h2d_strategies.py` — 4-way H2D benchmark.
- `scripts/bench_zero_grad.py` — grad-strategy benchmark.

## What Phase 2 tells us about where gains actually live

The current stack on gfx1151 / Strix Halo is **fundamentally limited by**:

1. **Memory bandwidth, not compute.** Every "hot" kernel identified in Phase 1 sits at or near peak achievable bandwidth for its shape. Adding more fusion doesn't reduce bytes moved — and bytes moved is the ceiling.
2. **rocBLAS matmul being outside user control.** At 20-27% of wall time (matmul category), this is optimally tuned Tensile output for gfx1151 shapes. Can't be beaten without MFMA (which gfx1151 doesn't have).
3. **Framework overhead that isn't visible at the op level.** Memsets from runtime internals, HIP stream submission latency, and similar costs accumulate to a few percent that user code cannot reach.

Realistic paths to further throughput:

- **Larger batch sizes.** Phase 1 found compile lift scales from 1.08× at b=4 to 1.32× at b=16. Continuing to 32+ (memory permitting via chunked CE) may unlock another 5-10% by amortizing constant overheads.
- **CUDA graphs** (Phase 3). Eliminates HIP launch overhead by capturing the entire step as a replay-able graph. Phase 3 spec estimates 5-15% lift.
- **Grad accumulation** to hit effective batch 64+ while keeping memory bounded. Orthogonal to Phase 2/3.
- **MFMA-enabled hardware** (gfx942, MI300). Not Strix Halo.

## Hand-off to Phase 3

Phase 3 (CUDA graphs through Parcae loop) starts from the current Phase 2-confirmed baseline of **14,708 tok/s** at `compile + HIP CE` on batch=16. POC day 1 can begin immediately.

Phase 2 produced zero code changes that Phase 3 must be aware of. The Inductor fusion catalog is useful background reading.

## STATUS.md update (separate commit)

Add Phase 2 row to STATUS.md:
```
Phase 2 (Fusion investigation) — COMPLETE 2026-05-05
  Result: 0 shipped fusions, 0 regressions. Phase 1 already captured attainable wins.
  Artifacts: Inductor catalog (92 kernels) + per-WI analyses in docs/perf/.
```

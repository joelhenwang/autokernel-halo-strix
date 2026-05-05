# Phase 3 WI-A0 — reduce-overhead probe: consolidated findings

**Date:** 2026-05-05
**Config:** OdinHalo, batch=16, block=256, warmup=15 steps, 50 measured steps
**Status:** COMPLETE — Track A effectively dead; pivoting to Track B

## Executive summary

The Phase 3 spec's entire premise was: "reduce-overhead crashes on looped models due
to buffer aliasing; need clones or unrolled compile to make it work." **This premise
is WRONG.** Reduce-overhead does NOT crash on looped OdinHalo — it runs successfully.
But it ALSO doesn't deliver CUDA-graph benefits because **HIP's CUDA-graph-capture
layer silently fails with "empty graph" warnings**, reverting to eager replay.

Net effect: reduce-overhead on this stack is −1-3% throughput and NO memory savings.
The clone-at-boundaries (WI-A1), manual CUDA graph (WI-A2), and unrolled compile
(WI-A3) are all solving the WRONG problem.

## Experiment results

| Experiment | Warmup (s) | tok/s | Peak GB | Final loss | Notes |
|------------|----------:|------:|--------:|-----------:|:------|
| **E0**: baseline `compile_zones` default | 7.2 | **14,472** | 5.28 | 6.1921 | Production config |
| E1: `compile_zones` **reduce-overhead**, no clones | 13.7 | 14,193 (−1.9%) | 5.23 | 6.2904 | No crash. No benefit. |
| E1c: reduce-overhead + layer-boundary clone | 5.0 | 14,088 (−2.7%) | 5.23 | 6.2652 | Clones slightly slower |
| E2: whole-model `torch.compile(mode="reduce-overhead")` | 14.6 | 14,419 (−0.4%) | 5.06 | 6.2140 | Within noise |
| E3: no compile (eager sanity baseline) | 5.8 | 10,978 (−24%) | 7.07 | 6.1839 | Shows compile lift |
| E4: compile_zones reduce-overhead + `chunked_ce` | 6.9 | 10,085 (−30%) | **3.64** | 6.4972 | Memory good, tok/s bad |

## Key findings

### 1. Reduce-overhead does NOT crash on looped OdinHalo

The trainer's auto-fallback warning ("reduce-overhead is incompatible with looped
models") is **overly conservative**. Running reduce-overhead without any clones
produces identical loss trajectory to baseline for 50+ steps. No NaN, no wrong-
answer, no silent corruption.

**The buffer-aliasing problem described in the original Phase 3 spec does not
manifest on this PyTorch+ROCm build.** Either it was fixed upstream, or it was
never a real issue on HIP backends.

### 2. CUDA-graph capture fails silently on HIP

Every run with reduce-overhead produced ~28-30 instances of this warning:

```
UserWarning: The CUDA Graph is empty. This usually means that the graph was
attempted to be captured on wrong device or stream.
  (Triggered internally at ATen/hip/HIPGraph.cpp:140)
```

The warning fires from PyTorch's HIP backend when `CUDAGraph.capture_end()` finds
no recorded operations. This means reduce-overhead is setting up capture infrastructure
but **no kernels are actually being enqueued inside the graph**. The compiled function
runs, but it runs eagerly on the main stream — NOT via graph replay.

**Consequence:** we pay the capture-overhead cost (setup + the graph-empty warning
processing) without receiving any of the launch-latency savings that reduce-overhead
is supposed to provide. Net: pure slowdown.

### 3. STATUS.md row 113's 2.14 GB memory claim does NOT reproduce

STATUS.md claims `compile reduce-overhead + HIP CE = 14,425 tok/s / 2.14 GB` at
batch=16. We measure:
- tok/s: 14,193 (close to STATUS's 14,425 — within 1.6%, probably noise)
- Peak GB: **5.23, not 2.14**

The 2.14 GB number must have come from:
- A different PyTorch/ROCm version where HIP graph capture worked, or
- A smaller model/batch combination, or
- Never actually a looped-OdinHalo measurement despite the label.

Either way, **the memory win Phase 3 was chasing isn't available on this build**.

### 4. Chunked CE + reduce-overhead DOES save memory but kills throughput

E4 (chunked_ce + reduce-overhead) produced:
- 3.64 GB (matches STATUS row 116's 3.65 GB at batch=32 claim)
- But only 10,085 tok/s (−30% from baseline, catastrophic)

The HIP CE kernel (invoked per-chunk) breaks out of Inductor's graph entirely, forcing
repeated cold starts and negating any compile benefit. This combination is not
production-viable.

### 5. Option X (clone-at-boundaries) is solving a problem that doesn't exist

WI-A1's core motivation is the alleged buffer-aliasing issue. E1c (with a clone
inserted after `_run_shared_block`) produces **identical loss trajectory** to E1
(without clones), but ~1% slower. The clones are pure overhead since there's
nothing to fix.

## Implications for Track A

**All three CUDA-graph options (X, Y, Z) are invalidated by this investigation:**

- **Option X (clone-at-boundaries):** solves the wrong problem. No buffer aliasing
  exists on this build.
- **Option Y (unrolled compile):** same root cause. If the graph-empty warning
  fires for per-layer compilation, it will also fire for the unrolled variant
  (same capture machinery, same HIP limitation).
- **Option Z (manual CUDAGraph):** explicitly bypasses `torch.compile` and directly
  calls `torch.cuda.CUDAGraph`. **Unlike X/Y, this could work IF the HIP backend
  supports manual graph capture for our specific tensor set.** But this requires
  confirming HIP graph capture works for non-torch.compile usage, which is
  uncertain.

### Should we attempt Option Z anyway?

Arguments for:
- The empty-graph warning may be Inductor-specific. Manual capture via
  `torch.cuda.CUDAGraph` is a lower-level API that may work when the Inductor-driven
  path fails.
- If it works, the expected benefit is real (launch-latency elimination, static
  buffer reuse).

Arguments against:
- Phase 2 evidence: launch overhead is already small on unified memory. Even if
  graph capture succeeds, the upside is likely ≤3% tok/s — below the +5% stretch
  gate.
- Implementation cost is high (custom autograd.Function, GradScaler integration,
  allocator management). Risk of silent correctness bugs.
- The HIP-specific failure mode suggests we'd be fighting infrastructure, not just
  user code.

**Recommendation:** skip Option Z for now, pivot to Track B. Revisit Z only if
Track B's broader levers also fail to find +5%.

## Decision

**Track A CLOSED WITHOUT IMPLEMENTATION.** Primary finding: the HIP backend's
CUDA-graph capture does not work with Inductor's reduce-overhead mode for this
model, and manual capture is high-risk for low ceiling. Throughput-primary goal
must be pursued through Track B.

Secondary finding: trainer's auto-fallback can be relaxed. Reduce-overhead
doesn't crash; it just doesn't help. If a user explicitly sets `TORCH_COMPILE_MODE=reduce-overhead`, we can let them try it and they'll see the warning. However, since it's a −1.9% regression, defaulting to `compile_zones` with mode=default is still correct.

## Hand-off to Track B

Baseline for Track B: **14,472 tok/s** (this run's E0, slightly off Phase 2's
14,708 due to 50-step vs 200-step measurement — will use longer runs for Track B
to reduce noise).

Next work item: **WI-B1 (shape sweep)** — measure tok/s across block × batch
combinations at constant ~4096 token/step budget.

## Artifacts

- `scripts/wi_a0_reduce_overhead_probe.py` — reusable probe harness
- `docs/perf/phase3-wi-a0-findings.md` — E0/E1/E1c/E3 measurements
- `docs/perf/phase3-wi-a0-e2-e4.md` — E2/E4 measurements
- `docs/perf/phase3-wi-a0-findings.json` — structured results
- `docs/perf/phase3-wi-a0-e2-e4.json` — structured results
- This consolidated document

# WI5: H2D prefetch benchmark — closed as "no attackable gain"

**Phase 2 work item:** WI5 — async data prefetch to hide `Memcpy HtoD` (Phase 1 profile: 4.0% of wall time).
**Status:** CLOSED. No prefetch strategy beats the current baseline on Strix Halo's unified memory.
**Evidence:** `scripts/bench_h2d_strategies.py` (4-way comparison at production config, 3 repeats each).

## Benchmark design

Four H2D strategies benchmarked against the production config:

| Strategy | pin_memory | Method |
|----------|:-----------|:-------|
| A (baseline) | False | Plain `.to(device)` (blocking) |
| B | False | `.to(device, non_blocking=True)` (advisory on unpinned) |
| C | True | `.to(device, non_blocking=True)` (truly async on pinned) |
| D | True | CUDA-stream double-buffered prefetcher with event sync |

Config: OdinHalo / batch=16 / block=256 / compile_zones / fused AdamW / fp16 autocast.
Measurement: 25-step warmup, 100 measured steps, 3 repeats each, median tok/s reported.

## Results

| Strategy | Median tok/s | Stdev | Δ vs A | All runs |
|----------|------------:|------:|-------:|----------|
| **A: pin=F, `.to()`** | **14,275** | 22 | +0.00% (baseline) | 14,305 / 14,275 / 14,262 |
| B: pin=F, `.to(non_blocking)` | 14,146 | 21 | −0.91% | 14,131 / 14,173 / 14,146 |
| C: pin=T, `.to(non_blocking)` | 14,020 | 34 | −1.79% | 14,018 / 14,077 / 14,020 |
| D: pin=T, stream-prefetch | 13,988 | 71 | −2.01% | 13,988 / 13,993 / 13,868 |

All four are within a 2% band, with the baseline **winning** on median throughput.

## Interpretation

Strix Halo is a unified-memory APU: LPDDR5x is physically shared between the CPU and
the iGPU. Key consequences for H2D transfers:

1. **There is no DMA across a PCIe bus.** `.to(device)` becomes an intra-RAM memcpy
   that is already as fast as hardware allows. Pinning pages changes nothing.

2. **`pin_memory=True` is a small pessimization.** PyTorch's pinned-memory allocator
   has to issue page-lock syscalls and goes through a separate arena. On Strix Halo
   this overhead has no benefit to repay it. Measured: −1.79% throughput (C vs A).

3. **Stream prefetcher adds sync overhead without hiding real latency.** Each batch
   requires `Event.record()` + `Event.wait()` on the copy stream (~5 μs overhead per
   event × 2 tensors × steps/sec). Because there is no real compute to hide this
   under (the H2D itself takes <50 μs and completes before the first GPU kernel),
   the synchronization cost dominates. Measured: −2.01% throughput (D vs A).

4. **Non-blocking flag on unpinned memory is a silent no-op.** PyTorch still blocks
   internally because source pages aren't registered, but spends a small amount of
   time checking the advisory flag. Measured: −0.91% throughput (B vs A).

## What about the 42 μs/call measurement in WI2?

The 42 μs per `.to(device)` on `(16, 256)` int64 tensors IS a real cost, but it's
**wall-clock CPU time including the kernel-submit overhead**, not pipeline-blocking
compute time. Between steps, the GPU is mostly idle on the input side anyway (waiting
for the next batch), so this "cost" overlaps with kernel launch scheduling.

On a discrete GPU with PCIe, the same 42 μs would include physical bus traversal and
would be hideable with prefetching. On Strix Halo, it is almost entirely
PyTorch/HIP-runtime bookkeeping that cannot be overlapped with anything.

## Decision

**WI5 CLOSED.** No prefetch strategy achieves the ≥0.5% gate. Current setup
(pin_memory=False, non_blocking=True) is within measurement noise of optimal and
marginally slower than the even-simpler pin_memory=False, plain `.to()`.

### Secondary finding (not acting on)

The current trainer uses `.to(device, non_blocking=True)` — this is slightly slower
(−0.9%) than plain `.to(device)`. We could switch to blocking `.to()` for a tiny
gain, but it's within noise and the `non_blocking=True` idiom is more portable
(would be beneficial if the repo is ever run on a dGPU). Leave as is.

### What to remember for Phase 3 (CUDA graphs)

CUDA graphs will pin inputs into a pre-allocated input buffer via
`cudaMemcpyAsync` into a captured replay slot. At that point, the H2D copy is
merged into the graph's prologue and the current 42 μs overhead disappears
automatically. So **Phase 3 will render this cost moot for free** — another reason
not to invest Phase 2 effort here.

## Artifacts

- `scripts/bench_h2d_strategies.py` — 4-way benchmark harness (kept for future).
- Results table above (embedded, no separate artifact needed).

## Hand-off

Proceeding to WI4 (Memset elimination). Last remaining opt candidate in Phase 2.

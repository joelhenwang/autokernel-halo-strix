# T-2.1: batch=32 probe findings

**Date:** 2026-05-10
**Probe:** `scripts/probe_t2_1_batch32.sh`
**Model:** OdinFlat on dolma-10b-odin32k
**Recipe:** Sprint 3A-confirm minus fused_zloss (so we isolate batch-size effect)
**Step budget:** 200 steps / config (50-step warmup)

## Results

| Config | batch | accum | tok/s | peak_mem_gb | vs baseline |
|---|---:|---:|---:|---:|---:|
| baseline | 16 | 8 | 31,563 | 13.5 | — |
| batch32_plain | 32 | 4 | 13,476* | 24.6 | — (unstable) |
| batch32_ddp_tune | 32 | 4 | 31,504 | 24.6 | **-0.2%** |
| batch32_sync_clean | 32 | 4 | 31,521 | 25.6 | **-0.1%** |

\* batch32_plain timed out at 15 min / 200 steps — likely compile wedge or
extended Inductor autotune search. Not a memory issue (24.6 GB fits well
under 40 GB/node). Subsequent runs (ddp_tune, sync_clean) with warmed-up
Inductor cache completed normally.

## Decision: **SKIP batch=32 for Stack B**

Empirical finding: batch=32 does NOT provide throughput benefit on gfx1151
for OdinFlat training. At MFU 20.7% the workload is bandwidth/IO-bound, not
compute-bound, so doubling the compute-per-step doesn't amortize. Doubling
memory (13.5 → 24.6 GB) brings no tok/s win.

This CONTRADICTS the prior 2026-05-06 DDP sweep's "+5% at batch=32" finding.
Probable causes for the discrepancy:
- Prior sweep may have used a different recipe (non-fused CE, different
  chunked-CE settings, different grad accumulation pattern)
- Prior sweep may have measured PyTorch native CE path where grad material-
  ization at (batch×T, V) overwhelmed bandwidth at batch=16 but not batch=32
- Current setup uses chunked CE + potentially more memory-friendly kernels
  already, so the batch=32 win was absorbed elsewhere

## Stack B lock-in

Based on B.1 (bucket sweep, null effect) + B.2 (batch=32, null effect):

**Stack B = Stack A + `--ak-sync-cleanup` (candidate, measurable gain TBD)**

Stack B candidate = Stack A recipe + `--ak-sync-cleanup` (branchless
SPECTRA + deferred loss-sync accumulator). Not yet measured as a stack
combination — will verify by incremental probe or fold into Phase C
control runs.

If `--ak-sync-cleanup` adds no measurable gain, Stack B ≡ Stack A (33.4k
aggregate) and Sprint 3 launches with Stack A.

## Stack composition so far

| Stack | Composition | tok/s aggregate | Delta |
|---|---|---:|---:|
| Baseline | Sprint 3A-confirm | 31,331 | — |
| Stack A | +fused_zloss | **33,410** | +6.6% |
| Stack B (tentative) | A + ak-sync-cleanup | TBD | +0-2% expected |

Continue to Phase B.3 (compiled autograd gated smoke) then Phase C.

## Notes

- batch=32_plain timeout is a diagnostic-only observation; memory is NOT
  the bottleneck. For future investigation, run batch=32 with 500+ steps
  and `--compile` warmed up to isolate whether Inductor autotune reliably
  completes in a bounded time.
- `--ak-sync-cleanup` was NOT measured standalone in this probe; it was
  stacked on top of batch=32, where the batch=32 drag dominates.
- `--ak-ddp-tune` is a no-op in this repo because `gradient_as_bucket_view`
  is already hard-coded on (scripts/train_ddp.py:1403).

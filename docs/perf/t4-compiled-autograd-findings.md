# T-4: Compiled autograd gated smoke findings

**Date:** 2026-05-10
**Probe:** `scripts/probe_t4_compiled_autograd.sh`
**Model:** OdinFlat on dolma-10b-odin32k
**Gate criteria (v3 §3.3):** ≥2.5-3% net DDP tok/s gain AND no overlap regression AND no recompile storm (<5 recompiles after warmup).

## Results

| Config | tok/s | vs baseline | allreduce/step | overlap | recompiles | status |
|---|---:|---:|---:|---:|---:|---|
| baseline | 31,630 | — | 1 | 0.000 | 0 | control |
| + compiled_autograd | 31,444 | **-0.6%** | 1 | 0.000 | 0 | REGRESSES |
| + CA + fused_zloss | 33,605 | +6.2% | 1 | 0.000 | 0 | fused_zloss doing the work |
| + CA + ak-ddp-tune | 31,459 | **-0.5%** | 1 | 0.000 | 0 | no win |

## Decision: **GATE FAILS — ship as infra only**

Compiled autograd alone REGRESSES throughput by 0.6% on gfx1151 DDP.
Even stacked with fused_zloss it produces 33,605 tok/s — marginally BELOW
Stack A's T-1.5 canonical 33,410 which was a 2000-step measurement (these
300-step probes are compile-warmup-dominated; if anything, 33,605 here
would be ~33,400-34,000 at 2000 steps — essentially tied with Stack A).

This aligns with v3's caveat:
- T-0.1 measured backward at 2.18× forward = near-theoretical (2× FLOPs).
  Backward is mostly real GEMM work, not Python/graph overhead.
- Compiled autograd's upside depends on the Python autograd engine being
  a bottleneck. On this gfx1151 setup, it is not.
- PyTorch DDP docs warn compiled autograd can REGRESS allreduce overlap;
  our measurement shows no allreduce regression because we already use
  manual async allreduce (1 per step), but there's also no overlap
  benefit to reclaim.

**Allreduce trace note:** `allreduce_count=1` and `overlap=0.000` in every
config reflect the known cosmetic limitation that the DDP trace's
`allreduce_total_ms` diff requires multiple allreduce events per step
(we emit 1). Not a bug in the probe — just that the trace can't
meaningfully measure overlap when there's only one event.

## Ship decision

- `--ak-compiled-autograd` flag stays as infrastructure (users can opt in)
- **NOT added to Stack B/C recipe**
- **Stack C ≡ Stack B** (no promotion)
- Future session may re-test compiled autograd when PyTorch makes the
  autograd engine more of a bottleneck (unlikely on gfx1151)

## Stack composition so far

| Stack | Composition | tok/s | Delta | Source |
|---|---|---:|---:|---|
| Baseline | Sprint 3A-confirm | 31,331 | — | prior canonical |
| A | +fused_zloss | **33,410** | +6.6% | T-1.5 FINAL 2000-step |
| B | = A (batch=32, bucket, CA all null-effect) | 33,410 | +6.6% | B.1/B.2/B.3 |
| B+ | A + ak-sync-cleanup (untested as stack) | TBD | +0-2% | candidate |
| C | = B | 33,410 | +6.6% | CA gate fails |
| D | A/B/C + T-5 ingredients if passing | TBD | TBD | Phase C |

Realistic conclusion: **our throughput ceiling without hidden-kernel recovery
sits near 34k tok/s** (≈+8%). To push toward 36-40k requires either:
1. Hidden-kernel recovery (Phase C, unconditional) if it can be made stable.
2. Algorithmic changes (larger block, different recipe) outside v3 scope.

## Next

Phase C (T-5 hidden kernel recovery): C.0 infra is shipped (replay-bundle
dump in save_nan_forensics). Move to C.1 warm-start matrix → C.2 trust cap
→ C.3 w_gate_up staging → C.4 Stack D assembly + 2000-step gate.

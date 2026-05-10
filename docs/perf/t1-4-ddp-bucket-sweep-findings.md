# T-1.4: DDP bucket_cap_mb sweep findings

**Date:** 2026-05-10
**Probe:** `scripts/probe_t1_ddp_bucket_sweep.sh`
**Configs:** bucket_cap_mb ∈ {8, 25, 50, 100}
**Steps per config:** 150 (with 50-step warmup — log points at step 50, 100, 150)
**Recipe:** Sprint 3A-confirm MINUS fused z-loss (isolates DDP effect)

## Results

| bucket_cap_mb | median tok/s | delta vs min |
|---:|---:|---:|
| 8 | 28,639 | +0.00% |
| 25 | 28,608 | -0.11% |
| 50 | 28,736 | +0.34% |
| 100 | **28,799** | +0.56% |

**Max spread: 0.67%** — all within noise floor. v3's prediction CONFIRMED: the repo's manual `allreduce_grads_async` path dominates; the DDP bucket subsystem sees only the final 1 allreduce per opt step, so bucket size tuning has negligible impact.

## Decision: no change

- **Keep default `bucket_cap_mb=25`** (no significant winner).
- `gradient_as_bucket_view=True` is already hard-coded in `scripts/train_ddp.py:1403` — confirming ~0.5-1 GB memory saving at negligible throughput cost.

**No update to `scripts/launch_sprint3a.sh` required.**

## Stack composition update

Stack B candidate ingredients:
- ✗ bucket_cap_mb tuning → null effect, skip
- ✓ `gradient_as_bucket_view=True` → already default
- ? batch=32 (Phase B.2 next)
- ✓ `--ak-sync-cleanup` (branchless SPECTRA + deferred loss) → will stack with A
- ✓ `--ak-ddp-tune` flag → gives `gradient_as_bucket_view=True` explicitly (no-op if already on)

## Caveats

- Probe was short (150 steps) so compile-warmup dilutes steady-state tok/s.
  Measured 28.6–28.8k range is ~15% below T-1.5's 33.4k because T-1.5 was 2000
  steps (compile amortized over more tokens). Relative comparison between
  configs is still valid because the warmup fraction is identical.
- Probe ran WITHOUT `--use-fused-zloss`; Stack A's +6.6% is orthogonal to
  bucket sizing and adds cleanly on top.
- `--assert-no-sync` was passed; zero violations observed across all 4 configs.

## Baseline context

```
Baseline Sprint 3A-confirm: 31,331 tok/s
Stack A (fused zloss):     33,410 tok/s  (+6.6%)   ← T-1.5 canonical locked
Stack B candidate:         (pending T-2.1 batch=32 probe)
```

## Next

Phase B.2: batch=32 probe. Expected per v3 prior sweep: +3-5% tok/s if memory fits (~+8 GB/node).

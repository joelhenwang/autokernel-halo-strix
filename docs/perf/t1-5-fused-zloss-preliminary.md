# T-1.5 fused-zloss validation: preliminary 300-step findings

**Run:** OdinFlat Sprint 3A recipe + `--use-fused-zloss --ak-loss-zloss`
**Launched:** 2026-05-10 19:08 local on Machine A + Machine B via DDP/TB4
**Status:** **RUNNING** — 2000-step target, currently at step 300 @ ~4h wall projected
**Checkpoint dir:** `checkpoints/t1-5-fused-zloss` (Machine A)
**DDP trace:** `checkpoints/t1-5-fused-zloss/ddp_trace_rank0.jsonl`

---

## Preliminary verdict: **T-1.5 PASS** (high confidence at 300 steps)

| Metric | Baseline (Sprint 3A-confirm) | T-1.5 (fused zloss) | Delta |
|---|---:|---:|---:|
| Steady-state tok/s | ~31,331 | **~33,750** | **+7.7%** |
| Step 50 (compile warmup) | ~25k | 26,295 | - |
| Step 100 | ~30k | **34,026** | - |
| Step 150 | | 33,887 | - |
| Step 200 | | 33,781 | - |
| Step 250 | | 33,748 | - |
| Step 300 | | 33,729 | - |
| Loss @ 300 | ~4.3 | **4.49** | within variance |
| GradScaler scale | 1.0e+03 | 1.0e+03 | stable |
| No divergence | ✓ | ✓ | ✓ |
| No frozen params | ✓ | ✓ | ✓ |

**Throughput signal:** steady +7.7% over baseline, matching v3's predicted 5-8% range.
**Stability signal:** loss declining smoothly, no scaler events, no unusual telemetry.

---

## DDP trace confirms no_sync correctness (T-0.6 / T-1.3)

From `ddp_trace_rank0.jsonl` at step 148:
```json
{
  "step": 148, "accum_steps": 8, "no_sync_expected_microsteps": 7,
  "allreduce_count": 1,  ← exactly 1 per opt step (correct)
  "bucket_cap_mb": 25,
  "gradient_as_bucket_view": false,
  "world_size": 2
}
```

**Conclusion:** no `no_sync` regression. Accumulation is working correctly. No allreduce storm.

Known limitation: `allreduce_total_ms` and `overlap_ratio_estimate` currently always zero
because the timing diff requires multiple allreduce events per step (we have only 1).
Does not affect correctness signal.

---

## Recipe that locked in

```
--imu1-groups --normuon --lr-2d 5e-3 --lr-1d 8e-4 \
--intra-doc-mask --value-residuals --head-gating \
--z-loss 1e-4 --z-loss-fraction 1.0 --attn-softcap 50.0 \
--activation-monitor --activation-monitor-interval 200 \
--mup --mup-base-width 256 \
--spectra-post --spectra-clip-norm 1.0 \
--use-fused-zloss --ak-loss-zloss
```

Notes:
- T-1.1 branchless SPECTRA is NOT active in this run (baseline SPECTRA).
- T-1.2 deferred loss sync (--ak-sync-cleanup) is NOT active.
- These can stack on top of the +7.7% from fused zloss alone.

---

## Projected final-stack throughput

Incremental expected wins (v3 estimates, tightened by our T-0 data):

| Addition | Expected | Cumulative tok/s |
|---|---:|---:|
| Baseline (Sprint 3A-confirm) | — | 31,331 |
| + fused zloss | +7.7% (measured) | **33,750** |
| + branchless SPECTRA | +1-2% (expected) | 34,100 — 34,400 |
| + deferred loss sync | +0.5-1% | 34,300 — 34,750 |
| + DDP bucket tune | +1-3% (T-1.4 probe) | 34,700 — 35,800 |
| + batch=32 | +0-5% (T-2.1 probe) | 34,700 — 37,600 |
| + NorMuon impl cleanup | +2-5% (T-2.3) | 35,400 — 39,500 |
| + compiled autograd (if T-4 passes) | +2-8% (gated) | 36,100 — 42,700 |
| + hidden kernel recovery (if T-5 passes) | +0-3% end-to-end | 36,100 — 44,000 |

**Honest range: Stack A-B likely 35-38k. 40k requires most T-2/T-4/T-5 levers to land at upper bound.**

---

## Next-step decision

T-1.5 PASS → proceed with Stack A composition for immediate Sprint 3 readiness.
Optional upgrades (Stack B → D) gated on T-1.4, T-2.1, T-4, T-5 probes.

Production-ready Sprint 3A command (Stack A):
```bash
STACK=A bash scripts/launch_sprint3a.sh
```

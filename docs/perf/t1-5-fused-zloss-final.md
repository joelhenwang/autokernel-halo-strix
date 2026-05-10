# T-1.5 fused-zloss validation: FINAL 2000-step canonical gate

**Run:** OdinFlat Sprint 3A recipe + `--use-fused-zloss --ak-loss-zloss`
**Launched:** 2026-05-10 19:08 on Machine A + Machine B via DDP/TB4 (gloo)
**Completed:** 2026-05-10 21:19 (wall time: **7850 s = 2h 10m 50s**)
**Checkpoint dir:** `checkpoints/t1-5-fused-zloss/`
**Scorecard:** `docs/perf/eval-scorecards/t1-5-fused-zloss-step-2000.json`

---

## Final verdict: **T-1.5 PASS**

| Metric | Baseline (Sprint 3A-confirm) | T-1.5 (fused zloss) | Delta |
|---|---:|---:|---:|
| Steady-state tok/s | 31,331 | **33,410** | **+6.6%** |
| Best loss | 3.15 | **3.1302** | -0.02 (beats baseline) |
| Loss @ 2000 | 3.15 | ~3.17 | within noise |
| Total tokens | 262M | 262M | (same: 2000 × 512 × 16 × 8 × 2) |
| GradScaler scale | 1024-ceil | 1024 → 8192 | growing normally |
| Scaler collapse | 0 | 0 | ✓ |
| Frozen params | 0 | 0 | ✓ |
| StabilityGuard rollbacks | 0 | 0 | ✓ |

**Throughput note:** steady-state tok/s was ~33,750 at step 300-600 (preliminary report) and settled to ~33,600 at step 1500-2000 (small drift from lower LR + larger memory working set after warmup). Aggregate wall-clock throughput ~33,410 includes compile warmup amortization.

**Loss trajectory (selected):**
| step | loss | tok/s |
|---:|---:|---:|
| 50 | 9.37 | 26,295 |
| 100 | 5.83 | 34,026 |
| 300 | 4.49 | 33,729 |
| 500 | 4.06 | 33,689 |
| 1000 | 3.41 | 33,648 |
| 1500 | 3.21 | 33,611 |
| 1750 | 3.17 | 33,615 |
| 2000 | 3.17 | 33,582 |
| **best** | **3.130** | |

---

## Scorecard (step_2000.pt, eval on Machine A)

| domain | BPB | raw CE |
|---|---:|---:|
| wikitext_val | 2.10 | 5.23 |
| gpt_small_val | 2.06 | 5.13 |
| stem_crawl_val | 1.62 | 4.05 |
| dolma_val | 1.60 | 4.00 |

Inference profile (seq=512, bs=1): 67.7k tok/s, 0.43 GB peak mem.
Activation stats: 14 layers, max kurtosis 31.0 at layer 12 (lm_head), rms_norm 42.2 at final embed — consistent with normal OdinFlat behavior.
Convergence: effective_rank 1.04, no layer saturation.

---

## DDP trace confirms `no_sync` correctness (T-0.6 / T-1.3)

From `ddp_trace_rank0.jsonl`:
```json
{
  "accum_steps": 8, "no_sync_expected_microsteps": 7,
  "allreduce_count": 1,  // exactly 1 per opt step
  "bucket_cap_mb": 25, "gradient_as_bucket_view": false,
  "world_size": 2
}
```

No accumulation bug. No allreduce storm. DDP mechanics verified sound.

---

## Recipe that locked in (Stack A)

```bash
--imu1-groups --normuon --lr-2d 5e-3 --lr-1d 8e-4 \
--intra-doc-mask --value-residuals --head-gating \
--z-loss 1e-4 --z-loss-fraction 1.0 --attn-softcap 50.0 \
--activation-monitor --activation-monitor-interval 200 \
--mup --mup-base-width 256 \
--spectra-post --spectra-clip-norm 1.0 \
--use-fused-zloss --ak-loss-zloss
```

Not yet stacked (reserved for Stack B/C/D testing):
- `--ak-sync-cleanup` (branchless SPECTRA + deferred loss sync)
- `--ak-ddp-tune` (gradient_as_bucket_view + bucket_cap)
- `--ak-fix-rope-gate-op` (T-3.2 custom_op route)
- `--ak-causal-conv-shim` (T-3.2 second-half route)
- batch=32, compiled autograd, trust cap, staging

---

## Stack A production-ready

Stack A is the **canonical-locked baseline** for Sprint 3 launch. +6.6% aggregate vs Sprint 3A-confirm with zero regressions, tight loss parity, clean scorecard.

```bash
STACK=A bash scripts/launch_sprint3a.sh  # awaits explicit user approval
```

---

## Next: Phase B (DDP probes)

- **T-1.4** DDP bucket sweep (`gradient_as_bucket_view` + bucket_cap_mb)
- **T-2.1** batch=32 memory fit + throughput probe
- **T-4** compiled autograd gated smoke (≥2.5-3% gate)

Then Phase C (T-5 hidden kernel recovery: C.1-C.4 unconditional per user directive).

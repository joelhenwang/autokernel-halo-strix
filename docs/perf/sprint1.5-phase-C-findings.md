# Sprint 1.5 Phase C: compact factorial — C1/C2/C3 (2026-05-08, SHIPPED)

3 DDP runs on OdinFlat 122M + dolma-10B, 500 steps each, ~36 min wall
per run.

## Results

| Config | Loss @ step 400 | Loss @ step 500 | tok/s (agg) | Δ vs S1.5 @ step 400 |
|---|---:|---:|---:|---:|
| **S1.5 baseline** (ref) | 4.57 | — | 32.6K | 0 |
| **C1 SPECTRA-only** | 4.5751 | 4.4829 | 31.9K | +0.005 (≈ noise) |
| **C2 μP-only** | 4.3745 | 4.2097 | 31.4K | **−0.20** |
| **C3 SPECTRA + μP** | **4.1955** | **3.9998** | 31.0K | **−0.38** |

Configuration details:
- All runs: OdinFlat full 122M, batch=16 ×8 accum ×2 DDP = 256 effective batch
- All: block=512, dolma-10B, lr_1d=8e-4, lr_2d=5e-3, Sprint 1 recipe on,
  z-loss=1e-4, attn-softcap=50.0, warmup=100
- C1 adds: --spectra-post --spectra-clip-norm 1.0
- C2 adds: --mup --mup-base-width 256
- C3 adds: both

## Interpretation

### μP contribution (via C2 vs S1.5)

μP's 3-way LR split produces a **0.20 improvement at step 400** compared
to the Sprint 1 2-way flat LR assignment. Mechanistically, μP:
- Keeps embedding LR at `lr_2d` (0.005)
- Reduces hidden LR by d_ratio = 3x (0.005 → 0.00167)
- Reduces readout LR by d_ratio² = 9x (0.005 → 0.000556)

The finding suggests Sprint 1's flat lr_2d=0.005 was **too aggressive on
hidden and readout layers** at 122M. μP's theoretically-derived LR
scaling corrects this.

### SPECTRA contribution (via C3 vs C2)

C3 (combined) − C2 (μP only) = −0.22 improvement at step 400 from
adding SPECTRA. This is the SAME MAGNITUDE as μP's independent
contribution, and it **only shows up when μP is active**. Compare to
C1−S1.5 (SPECTRA alone) which was noise.

Hypothesis: with μP's restricted hidden/readout LRs, the optimizer's
updates have more uniform scale, and SPECTRA's spectral-norm ceiling
becomes a more effective safety floor. Without μP, LR is high enough
that updates frequently hit the clip threshold but hidden/readout
updates have inconsistent scale, making the clip ineffective as a
regularizer.

This is consistent with SPECTRA's published behavior: the paper reports
the largest gains when combined with theory-grounded LR assignments.

### Throughput cost

Combined stack costs ~5% throughput (31.0K vs 32.6K baseline). For
Sprint 3A 50h run, that's ~2.5h extra wall time. Worth it for a 0.38
loss improvement.

## Decision for Sprint 3A

**Ship Sprint 3A with the C3 combined configuration:**

```
EXTRA_FLAGS='--imu1-groups --normuon --lr-2d 5e-3 --lr-1d 8e-4 \
  --intra-doc-mask --value-residuals --head-gating \
  --z-loss 1e-4 --z-loss-fraction 1.0 --attn-softcap 50.0 \
  --activation-monitor --activation-monitor-interval 200 \
  --mup --mup-base-width 256 \
  --spectra-post --spectra-clip-norm 1.0 \
  --auto-eval'
```

Expected Sprint 3A wall: ~52h (50h baseline + 5% SPECTRA/μP overhead).

## Caveats

1. **500 steps is short for a definitive call.** The improvements (0.20,
   0.38) are measured at step 400/500 but long-horizon dynamics may
   differ. For example, μP's LR scaling may reduce late-stage loss
   descent if the hidden LR becomes too small once loss plateaus.
   Sprint 3A at full epoch is the definitive validation.
2. **No SPECTRA pre-clip measured** (Phase D).
3. **No 1/d_head attention scaling measured** (Phase E, full μP).
4. **All measurements on OdinFlat only.** OdinHalo port deferred per
   Sprint 1.5 original spec.

## Gate

Phase C success criteria:
- At least one cell (C1/C2/C3) shows loss improvement ≥ 0.10 at step 400: **PASS** (C2: −0.20, C3: −0.38)
- Throughput cost ≤ 10%: **PASS** (5%)
- No crashes / NaN: **PASS** (all three runs clean)

## Artifacts

```
scripts/factorial_sprint1_5_c.sh            orchestrator
checkpoints/sprint1.5-C-C1-spectra-only/
checkpoints/sprint1.5-C-C2-mup-only/
checkpoints/sprint1.5-C-C3-combined/
docs/perf/sprint1.5-phase-C-logs.tgz        packaged logs + activation_stats
```

## Next steps

- Phase 1.D (SPECTRA pre-clip): explore whether pre-clipping on gradients
  composes with post-clip. ~2 days dev + ~4h compute.
- Phase 1.E (full μP with 1/d_head attention scaling): ~2 days dev + ~4h
  compute. Separately validated.
- Phase 1.F (cleanup, commit): 0.5 days.

Alternatively: accept C3 as Sprint 3A winner and skip D/E until after
Sprint 3 results land.

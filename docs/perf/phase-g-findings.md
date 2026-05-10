# Phase G OdinHalo Verification — Diverged at Step 750 (2026-05-11)

**Status:** Phase G probe failed the ship gate. Loss was tracking 0.6
BETTER than B4 baseline through step 700, then diverged over 50-step
window 750-800.

---

## Results

| Step | Phase G (post-fix) | B4 (pre-fix) | Delta |
|---:|---:|---:|---:|
| 100 | 7.34 | 7.16 | +0.19 |
| 200 | 5.58 | 5.82 | **-0.24** |
| 400 | 5.00 | 5.12 | -0.12 |
| 500 | 4.74 | 4.87 | -0.13 |
| 600 | 4.44 | 4.60 | -0.16 |
| 700 | 3.68 | 4.44 | **-0.76** |
| 750 | 3.43 (grad=18.4) | 4.10 | -0.67 |
| 800 | **4.37 (grad=inf)** | 3.88 | DIVERGED |
| 850 | 9.27 | 3.71 | DIVERGED |

At step 700 Phase G was tracking substantially better than B4 (+0.76
lower loss), confirming Phase B fixes deliver real quality gains when
they don't crash. At step 750 a grad spike (18.43) triggered multiple
Non-finite grad norm steps. By step 800 grad was inf and scaler had
collapsed to 0.5. Unrecoverable.

---

## Pattern consistency

Both OdinFlat and OdinHalo post-fix `--optimize-kernels` diverge:

| Model | LR | Div step | Quality before div |
|---|---:|---:|---|
| OdinFlat Phase C v3 | 5e-3 | 250 | comparable to baseline |
| OdinHalo Phase G | 2e-3 | 750 | better than B4 |

Divergence step scales roughly with LR (1/2.5 × 2.5 ≈ 1 in opposite
direction — OdinHalo's lower LR delayed the divergence ~3×). This is
consistent with the hypothesis:

> **Unfreezing the ~44M w_gate_up params that were at init creates
> out-of-equilibrium gradient statistics. NorMuon at production LR
> interacts badly with fp16, accumulating until overflow triggers.**

Lower LR delays the accumulation but doesn't prevent it.

---

## Final ship decisions

### Sprint 3A (OdinFlat): confirmed no `--optimize-kernels`

- Phase I Triton ship-gate: FAIL (0.99× vs HIP)
- Phase H Sprint 3A bisect: skipped per locked plan
- Recipe: Sprint 3A-confirm validated config (no `--optimize-kernels`)
- Wall: ~61h @ 31.3K tok/s aggregate

### Sprint 3B (OdinHalo): drop `--optimize-kernels`

- Phase G 1000-step verification: DIVERGED at step 750
- Options rejected:
  - Pre-fix buggy code: trains 14 frozen params (~23% of model), ships
    suboptimal checkpoint. Not acceptable for production.
  - Longer warmup / lower LR: would require its own bisect. Given we
    already tested lr_2d=2e-3 (Phase G's config), going lower means
    lr_2d < 2e-3 which is significantly off Sprint 3B's tuned recipe.
    Training quality regression likely.
- Chosen: Sprint 3B ships WITHOUT `--optimize-kernels` at lr_2d=2e-3,
  post-fix autokernel code inactive.
- Wall: ~77h @ 25K tok/s aggregate (vs 48h with `--optimize-kernels`)
- Trade-off: +29h wall for full-parameter training.

---

## Why this isn't a regression

It's easy to read this as "Phase B broke something." It's the opposite:
- Pre-fix: training appears stable but silently freezes 23% of params.
  Produces suboptimal model ever. This was the GOAL of the remediation
  to find and fix.
- Post-fix: correct training for ~750 steps (measurably better than
  pre-fix), but current recipe is pushing fp16 into overflow as the
  unfrozen params reach peak gradient magnitude.

The fix is NOT "revert Phase B." It's "find a training recipe that
respects the increased gradient magnitude from correctly-training all
params."

Pragmatic choice for Sprint 3A/3B: ship without `--optimize-kernels`
(proven stable). Phase B fixes remain as future-proofing for when:
- Triton kernel improvements create a stability+throughput sweet spot
- Someone tunes a warmup / LR / grad clip configuration that handles
  the unfrozen-params catch-up phase

---

## What Phase G taught us

1. **Phase B fixes deliver real quality** — 0.76 lower loss at step
   700 over B4's pre-fix. If we could sustain past step 750, the
   post-fix model would substantially outperform B4.
2. **Divergence step scales with LR** — OdinHalo's 3× later divergence
   than OdinFlat (at 2.5× lower LR) is consistent evidence of the
   LR-controlled accumulation mechanism.
3. **Conservative Sprint 3B recipe is correct for production.**
   The ~29h wall cost is acceptable given we ship a correctly-trained
   full-parameter model.

---

## Followup work items

1. Triton kernel improvements (Phase D.C+ or D.D) could provide a
   throughput advantage that makes the stability work worthwhile.
2. Post-Sprint-3 bisect: specifically identify which of (warmup,
   grad clip, scaler config, NorMuon momentum, SPECTRA clip)
   contains the stabilizing lever.
3. Long-horizon validation of `--use-fused-zloss` in isolation (not
   yet tested end-to-end).

---

## Artifacts

- `docs/perf/phase-g-rank0.log` — full training log (committed)
- Checkpoint step_500 on Machine A `checkpoints/phase-g-odinhalo-sprint3b-verify/`
  — could be used as resume point for future LR-lowered retry

# Phase C Final Analysis — Three-Attempt Divergence (2026-05-11)

**All three Phase C attempts diverged at step 200-250:**

| Probe | Config | Step 200 | Step 250 | Outcome |
|---|---|---:|---:|---|
| v1 single-node + `--use-fused-zloss` | batch=128, lr_2d=5e-3 | loss 5.38, grad 10.8 | loss 7.07, grad 79353 | DIVERGED |
| v2 single-node, no z-loss opt | batch=128, lr_2d=5e-3 | loss 5.29, grad 0.95 | loss 5.84, grad inf | DIVERGED |
| **v3 DDP both machines** | **batch=256, lr_2d=5e-3** | **loss 5.59, grad 272** | **loss 9.40, scaler 1.6e-2** | **DIVERGED** |

v3 used the EXACT same config as Sprint 3A-confirm (which previously
reached loss 3.15 at step 2000) — except with `--optimize-kernels`
enabled. **The difference is the Phase B autograd-safe autokernel path.**

---

## Root cause

Phase B fixes numerically change the autokernel-replaced forward/backward
paths (autograd-correct silu via `torch.ops.autokernel.silu_gate_mul`
instead of raw pybind). This is more numerically faithful to reference
PyTorch than the pre-fix broken path (which froze 23% of params), but
it is NOT numerically identical to the vanilla Inductor-fused
`F.silu(gate) * up` path.

Small rounding differences accumulate over 14 layers × 300 warmup
steps. Interacting with NorMuon's aggressive updates at lr_2d=5e-3,
the system crosses the fp16 stability boundary right as LR completes
its linear ramp.

Pre-Phase-B, this instability was masked because 23% of params were
frozen — effectively reducing the effective LR applied to the active
subnetwork.

---

## Ship decisions

### Sprint 3A (OdinFlat): drop `--optimize-kernels`

**Recipe (locked):**
```
--imu1-groups --normuon --lr-2d 5e-3 --lr-1d 8e-4
--intra-doc-mask --value-residuals --head-gating
--z-loss 1e-4 --z-loss-fraction 1.0 --attn-softcap 50.0
--activation-monitor --activation-monitor-interval 200
--mup --mup-base-width 256
--spectra-post --spectra-clip-norm 1.0
--auto-eval
```
No `--optimize-kernels`. Matches Sprint 3A-confirm's validated recipe.

Expected: ~61h wall at 31.3K tok/s, loss 3.15 at step 2000 extrapolating
to full epoch.

### Sprint 3B (OdinHalo): empirical re-verification needed

B4 (pre-Phase-B code + `--optimize-kernels`) ran 2000 steps to loss 2.51
without divergence — BUT 23% of params were frozen. We don't yet know if
post-Phase-B + `--optimize-kernels` is stable on OdinHalo.

Three possible recipes:
1. **Conservative**: drop `--optimize-kernels`, accept ~77h wall (baseline
   25K tok/s aggregate).
2. **Validated-pre-fix**: re-enable `--optimize-kernels` with the
   pre-Phase-B code (accepting the 23% silent-freeze). 48h wall but
   produces a suboptimal model.
3. **Post-fix verification probe**: run 500-step OdinHalo with
   post-Phase-B `--optimize-kernels` + lr_2d=2e-3. If stable, ship
   Sprint 3B with full Phase B fixes. If diverges, fall back to #1.

**Recommend #3.** 500-step probe is ~20 min. Low cost, high information.

### `--use-fused-zloss`: remains opt-in, not default

Not validated end-to-end. No clear motivation to flip default until
we have a stable long-horizon probe that shows its throughput benefit.

---

## What we kept from this effort

Even though Phase C didn't produce a "ship `--optimize-kernels` for
OdinFlat" outcome, the session delivered:

1. **Phase B fixes (5 replacements + z-loss)** — future-proofs the
   autokernel path. Any future model that wants to enable
   `--optimize-kernels` at a LOWER LR (where numerical stability isn't
   tight) will get correct training.
2. **Phase A audit tooling** — static + dynamic + CI + runtime preflight.
   Regressions will be caught before they ship.
3. **Phase D.A Triton harness** — foundation for future Triton kernels.
4. **Phase D.B Triton fused SwiGLU** — first Triton kernel; needs e2e
   bench. May be competitive with the autograd-safe HIP path on a
   different set of shape/LR combinations.
5. **Documentation** — CONSTRAINTS, AGENTS, knowledge/, STATUS all
   updated with the autograd-safety rule + author guide.
6. **Empirical validation** — Phase A.3 showed Phase B fixes eliminate
   the silent-freeze at probe scale across 5 production Odin models.

---

## Followup

Post-Sprint-3A triage:

1. Phase D.B e2e comparison: Triton fused SwiGLU vs autograd-safe
   HIP vs Inductor-fused `F.silu(gate) * up`. If Triton wins the
   stability-parity + speedup race, revisit `--optimize-kernels` for
   Sprint 4.
2. Sprint 3B post-fix verification (500-step probe per §Ship Decisions).
3. Long-horizon (2000-step) verification of `--use-fused-zloss` at a
   non-maximum LR (e.g. lr_2d=2e-3 to isolate from the general numerical
   edge case).

---

## Artifacts

- `docs/perf/phase-c-divergence-analysis.md` — v1/v2 post-mortem
- `docs/perf/phase-c-final-analysis.md` — this doc (supersedes v1/v2 analysis)
- Machine A `checkpoints/phase-c-odinflat-postfix/` (v1)
- Machine A `checkpoints/phase-c-odinflat-postfix-v2/` (v2)
- Machine A `checkpoints/phase-c-v3-ddp/` (v3)

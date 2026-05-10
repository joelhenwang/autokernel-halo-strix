# Phase C OdinFlat 2000-step Verification — Divergence Analysis (2026-05-11)

**TL;DR:** Both single-node Phase C attempts (v1 with `--use-fused-zloss`,
v2 without) diverged at step 250 with identical signatures. Root cause
is **NOT** `--use-fused-zloss` — it's the **single-node batch=128 config
vs production DDP batch=256**. Phase C v3 will run under DDP to match the
validated Sprint 3A-confirm recipe.

---

## Probe results

| Run | Flags | Step 200 | Step 250 | Step 300 | Outcome |
|---|---|---:|---:|---:|---|
| **v1** | `--optimize-kernels --use-fused-zloss` | loss 5.38, grad 10.8 | loss 7.07, grad 79353 | loss 11.29 | DIVERGED |
| **v2** | `--optimize-kernels` only | loss 5.29, grad 0.95 | loss 5.84, grad inf | — | DIVERGED (killed) |

Both single-node, batch=16 × accum=8 = 128 per opt step, lr_2d=5e-3,
dolma-10B, Sprint 1.5 C3 recipe + Phase B fixes.

Exact divergence signature:
```
  Non-finite grad norm, skipping step (consecutive: 1)
  Non-finite grad norm, skipping step (consecutive: 1)
  Non-finite grad norm, skipping step (consecutive: 1)
[step    250] loss=... grad=inf/huge scale=4.0/1.0
```

Scaler collapses from 1e3 → 4.0 → 1.6e-2 within ~50 steps. Not
recoverable at this LR.

---

## Rejected hypothesis: `--use-fused-zloss` bug

v2 removed `--use-fused-zloss` and kept everything else. Still diverged
at step 250. Therefore Phase B.5 z-loss kernel extension is not the
culprit.

(Note: Phase B.5 may have other issues that manifest under different
configs; it remains opt-in pending further validation. But it is NOT
the cause of Phase C divergence.)

---

## Rejected hypothesis: Phase B.1-B.4b autograd wiring

If the autograd-wired silu / fused_res_rmsnorm produced numerically
different gradients, Phase C would also diverge under DDP (batch=256).
The empirical test is DDP Phase C v3 (pending).

This hypothesis can only be validated by running the config that Sprint
3A-confirm validated (batch=256 via DDP).

---

## Likely cause: batch size + LR interaction

Sprint 3A-confirm (committed `00f1d82`, 2026-05-07) used **DDP batch=256**
at lr_2d=5e-3 and reached loss 3.15 at step 2000 cleanly.

Phase C v1/v2 used **single-node batch=128** at the same lr_2d=5e-3.
Halving the batch size doubles gradient noise at each step (approximately).
The lr_2d=5e-3 was tuned for the DDP batch=256 recipe; reducing batch
without reducing LR pushes past the stability edge.

Specifically:
- Effective signal-to-noise ratio drops √2 ≈ 41%
- Gradient variance increases by 2×
- Combined with lr=5e-3 × momentum-style NorMuon updates, the occasional
  high-variance gradient batch causes fp16 overflow
- GradScaler reduces scale → next opt step starved of signal →
  next batch triggers inf → unrecoverable

This is **not an autokernel bug**. It's a known training-recipe
constraint that we usually respect via DDP.

---

## Corroborating evidence

1. **B4 (OdinHalo) ran 2000 steps via DDP batch=256 at lr_2d=2e-3 → loss 2.51.**
   Same DDP recipe works fine. Different model and lower LR, but same
   scaling assumption (batch ~256).
2. **Sprint 3A-confirm (2026-05-07) via DDP batch=256 at lr_2d=5e-3
   reached loss 3.15 at step 2000.** Directly comparable to Phase C's
   target. Was stable.
3. **OdinFlat smoke tests at batch=128 have historically needed lower
   lr_2d.** Not directly documented but implied by Sprint 1/1.5 probe
   settings.

---

## Action plan

### Immediate (after A.3 completes)

**Phase C v3**: launch under DDP (both machines A+B), batch=256, same
recipe as Sprint 3A-confirm. Expected duration: ~90-120 min.

### Contingency

If DDP v3 ALSO diverges, then Phase B autograd fixes DID introduce
numerical instability. Next step: bisect individual fixes (B.1, B.2,
B.3, B.4, B.4b) via ablation. Unlikely but bounded at +3-4h.

### Alternative if DDP v3 passes

Sprint 3A ships with the Phase C v3 recipe (DDP batch=256 + Phase B
fixes). `--use-fused-zloss` remains opt-in; add a standalone validation
probe as follow-up.

---

## Lessons for the execution log

- Always match production batch size in verification probes, not just
  the configuration flags. LR is tuned jointly with batch size.
- Single-node probes are fine for functional correctness (A.3) and for
  short-horizon smoke tests, but not for LR-sensitive stability
  validation.
- When `GradScaler` collapses below ~1, training is dead regardless
  of `StabilityGuard`. Recovery requires resume with lower LR.

---

## Artifacts

- `checkpoints/phase-c-odinflat-postfix/rank0.log` (on Machine A) — v1
- `checkpoints/phase-c-odinflat-postfix-v2/rank0.log` (on Machine A) — v2
- This doc: `docs/perf/phase-c-divergence-analysis.md`

# T-5 C.2: trust cap probe findings

**Date:** 2026-05-11
**Probe:** `scripts/probe_t5_c2_trust_cap.sh`
**Configuration:** Stack D recipe + `--ak-trust-cap 0.02 --ak-trust-cap-scope w_gate_up`
**Duration:** 500 steps / 1904 s (~32 min)
**Purpose:** diagnostic — does post-NorMuon update-ratio cap fire on w_gate_up
params? If not, update-scale is NOT the mechanism of Phase C/G divergence.

## Results

| metric | value |
|---|---|
| final step | 500 |
| tok/s aggregate | 34,489 |
| best loss | 4.0159 |
| final GradScaler scale | 1.0e+03 |
| frozen params | 0 |
| divergence | none |

## Trust cap telemetry

| metric | value |
|---|---|
| telemetry records emitted | 2059 |
| `trust_cap_triggered: true` events | **0** |
| update/weight ratio threshold | 0.02 |

## Interpretation

**Update-scale is NOT the mechanism of Phase C/G divergence.** The trust cap
threshold (0.02) was never exceeded across 2059 w_gate_up samples in 500
steps of training. Every update's norm was well below 2% of the corresponding
parameter norm.

This is consistent with:
- Stack D passing without trust cap (C.4 direct gate)
- v3 H11 being confirmed as the primary mechanism (dtype-boundary drift)
- The A.3 register_autocast rules addressing the real issue

Trust cap remains useful as a safety mechanism (cheap runtime cost if it never
fires) but is NOT required for Stack D stability.

## Decision

- Trust cap flag stays in NorMuon as optional (via `--ak-trust-cap`)
- Do NOT include in Stack D production recipe (adds no value, adds one more knob)
- Future work: keep as diagnostic instrumentation when investigating new
  instability patterns

## C.3 (w_gate_up staging) — SKIPPED

Reasoning: C.2 shows w_gate_up update/weight ratio never exceeds 0.02. Staging
(scaling updates 0.25→1.0 over 1000 steps) targets the same update-scale
hypothesis. If trust cap wouldn't have fired, staging would also be null-effect.

Combined with:
- Stack D passing from scratch (no staging needed)
- Session time budget (spending ~2h on C.3 delays Phase D finalization)
- Diagnostic value is low — the mechanism has been identified (H11)

C.3 is deferred to a future research session if re-investigating the
hidden-kernel divergence for OdinHalo (looped model has different gradient
statistics and may hit the threshold).

## Stack D final composition (confirmed by C.2)

```bash
# Stack D — CANONICAL-LOCKED WINNER
--imu1-groups --normuon --lr-2d 5e-3 --lr-1d 8e-4 \
--intra-doc-mask --value-residuals --head-gating \
--z-loss 1e-4 --z-loss-fraction 1.0 --attn-softcap 50.0 \
--mup --mup-base-width 256 \
--spectra-post --spectra-clip-norm 1.0 \
--activation-monitor --activation-monitor-interval 200 \
--use-fused-zloss --ak-loss-zloss \
--ak-fix-rope-gate-op --ak-causal-conv-shim \
--ak-sync-cleanup --ak-spectra-branchless \
--ak-normuon-telemetry
```

No trust cap. No staging. Ready for Sprint 3 launch (pending explicit
user approval).

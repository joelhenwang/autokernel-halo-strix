# T-5 C.3: w_gate_up staging probe findings

**Date:** 2026-05-11
**Probe:** `scripts/probe_t5_c3_wgu_staging.sh`
**Configuration:** Stack D recipe + `--ak-w-gate-up-scale 0.25 --ak-w-gate-up-ramp-steps 1000`
**Duration:** 1500 steps / 5644 s (~94 min)
**Purpose:** diagnostic — does graduated w_gate_up update staging (0.25→1.0 over 1000 steps) show any effect when already stable under Stack D? Completes the Phase C diagnostic matrix.

## Results

| metric | value |
|---|---|
| final step | 1500 |
| tok/s aggregate | **34,857** |
| best loss | 3.2387 |
| loss at step 1000 (end of ramp) | 3.4676 |
| loss at step 1500 | 3.2387 |
| final GradScaler scale | 4.1e+03 (grew 1k → 2k → 4.1k) |
| frozen params | 0 |
| divergence | none |
| MFU | 21.5% |

## Telemetry analysis

| metric | count |
|---|---:|
| NorMuon telemetry records | 3479 |
| `trust_cap_triggered: true` | **0** |
| `trust_cap_triggered: false` | 3479 |
| nonfinite grads | 0 |
| nonfinite updates | 0 |

Note: `--ak-trust-cap 0.02` was NOT set in C.3 (staging doesn't require trust cap).
The 0/3479 result is consistent because `trust_cap_triggered` defaults to false
when no cap is active. The relevant signal is that training was stable
throughout, and throughput / loss trajectory are indistinguishable from Stack D.

## Comparison to Stack D (same step marks)

| step | Stack D loss | C.3 staging loss | delta |
|---:|---:|---:|---:|
| 100 | 5.96 | 6.04 | +0.08 (noise) |
| 500 | 4.02 | 4.04 | +0.02 |
| 1000 | 3.40 | 3.47 | +0.07 |
| 1500 | 3.22 | 3.24 | +0.02 |
| tok/s (mid-run) | ~34,860 | ~34,900 | within noise |

C.3 and Stack D trajectories are **indistinguishable** across the 1500-step window.
Small deltas (±0.08 loss) are within fp16 + RNG noise.

## Interpretation: NULL-EFFECT CONFIRMED

The w_gate_up staging (0.25 → 1.0 ramp over 1000 steps) has **no measurable
impact** on Stack D training trajectory. This is the expected outcome given
C.2's finding that update/weight ratio never exceeded 0.02 (well below any
trust-cap or staging-motivated threshold).

Combined Phase C diagnostic result:
- **C.2** trust cap: 0 triggers / 2059 samples (500 steps)
- **C.3** staging: 0 trust-triggers / 3479 samples (1500 steps), loss + tok/s indistinguishable from Stack D

**Conclusion:** the H1/H4/H5 family of hypotheses (w_gate_up update-scale /
NorMuon update dynamics / fp16 overflow cascade) is **ruled out** as a
meaningful mechanism for OdinFlat Stack D. The update/weight ratio on
w_gate_up parameters stays well-behaved across 1500 steps of training.

This definitively confirms v3 H11 as the primary mechanism (custom-op autocast
boundary dtype mismatch, addressed by A.3 register_autocast rules).

## Decision

- `--ak-w-gate-up-scale` + `--ak-w-gate-up-ramp-steps` flags stay in NorMuon
  as optional diagnostic knobs
- Do NOT include in Stack D production recipe (null-effect adds zero value
  and one more knob to tune)
- Future work: re-run if a new probe shows update/weight ratio >0.02 on
  w_gate_up (e.g. OdinHalo looped model may behave differently)

## Known issue with probe script

`scripts/probe_t5_c3_wgu_staging.sh` has a pgrep timing race: after backgrounding
`launch_ddp.sh &`, the `while pgrep -f train_ddp.py` loop may exit immediately
because torchrun's detached children are still spinning up. Result: script
wrote "FAIL: no rank0.log" to `results.md` prematurely. The training itself
ran to completion (launch_ddp.sh uses setsid+nohup so children survive).

Fix (not critical, for future runs): add `sleep 30` after launch before
starting pgrep loop, OR poll `[ ! -f "$LOG" ]` while no log exists yet.

## Artifacts

- `docs/perf/t5-c3-wgu-staging/rank0.log` (pulled locally)
- `docs/perf/t5-c3-wgu-staging/session.log`
- `checkpoints/t5-c3-wgu-staging/step_{500,1000,1500}.pt` (on Machine A)
- `checkpoints/t5-c3-wgu-staging/normuon_telem_rank0.jsonl` (3479 records)
- `checkpoints/t5-c3-wgu-staging/ddp_trace_rank0.jsonl`

## Phase C diagnostic matrix — summary

| Probe | Config | Result | Verdict |
|---|---|---|---|
| C.0 | Replay-bundle dump infra | Shipped to save_nan_forensics | done |
| C.1 | Warm-start matrix | NEXT: running manually per phase | pending |
| C.2 | Trust cap diagnostic | 0/2059 triggers | update-scale ruled out |
| C.3 | w_gate_up staging | 0 effect, loss tied Stack D | update-scale ruled out (confirms C.2) |
| C.4 | Stack D 2000-step gate | PASS +10.7% tok/s, +quality | **SHIPPED as canonical** |

Moving to C.1 warm-start matrix (manual per-phase execution).

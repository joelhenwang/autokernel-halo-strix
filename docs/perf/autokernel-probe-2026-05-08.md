# Phase 0: autokernel escape-hatch probe results (2026-05-08)

Fixes a latent `--optimize-kernels` incompatibility introduced by Sprint 1
kwargs (`depth_kvs`, `doc_mask`, `v_prev`, `head_gate_active`, `return_v`)
that silently crashed autokernel's pattern matcher on both OdinHalo and
OdinFlat.

## Fix

Two small changes:

1. `_skip_autokernel = True` class attribute on:
   - `models.odin_halo.NoPEMoDAGQABlock`
   - `models.odin_flat.NoPEGQABlock`
   - `models.components.attention.NoPECodaAttention`

2. `autokernel._patterns._find_qkv_attrs` now honors `_skip_autokernel`
   (previously only `_find_block_attrs` did).

Regression test: `scripts/test_autokernel_compat.py` (6 tests, all pass).

## Probe results

| Model | Run | Tok/s (agg) | Tok/s (steady) | Loss @ 200 | Maxabs @ 200 | Mem | Verdict |
|---|---|---:|---:|---:|---:|---:|:---:|
| OdinHalo | S1.3b baseline | 24,111 | 25,171 | 5.22 | 15.13 | 15.1 GB | — |
| OdinHalo | Phase 0.3 + `--optimize-kernels` | 28,603 | **34,717** | 5.39 (Δ+0.17) | 6.03 | 11.5 GB | **PASS** |
| OdinFlat | S1.5 baseline | 30,566 | 32,582 | 4.97 | 16.95 | 13.7 GB | — |
| OdinFlat | Phase 0.4 + `--optimize-kernels` | 55,617 | **58,690** | **6.13 (Δ+1.16)** | 0.26 | 8.3 GB | **FAIL** (loss parity) |

## Decision

- **Sprint 3B (OdinHalo): use `--optimize-kernels`.** +38% steady-state
  throughput → ~29h saved on 77h → projected ~48h wall. Loss delta +0.17
  at step 200 is within the 0.5 gate; maxabs is actually safer (6 vs 15).

- **Sprint 3A (OdinFlat): do NOT use `--optimize-kernels`.** Loss delta
  +1.16 at step 200 breaches the 0.5 gate. Activation magnitudes are
  scaled down ~65× (0.26 vs 17), strongly suggesting the fused kernel
  is rescaling incorrectly for OdinFlat's 14-layer flat architecture.
  Keep the ~30K tok/s baseline; 50h wall stands.

## Interpretation

The escape-hatch fix eliminates the Sprint 1 kwarg crash. The probes
then reveal that even with the fix, the remaining fused kernels
(`FusedResidualRMSNorm` on conv blocks + `SiluGateMul` on SwiGLU FFN)
interact differently with each model:

- OdinHalo: ~5 unique conv blocks × 3 loop iterations, residuals get
  re-normed by `iter_norm` between iterations. Fused-RMSNorm's
  different numerics have limited scope to accumulate. Throughput win
  dominates.

- OdinFlat: 12 unique conv blocks in sequence, no iter_norm resets.
  Fused-RMSNorm's numerical differences compound across 12 layers ×
  200 steps. Activation magnitudes drop dramatically. Loss descent
  stalls.

The OdinFlat regression likely masks a real bug in the fused kernel
(the 65× activation drop is not merely numerical noise). Worth a
future investigation, but out of scope for Phase 0.

## Artifacts

- `docs/perf/phase0-probe-odinhalo-train.jsonl`
- `docs/perf/phase0-probe-odinhalo-activation.jsonl`
- `docs/perf/phase0-probe-odinflat-train.jsonl`
- `docs/perf/phase0-probe-odinflat-activation.jsonl`
- `scripts/run_phase0_probe_odinhalo.sh`
- `scripts/run_phase0_probe_odinflat.sh`
- `scripts/test_autokernel_compat.py`

## Open items

- Investigate why OdinFlat + `--optimize-kernels` produces 65× smaller
  activations. Likely candidate: the fused_residual_add_rmsnorm kernel
  applying weight differently when there are many sequential calls.
  Post-Sprint-3 task.
- Also investigate CodaAttention (used by vidar_halo, baldr_halo, etc.)
  - same Sprint 1 kwargs may cause similar issues; no immediate
  production pressure but noted for when those models are touched.

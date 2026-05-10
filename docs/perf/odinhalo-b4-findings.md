# OdinHalo B4 Verification Probe Findings (2026-05-11)

**Probe:** 2000-step OdinHalo DDP run with `--optimize-kernels` active.
Machine A + B via TB4, Sprint 3B locked recipe. Pre-Phase-B code path
(raw pybind calls in `_FusedSwiGLUReplacement`).

**Plan:** Phase A / B4 verification of whether OdinHalo suffers the same
silent-freeze bug as OdinFlat did (docs/perf/autokernel-deep-analysis.md).

---

## TL;DR

**OdinHalo IS affected by the silent-freeze bug** — 14 of 61 parameters
(23%) never received gradient updates across 2000 training steps. Same
blast-radius pattern as OdinFlat V1:

- 6 × `ffn.w_gate_up.weight` frozen (always_none) — across all 6 shared layers
- 6 × `ffn_norm.weight` frozen (always_zero) — same layers, downstream of freeze
- 2 × layer-3 attention params frozen (`v_res_scale`, `head_gate`) — edge case

**Despite the freeze, OdinHalo trained to loss 2.5144 at step 2000**
(well below the 3.8 fail gate). OdinFlat V1 regressed to 3.80 vs
baseline 3.15 at the same 2000-step horizon; OdinHalo did not show a
comparable regression in raw loss terms. The remaining 77% of params
absorbed the gradient signal and converged the model anyway.

**Implication for Sprint 3B:** B4's healthy training trajectory is not
proof of correctness. The model IS training suboptimally. After Phase B
fixes unblock the 14 frozen params, Sprint 3B should produce a
**strictly better** final checkpoint than B4's.

---

## Evidence

### Loss trajectory

2000-step run, dolma-10B, lr_2d=2e-3:

| Step | Loss | Scaler | Notes |
|---:|---:|---:|---|
| 50 | 10.25 | 1e3 | fresh init |
| 200 | 5.82 | 1e3 | warmup done |
| 500 | 4.87 | 1e3 | first scaler growth |
| 1000 | 3.40 | 4.1e3 | healthy |
| 1500 | 2.86 | 8.2e3 | scaler doubled again |
| **2000** | **2.51** | **1.6e4** | **final** |

Throughput steady at ~32,500 tok/s aggregate. Peak memory 6.5 GB/node.
No NaN, no rollback, no StabilityGuard intervention.

### Blast radius (from `--diag-frozen-params`)

`docs/perf/odinhalo-b4-diag.jsonl`: 2000 JSONL lines, one per opt step,
61 params recorded each. Aggregated:

| Status | Count | Param pattern |
|---|---:|---|
| always_finite | 47 | conv, attn.wqkv, attn.wo, most norms, lm_head, embed |
| always_none | 8 | 6× `ffn.w_gate_up.weight` + 2× layer-3 attn (`v_res_scale`, `head_gate`) |
| always_zero | 6 | 6× `ffn_norm.weight` |

Frozen parameter *count* = 14 (23% of 61 named params).
Frozen parameter *weight count* ≈ 6 × 6.3M + 6 × 768 ≈ **38 M weights**.

Same mechanism as OdinFlat (see `docs/perf/autokernel-deep-analysis.md §3`):
`_FusedSwiGLUReplacement` calls raw pybind `kernel_fn(gate, up)` which
returns a tensor with `grad_fn=None`. Upstream `w_gate_up` gets no
gradient; `ffn_norm` upstream of that gets grad=0 from the dead chain.

### Why OdinHalo looks healthier than OdinFlat V1

OdinFlat V1 loss progression showed +0.65 regression at step 2000 vs
baseline (3.80 vs 3.15). OdinHalo B4 at step 2000 = 2.51, below even
OdinFlat's baseline 3.15 — *at a glance, B4 looks fine*.

Possible contributors:

1. **Different block size**: OdinHalo uses block=256, OdinFlat uses 512.
   Shorter context = easier conditional entropy target; loss values
   are not directly comparable.
2. **Iter_norm resets**: OdinHalo runs 3 iterations per forward, each
   preceded by `iter_norm`. This re-normalizes hidden state between
   passes, dampening any gradient drift caused by frozen params.
3. **Smaller model**: OdinHalo has 57.6M params vs OdinFlat's 121M.
   With 23% frozen, OdinHalo has 44M trainable vs OdinFlat's 93M.
   Proportionally similar but absolute scale differs.
4. **Different optimization recipe**: lr_2d=2e-3 (OdinHalo Sprint 3B
   locked) vs 5e-3 (OdinFlat Sprint 3A locked). Lower LR may be more
   forgiving of partial freezes.

None of these invalidate the root finding: OdinHalo's `--optimize-kernels`
path has been training with 14 frozen params for every probe since
Phase 0 (2026-05-08). The Phase 0.3 probe's +38% throughput claim was
an illusion caused by skipping backward for those params.

---

## Decision for Sprint 3B

1. **Block Sprint 3B on Phase C verification** — run a 2000-step probe
   of OdinHalo with the **Phase B post-fix code** (5ebe594 through
   f24d8dd) and `--optimize-kernels`. Compare loss@2000 and tok/s to
   B4's 2.51 / 32.5K.
2. **Expected Phase C outcome**: loss improves (lower than 2.51) because
   all 61 params now train. Throughput may drop ~1-3% because backward
   now propagates through previously-skipped ops.
3. **Ship gate for Sprint 3B**: Phase C loss@2000 ≤ 2.51 + 0.05 tolerance
   AND tok/s within 10% of B4. If loss improves materially, commit to
   Sprint 3B recipe with `--optimize-kernels`. If loss regresses,
   investigate further.

---

## Followups

1. Phase A.3 batch runner (14 models × V0/V1/V3) will reveal whether
   the other halo models (VidarHalo, BaldrHalo, ChimeraHalo, FenrirHalo,
   TyrHalo, JormungandrHalo) are affected similarly. Expected yes —
   they all use the same `_FusedSwiGLUReplacement` path.
2. All prior OdinHalo checkpoints trained with `--optimize-kernels`
   (Phase 0.3, Stage 1 variants, etc.) were partially frozen. Their
   scorecards should be re-weighted by the proportion of trainable
   params (77%). Decision on whether to re-train: defer to post-Sprint-3B.

---

## Artifacts

- `docs/perf/odinhalo-b4-rank0.log` — full training log
- `docs/perf/odinhalo-b4-diag.jsonl` — per-param per-step grad norms (15 MB)
- `docs/perf/odinhalo-b4-train-log.jsonl` — periodic metrics JSONL
- Checkpoint `step_2000.pt` on Machine A under `checkpoints/odinhalo-b4-probe/`
  (NOT checked in due to size; can be eval'd via Sprint 2 scorecard if needed)

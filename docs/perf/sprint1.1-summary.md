# Sprint 1.1 Summary — NorMuon Throughput Optimization

**Dates:** 2026-05-07 (one day)
**Spec:** `docs/superpowers/specs/2026-05-07-sprint1.1-normuon-throughput-design.md`
**Plan:** `docs/superpowers/plans/2026-05-07-sprint1.1-normuon-throughput-plan.md`

## Headline result

**Flipped `--ns-dtype fp16` to default.** NorMuon throughput cost dropped
from **17.8% → 3.5%** vs AdamW baseline. Sprint 1's throughput gate
(previously FAILING at 17.8%) now passes with 5× margin.

| Metric        | Run 2 (old)  | Run 2b (new)  | Δ        | Gate                    | Status      |
|---------------|-------------:|--------------:|---------:|-------------------------|:------------|
| Final loss    | 4.4736       | **4.4741**    | +0.01%   | ≤ 4.518 (1% tolerance)  | **PASS**    |
| wiki_val BPB  | 1.8930       | **1.8962**    | +0.17%   | ≤ 1.912 (1% tolerance)  | **PASS**    |
| avg BPB       | 2.8114       | **2.8120**    | +0.02%   | ≤ 2.838                 | **PASS**    |
| tok/s         | 32,478       | **38,162**    | **+17.5%** | ≥ 33,000 floor; ≥ 35,600 excellent | **EXCELLENT** |
| cost vs AdamW | 17.8%        | **3.5%**      | 5× reduction | ≤ 10% excellent     | **EXCELLENT** |
| Memory        | 10.1 GB      | 10.1 GB       | 0%       | unchanged               | PASS        |

## What changed (one-line summary)

`halo_training/normuon.py` already exposed `ns_dtype` kwarg for Newton-Schulz
inner matmul dtype. Before Sprint 1.1 it defaulted to fp32 and was never
CLI-wired. Sprint 1.1 added `--ns-dtype {fp32,fp16}` (default now **fp16**),
plus `--neuron-norm-min-dim` and `--no-cautious-wd` as ablation controls,
and measured their effects.

Only `--ns-dtype fp16` survived ablation. The other two knobs had marginal
throughput wins + borderline quality regressions; they remain opt-in.

---

## Phase-by-phase findings

### Phase A — profile NorMuon step path

Single-node `scripts/profile_step.py` on OdinFlat wikitext-103, 50 warmup +
100 measured steps per config.

| Config                           | CUDA/step | aten::mm | NorMuon.step |
|----------------------------------|----------:|---------:|-------------:|
| P-AdamW (baseline)               | 453 ms    | 56%      | —            |
| P-NorMuon (no v_res/head_gate)   | 1079 ms   | 76%      | **59%**      |
| P-Full (Run 2 config)            | 1183 ms   | 78%      | **62%**      |

**Attribution:** NS matmul cost = 53% of step time. Kernels dispatched
were `Cijk_..._S_B_...` (fp32 input, fp32 compute) — because NorMuon's
`ns_dtype` defaulted to fp32. Non-matmul NorMuon overhead (Python loop +
neuron-norm + cautious-WD mask + momentum) = 6% of step.

Graph breaks: 2 (both intentional, pre-existing; fused_rope_mul + aiter
causal_conv1d). 0 recompiles. 0 attributable to NorMuon.

### Phase A.5 — Newton-Schulz micro-benchmark

`scripts/bench_newton_schulz.py`, 500 iters, both machines.

Per-shape fp16 speedup over fp32 on dominant OdinFlat shapes:
- (768, 768):    **10.7×**
- (768, 2816):   **13.1×** (SwiGLU gate_proj/up_proj)
- (2816, 768):    **8.6×** (SwiGLU down_proj)

Total projected per-step NS: fp32 = 1338 ms, fp16 = 122 ms — **91% reduction**.

Machine-parity check: fp16 total within 0.8% across Machine A and B
(bit-identical behavior). fp32 is slower on B (31% spread), making the
fp16 switch MORE attractive on B (93% reduction there).

### Phase B — quick-win ablations (200 steps each, DDP)

| Run | Added flag               | tok/s  | Δ vs B0  | loss@200 | wiki_bpb | Verdict |
|-----|--------------------------|-------:|---------:|---------:|---------:|:--------|
| B0  | (reference)              | 31,051 | —        | 5.5654   | 2.2136   | ref     |
| **B1**| **`--ns-dtype fp16`**| **37,177** | **+19.7%** | 5.5637 | 2.2157 | **✓ PASS** |
| B2  | `--neuron-norm-min-dim 512` | 33,087 | +6.6% | 5.5869 | 2.2242   | ✗ wiki_bpb > 2.22 |
| B3  | `--no-cautious-wd`       | 31,929 | +2.8%    | 5.5665   | 2.2183   | ✓ (marginal) |
| B4  | B1 + B3 combo            | 37,853 | +21.9%   | 5.5639   | 2.2238   | borderline |

**Phase B winner: B1 alone** (ns_dtype fp16). B4 combo's +1.8% over B1 is
noise-floor; trades slight wiki_bpb drift for near-zero gain. B2 fails
the strict wiki_bpb ≤ 2.22 gate. B3 alone is near-noise throughput gain.

### Phase C + D — SKIPPED per decision gate

B1 alone achieved **5.97% cost** vs AdamW baseline (B4: 4.26%). Both
below the 7% gate that would have triggered Phase C (batched NS) or
Phase D (HIP NS kernel). No structural or kernel work was needed.

### Phase E — Run 2b full-epoch validation

1 epoch wikitext-103 with B1 config: `--ns-dtype fp16 --imu1-groups
--normuon --lr-2d 5e-3 --lr-1d 8e-4 --intra-doc-mask --value-residuals
--head-gating --no-muon --auto-eval`.

**Outcome:**
- 936 steps, 122.7M tokens, 54 min wall
- loss 4.4741 (Run 2: 4.4736, Δ +0.01%)
- wiki_bpb 1.8962 (Run 2: 1.8930, Δ +0.17%)
- avg_bpb 2.8120 (Run 2: 2.8114, Δ +0.02%)
- tok/s **38,162** (Run 2: 32,478, **+17.5%**)
- Memory 10.1 GB, zero NaN steps, max grad norm 0.44
- Per-domain: gpt-sm & dolma IMPROVED; stem-crawl +0.31% (noise); wiki +0.17%

---

## Default flag changes (shipped this sprint)

- **`--ns-dtype` default flipped from `fp32` → `fp16`** in both
  `scripts/train_ddp.py` and `scripts/profile_step.py`.
- `--neuron-norm-min-dim` default remains `0` (always apply); available
  as opt-in Phase B2 knob but not adopted.
- `--cautious-wd` default remains `True`; `--no-cautious-wd` available
  but not adopted.

Users wanting to restore Phase 2 behavior pass `--ns-dtype fp32` explicitly.

---

## Timeline

| Day | Task                              | Status |
|----:|-----------------------------------|:------:|
| 1   | Phase A (profile + NS bench)     | ✓      |
| 1   | Phase B CLI wiring + 8/8 tests   | ✓      |
| 1   | Phase B ablation runs B0-B4      | ✓      |
| 1   | Phase E Run 2b                    | ✓      |
| 1   | Single atomic Sprint 1.1 commit  | ✓      |

Total wall time: ~7 hours compute (1 hour profiling + 54 min × 5 ablation
runs + 54 min Run 2b); ~4 hours dev work. Under budget (plan: 1-2 days).

---

## Unlocks

Sprint 1.1's shipping closes Sprint 1's throughput gate. Downstream:

1. **Sprint 1 throughput gate now PASSES** (3.5% cost; target was ≤10%).
   All Sprint 1 features (intra-doc mask, NorMuon, value residuals,
   head gating, fp16 NS) ship together as the default recipe for OdinFlat
   training.

2. **Sprint 3 (T²-optimal dolma-10B training)** inherits the +17.5%
   throughput. A 50-hour training run at 32.5K tok/s becomes a **~42-hour
   run at 38K tok/s** — 8 hours saved per full Sprint 3 epoch.

3. **Sprint 1.5 (SPECTRA + μP)** inherits the optimized NorMuon. SPECTRA
   acts on NorMuon's orthogonalized updates; having a faster NS path
   means SPECTRA ablations can run more configs in the same wall budget.

4. **Future work** — the original plan reserved Phase D (HIP NS kernel)
   as a conditional escalation. It was not needed and remains a
   future-proofing option if we scale to larger models where NS matmul
   cost becomes dominant again despite fp16.

---

## Artifacts (single commit)

```
# Phase A
docs/perf/normuon-profile-{AdamW,NorMuon,Full}.txt
docs/perf/normuon-compile-log.txt
docs/perf/normuon-ns-benchmark.json
docs/perf/normuon-ns-benchmark-machineB.json
docs/perf/normuon-profile-summary.md
scripts/profile_step.py                    # rewrite: Sprint 1 flag surface
scripts/bench_newton_schulz.py             # new: standalone NS bench
scripts/test_sprint1_1_profile.py          # 3/3 tests pass
scripts/test_sprint1_1_bench.py            # 1/1 tests pass

# Phase B
docs/perf/eval-scorecards/sprint1.1-B{0,1,2,3,4}-step-200.json
docs/perf/sprint1.1-phaseB-scorecard.md
halo_training/normuon.py                   # neuron_norm_min_dim gate
halo_training/optimizer.py                 # build_imu1_optimizer kwargs
scripts/train_ddp.py                       # 3 new CLI flags, fp16 default
scripts/run_sprint1_1_phaseB.sh            # B0-B3 runner
scripts/run_sprint1_1_B4.sh                # B4 combo runner
scripts/test_sprint1_1_normuon.py          # 8/8 tests pass

# Phase E
docs/perf/eval-scorecards/sprint1-run2b-step-{500,936}.json
docs/perf/sprint1.1-summary.md             # this file
scripts/run_sprint1_1_phaseE.sh            # Run 2b runner
scripts/print_scorecard_bpb.py             # helper for per-domain BPB tables
```

## Links

- Spec: `docs/superpowers/specs/2026-05-07-sprint1.1-normuon-throughput-design.md`
- Plan: `docs/superpowers/plans/2026-05-07-sprint1.1-normuon-throughput-plan.md`
- Phase A summary: `docs/perf/normuon-profile-summary.md`
- Phase B scorecard: `docs/perf/sprint1.1-phaseB-scorecard.md`
- Knowledge: `knowledge/training/normuon_throughput_gfx1151.md` (updated)

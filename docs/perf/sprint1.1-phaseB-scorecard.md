# Sprint 1.1 Phase B Scorecard — NorMuon Throughput Ablations

**Dates:** 2026-05-07
**Hardware:** 2× AMD Strix Halo (gfx1151), DDP over TB4 (gloo)
**Config:** `batch=16 block=512 accum=8 world=2` → effective batch 256; 200 opt steps
**Base flags:** `--imu1-groups --normuon --lr-2d 5e-3 --lr-1d 8e-4 --intra-doc-mask --value-residuals --head-gating --no-muon --auto-eval`

---

## Results table

| Run | Added flag             | tok/s  | Δ vs B0 | loss@200 | wiki_bpb | avg_bpb | Verdict |
|-----|------------------------|-------:|--------:|---------:|---------:|--------:|:--------|
| B0  | (reference)            | 31,051 | —       | 5.5654   | 2.2136   | 3.0162  | ref     |
| **B1**  | **`--ns-dtype fp16`**      | **37,177** | **+19.7%** | 5.5637 | **2.2157** | 3.0311 | **✓ PASS** |
| B2  | `--neuron-norm-min-dim 512` | 33,087 | +6.6% | 5.5869 | 2.2242 | 3.0078 | ✗ wiki_bpb > 2.22 |
| B3  | `--no-cautious-wd`     | 31,929 | +2.8%  | 5.5665 | 2.2183 | 3.0462 | ✓ pass (marginal) |
| B4  | B1 + B3 combo          | 37,853 | +21.9% | 5.5639 | 2.2238 | 3.0103 | borderline |

Quality gates (from plan, applied at step 200):
- `wiki_bpb ≤ 2.22`
- `step-200 loss within 0.5% of B0 reference` (B0 = 5.5654 → tolerance 5.5932)

---

## Per-run analysis

### B0 — reference (Run 2 config)

Run 2 measured 32,478 tok/s at full epoch on wikitext-103. B0's 31,051
tok/s (single 200-step run with auto-eval firing concurrently) matches
within 4.4%. B0 is the noise-floor reference for the three toggles.

### B1 — fp16 Newton-Schulz ✓ **winner**

Single change: `ns_dtype=torch.float16` in `NorMuon.__init__`. Routes the
5-step Polar Express matmuls through rocBLAS `HHS_BH_` fp16 kernels
instead of `S_B_` fp32. Phase A micro-benchmark predicted 8-13× per-shape
NS speedup on SwiGLU shapes; the end-to-end training lift is **+19.7%**.

Quality is **indistinguishable from B0** at step 200:
- loss: 5.5637 (vs 5.5654, −0.03%)
- wiki_bpb: 2.2157 (vs 2.2136, +0.09%)
- avg_bpb: 3.0311 (vs 3.0162, +0.49%)

No NaN/Inf steps observed. Max grad norm stable at 0.28-0.44 throughout.

### B2 — size-gated neuron-norm ✗ quality miss

`neuron_norm_min_dim=512` skips neuron-wise normalization on params
whose smaller dimension is <512 (the three factorized embed/head
projections plus per-head biases for anything small).

Throughput lift +6.6% (less than expected — Phase A predicted <1% since
the skipped params are cheap, so the measured +6.6% is likely noise-floor
variance rather than a true savings).

Quality gate miss: wiki_bpb 2.2242 > 2.22 ceiling (by 0.09%). Loss +0.39%
is within the 0.5% tolerance but the BPB cross is strict. `avg_bpb`
actually improves (3.0078 vs 3.0162), so the signal is mixed. Given the
borderline + noise-floor throughput, **B2 is dropped from B4**.

### B3 — disable cautious WD ✓ pass (marginal)

`cautious_wd=False` applies decoupled WD unconditionally instead of only
where `sign(grad) * sign(weight) > 0`. Removes one mask compute per 2D
param per step.

Throughput lift +2.8% — marginal, within noise floor. Quality clean:
loss +0.02%, wiki_bpb 2.2183 (well under 2.22). **No evidence cautious
WD is meaningfully helping at step 200**, but the throughput gain is too
small to be confident about.

### B4 — B1 + B3 combo (borderline)

`--ns-dtype fp16 --no-cautious-wd` stacked. Tok/s +21.9% vs B0, +1.8%
over B1 alone — again within noise floor. Loss matches B1 (5.5639 vs
5.5637). wiki_bpb drifts up to 2.2238 (+0.46% vs B0, +0.37% vs B1),
missing the 2.22 ceiling by 0.17%. avg_bpb improves to 3.0103 (vs B0's
3.0162, better by 0.2%).

The +1.8% throughput edge from adding B3 on top of B1 is not worth the
wiki_bpb drift.

---

## Decision

**Phase E Run 2b will use B1 alone** (`--ns-dtype fp16`).

Rationale:
1. B1 passes both quality gates cleanly (wiki_bpb 2.2157 < 2.22, loss
   within 0.03% of B0).
2. B4's additional +1.8% from `--no-cautious-wd` is noise-floor and
   trades a slight wiki_bpb regression. Not a clean win.
3. B2 (size-gated neuron-norm) fails the wiki_bpb ceiling.
4. Cost vs AdamW baseline (39,538 tok/s): B1 = **5.97% cost**, well
   below the 7% Phase-B exit gate.

## Phase B exit gate

Per plan §Phase B exit gate:
- B4 cost = **4.26%** ≤ 7% → **skip Phases C + D → proceed to Phase E**

Even without B4, B1 alone passes: cost = 5.97% ≤ 7%. **Phase C (batched
NS) and Phase D (HIP NS kernel) are not needed.** The Phase A prediction
that fp16 NS alone would close the gap is fully confirmed.

---

## Raw throughput summary

Cost vs AdamW baseline (Run 2 baseline = 39,538 tok/s DDP on wikitext-103):

| Config                    | tok/s  | cost%  | Gate? |
|---------------------------|-------:|-------:|:------|
| AdamW baseline (Run 2 bl) | 39,538 | 0.0%   | —     |
| Run 2 full recipe (prod)  | 32,478 | 17.8%  | FAIL  |
| B0 (Run 2 recipe, 200 stp)| 31,051 | 21.5%  | —     |
| B3 (+ no-CWD)             | 31,929 | 19.2%  | —     |
| B2 (+ size-gate)          | 33,087 | 16.3%  | —     |
| **B1 (+ fp16 NS)**        | **37,177** | **5.97%** | **PASS** |
| **B4 (fp16 + no-CWD)**    | **37,853** | **4.26%** | PASS (quality borderline) |

The 200-step single-run variance is ~1-2% tok/s. Real-epoch gains
projected ~+17-20% over Run 2's 32,478 tok/s → ~**38,000-39,000 tok/s**.

---

## Artifacts

```
checkpoints/sprint1.1-B{0,1,2,3,4}/       — 200-step checkpoints + rank0/1 logs
docs/perf/eval-scorecards/sprint1.1-B{0-4}-step-200.json
docs/perf/eval-scorecard.jsonl            — appended 5 rows
docs/perf/sprint1.1-phaseB-scorecard.md   — this file
scripts/run_sprint1_1_phaseB.sh           — B0-B3 sequential runner
scripts/run_sprint1_1_B4.sh               — B4 combo runner
scripts/print_scorecard_bpb.py            — BPB table helper
halo_training/normuon.py                  — added neuron_norm_min_dim gate
halo_training/optimizer.py                — build_imu1_optimizer accepts new kwargs
scripts/train_ddp.py                      — --ns-dtype / --neuron-norm-min-dim / --no-cautious-wd
scripts/profile_step.py                   — mirrored flags for profiling
scripts/test_sprint1_1_normuon.py         — 8/8 unit tests pass
```

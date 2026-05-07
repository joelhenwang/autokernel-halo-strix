## Current Training Status

> **Read this file before every training launch or decision.**
> Update after every run completes or fails.

---

## fp16 Stability Hardening (2026-05-07, SHIPPED inline)

**Status:** COMPLETE. Inline follow-up to the dolma-10B NaN incident
(2-epoch OdinHalo loss + grad NaN). No Sprint spec/plan; single commit.
User explicitly declined validation run — forensic reasoning only.

### What shipped

**Prevention (Ring 1):**
- `--z-loss <w> --z-loss-fraction <f>` in `train_ddp.py` (mirrors `halo_training/trainer.py`, now available in DDP path)
- `iter_scales.clamp(-4, 4)` at forward time in OdinHalo + VidarHalo. Parameter unchanged; checkpoint-compatible both ways.
- GradScaler `growth_interval` default 2000 → 500 (slower scale runaway)
- `--max-grad-norm` auto-tightens 1.0 → 0.8 when `--resume-from` is set
- `--attn-softcap <c>` opt-in pre-softmax tanh cap in Attention / CodaAttention / NoPECodaAttention (all use shared `_attention_core` helper). 0 = SDPA fast path (no regression).
- `--activation-monitor` opt-in: per-layer maxabs + fp16_headroom to `$CKPT_DIR/activation_stats.jsonl`. New module `halo_training/activation_monitor.py`.

**Response (Ring 2):**
- NaN forensics dump (R1): on `StabilityGuard` trigger, saves `$CKPT_DIR/nan_dump_step_N.pt` with offending batch, scaler state, grad norm history (last 50), per-param weight maxabs, activation stats. Diagnostic for post-mortem; does not block rollback.
- Rollback also halves `scaler.growth_interval` (R3). Floor 100.
- `scaler.get_scale()` in periodic log line + warning when > 16384 (R5). JSONL gets `scaler_scale` key.

**Docs:**
- NEW: `knowledge/training/fp16_stability_gfx1151.md` (root cause + every knob + forensics schema + diagnostic playbook)
- AGENTS.md training-gotchas entry added
- CONSTRAINTS.md adds `fp16 Stability` checklist section + iter_scales clamp entry

### Tests

`scripts/test_fp16_stability.py` — **14/14 pass**. Covers every knob
above + backward-compat of `StabilityGuard.rollback(scaler=None)`.

All existing test suites regressed clean:
  - test_sprint1_phase1.py: 15/15
  - test_sprint1_phase2.py: 15/15
  - test_sprint1_1_profile.py: 3/3
  - test_sprint1_1_bench.py: 1/1
  - test_sprint1_1_normuon.py: 8/8
  - test_fp16_stability.py: 14/14
  - Total: 56/56

### Explicitly NOT shipped

- `--bf16` flag: user call — bf16 not compatible with gfx1151 / our stack.
- Reproduction / smoke run of 2-epoch dolma-10B: user declined validation.
- Data-pointer advance on rollback (R2): punted.
- bf16 escalation ladder on rollback 3+ (R4): punted.

### Recommended flags for future long-horizon runs

```bash
EXTRA_FLAGS="--z-loss 1e-4 --z-loss-fraction 0.4 \
             --attn-softcap 50.0 \
             --activation-monitor \
             --max-grad-norm 0.8"
```

For resumed runs, `--resume-from` auto-tightens grad-norm.

---

## Sprint 1.1: NorMuon Throughput Optimization (2026-05-07, SHIPPED)

**Status:** COMPLETE. All quality gates PASS, throughput target exceeded.
Default flag `--ns-dtype fp16` flipped on in `train_ddp.py` +
`profile_step.py`. Single-commit delivery.

### Headline result

| Metric | Run 2 (fp32 NS) | **Run 2b (fp16 NS)** | Δ | Gate | Status |
|---|---:|---:|---:|:---|:---:|
| Final loss | 4.4736 | **4.4741** | +0.01% | ≤ 4.518 | PASS |
| wiki_val BPB | 1.8930 | **1.8962** | +0.17% | ≤ 1.912 | PASS |
| avg BPB | 2.8114 | **2.8120** | +0.02% | ≤ 2.838 | PASS |
| tok/s | 32,478 | **38,162** | **+17.5%** | ≥ 33K floor | EXCELLENT |
| cost vs AdamW | 17.8% | **3.5%** | 5× reduction | ≤ 10% excellent | EXCELLENT |
| Memory | 10.1 GB | 10.1 GB | 0% | unchanged | PASS |

### Root cause

NorMuon's Newton-Schulz iteration ran fp32 matmuls (rocBLAS `Cijk_..._S_B_...`)
by default. Phase A profile: NS fp32 matmul = 53% of step time. Phase A.5
micro-bench: switching to fp16 gives 8-13× speedup on SwiGLU shapes,
4× on smaller projections. Machine-parity confirmed.

### What shipped

- `--ns-dtype {fp16,fp32}` CLI flag (default fp16) in `train_ddp.py` and
  `profile_step.py`
- `--neuron-norm-min-dim` and `--no-cautious-wd` as opt-in ablation knobs
- `halo_training/normuon.py::NorMuon` accepts `neuron_norm_min_dim` +
  respects it in `_normuon_step`
- `halo_training/optimizer.py::build_imu1_optimizer` forwards all 3 kwargs
- Single-node `scripts/profile_step.py` rewritten to accept Sprint 1 flags
- `scripts/bench_newton_schulz.py` standalone NS micro-benchmark
- 12/12 unit tests pass (test_sprint1_1_profile.py + test_sprint1_1_bench.py
  + test_sprint1_1_normuon.py)

### Phase outcomes

| Phase | Plan | Status |
|:-----:|------|:------:|
| A | Profile NorMuon step path (3 configs) | ✓ DONE |
| A.5 | Compile events + NS micro-bench + summary | ✓ DONE |
| B | Quick-win ablations B0-B4 | ✓ DONE |
| C | Batched NS (structural) | **SKIPPED** (cost already ≤ 7%) |
| D | HIP NS kernel | **SKIPPED** (cost already ≤ 7%) |
| E | Run 2b full-epoch validation | ✓ DONE |

### Artifacts

```
docs/perf/normuon-profile-{AdamW,NorMuon,Full}.txt   — Phase A raw profiles
docs/perf/normuon-compile-log.txt                     — Phase A.5.1 dynamo events
docs/perf/normuon-ns-benchmark{,-machineB}.json       — Phase A.5.2 NS bench (both machines)
docs/perf/normuon-profile-summary.md                  — Phase A.5.3 attribution doc
docs/perf/sprint1.1-phaseB-scorecard.md               — Phase B 5-run scorecard
docs/perf/sprint1.1-summary.md                        — Sprint-level summary
docs/perf/eval-scorecards/sprint1.1-B{0,1,2,3,4}-step-200.json
docs/perf/eval-scorecards/sprint1-run2b-step-{500,936}.json
scripts/profile_step.py  bench_newton_schulz.py  run_sprint1_1_*.sh
scripts/test_sprint1_1_{profile,bench,normuon}.py     — 12/12 tests pass
knowledge/training/normuon_throughput_gfx1151.md      — updated with Run 2b numbers
```

### Unlocks

- **Sprint 1 throughput gate now PASSES** (3.5% cost; target ≤ 10%). Ship
  Sprint 1 recipe as default for OdinFlat training.
- **Sprint 3 (T²-optimal dolma-10B)** inherits +17.5% throughput — saves
  ~8 hours per full epoch at 50-hour run length.
- **Sprint 1.5 (SPECTRA + μP)** inherits optimized NorMuon configuration.

---

## Sprint 1 Phase 5: Run 2 Full Recipe (2026-05-06)

**Status:** COMPLETE. Quality gate PASS, throughput gate FAIL → Sprint 1.1 follows.

Configuration:
  --intra-doc-mask --imu1-groups --normuon --lr-2d 5e-3 --lr-1d 8e-4
  --value-residuals --head-gating --no-muon --auto-eval

Results (step 936, 1 epoch wikitext):
  Final train loss:  4.4736   (−6.8% vs baseline 4.7975)  ✅ PASS
  wikitext_val BPB:  1.893    (−1.5% vs baseline 1.9214)  ⚠️ beats baseline, misses spec's −3.8% target of 1.73
  gpt_small_val BPB: 2.8327   (−1.1%)
  stem_crawl_val BPB: 3.4314  (−3.0%)
  dolma_val BPB:     3.0883   (−0.7%)
  avg BPB:           2.810    (−1.8%)
  Throughput:        32,478 tok/s  (17.8% cost)           ❌ FAIL spec's ≤7% gate
  Memory:            10.1 GB  (−1.9%)                      ✅ PASS
  Wall time:         63 min
  Stability:         zero NaN, max grad norm < 2           ✅ PASS
  Fallback:          --no-normuon path still works         ✅ PASS

Gate evaluation: 4/6 criteria pass, 1 partial (BPB beats baseline but
doesn't hit the −3.8% target), 1 fail (throughput). Quality gains are
real and substantial; the failure mode is NorMuon's PyTorch-op NS being
expensive on gfx1151 (13-16% pure-NorMuon cost; +2 pp for value residuals
and head gating combined).

Decision: launch Sprint 1.1 (NorMuon throughput optimization sub-sprint)
before flipping any defaults in the CLI. Ship Sprint 1 + 1.1 together.

Scorecards:
  docs/perf/eval-scorecards/sprint1-run2-step-500.json
  docs/perf/eval-scorecards/sprint1-run2-step-936.json

---

## Sprint 1 Phase 4: NorMuon LR Probe (2026-05-06)

**Status:** COMPLETE. Winning config: lr_2d=5e-3, lr_1d=8e-4 (Probe C).

Three 200-step DDP probes with NorMuon + free wins, swept lr_2d:

| Config | lr_2d | lr_1d | step 200 loss | step 200 wiki_bpb | tok/s | cost% |
|--------|------:|------:|--------------:|------------------:|------:|------:|
| Baseline (AdamW) | 8e-4 | 8e-4 | 5.9045 | — | 39,925 | 0% |
| Run 1b (AdamW+fw) | 8e-4 | 8e-4 | 5.8477 | — | 39,535 | 1.0% |
| Probe A (NorMuon) | 8e-4 | 8e-4 | 5.6733 | 2.2376 | 34,363 | 13.1% |
| Probe B (NorMuon) | 2e-3 | 8e-4 | 5.5967 | 2.2163 | 34,596 | 12.5% |
| **Probe C (NorMuon)** | **5e-3** | 8e-4 | **5.5617** | **2.2078** | 33,234 | 15.9% |

Probe C wins on all 4 BPB domains + training loss. Higher lr_2d gives
NorMuon more "room to move" relative to AdamW's more conservative update
magnitude. Downside: throughput drops further at higher lr because
larger updates trigger more cautious-WD sign flips.

Used Probe C config for Phase 5 Run 2.

Also fixed a pre-existing bug: `scripts/train_ddp.py` `_complete_step`
raised `AssertionError: No inf checks were recorded` when `--max-steps`
terminated mid-accumulation. Wrapped in try/except with clean warning.

Scorecards:
  docs/perf/eval-scorecards/sprint1-probe-{A,B,C}-step-200.json

---

## Sprint 1 Phase 3: Run 1 + Run 1b Free-Wins Validation (2026-05-06)

**Status:** Diagnosis confirmed. Free wins' LR split hurt; even without
split, BPB is neutral-to-slight-regress at 1-epoch wikitext. Decision:
move to Phase 4 focused on NorMuon (the real Sprint 1 lever).

### Three-way A/B (block=512, 1 epoch, 936 opt steps, matched config)

| Metric | Run 1 (split 3.36x) | Run 1b (no split) | Baseline (AdamW) |
|---|---:|---:|---:|
| Final train loss | 4.7907 | **4.6295** | 4.7975 |
| wikitext_val BPB | 1.9835 | 1.9253 | **1.9214** |
| gpt_small_val BPB | 2.9374 | 2.8754 | **2.8634** |
| stem_crawl_val BPB | 3.6401 | 3.541 | **3.5361** |
| dolma_val BPB | 3.2646 | 3.1666 | **3.1094** |
| avg BPB | 2.9564 | 2.877 | **2.861** |
| Throughput | 39,045 | 39,104 | 39,538 |

Config deltas:
  Baseline:    no Sprint 1 features, single-group AdamW lr=8e-4 wd=0.1
  Run 1:       intra-doc-mask + IMU-1 grouping lr_2d=8e-4 lr_1d=3e-4 + LN scaling
  Run 1b:      intra-doc-mask + IMU-1 grouping lr_2d=8e-4 lr_1d=8e-4 + LN scaling

### Findings

1. **LR split was the Run 1 regression culprit.** Dropping lr_1d to 3e-4
   starved 1D group (embed/norms/head); Run 1b with unified lr=8e-4
   recovers parity and then some.

2. **Train loss != generalization.** Run 1b achieves −3.5% training
   loss vs baseline but scores +0.2-1.8% WORSE BPB on all 4 held-out
   domains. Classic overfit signature: intra-doc-mask + LN scaling
   + no-WD-on-embed help the model fit the training stream but don't
   improve tail-slice BPB at this scale.

3. **At 1-epoch wikitext (122M tokens) the free wins don't pay out.**
   The regularization signal requires more data / epochs to materialize.
   For Sprint 3's dolma-10B (57x tokens) the story may differ; for now,
   don't rely on free wins as the loss-improvement driver.

4. **Throughput is essentially free for all three configs** (39-39.5K
   tok/s = <2% spread). Free wins and IMU-1 grouping have no measurable
   compute cost.

### Decision for Phase 4

Original plan: 3-config LR sweep for NorMuon at IMU-1's absolute LRs
(0.015-0.030). Revised plan: sweep lr_2d at our AdamW-scale LRs since
NorMuon's effective step size per our param scale is different from
IMU-1's 430M setup.

Probe strategy:
  Baseline reference: Baseline block=512 final loss 4.7975, BPB 1.9214
  Run 1b reference:   lower training loss but flat BPB

  Config A: lr_2d=8e-4 lr_1d=8e-4 (NorMuon at AdamW-baseline LR)
  Config B: lr_2d=2e-3 lr_1d=8e-4 (2.5x higher lr_2d for NorMuon)
  Config C: lr_2d=5e-3 lr_1d=8e-4 (6x higher lr_2d)

All include free wins (intra-doc-mask + LN scaling + no-WD-on-embed).
200 steps each. Winner = lowest mean wiki_val BPB at step 200 OR
lowest step-200 training loss (both measured via auto-eval).

Gate: >= 1 NorMuon config beats baseline wikitext_val BPB of 1.921.
If no NorMuon config beats baseline, Sprint 1's gate fails and we
fall back to "ship free wins + IMU-1 grouping without split as the
new default" rather than NorMuon.

Scorecards committed:
  sprint1-run1-step-{500,936}.json
  sprint1-run1b-step-{500,936}.json
  sprint1-baseline-block512-step-{500,936}.json

---

## Sprint 1 Phase 2: NorMuon + Architectural Additions (2026-05-06)

Commit: 75cd918

Delivers IMU-1 Phase 2 features:
  - halo_training/normuon.py — NorMuon optimizer (Muon + neuron-wise norm + Cautious WD)
  - models/components/attention.py — value residual (v_res_scale) + per-head gating
  - models/odin_flat.py — use_value_residuals, use_head_gating flags + v_prev threading
  - scripts/train_ddp.py — three-flag model configuration

Tests: 15/15 Phase 1 + 15/15 Phase 2 + 22/22 Sprint 2 = 52 green.

Integration: 50-step DDP smoke with ALL features ON:
  NorMuon(2D, n=52.5M, lr=0.005) + AdamW(1D, n=69.2M, lr=0.001)
  use_intra_doc_mask + use_value_residuals + use_head_gating = ON
  loss=9.25 (step 50 pre-warmup), tok/s=26,274, mem=10.1 GB

NorMuon throughput cost ~34% at step 50 (pre-warmup); will measure
steady-state in Phase 4 LR probe.

---

## Sprint 1 Phase 1: Free-Wins Infrastructure (2026-05-06)

Commit: afd7dcb

Delivers IMU-1 Phase 1 features:
  - doc_ids in PreTokenizedDataset / BabyLMDataset (3-tuple return)
  - split_params_2d_vs_1d + build_imu1_optimizer (AdamW two-group)
  - LayerNorm scaling init (1/sqrt(layer_idx+1))
  - Intra-document attention mask in NoPECodaAttention
  - doc_ids plumbing through trainer/smoke/evaluate/streaming/train_ddp
  - 8 new CLI flags (--intra-doc-mask, --imu1-groups, --lr-2d, --lr-1d,
    --normuon, --value-residuals, --head-gating)

Tests: 15 unit tests (scripts/test_sprint1_phase1.py); smoke test passes
end-to-end with all flags; DDP 30-step smoke validates full integration.

---

## Sprint 2: Evaluation Scorecard Infrastructure (2026-05-06)

Per-checkpoint multi-dimensional scorecard shipped. Gate C → B CLEARED.

**Spec:** [docs/superpowers/specs/2026-05-06-sprint2-eval-scorecard-design.md](docs/superpowers/specs/2026-05-06-sprint2-eval-scorecard-design.md)
**Plan:** [docs/superpowers/plans/2026-05-06-sprint2-eval-scorecard-plan.md](docs/superpowers/plans/2026-05-06-sprint2-eval-scorecard-plan.md)

### Infrastructure delivered

- `scripts/eval_checkpoint.py` — CLI entry point, all-by-default evaluator dispatch with `--skip-*` opt-outs
- `halo_training/eval/` — 5 evaluators + common helpers + scorecard schema
  - `common.py` — checkpoint/model/tokenizer loading, `_orig_mod.` stripping, validation split discovery
  - `scorecard.py` — schema v1.0, JSON/JSONL assembly
  - `per_domain_bpb.py` — BPB on tail slices of wikitext/gpt-small/stem-crawl/dolma
  - `sampling.py` — wraps refactored `ablate_odin_flat_sampling` to extract distinct-2 / self-PPL at winning config
  - `inference_profile.py` — tok/s + peak memory at seq={256, 512, 1024}, batch=1
  - `sample_pack.py` — 20-prompt deterministic regression with hash + prior-checkpoint diff
  - `activation_stats.py` — per-layer kurtosis / RMS via forward hooks (auto-skips when model lacks `.layers`)
- `evals/sample_pack_v1.txt` — 20 frozen prompts (never mutate; bump to v2 for changes)
- `scripts/train_ddp.py --auto-eval` — detached-subprocess hook after every `save_checkpoint`
- `scripts/launch_ddp.sh` — `EXTRA_FLAGS` env var flows through to both ranks
- `scripts/test_eval_scorecard.py` — 22 unit tests, all green on both machines
- `scripts/test_auto_eval_spawn.py` — Phase 5 spawn smoke-test helper
- `scripts/compare_parity.py` — cross-machine scorecard diff tool

### Scope changes from original design

- **Int4 BPB DROPPED** (2026-05-06 revision) — per-tensor symmetric int4 too crude as a deployment-readiness indicator at 122M scale; no near-term deployment path. Can be reintroduced later as standalone `scripts/quantize_eval.py` or re-added to scorecard.
- lm-evaluation-harness benchmarks (HellaSwag, ARC, MMLU, PIQA, BLiMP) deferred to Sprint 4 (post-training) as originally planned.

### Retroactive validation (Phase 6)

Three reference scorecards committed:

| Checkpoint | Wall time | BPB (wiki / gpt-small / stem / dolma) | distinct_2 | self_ppl |
|---|---:|---|---:|---:|
| `odin-flat-wikitext-ddp/step_1869.pt` | 33.6s | 1.80 / 3.01 / 3.43 / 3.14 | 0.765 | 9.84 |
| `odin-flat-stem-crawl-ddp/step_4046.pt` | 67.8s | 2.52 / 2.47 / 1.63 / 2.13 | 0.535 | 6.82 |
| `odin-halo-wikitext-ddp/step_1869.pt` | 34.0s | 1.97 / 3.08 / 3.42 / 2.81 | 0.990 | 14.12 |

- Catastrophic-forgetting pattern visible: stem-crawl checkpoint has low BPB on stem (1.63) but high on wikitext (2.52).
- OdinHalo shows higher distinct-2 (0.99 vs OdinFlat's 0.77) — its looped architecture produces more diverse continuations at winning config.
- Activation stats skip gracefully on OdinHalo ("no `.layers` iterable") — as designed.

### Machine parity (Phase 6c)

Same `odin-flat-wikitext-ddp/step_1869.pt` eval'd on Machine A and Machine B:

| Metric | Machine A | Machine B | Delta |
|---|---:|---:|---:|
| wikitext_val BPB | 1.8013 | 1.8013 | +0.00% |
| gpt_small_val BPB | 3.0087 | 3.0087 | +0.00% |
| stem_crawl_val BPB | 3.4344 | 3.4344 | +0.00% |
| distinct_2 | 0.7648 | 0.7648 | +0.00% |
| self_ppl | 9.84 | 9.84 | +0.00% |
| sample_pack hash | sha256:356b6389b3b52db0 | sha256:356b6389b3b52db0 | identical |
| tok_s_seq512_bs1 | 59,156 | 58,671 | -0.82% |
| peak_mem_gb_seq512 | 0.432 | 0.433 | +0.23% |

All within ±5% budget; BPB + sampling + sample-pack hashes bit-identical across machines.
dolma_val not present on Machine B (deliberately Machine-A-only per AGENTS.md).

### Usage

```bash
# Basic (single checkpoint)
EVAL_MACHINE=a python scripts/eval_checkpoint.py \
    --checkpoint checkpoints/odin-flat-wikitext-ddp/step_1869.pt \
    --model models/odin_flat.py --class-name OdinFlat

# Selective evaluators
python scripts/eval_checkpoint.py --checkpoint ... \
    --skip-sample-pack --skip-activation-stats

# Auto-trigger during DDP training
EXTRA_FLAGS='--auto-eval' bash scripts/launch_ddp.sh

# Machine parity check
bash scripts/compare_parity.py docs/perf/eval-scorecards/<name>.json /tmp/parity-b/<name>.json
```

**Outputs:**
- `docs/perf/eval-scorecards/<name>.json` — full scorecard per checkpoint
- `docs/perf/eval-scorecard.jsonl` — one-line-per-run rolling index (jq/grep-friendly)

**Gate unlocked:** Sprint 3 (T²-optimal dolma training) can run with `--auto-eval` for per-checkpoint visibility during the 50-hour run. Sprint 1 and Sprint 1.5 ablations now have a multi-dimensional measurement harness.

---

## CE Kernel Optimization Stack (2026-05-05)

Two-phase rewrite of cross-entropy path on OdinHalo (V=32768, B=4, T=256).

**Phase 1 (`kernel.py`):** Online softmax single-pass CE kernel with fused/tiny modes.
  - Features baked in-kernel: `logit_softcap`, `z_loss`, `ignore_index`, `label_smoothing`
  - `kernel_fn` fast path (forward-only) for bench.py compatibility
  - `ce_full()` entry point with kwargs
  - **Isolated perf (B=4096, V=32768)**: fwd 2.32×, bwd 1.54×, **fwd+bwd 1.66×** vs PyTorch
  - Correctness: 5-stage bench + 16 feature tests all PASS

**Phase 2 (`kernels/hip/chunked_linear_cross_entropy.py` rewritten):**
Avoids materializing `[N, V]` logits tensor by chunked linear+CE+grad flow.
  - Uses Phase 1 HIP kernel per chunk with pre-scaled grad (no bwd multiply pass)
  - fp16 matmul (drops fp32 from prior impl)
  - Supports softcap, label_smoothing, ignore_index, z_loss
  - Tied weight grads handled via PyTorch autograd accumulation
  - CLI flag: `--chunked-ce`
  - Model integration: `FactorizedLMHead.use_chunked_ce` → returns `h_low` in training
  - Gradient parity vs fp32 reference: loss_rel ~1e-7, grad_rel ~1e-3 (fp16 matmul floor)

**Production OdinHalo (batch=4, T=256, V=32768) ablation:**

| Config                                    | tok/s  | Memory  | Speedup |
|-------------------------------------------|--------|---------|---------|
| Baseline (PyTorch CE)                     | 9,807  | 1.93 GB | 1.000×  |
| + Phase 1 HIP CE                          | 9,857  | 1.74 GB | 1.005×  |
| + Phase 2 Chunked CE                      | 11,455 | 1.56 GB | 1.168×  |
| + RoPE fusion (PyTorch CE)                | 11,295 | 1.96 GB | 1.152×  |
| + RoPE + HIP CE  (**best tok/s**)         | 11,704 | 1.76 GB | **1.193×** |
| + RoPE + Chunked CE (**best memory**)     | 11,366 | 1.59 GB | 1.159× (-343 MB) |

**Files:**
- `kernel.py` — rewritten Phase 1 kernel (online softmax + all features)
- `kernels/hip/chunked_linear_cross_entropy.py` — Phase 2 chunked linear+CE
- `models/components/embeddings.py` — FactorizedLMHead gained `forward_hlow()` + `use_chunked_ce` flag
- `models/odin_halo.py` — conditional h_low return + `logit_softcap` attribute
- `halo_training/trainer.py` — chunked_ce wired with softcap/z_loss/label_smoothing passthrough
- `halo_training/cli.py` — `--chunked-ce` opt-in flag, auto-propagates `use_chunked_ce=True`
- `scripts/test_ce_features.py` — 16 feature correctness cases
- `scripts/test_chunked_ce.py` — gradient parity + memory test
- `scripts/test_odin_chunked.py` — end-to-end OdinHalo integration test
- `scripts/ablation_full.py` — full stack ablation
- `scripts/ablation_modes.py` — fused vs tiny mode comparison

---


## Compile × Kernel Ablation + RoPE Bug Fix (2026-05-05 later)

Critical bug fix + comprehensive compile ablation.

### RoPE non-contiguous bug (fixed in `models/components/conv_blocks.py`)
`freqs_cis.real[:T, :pairs].float()` — `.real` on a complex tensor returns a
**non-contiguous view** (complex memory is interleaved [real, imag] fp32 pairs,
so `.real` has stride 2). The HIP `fused_rope_gate_mul` kernel reads this as
contiguous, effectively reading imag values as cos at odd positions → garbled
RoPE rotation. Silent miscompute; all halo models using this path were training
with WRONG positional encoding. Added `.contiguous()` in both HIP call sites.

Isolated RoPE+gate output diff (fixed vs buggy HIP): **max_err = 13.7** (not noise).
Isolated RoPE+gate output diff (fixed HIP vs native): max_err = 1e-3 (fp16 noise).

### Compile investigation results

OdinHalo has 4 graph breaks per HyPEShortConvBlock at default compile:
1. HIP `fused_rope_gate_mul` (wrapped with `@torch.compiler.disable`)
2-4. `causal_conv1d_fn` calls to DaoAILab C++ extension (non-contiguous `out=` tensor)

Added **compile-friendly path** (`HyPEShortConvBlock._compile_friendly` flag) with:
- Native PyTorch RoPE + gate multiply (no custom kernel)
- Manual causal conv via `F.conv1d` (no DaoAILab extension)
- Result: 0 graph breaks, compiles with `fullgraph=True`
- Accessible via `model.compile_zones_friendly()`

**However, 0 breaks did NOT materially speed up compile** — compile-friendly
path runs ~ the same as default compile (both ~1.08× eager at batch=4). Root
cause: HIP kernels are faster than the Inductor-generated triton for their
specific operations, even with graph breaks.

### Batch size sensitivity — MAJOR finding

Compile lift grows dramatically with batch size. At small batches, kernel launch
overhead dominates and compile can't help much. At larger batches, the overhead
is amortized and Inductor's fusion + matmul autotuning pays off.

### Full stack ablation (OdinHalo V=32768, 400 steps, 200 warmup, fused AdamW)

| Config | batch=4 tok/s | batch=16 tok/s | Lift (bs=16) |
|--------|---------------|----------------|--------------|
| Baseline (PyTorch CE, no fusion)            |  9,793 | 11,145 | 1.000× |
| + HIP CE (tiny)                             | 10,053 | 11,549 | 1.036× |
| + HIP CE + RoPE fusion                      | 10,058 | 11,203 | 1.005× |
| + HIP CE + RoPE + Chunked CE                |  9,790 | 10,959 | 0.983× |
| compile + PyTorch CE                        | 10,745 | 14,108 | 1.266× |
| **compile + HIP CE (best tok/s)**           | **11,066** | **14,682** | **1.317×** |
| compile + HIP CE + RoPE fusion              | 11,047 | 14,506 | 1.302× |
| **compile + HIP CE + RoPE + Chunked**       | 10,690 | 14,228 | 1.277× |
| *(best mem at batch=16)*                    | 1.95 GB | **3.89 GB** | (vs 6.60 GB) |

### reduce-overhead mode (CUDA graphs) — best memory configurations

| batch | Config | tok/s | Peak mem | Notes |
|-------|--------|-------|----------|-------|
| 16 | compile default + HIP CE               | 14,682 | 4.83 GB | best tok/s at bs=16 |
| 16 | compile reduce-overhead + HIP CE       | 14,425 | **2.14 GB** | -2.7 GB for 1.8% tok/s loss |
| 16 | compile reduce-overhead + Chunked CE   | 13,933 | **1.67 GB** | absolute lowest memory |
| 32 | compile default + HIP CE               | 13,967 | 9.72 GB | throughput plateaus |
| 32 | compile reduce-overhead + Chunked CE   | 12,736 | 3.65 GB | largest effective batch at low mem |

Throughput plateaus at batch≥16; going to batch=32 doesn't help (GPU saturated).

### reduce-overhead limitations (IMPORTANT)

`TORCH_COMPILE_MODE=reduce-overhead` is **NOT supported with looped models**
(HALO family with `compile_zones`). CUDA graph buffer reuse across Parcae
iterations invalidates saved activations for backward. The trainer detects this
and auto-falls-back to default mode with a warning:

```
WARNING: reduce-overhead is incompatible with looped models
  (buffer reuse across Parcae iterations). Falling back to default.
```

Similarly, `reduce-overhead + --chunked-ce` is unsupported (auto-disabled with
warning). Use case: memory-savings are available via isolated-benchmark reduce-
overhead testing, but for production training, use default compile mode.

For non-looped models (e.g., plain Llama), reduce-overhead works fine.

### Chunk size tuning for ChunkedLinearCrossEntropyLoss

| chunk_size | tok/s (compiled) | Peak mem | Note |
|-----------:|-----------------:|---------:|:-----|
| 128 | 13,731 | 3.86 GB | more python overhead |
| 256 | 14,117 | 3.89 GB | prior default |
| **512** | **14,303** | 3.96 GB | **new default, sweet spot** |
| 1024 | 12,397 | 4.09 GB | matmul shapes start to hurt |
| 4096 | 12,358 | 4.90 GB | loses memory benefit |

## Phase 1 Quick-Wins Results (2026-05-05)

Phase 1 spec: `docs/superpowers/specs/2026-05-05-phase1-quick-wins-design.md`.
Plan: `docs/superpowers/plans/2026-05-05-phase1-quick-wins-plan.md`.

### Summary

| Work item | Status | Measured effect |
|-----------|:------:|:----------------|
| W1 Deep profile tooling | ✓ shipped | `docs/perf/odinhalo-profile-2026-05-05-compile/profiler.md` generated (drives Phase 2) |
| W5 Residual dedup in HyPEShortConvBlock | ✓ shipped | Inductor already CSE'd under compile — neutral throughput |
| W6 DataLoader: `--num-workers` + `non_blocking=True` | ✓ shipped | Neutral (Strix unified memory; pinning+non_blocking ~no-op) |
| W2 Lion optimizer + `--lion` CLI | ✓ shipped | Opt-in flag; smoke test passes |
| W3 Optimizer shootout (4-way) | ✓ shipped | See table below; AdamW wins tok/s, Muon wins final loss |
| W4 `compile(optimizer.step)` experiment | ✓ shipped | No benefit (0.997×); fused AdamW already single-kernel |
| CLion (Cautious Lion, arXiv:2604.14587) | ✓ shipped | Added as part of W3 optimizer suite; per-coord gate default |

### Shootout: OdinHalo V=32768 batch=16, 400 steps (200 warmup, 200 measured)

| Optimizer | tok/s | Peak GB | Init loss | Final loss | Δ loss |
|-----------|------:|--------:|----------:|-----------:|-------:|
| **AdamW (fused)** | **13,991** | 5.67 | 4.751 | **4.181** | −0.570 |
| Muon | 3,958 | 5.47 | 4.474 | **4.072** | −0.402 |
| Lion | 13,695 | 5.44 | 5.314 | 4.311 | −1.003 |
| CLion (per_coord, ν=1e-6) | 13,431 | 5.44 | 5.291 | 4.313 | −0.977 |

**Winner:** AdamW (highest tok/s + reasonable loss).
**Lowest loss:** Muon, but 3.5× slower step (Newton-Schulz iteration).
**Largest loss reduction:** Lion, but higher starting loss (sign-update at LR=3e-5 is coarse).
**CLion vs Lion:** essentially identical at ν=1e-6 (per-coord gate rarely triggers identity
path at this threshold). Paper's claimed generalization advantage needs longer runs + test-set
eval to observe.

### CLion implementation notes (arXiv:2604.14587)

Paper's Algorithm 2 specifies whole-tensor gating: use `sign(c)` iff the minimum non-zero
absolute value of `c` exceeds ν. At scale (OdinHalo 57.6M params, median |c|≈1e-5 per
tensor), **this gate almost never fires** — any single tiny gradient component fails the
check, forcing every tensor through the identity path at Lion's tiny LR → no learning.

Figure 1(d) of the paper illustrates CLion as a per-coordinate active function (identity
for small |c|, sign for larger). We default to this interpretation (`gate_mode="per_coord"`)
and offer `gate_mode="per_tensor"` for paper-faithfulness. The per-coord interpretation is
both useful in practice and consistent with the figure.

Default ν for OdinHalo-scale training: **1e-6** (below typical |c|≈1e-5 so sign fires for
~90%+ of coords). Paper's Theorem 2 threshold of 1.0 is only appropriate if gradients are
well-scaled to O(1); it is not the right value for modern LLM training with GradScaler.

### Regression check (post-Phase-1)

| Config | batch=16 tok/s | Peak GB | vs pre-Phase-1 |
|--------|---------------:|--------:|---------------:|
| compile + HIP CE | 14,577 | 4.83 | −0.7% (within noise) |
| compile + HIP CE + Chunked CE | 14,163 | 3.89 | −0.5% (within noise) |

All 8 halo model variants still pass single-step training test with `--chunked-ce`.

### Next steps

Phase 1 did NOT deliver a throughput lift because the shootout winner (AdamW) was already
the default. The value Phase 1 delivered:

1. **Profile artifact** (W1) for Phase 2 fusion investigation.
2. **Optimizer options** (Lion, CLion) as opt-in flags for future experiments.
3. **Confirmed no regression** from refactors.
4. **Confirmed `compile(optimizer.step)` has no benefit** — document as deferred.

Phase 2 (fusion investigation) should use the W1 profile as its starting point.

---

## Phase 2 Fusion Investigation Results (2026-05-05)

Phase 2 spec: `docs/superpowers/specs/2026-05-05-phase2-fusion-investigation-design.md`.
Plan: `docs/superpowers/plans/2026-05-05-phase2-fusion-investigation-plan.md`.
Summary: `docs/perf/phase2-summary-2026-05-05.md`.

### Summary

Six work items evaluated in ~4 hours. **Zero fusions shipped. Zero regressions.**
Phase 1 already captured all attainable wins on this stack.

| WI | Target (% of Phase 1 wall) | Outcome |
|----|----------------------------|:--------|
| WI1 | `triton_poi_fused__to_copy_mul_transpose_view_{7,8}` (9.1%) | CLOSED — already optimal Inductor fusion of 5 ops (RoPE+cast+QKV gather). Memory-bandwidth-bound. |
| WI2 | `aten::add_` + `aten::copy_` (9.3%) | CLOSED — 67% add_ is autograd weight-grad accumulation; 90% copy_ is input H2D upload. |
| WI3 | `aten::embedding_dense_backward` (4.1%) | DEFERRED — already near 1.3 TB/s bandwidth limit; tied-embedding autograd fusion too risky. |
| WI4 | `Memset (Device)` (4.1%) | CLOSED — framework-internal (rocBLAS scratch, fused_adamw, GradScaler); no user-reachable source. |
| WI5 | `Memcpy HtoD` (4.0%) | CLOSED — all 4 H2D strategies slower than baseline on unified memory. |
| WI6 | Inductor fusion catalog | SHIPPED — 92 unique triton kernels documented (up to 24 ops fused per kernel). |

### Key findings

1. **Inductor fuses aggressively under `compile_zones`:** 92 triton kernels cover
   nearly every elementwise chain in the model. `mul` appears in 81 kernels, `add` in 33.
   Writing custom HIP kernels for patterns already in the catalog would yield no speedup.

2. **Unified-memory H2D wisdom is inverted on Strix Halo.** Prefetching, pinned memory, and
   non-blocking copies all REGRESS throughput by 1-2%. Current `pin_memory=False +
   non_blocking=True` is the local optimum.

3. **Gradient lifecycle alternatives regress.** `set_to_none=True` beats `set_to_none=False`
   and pre-allocated foreach_zero_ by ~2%. Caching allocator already handles this well.

4. **The 4.1% Memsets are invisible to user code.** Every `aten::zero_` reports 0 μs.
   The actual Memsets come from rocBLAS/fused_adamw/GradScaler internals.

5. **Post-Phase-1 the stack is memory-bandwidth-limited at nearly every hot op.**
   Each of the 5% profile entries is already near its shape's theoretical bandwidth ceiling.
   Further gains require CUDA graphs (Phase 3) or batch-size scaling, not more fusion.

### Baseline confirmation

Post-Phase-2 ablation matches Phase 1 (no regressions):

| Config | Batch=16 tok/s | Peak GB |
|--------|--------------:|--------:|
| **compile + HIP CE (best throughput)** | **14,708** | 4.83 |
| compile + HIP CE + RoPE + Chunked CE (best memory) | 14,152 | **3.89** |

### Permanent artifacts delivered

- `docs/perf/inductor-fusion-catalog.md` + `.json` — 92-kernel structured catalog.
- `docs/perf/wi{1,2,3,4,5}-*.md` — per-WI analyses with closure rationales.
- `docs/perf/kernel-bodies-c1.txt` — triton source for the 9.1% kernel.
- `docs/perf/phase2-summary-2026-05-05.md` — consolidated summary.

Reusable tooling:
- `scripts/dump_inductor_output.py`, `parse_inductor_cache.py`, `extract_kernel_body.py`.
- `scripts/profile_shape_calls.py` — shape-annotated per-op profile.
- `scripts/bench_h2d_strategies.py`, `bench_zero_grad.py` — ablation harnesses.

### Next step: Phase 3 (CUDA graphs through Parcae)

Phase 3 starts from confirmed baseline **14,708 tok/s** at `compile + HIP CE`.
Expected lift: 5-15% by eliminating HIP launch overhead via graph capture.

---

## Phase 3 Throughput Investigation Results (2026-05-05)

Phase 3 spec: `docs/superpowers/specs/2026-05-05-phase3-cuda-graphs-parcae-design.md`.
Plan (revised): `docs/superpowers/plans/2026-05-05-phase3-throughput-plan.md`.
Summary: `docs/perf/phase3-summary-2026-05-05.md`.

### Summary

Phase 3 pivoted from CUDA-graphs-only to broader throughput search after WI-A0
refuted the original spec's premise. Shipped: **`TORCH_COMPILE_MODE=max-autotune`**
for +5.17% OdinHalo steady-state throughput.

| WI | Target | Outcome |
|----|--------|:--------|
| WI-A0 | Investigate reduce-overhead claim | CLOSED — HIP graph capture fails silently; no benefit. STATUS row 113's 2.14 GB claim does NOT reproduce. |
| WI-A1/A2/A3 | Clone-at-boundaries / manual graphs / unrolled | CANCELLED by A0 findings (solving non-existent problem). |
| WI-B1 | Shape sweep | CLOSED — all shapes within ±3% noise band. |
| WI-B2 | Compile-per-iter / whole-model | CLOSED — marginal +1.3% below shipping gate. |
| **WI-B3** | **max-autotune** | **SHIPPED — +5.17%** verified via 200-step parity test. |
| WI-B4/B5 | aiter / rocBLAS audit | Not executed (user scope: OdinHalo only). |

### Key findings

1. **reduce-overhead on HIP silently fails graph capture.** The trainer's
   auto-fallback warning was based on a premise that doesn't match reality:
   loops don't cause aliasing, but HIP's CUDA-graph backend produces
   "empty graph" warnings and runs eagerly — net −1.8% throughput.

2. **Inductor fusion is already saturated (Phase 2 finding reaffirmed).**
   WI-B2 showed widening the compile scope either regresses (wider function
   hits Python-container graph breaks) or gives only +1.3% (whole-model).

3. **max-autotune finds real pointwise-kernel tile wins.** rocBLAS still wins
   over triton_mm for all matmuls. But autotune explores `num_warps`, `num_stages`,
   `BLOCK_M/N/K` settings for the 92 fused pointwise triton kernels identified
   in Phase 2 WI6, and finds faster configurations than default.

### Winner config

For OdinHalo production training:

```bash
TORCH_COMPILE_MODE=max-autotune python -m halo_training \
  --model models/odin_halo.py --class-name OdinHalo --compile \
  --dataset datasets/dolma-10b-odin32k.bin --epochs 1 \
  ... (other flags)
```

First-compile: ~2 min (one-time autotune search).
Warm-cache subsequent: ~9 s.

### Verification

200-step training run with same seed:

| Metric | compile_zones default | max-autotune | Delta |
|--------|---------------------:|-------------:|------:|
| Steady-state tok/s (steps 50-199) | 14,018 | **14,742** | **+5.17%** |
| Max |loss Δ| over 200 steps | — | — | 0.2085 (within fp16 noise) |
| Mean |loss Δ| | — | — | 0.0862 |
| Final loss (step 200) | 4.5037 | 4.6052 | 0.1015 |
| Peak memory | 5.28 GB | 5.23 GB | comparable |

### Code shipped

- `models/odin_halo.py::OdinHaloBase.compile_zones(mode=None)` — env-var threading
- `models/odin_halo.py::OdinHaloBase.compile_zones_friendly(mode=None)` — same
- `halo_training/trainer.py` — thread `TORCH_COMPILE_MODE` through compile_zones
  with TypeError fallback for other HALO models (old bare signature).
- Docs: AGENTS.md compile strategy section updated.

### Artifacts

- 5 per-WI analyses in `docs/perf/phase3-wi-*.md`
- 5 reusable benchmark scripts in `scripts/wi_*.py`
- Summary: `docs/perf/phase3-summary-2026-05-05.md`

---

## OdinFlat: Flat (Non-Looped) Variant + DDP Training (2026-05-05)

### Model

`models/odin_flat.py` — **OdinFlat**: 14-layer flat hybrid LM, **121.7M params** (all unique).
Same block internals as OdinHalo (HyPE conv gate, NoPE GQA with XSA, SwiGLU, factorized embed/head,
logit softcap 30). Difference: no weight sharing, no Parcae loop, no MoDA, no injection/skip/iter machinery.

Architecture: `FactorizedEmbed → [12 HyPEShortConvBlock + 2 NoPEGQABlock] → FactorizedLMHead`
- GQA at positions 6 and 13 (center + end, mirroring looped's center-of-iteration pattern)
- 14 forward passes per step (vs looped's 18 = 6 shared × 3 iters)
- All compile infrastructure identical to OdinHalo (`compile_zones(mode=...)`)

### Throughput comparison (500 steps, block=256, batch=16, max-autotune-no-cudagraphs)

| Model | Unique params | Eff. params | Steady-state tok/s | Memory | Delta |
|-------|-------------:|------------:|-------------------:|-------:|------:|
| **OdinFlat** | 121.7M | 121.7M | **19,400** | 6.1 GB | **+27.5%** |
| OdinHalo (looped) | 57.6M | ~157M | 15,220 | 5.9 GB | baseline |

Flat is faster because: fewer forward passes (14 vs 18), no iteration overhead (injection, iter_norm,
skip gates, MoDA depth_kv routing), better Inductor fusion (simple sequential loop vs Python control flow).

### DDP training (active run)

**Config:**
```bash
# Launch via scripts/launch_ddp.sh (Machine A orchestrates both)
TORCH_COMPILE_MODE=max-autotune-no-cudagraphs
torchrun --nnodes=2 --nproc_per_node=1
  --model models/odin_flat.py --class-name OdinFlat
  --dataset datasets/wikitext-103-odin32k.bin (123M tokens, 0.25 GB)
  --epochs 1 --block-size 256 --batch-size 16 --accum-steps 8
  --compile --no-muon --lr 8e-4 --backend gloo
```

- **Effective batch:** 16 × 8 × 2 nodes = **256 tokens/step × 256 block = 65,536 tokens/step**
- **Network:** Thunderbolt 4 (10.77.0.x), gloo backend
- **Optimizer:** AdamW (fused=True), cosine schedule, warmup=300 steps
- **Total steps:** 1,869 optimizer steps (1 epoch over 123M tokens at eff batch 256)
- **Checkpoints:** every 500 steps → `checkpoints/odin-flat-wikitext-ddp/`

**DDP training (complete — 2026-05-05):**

Two runs were done back-to-back due to mid-run SSH disconnect. The second run
resumed from `step_1000.pt` with a fresh optimizer, effectively giving OdinFlat
an extra ~1,000 steps of training beyond the original epoch schedule.

| Step | Loss | BPB | Aggregate tok/s | MFU | Per-node memory |
|-----:|-----:|----:|----------------:|----:|----------------:|
| 50 | 7.01 | 2.81 | 34,076 | 20.9% | 6.6 GB (fresh-opt shock) |
| 500 | 4.93 | 1.97 | 39,341 | 24.2% | 6.6 GB |
| 1000 | 4.69 | 1.88 | 39,269 | 24.1% | 6.6 GB |
| 1500 | 4.54 | 1.82 | 39,169 | 24.1% | 6.6 GB |
| **1869 (final)** | **4.47** | **1.79** | **39,110** | **24.0%** | **6.6 GB** |

```
Done: 1869 steps, 122,503,168 tokens in 3132s (39,110 tok/s), best loss=4.4698
```

**Aggregate steady-state: 39,110 tok/s** over full epoch (per-node: ~19,555 tok/s).
DDP scaling: 39,110 / (19,400 × 2) = **100.8%** — gloo/TB4 overhead negligible.

Checkpoints saved at steps 500, 1000, 1500, 1869 (each ~1.46 GB).
Clean exit, no grad spikes, no StabilityGuard rollbacks.

**Launch script fix (2026-05-05):** `scripts/launch_ddp.sh` now uses
`setsid nohup ... < /dev/null` to fully detach both ranks. Verified: `torchrun`
gets reparented to init (PPID=1) and survives SSH disconnect. First run died at
step 1250 when the local SSH session died during a Windows Update restart;
detached relaunch completed without further issue.

### OdinHalo (looped) DDP run on same config (2026-05-05)

Ran the same 1-epoch wikitext-103 DDP config for the looped OdinHalo
variant (57.6M unique / ~157M effective) using the now-parameterized
`launch_ddp.sh` (`MODEL=models/odin_halo.py CLASS=OdinHalo ...`).

```
Done: 1869 steps, 122,503,168 tokens in 4089s (29,957 tok/s), best loss=4.7121
```

| Metric | OdinFlat | OdinHalo | Delta |
|--------|--------:|--------:|------:|
| Aggregate tok/s | 39,110 | 29,957 | OdinFlat +30.6% |
| Final loss | **4.4698** | 4.7121 | OdinFlat −0.24 |
| Final BPB | 1.791 | 1.888 | OdinFlat −5.1% |
| Wall time | 52 min | 68 min | OdinFlat −31% |
| Per-node memory | 6.6 GB | 6.2 GB | OdinHalo −6.5% |
| Checkpoint size | 1.46 GB | 691 MB | OdinHalo −53% |
| MFU (raw) | 24.0% | 8.8% | — (see note) |

**MFU note:** The raw MFU formula uses parameter count, which under-credits
weight-sharing models. At effective-param count (157M for OdinHalo), effective
MFU is ~24%, matching OdinFlat.

**Loss gap narrowed during training:** +0.56 at step 500 → +0.24 at epoch end.
OdinHalo's weight sharing plays out as slower-but-steeper learning per token.
At 1× params in tokens this is still insufficient to overtake OdinFlat, but
at Chinchilla-optimal budgets (20×+) OdinHalo is expected to catch up or
surpass due to its implicit regularization.

**Loss trajectory:**
| Step | OdinFlat | OdinHalo | Gap |
|-----:|---------:|---------:|----:|
| 500 | 4.93 | 5.49 | +0.56 |
| 1000 | 4.69 | 4.99 | +0.30 |
| 1500 | 4.54 | 4.78 | +0.24 |
| 1869 | **4.47** | **4.71** | +0.24 |

Checkpoints at steps 500, 1000, 1500, 1869 saved under
`checkpoints/odin-halo-wikitext-ddp/` (691 MB each).

---

## DDP Config Sweep (2026-05-06)

Full sweep report: `docs/perf/ddp-sweep-2026-05-06.md`.
Raw data: `docs/perf/ddp-sweep-{a,b}-2026-05-06.jsonl`.

### Headline findings

| Sweep | Finding |
|-------|---------|
| **num_workers** ∈ {4, 8, 12, 14} | **Flat** — 19,585–19,664 tok/s (±0.4%). Dataloader is not the bottleneck. |
| **batch × accum** (12-config cartesian) | **batch dominates** (+17% from 8→32); accum marginal (+3%). Best: batch=32 (9.8 GB). |
| **block_size** ∈ {256, 512, 1024} | **block=512 wins +4%** (20,408 vs 19,599 at 256, 19,779 at 1024). |

### Default changes committed

- `launch_ddp.sh`: `BLOCK=256 → 512`, `NUM_WORKERS=4 → 12`
- `train_ddp.py` argparse: `--block-size` default 256 → 512, `--num-workers` default 4 → 12
- All other defaults unchanged (batch=16, accum=8, warmup=300, max_grad_norm=1.0)

### Context-dependent overrides documented in AGENTS.md

| Scenario | Override |
|---|---|
| Max throughput, memory rich | `BATCH=32 ACCUM=8` |
| Longer context | `BLOCK=1024` |
| Smoother gradients | `ACCUM=16` or `32` |
| Memory-constrained | `BLOCK=256 BATCH=8` |
| Resumed training | `LR=6e-4 MAX_GRAD_NORM=0.8 WARMUP_STEPS=500` |

### Key artifacts

- `scripts/sweep_runner.py` — reusable single-node sweep harness
- `scripts/sweep_configs_a.json`, `scripts/sweep_configs_b.json` — config lists

### Methodology

- Parallel sweeps: Machine A (12 configs, 76 min) + Machine B (7 configs, 37 min).
  Machine A had the batch × accum cartesian (heavier workload per config);
  Machine B had num_workers + block_size (lighter but slower for block=1024).
- Single-node measurements (halo_training CLI, not DDP). Does not capture
  allreduce effects; DDP should see equal-or-greater accum_steps benefit.
- 100 steps/config with `TORCH_COMPILE_MODE=max-autotune-no-cudagraphs`.
  Measurement window: steps 30–100 (8 log_interval=10 samples).
- Model: OdinFlat. Dataset: wikitext-103-odin32k.bin.

---

## Continued Training Pipeline (2026-05-05 → 2026-05-06)

Both OdinFlat and OdinHalo are on a shared "resume-and-extend" training trajectory,
each chaining from wikitext → gpt-training-small → stem-crawl-solo via `--resume-from`.
Each resume loads weights only with a fresh optimizer (brief loss spike during warmup
that recovers by ~step 500).

### OdinFlat trajectory

| Phase | Dataset | Tokens | Final loss | Wall (DDP) | Checkpoint dir |
|-------|---------|-------:|-----------:|-----------:|----------------|
| 1 | wikitext-103-odin32k | 123M × 1 ep | **4.47** (BPB 1.79) | 52 min | `odin-flat-wikitext-ddp/step_1869.pt` |
| 2 | gpt-training-small-odin32k (resumed) | 296M × 1 ep | **5.07** (BPB 2.03) | 2h 5min | `odin-flat-gpt-small-ddp/step_1128.pt` |
| 3 | stem-crawl-solo-odin32k (resumed, in progress) | 531M × 1 ep | — | ETA ~3.7 hr | `odin-flat-stem-crawl-ddp/` |

Cumulative tokens seen: **~950M** by end of Phase 3.

### OdinHalo trajectory

| Phase | Dataset | Tokens | Final loss | Wall (DDP) | Checkpoint dir |
|-------|---------|-------:|-----------:|-----------:|----------------|
| 1 | wikitext-103-odin32k | 123M × 1 ep | **4.71** (BPB 1.89) | 68 min | `odin-halo-wikitext-ddp/step_1869.pt` |
| 2 | gpt-training-small-odin32k (resumed) | 296M × 1 ep | **5.10** (BPB 2.04) | 2h 45min | `odin-halo-gpt-small-ddp/step_2257.pt` |

### Key observation: loss gap collapse

| Phase | OdinFlat loss | OdinHalo loss | Gap |
|-------|-------------:|-------------:|----:|
| Wikitext (1 epoch fresh) | 4.47 | 4.71 | **+0.24** |
| Gpt-small (1 epoch resumed) | 5.07 | 5.10 | **+0.03** |

Gap narrowed 87% after one additional epoch. Consistent with the weight-sharing
regularization hypothesis: OdinHalo's 3× param reuse pays off more as training
budget grows. At Chinchilla-optimal scale (~20× tokens/params), OdinHalo is
expected to catch or surpass OdinFlat on held-out loss. At current 1–4×
params-in-tokens, OdinFlat still wins absolute loss and is +31% faster throughput.

### Dataset availability (all odin-32k tokenized, on both machines)

| Dataset | Tokens | Size | Status |
|---------|-------:|-----:|--------|
| `babylm-odin32k.bin` | 17M | 34 MB | Machine A only (smoke tests) |
| `wikitext-103-odin32k.bin` | 123M | 246 MB | Both machines |
| `gpt-training-small-odin32k.bin` | 296M | 593 MB | Both machines |
| **`stem-crawl-solo-odin32k.bin`** | **531M** | **1.06 GB** | **Both machines** (new, tokenized 2026-05-06) |
| `dolma-10b-odin32k.bin` | 6.9B | 13.7 GB | Machine A only |

Vidar-32k tokenized datasets exist but require retokenization to be used with
OdinHalo/OdinFlat (odin-32k and vidar-32k are different vocabularies).

### Sampling ablation findings (diagnostic)

Ran `scripts/ablate_odin_flat_sampling.py` on both OdinFlat and OdinHalo final
checkpoints for each trajectory phase. The winning sampling config tracks
training progress:

| Model/Phase | temp | rep_pen | top_p | top_k | dist2 | self-PPL |
|-------------|-----:|--------:|------:|------:|------:|---------:|
| OdinFlat on wikitext (step_1869) | 0.6 | 1.00 | 1.0 (off) | 0 (off) | 0.765 | 9.84 |
| OdinHalo on wikitext (step_1869) | 0.6 | 1.15 | 0.95 | 40 | 0.990 | 14.11 |
| OdinHalo on +gpt-small (step_2257) | 0.6 | 1.00 | 0.95 | 0 (off) | 0.699 | 11.33 |

**Diagnostic pattern:** well-trained models prefer unconstrained sampling
(no `rep_pen`, no `top_k` filtering); looser distributions benefit from tail
clipping. OdinHalo's sampling config loosened after gpt-small training
(`rep_pen: 1.15 → 1.00`, `top_k: 40 → 0`) — signal that the model's
distribution tightened, consistent with the loss trajectory.

### Active run: OdinFlat on stem-crawl-solo

Launched 2026-05-06 via `launch_ddp.sh` with new sweep-derived defaults (block=512,
num_workers=12) + resumed-training overrides (lr=6e-4, warmup=500, grad_clip=0.8).

```
Resume:  checkpoints/odin-flat-gpt-small-ddp/step_1128.pt
Config:  batch=16 × accum=8 × 2 nodes, block=512
         eff_batch=256 seqs × 512 tokens = 131,072 tok/step
Steps:   4,046 (1 epoch over 531M tokens)
Early:   step 50 loss=5.93, 37,384 tok/s aggregate, 10.3 GB/node
ETA:     ~3.7 hours total wall time
Cadence: checkpoint every 500 steps → ~9 intermediate saves
```

Monitor: `bash run_remote.sh "tail -3 checkpoints/odin-flat-stem-crawl-ddp/rank0.log"`

### Research deep-dive (2026-05-06)

Two synthesis documents:
1. `docs/research/small-lm-research-2026-05-06.md` — focused (SmolLM3 + APO recipe)
2. `docs/research/broad-research-synthesis-2026-05-06.md` — **broad** (240+ papers,
   2025-01 → 2026-04, `hf papers` search corpus)

**Top-5 recommendations (from broad synthesis):**
1. **NorMuon + Cautious Weight Decay** optimizer replacement (−3.85% loss vs AdamW, IMU-1 validated at 430M)
2. **Add value residuals + LayerNorm scaling + per-head gating** to OdinFlat blocks (IMU-1 recipe, additive gains)
3. **Intra-document attention masking + remove WD from embeddings** (free stability wins)
4. **Build minimal SFT + ORPO/APO + F-GRPO RLVR pipeline** (currently ZERO post-training infra)
5. **T²-optimal retraining on dolma-10b-odin32k** (6.9B tokens = 57× ratio, justifies overtraining per T² scaling laws)

**Most surprising new finding:** T² scaling (arXiv:2604.01411) shows that when
inference cost is included, optimal pretraining is **deep into the overtraining
regime**, beyond where Chinchilla suggests. Our ~7× ratio for OdinFlat is
deeply under-trained by this measure.

**Biggest gap:** Zero post-training infrastructure. Comprehensive survey in
`knowledge/training/instruct_alignment_techniques_2025_2026.md` is unused.

**Second biggest gap:** Single-metric (loss/BPB) evaluation. Missing per-domain
BPB, capability probes (BLiMP), sample-pack regression, quantized BPB.

**GRPO family worth noting:** F-GRPO (Focal-loss), Scaf-GRPO (scaffolding),
GRPO-SG (sharpness), f-GRPO (divergence-based), Apriel-Reasoner (difficulty-
aware length penalty). All released 2025-10 through 2026-04.

---


### Important: max-autotune vs max-autotune-no-cudagraphs

`max-autotune` crashes during trainer backward pass with gradient accumulation
(`accum_steps > 1`) due to CUDA graph buffer overwrite conflict:
```
RuntimeError: Error: accessing tensor output of CUDAGraphs that has been overwritten
by a subsequent run.
```

**Fix:** Use `TORCH_COMPILE_MODE=max-autotune-no-cudagraphs` for any training with
`accum_steps > 1`. Gets the same Triton kernel autotuning benefit without attempting
CUDA graph capture (which fails on HIP anyway per Phase 3 WI-A0).

The smoke test (no gradient accumulation) works fine with plain `max-autotune`.

### Artifacts

- `models/odin_flat.py` — OdinFlat, OdinFlatAblation, OdinFlatMini, NoPEGQABlock
- `scripts/launch_ddp.sh` — one-command DDP launcher (orchestrates both machines via SSH)
- `scripts/tcp_test.py` — TB4 connectivity diagnostic
- `datasets/wikitext-103-odin32k.bin` — 123M tokens pretokenized with odin-32k
- `docs/perf/flat-vs-looped-odin-2026-05-05.md` — throughput comparison report
- `checkpoints/odin-flat-wikitext-ddp/` — active training checkpoints

---



### Chunked CE extension to all halo models (session 2)

Extended `use_chunked_ce=True` ctor arg to: VidarHalo, FenrirHalo, TyrHalo, BaldrHalo, ChimeraHalo
(previously only OdinHalo). All 6 halo families + mini variants verified in
`scripts/test_all_models_chunked_train.py`.

### When to use Chunked CE — model-size dependent

Chunked CE overhead (16 chunks × 4 GEMMs = 64 launches per step) is fixed cost.
It becomes worthwhile only when:
- Rest-of-model compute dominates (larger models, more Parcae iterations)
- Memory headroom is tight (bigger batches, longer sequences)

**Measured at batch=16, block=256:**

| Model | PyTorch CE | Chunked CE | Δ tok/s | Δ mem | Recommend |
|-------|-----------:|-----------:|--------:|------:|:----------|
| **OdinHalo** (57M, 3 iter)  | 14,108 | 14,228 | +0.8% | -1.7 GB | ✓ use chunked |
| **VidarHalo** (47M, 2 iter) | 27,334 | 11,207 | **-59%** | -1.4 GB | ✗ skip chunked |

The rule: if your model is already GPU-saturated on compute (OdinHalo at bs=16),
chunked CE frees up memory at ~0% cost. If your model is memory-bound and the
CE path is proportionally large (VidarHalo), chunked CE's Python loop overhead
dominates the savings.

### Key takeaways
1. **For production training use batch=16 + compile + HIP CE (tiny)**.
2. **For memory-constrained setups** (larger batches, longer sequences): add
   `TORCH_COMPILE_MODE=reduce-overhead` for 50%+ memory savings at <2% throughput cost.
3. **For extreme memory savings** (1.67 GB): combine reduce-overhead + `--chunked-ce`.
4. **RoPE HIP fusion and RoPE Inductor fusion are functionally equivalent** when
   compile is on; no need to force HIP fusion under compile.
5. **Fused AdamW** (`torch.optim.AdamW(..., fused=True)`) is +12% at batch=4.
   Already default in `halo_training/optimizer.py`.
6. **Trainer now auto-uses `compile_zones`** for looped models when `--compile` set.
7. **`TORCH_COMPILE_MODE` env var** switches between default/reduce-overhead/max-autotune.

### Scripts (this session)
- `scripts/ablation_final.py` — comprehensive compile × kernel × batch_size
- `scripts/ablation_compile.py` — compile-friendly vs default compile
- `scripts/ablation_compile_modes.py` — mode={default, reduce-overhead, max-autotune}
- `scripts/ablation_optimizer.py` — fused AdamW + grad_clip + batch size
- `scripts/diag_compile.py`, `scripts/diag_compile_v2.py` — graph-break diagnosis
- `scripts/test_compile_friendly_parity.py` — output parity tests
- `scripts/trace_rope_in_block.py` — proved RoPE non-contig bug
- `scripts/mini_rope_repro.py` — isolated RoPE math verification
- `scripts/profile_step.py` — torch.profiler dump showing matmul=68%, AdamW=19%

---


## Active Model

**ODIN-HALO** (`models/odin_halo.py`, class `OdinHalo`) ← **NEW**
- 57.6M unique / ~156M effective params
- d=768, 6 shared layers (5 HyPEShortConv + 1 NoPE-GQA) × 3 Parcae iterations
- HyPE: NoPE attention (content-only) + RoPE on conv gate
- No momentum, iteration skip connections, logit softcap=30
- Tokenizer: `tokenizers/odin-32k/tokenizer.json` (EOS=0, PAD=1, vocab=32768)
- Dataset: `datasets/dolma-10b-odin32k.bin` (6.8B tokens, 13.7 GB, Machine A only)

**VIDAR-HALO** (`models/vidar_halo.py`, class `VidarHalo`)
- 47.0M unique / 95M effective params
- d=768, 4 shared layers (3 ShortConv + 1 MoDA-GQA) × 2 Parcae iterations
- No momentum (direct residuals) — 22% faster than momentum variants
- Tokenizer: `tokenizers/vidar-32k/tokenizer.json` (EOS=0, PAD=1, vocab=32000)
- Dataset: `datasets/dolma-10b-vidar32k.bin` (6.9B tokens, 13.8 GB, both machines)

**Variants:**
- `VidarHalo` — 32K vocab (production)
- `VidarHaloGPT2` — 50257 vocab (for GPT-2 .bin files like stem-crawl-solo.bin)
- `VidarHaloMini` — d=128 smoke test (~1.7M params)

**Secondary: FENRIR-HALO** (`models/fenrir_halo.py`, class `FenrirHalo`)
- 80.8M unique, d=640, 10 shared layers × Parcae mean=3
- Velocity clamp ±8.0 added (fix for NaN at step 30800)
- GPT-2 tokenizer (50257 vocab)

---

## Active Training Runs

| Machine | Model | Dataset | Config | Step | Loss | tok/s | Started | Status |
|---------|-------|---------|--------|------|------|-------|---------|--------|
| — | — | — | — | — | — | — | — | idle |

## Latest Training Run (2026-05-04)

VIDAR-HALO DDP, 32K tokenizer, compile + autokernel, AdamW, lr=0.001, warmup=300, stem-crawl-vidar32k:

| Step | Loss | BPB | tok/s (instantaneous) | Memory |
|------|------|-----|----------------------|--------|
| 400 | 8.50 | 3.41 | ~26,000 | 7.1 GB |

Pre-compile required: `python scripts/precompile_kernels.py --model models/vidar_halo.py --class-name VidarHalo` on each machine before DDP launch.

---

## Best Checkpoints (verified clean)

| Model | Path | Loss/BPB | Dataset | Tokens Seen | Notes |
|-------|------|----------|---------|-------------|-------|
| FenrirHalo | `fenrir_halo_babylm/step_2566.pt` | 3.16 / 1.26 | BabyLM (16M) | ~16M | Clean, well-converged |
| FenrirHalo | `fenrir_halo_dolma/step_15000.pt` | ~4.2 / ~1.7 | dolma-10b (7B) | ~500M | Last clean before NaN zone |
| TyrHaloLight | `tyr_light_e1_v3/step_8299.pt` | 5.08 / 2.10 | stem-crawl (544M) | 544M | Epoch 1 winner vs BALDR |
| BaldrHalo | `baldr_halo_e1_lr002/step_8299.pt` | 6.87 / 2.75 | stem-crawl (544M) | 544M | Lost to TyrHaloLight |
| VidarHalo | `vidar_smoke_32k/step_300.pt` | ~25.8 / ~10.4 | dolma-10b-vidar32k | ~10M | Smoke test only (300 steps, eager) |

---

## Corrupted / Unusable Checkpoints

| Path | Reason |
|------|--------|
| `fenrir_halo_dolma/step_25000.pt` | Trained without velocity clamp; NaN on resume |
| `fenrir_halo_dolma/step_30000.pt` | Same — weights damaged by unclamped velocity |
| `fenrir_halo_dolma/step_105000.pt` | Deep NaN; all steps after 30800 are corrupted |
| `fenrir_halo_dolma_r2/*` | Grad NaN at step 3200; short run, not useful |

---

## Known Issues & Hard Rules

1. **Always use `--optimize-kernels`** for real training. Without it: 10K tok/s. With it: 35K tok/s. The difference is autokernel fusing RMSNorm+SwiGLU+QKV.
2. **Always use .sh scripts** for remote commands: `run_remote.sh` (Machine A), `run_remote_b.sh` (Machine B). Never raw SSH.
3. **bf16 is NOT supported** on gfx1151. Always fp16 or fp32. bf16 is 24% slower, compile crashes.
4. **DDP smoke tests**: `--max-steps 300 --time-budget 20`. Never launch full epoch for smoke testing.
5. **Single-machine smoke tests**: `python -m halo_training ... --smoke` (200 steps built-in).
6. **Large .bin files use memmap** — no bulk np.fromfile for datasets >1GB. Both `data.py` and `train_ddp.py` use zero-copy memmap.
7. **FENRIR checkpoints after step_25000 are corrupted** — trained without velocity clamp. Use step_15000 or earlier.
8. **Custom tokenizer EOS=0** (not 50256). Check tokenizer config before hardcoding EOS.

---

## Throughput Reference

| Config | Model | tok/s | Notes |
|--------|-------|-------|-------|
| Single eager | VidarHalo (47M) | 18,929 | No compile, no autokernel |
| Single compiled (fwd+bwd only) | VidarHalo | 31,362 | No optimizer overhead |
| Single AdamW+compile | VidarHalo | ~16,800 | CE on vocab=50257 was bottleneck |
| DDP AdamW (no autokernel) | VidarHaloGPT2 | 34,541 global | 2 machines, TB4, vocab=50257 |
| DDP AdamW+compile, 32K tok | VidarHalo | ~26,000 global | 2 machines, TB4, no autokernel |
| DDP AdamW+compile+autokernel, 32K | VidarHalo | ~26,000 global | 2 machines, TB4, 7.1GB mem |
| Single compile+autokernel | VidarHalo | 7,100 | Per-machine instantaneous (isolated bench) |
| DDP Muon+compile+autokernel | FenrirHalo (80M) | ~26,000 global\* | Original dolma run (\*35K was inflated cumulative avg) |

**Note:** Prior 63K and 41K numbers were cumulative averages inflated by Inductor warmup ramp. Real instantaneous throughput is ~13K per machine, ~26K global DDP. Metric fixed to instantaneous in train_ddp.py.

### Ablation Throughput (VidarHaloAblation, d=768, 2L×2iter, 30M, single machine)

| Config | tok/s | Memory | Notes |
|--------|-------|--------|-------|
| bs=32 accum=2 eager (no AK) | **10,333** | 7.9 GB | **Use for Tier S** — zero startup cost |
| bs=32 accum=2 eager+AK | 7,200 | 16.9 GB | AK adds chunked CE (30+ min compile) |
| bs=32 accum=2 compiled+AK | 11,104 | 7.9 GB | +7% but 25+ min Inductor compile |

**Tier S config: eager, no compile, no AK. 10.3K tok/s, BabyLM 1ep ≈ 27 min.**

Key findings:
- MTP dropped: 45% throughput cost, no quality evidence at sub-100M scale
- torch.rms_norm replaces HIP RMSNorm: 15.5ms→2.1ms per call (7.5x)
- Depth-reduced (2L) not width-reduced (d=384): same GEMM shapes as production
- ~10K ceiling for d=768 single machine eager — bandwidth bound at 240 GB/s
- Compile/AK only worth it for runs >1 hour (Tier M/V) where compile cost amortizes

### First Ablation Results (Tier S, BabyLM 1ep, d=768 2L×2iter)

| Config | Final Loss | BPB | tok/s | Notes |
|--------|-----------|-----|-------|-------|
| Baseline | 6.73 | 2.698 | 7.2K | With AK (first run) |
| P1a (Polar-Express NS) | 6.72 | 2.693 | 7.2K | -0.01 loss (noise) |

**P1a verdict:** No meaningful gain at screening scale. Keep for Tier M test (may help more with longer training + larger matrices).

---

## Machine Info

| Machine | SSH | Venv | Project | TB4 IP | GPU |
|---------|-----|------|---------|--------|-----|
| A (rank 0) | `run_remote.sh` | `~/Desktop/ai_lab/.venv/` | `~/Desktop/ai_lab/autokernel-halo-strix/` | 10.77.0.1 | gfx1151 |
| B (rank 1) | `run_remote_b.sh` | `~/Desktop/comfyui-rocm7.12/.venv/` | `~/Desktop/comfyui-rocm7.12/autokernel-halo-strix/` | 10.77.0.2 | gfx1151 |

DDP: `GLOO_SOCKET_IFNAME=thunderbolt0`, `MASTER_ADDR=10.77.0.1`, backend=gloo.

---

*Last updated: 2026-05-05*

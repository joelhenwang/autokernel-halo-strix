# Training Evolution Pipeline: 5-Stage Funnel

**Date:** 2026-04-10
**Status:** Design approved
**Goal:** Systematically screen architecture hypotheses through increasingly expensive training stages, culminating in a 250M model that beats LFM2.5-350M on standard benchmarks, then instruction-tune for on-device Strix Halo deployment.

---

## Context

We have 22 architecture hypotheses, verified kernel optimizations (28% speedup via autokernel), and PLE/MatFormer ablation results. BabyLM (16M tokens) is too small to produce competitive models. We need a structured funnel that:

1. Eliminates bad architectures cheaply (minutes, not hours)
2. Validates promising ones on progressively larger/better datasets
3. Produces benchmark-competitive final models
4. Enables future instruction tuning for on-device assistant use

Key findings informing this design:
- torch.compile + autokernel = 43K tok/s on LlamaModel 124.7M (54% MFU)
- "Fast" threshold: >20K tok/s with compile+autokernel. <10K = bad architecture.
- EOS token bug fixed (documents now separated by `<|endoftext|>` token 50256)
- LFM2.5-350M uses weight tying + hybrid conv/attention, no embedding tricks needed
- MatFormer is free (+0.2% tok/s, negligible quality cost, free elastic inference)

---

## 5-Stage Funnel

```
Stage 0: Smoke (10 min, smoke-test-dataset)
  Gate: tok/s > 20K + loss decreasing (post-warmup)
  ↓
Stage 1: BabyLM (1 epoch, babylm-strict-small, ~16M tokens)
  Gate: Manual judgment on loss & BPB
  ↓
Stage 2: GPT-Training-Small (1 epoch + eval v1)
  Gate: Manual judgment on per-domain perplexity
  ↓
Stage 3: Dolma Mix 10B (2 epochs + eval v2)
  Gate: Manual judgment on benchmark scores
  ↓
Stage 4: Dolma Mix 100B (2 epochs + eval v2)
  → Final pretrained model → instruction tuning (Phase 2)
```

---

## Directory Structure

```
results/
  <model_name>/                       # e.g., "tempest", "virtuoso_ple_a"
    stage_0_smoke/
      metrics.json
      summary.md
      train.log
    stage_1_babylm/
      metrics.json
      training_curve.csv
      summary.md
      train.log
      checkpoint/
    stage_2_gpt_small/
      metrics.json
      training_curve.csv
      eval_v1.json
      summary.md
      train.log
      checkpoint/
    stage_3_dolma_10b/
      metrics.json
      training_curve.csv
      eval_v2.json
      summary.md
      train.log
      checkpoint/
    stage_4_dolma_100b/
      metrics.json
      training_curve.csv
      eval_v2.json
      summary.md
      train.log
      checkpoint/
```

---

## Stage Scripts

All scripts are Python, under `scripts/`. Each follows the same pattern:

```python
# Usage: python scripts/stage_X.py --model models/tempest.py --class-name Tempest \
#        [--resume results/tempest/stage_Y/checkpoint]

1. Parse args (model, class-name, resume)
2. Derive model_name from class-name (lowercase, e.g., "Tempest" -> "tempest")
3. Create results/<model_name>/stage_X/ directory
4. Load model (+ resume from checkpoint if provided)
5. torch.compile(model)
6. autokernel.optimize(model, training=True)
7. Train (dataset, epochs/time_budget, warmup_steps)
8. Save: metrics.json, training_curve.csv, summary.md, train.log, checkpoint/
9. Run eval if applicable (v1 for stage 2, v2 for stages 3-4)
10. Print summary + gate recommendation
```

### Script inventory

| Script | Dataset | Time/Epochs | Warmup | Eval | Checkpoints |
|--------|---------|-------------|--------|------|-------------|
| `scripts/stage_0_smoke.py` | smoke-test-dataset | 10 min | 50 steps | None | No |
| `scripts/stage_1_babylm.py` | babylm-strict-small | 1 epoch | 100 steps | None | Yes |
| `scripts/stage_2_gpt_small.py` | gpt-training-small | 1 epoch | 200 steps | v1 | Yes |
| `scripts/stage_3_dolma_10b.py` | Dolma Mix 10B | 2 epochs | 1,000 steps | v2 | Yes |
| `scripts/stage_4_dolma_100b.py` | Dolma Mix 100B | 2 epochs | 2,000 steps | v2 | Yes |

---

## Shared Training Config

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| batch_size | 16 | L2 cache sweet spot (verified) |
| accum_steps | 4 | Effective batch = 64 |
| lr | 8e-4 | Verified on AMADEUS and Tempest |
| block_size | 1024 | Matches all architecture max_seq_len |
| optimizer | AdamW(fused=True) | Best for <2B models |
| scheduler | Cosine annealing | Standard, with warmup |
| torch.compile | Yes (mode="default") | 2-3x MFU boost |
| autokernel | Yes (optimize_kernels=True) | 28% speedup verified |
| EOS tokens | Yes (token 50256 between docs) | Fixed in data.py |
| tokenizer | tiktoken GPT-2 (vocab=50257) | Standard |

---

## Gate Criteria

### Stage 0 Gate: Fast + Loss Decreasing

- **Fast:** tok/s > 20K (with torch.compile + autokernel)
- **Acceptable:** tok/s 10K-20K (needs optimization work before advancing)
- **Bad:** tok/s < 10K (bad architecture or not optimized — do not advance)
- **Loss decreasing:** `mean(loss[warmup_end : midpoint]) > mean(loss[midpoint : end])`
- Warmup steps excluded from the loss comparison

### Stages 1-4 Gates: Manual Judgment

You review `summary.md` + `metrics.json` (+ `eval_v1.json`/`eval_v2.json` where applicable) and decide whether to advance. No automated thresholds — your judgment on whether the model shows promise.

---

## Output Files

### metrics.json
```json
{
  "model_name": "tempest",
  "class_name": "Tempest",
  "params": 244500000,
  "stage": 0,
  "dataset": "smoke-test-dataset",
  "tok_s": 21500,
  "mfu": 0.42,
  "final_loss": 3.21,
  "bpb": 4.63,
  "steps": 200,
  "tokens_seen": 1638400,
  "epochs": null,
  "peak_memory_gb": 26.6,
  "warmup_steps": 50,
  "elapsed_s": 600,
  "gate_passed": true,
  "gate_reason": "tok/s=21500 > 20000, loss decreasing"
}
```

### training_curve.csv
```csv
step,loss,bpb,lr,grad_norm,tok_s,mfu,memory_gb
10,10.63,4.26,8.00e-05,2.94,21000,0.41,26.6
20,8.45,3.39,1.60e-04,3.09,21200,0.42,26.6
...
```

### summary.md
```markdown
# Stage 0: Smoke Test — Tempest

**Model:** Tempest (244.5M params)
**Dataset:** smoke-test-dataset (10 min budget)
**Config:** batch=16, accum=4, lr=8e-4, compile=True, autokernel=True

## Results
- **Throughput:** 21,500 tok/s (42% MFU)
- **Final loss:** 3.21 (BPB: 4.63)
- **Memory:** 26.6 GB peak
- **Steps:** 200 in 600s

## Loss Curve (post-warmup)
- First half avg: 6.82
- Last half avg: 3.95
- Decreasing: Yes

## Gate: PASS
- tok/s 21,500 > 20,000 threshold
- Loss decreasing after warmup

## Notes
(Manual observations go here)
```

### train.log
Full stdout/stderr captured during training.

---

## Eval Framework v1 (Stage 2+)

Cheap automated metrics. Output: `eval_v1.json`

| Metric | Method | Purpose |
|--------|--------|---------|
| Per-domain perplexity | Split dataset by source, compute BPB on each | Catch domain-specific degradation |
| n-gram repetition rate | Sample 100 generations, measure 4-gram repeat ratio | Detect doom-looping tendency |
| EOS prediction accuracy | Measure loss specifically on EOS token positions | Verify document boundary learning |

```json
{
  "per_domain_bpb": {
    "web": 5.12,
    "books": 4.87,
    "wikipedia": 4.23,
    "code": 6.45,
    "subtitles": 5.89
  },
  "repetition_4gram_rate": 0.023,
  "eos_accuracy": 0.87
}
```

---

## Eval Framework v2 (Stage 3+)

Full evaluation. Output: `eval_v2.json`. Includes everything from v1 plus:

| Metric | Method | Purpose |
|--------|--------|---------|
| HellaSwag (0-shot) | lm-evaluation-harness | Commonsense reasoning |
| ARC-Easy (0-shot) | lm-evaluation-harness | Science knowledge (easy) |
| ARC-Challenge (25-shot) | lm-evaluation-harness | Science knowledge (hard) |
| WinoGrande (5-shot) | lm-evaluation-harness | Coreference resolution |
| MMLU (5-shot) | lm-evaluation-harness | Broad knowledge |
| BLiMP | lm-evaluation-harness | Grammatical competence |
| Fixed prompt pack | 20-30 hand-picked prompts, saved generations | Qualitative regression |
| Quantized eval | int4 quantization, re-measure BPB + HellaSwag | Deployment readiness |

```json
{
  "v1_metrics": { "...": "..." },
  "benchmarks": {
    "hellaswag_0shot": 0.38,
    "arc_easy_0shot": 0.52,
    "arc_challenge_25shot": 0.28,
    "winogrande_5shot": 0.54,
    "mmlu_5shot": 0.26,
    "blimp": 0.72
  },
  "quantized": {
    "int4_bpb": 5.34,
    "int4_hellaswag": 0.35,
    "bpb_degradation_pct": 2.1
  },
  "prompt_pack": "results/<model>/stage_X/generations/"
}
```

### LFM2.5-350M Benchmark Targets (beat these)

| Benchmark | LFM2.5-350M Score | Source |
|-----------|-------------------|--------|
| HellaSwag | TBD (fetch from HF) | HuggingFace model card |
| ARC-Easy | TBD | |
| ARC-Challenge | TBD | |
| MMLU | TBD | |

These targets will be populated when we set up eval v2.

---

## Warmup Schedule

| Stage | Dataset Size | Warmup Steps | % of Training |
|-------|-------------|-------------|---------------|
| 0 | ~1M tokens | 50 | ~5% |
| 1 | ~16M tokens | 100 | ~2% |
| 2 | ~100M tokens (est.) | 200 | ~1% |
| 3 | 10B tokens | 1,000 | ~0.1% |
| 4 | 100B tokens | 2,000 | ~0.05% |

Warmup is linear ramp from 0 to base_lr. Warmup steps excluded from loss gate comparison (first-half vs last-half).

---

## Checkpoint Strategy

| Stage | Save Checkpoint? | When | Resume From |
|-------|-----------------|------|-------------|
| 0 | No | N/A | N/A |
| 1 | Yes | End of epoch | Fresh init |
| 2 | Yes | End of epoch | Stage 1 checkpoint (optional) or fresh |
| 3 | Yes | Every 5K steps + end | Stage 2 checkpoint (optional) or fresh |
| 4 | Yes | Every 10K steps + end | Stage 3 checkpoint (recommended) or fresh |

Intermediate checkpoints (stages 3-4) enable:
- Recovery from crashes during long training
- Eval at multiple points during training
- Selecting best checkpoint (not just final)

---

## Files to Create

| File | Purpose |
|------|---------|
| `scripts/stage_0_smoke.py` | 10-min smoke test with compile+autokernel |
| `scripts/stage_1_babylm.py` | 1 epoch BabyLM training |
| `scripts/stage_2_gpt_small.py` | 1 epoch GPT-training-small + eval v1 |
| `scripts/stage_3_dolma_10b.py` | 2 epoch Dolma 10B + eval v2 |
| `scripts/stage_4_dolma_100b.py` | 2 epoch Dolma 100B + eval v2 |
| `scripts/eval_v1.py` | Per-domain perplexity + repetition metrics |
| `scripts/eval_v2.py` | Full benchmark harness wrapper |
| `scripts/funnel_summary.py` | Utility: print comparison table across all models/stages |

## Files to Modify

| File | Change |
|------|--------|
| `halo_training/data.py` | Already fixed: EOS tokens between documents |
| `halo_training/trainer.py` | Add training_curve.csv logging, return structured metrics dict |

---

## Verification

1. Run `stage_0_smoke.py` on Tempest with compile+autokernel — should hit >20K tok/s
2. Run `stage_0_smoke.py` on VirtuosoMatFormer — should also pass
3. Run `stage_1_babylm.py` on Tempest — verify metrics.json, training_curve.csv, summary.md, checkpoint/ all created
4. Verify EOS fix: sample tokenized data, confirm token 50256 appears between documents
5. Run `funnel_summary.py` — should produce comparison table even with partial results

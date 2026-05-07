---
title: "Self-Distillation Spec: OdinFlat → OdinHalo"
domain: specs
type: spec
status: design-ready
sprint: v3.1
tags: [distillation, odin, halo, self-distillation, fp16, gfx1151]
related:
  - ../plans/2026-05-07-v3-experiment-roadmap.md
  - ../../../knowledge/architectures/v3_speculative_directions_2026.md
  - ../../../knowledge/training/fp16_stability_gfx1151.md
---

# Self-Distillation Spec: OdinFlat → OdinHalo (2026-05-07)

## Status

**Design-ready. No code yet.** Spec for the v3.1 phase E1 experiment:
use the already-trained OdinFlat wikitext-103 checkpoint as a teacher
to improve the OdinHalo wikitext-103 student. The teacher-student pair
is maximally de-risked because both were trained on the same 123M
tokens with the same tokenizer, and the student is the weaker of the
two (5.1% BPB gap) with plenty of room to close.

Blocks on nothing. Can run in parallel with FrankenMoE-Loop v2
implementation.

## 1. Goal and scope

### Primary goal

Close **≥50% of the OdinFlat→OdinHalo BPB gap** on wikitext-103 at
matched 1-epoch training budget. OdinFlat finished at BPB 1.79;
OdinHalo at BPB 1.88. Target: OdinHalo-distilled at BPB ≤ **1.835**
(half the 0.09 gap).

### Secondary goal

Validate distillation as a v3 primitive. If wikitext-103 passes the
gate, the same machinery becomes the Sprint 3 tool for dolma-10B.

### Non-goals

- Multi-teacher distillation (ensemble or rotating).
- Intermediate-feature / hint matching (requires layer-correspondence
  which is hard between 14-layer flat and 6-shared-×-3-iter looped).
- SFT / instruction distillation (this is pretraining-only).
- Teacher fine-tuning during distillation (teacher is frozen).

## 2. Why this works on gfx1151

### Teacher footprint

- OdinFlat is 122M fp16 parameters ≈ 244 MB on disk, ~170 MB resident
  in fp16 GPU memory (some weight tying).
- Teacher runs in `no_grad` fp16 so no activation graph is retained;
  transient activation memory peaks around 1.5 GB during the
  concurrent student forward.
- At our current 6.2 GB per-rank OdinHalo steady state, absorbing
  ~1.7 GB of teacher overhead lands around 7.9 GB per rank —
  comfortably within our 32 GB unified-memory budget.

### Throughput cost

Expected 1.4–1.7× step time. Back-of-envelope:

- Teacher forward ≈ 0.55× student forward (122M flat vs 157M
  effective looped, flat is faster per token; OdinFlat benched at
  19.4K tok/s single-node vs OdinHalo 15.2K).
- No teacher backward.
- KL loss on `[B, T, V=32768]` logits chunked through the same path
  as `ChunkedLinearCrossEntropyLoss` ≈ 1.1× vs CE.
- Net ≈ 1 + 0.55 + 0.1 ≈ **1.65× step time**.

With NorMuon fp16 (17.5% tok/s win from Sprint 1.1) + `max-autotune-
no-cudagraphs` already in place, the budget allows this. Expected
distilled aggregate throughput ≈ 39,110 / 1.65 ≈ **23,700 tok/s**
(vs 29,957 baseline OdinHalo).

### fp16 considerations

- **Teacher logits in fp16**: safe for top-k path (bounded values
  after softmax); riskier for full-vocab KL because log-softmax over
  32K dims can accumulate in fp16. **Mitigation**: promote teacher
  log-probs to fp32 for the KL computation. Memory cost: 2× on a
  chunk's worth of log-probs only (chunk=512 tokens → ~16 MB fp32
  transient).
- **Student logits** go through the same `ChunkedLinearCrossEntropyLoss`
  path today (never materialize full `[N, V]` fp16). Distillation
  must preserve this.
- **Temperature scaling** (T > 1) softens the teacher distribution;
  compute `teacher_logits / T` in fp32 before softmax to avoid
  scale-induced underflow.

## 3. Three distillation variants

### V1 — Full-vocab logit KL (primary signal check)

Loss:

```
L_distill = T^2 * KL(softmax(teacher_logits / T) || softmax(student_logits / T))
L_total   = L_distill
```

where `T = 2.0` (standard Hinton distillation default). The `T^2`
factor restores the gradient magnitude that temperature scaling
removes.

- Throughput cost: highest of the three variants (~1.65× step time).
- Signal: richest (full teacher distribution).
- fp16 risk: temperature-scaled softmax over 32K tokens requires
  fp32 promotion for numerical safety.

### V2 — KL + CE mix (recommended default)

Loss:

```
L_total = α * L_distill + (1 - α) * L_CE_standard
```

with `α = 0.5`, `T = 2.0`. Standard-recipe balance: half the gradient
from teacher, half from ground-truth targets.

- Throughput cost: same as V1 (both forward paths active anyway).
- Signal: more stable than pure KL (CE grounds the student when the
  teacher has flat distributions).
- Literature precedent: most knowledge-distillation wins since 2015
  use a KL+CE mix; pure KL tends to drift.

### V3 — Top-k logit KL (throughput variant)

Loss:

```
topk_values, topk_idx = teacher_logits.topk(k=32, dim=-1)
teacher_probs = softmax(topk_values / T, dim=-1)
student_topk  = student_logits.gather(dim=-1, index=topk_idx)
student_log_probs = log_softmax(student_topk / T, dim=-1)
L_distill = T^2 * -(teacher_probs * student_log_probs).sum(-1).mean()
```

- Throughput cost: ~1.4× step time (cheaper: only k=32 values
  per position to gather and compute).
- Signal: sparser but captures the teacher's high-confidence modes.
- fp16 risk: minimal (small dimension, easy to promote).

### Recommended default

**V2 (KL + CE mix) as primary**, V3 as a throughput-constrained
fallback if V2 proves too slow, V1 as a research variant to run
once for comparison.

## 4. Implementation plan

### 4.1 New modules

#### `halo_training/distill.py`

```python
class TeacherWrapper(nn.Module):
    """Loads a checkpoint in fp16, freezes params, compiles forward.

    Args:
        checkpoint_path: path to a .pt state dict.
        model_factory: callable returning the model skeleton (matches the
            saved arch config).
        compile_mode: str or None. If None, reads TORCH_COMPILE_MODE env.
        top_k: if not None, forward returns top-k logits + indices instead
            of full logits (V3 path).

    Forward returns:
        - If top_k is None: logits [B, T, V] in fp16.
        - If top_k is set: (topk_logits, topk_indices) both [B, T, k].
    """
```

Implementation notes:

- Load checkpoint, strip `_orig_mod.` prefix if present (compile
  artifact), load into model skeleton.
- `model.eval()` + `for p in model.parameters(): p.requires_grad = False`.
- `model.half()` to force fp16.
- Wrap forward in `torch.inference_mode()` — no autograd graph at all.
- Optionally call `model.compile_zones(mode)` or
  `model.compile_zones_friendly(mode)` if the model supports it
  (OdinFlat does).
- **Critical**: forbid the wrapped model from using `use_chunked_ce`.
  Teacher must return full logits (or top-k); student can still use
  chunked CE for its own CE term.

#### `halo_training/distill_loss.py`

```python
class ChunkedDistillLoss:
    """Chunk-based KL + CE loss that never materializes [N, V] twice.

    Args:
        mode: one of "kl", "kl_ce", "topk".
        alpha: weight on KL for "kl_ce" (default 0.5).
        temperature: softmax temperature (default 2.0).
        top_k: only used in "topk" mode (default 32).
        chunk_size: tokens per chunk (default 512, matches AGENTS.md).
        logit_softcap: apply softcap to student logits? 30.0 for OdinHalo.
        z_loss_weight: optional z-loss on student (default 0.0).
        label_smoothing: only applies to the CE term (default 0.0).
        ignore_index: default -100.

    Forward inputs:
        student_hlow: [B*T, d_low] from student's FactorizedLMHead.forward_hlow
        student_lm_weight: [V, d_low] tied weight for student head
        teacher_output: logits [B*T, V] or (topk_values, topk_idx) tuple
        targets: [B*T] ground truth token ids (for CE term)

    Returns:
        dict with "loss", "kl_loss", "ce_loss" for logging.
    """
```

Implementation notes:

- Chunking mirrors `kernels.hip.chunked_linear_cross_entropy` pattern:
  iterate over token positions in chunks of 512, compute
  `student_logits_chunk = student_hlow_chunk @ student_lm_weight.T`
  per chunk, apply softcap, compute KL+CE against the chunk's
  teacher slice, accumulate scalar loss.
- For V1/V2 full KL, fp32 promotion: `teacher_lp = (teacher_logits /
  T).log_softmax(-1).float()`, same for student (on the chunk). KL
  sum then `* T^2`.
- For V3 top-k, no fp32 promotion needed; the reduction is over 32
  elements.
- Gradient only flows through `student_hlow` and tied `student_lm_weight`
  (via PyTorch autograd; teacher output is detached by virtue of
  `inference_mode`).
- z-loss term pass-through: `(student_logsumexp)^2 * z_weight`, added
  once per chunk.

#### `halo_training/cli.py` additions

```
--distill-teacher PATH       Path to teacher checkpoint (.pt)
--distill-teacher-model PATH Path to teacher model file
--distill-teacher-class STR  Teacher class name
--distill-mode {kl,kl_ce,topk}   (default: kl_ce)
--distill-alpha FLOAT        Weight on KL term (default: 0.5, kl_ce only)
--distill-temperature FLOAT  Softmax temperature (default: 2.0)
--distill-topk INT           Top-k for topk mode (default: 32)
```

All optional; when `--distill-teacher` is absent, trainer takes the
non-distilled path exactly as today.

#### `halo_training/trainer.py` additions

Minimal surface:

```python
# Init (once at trainer start, when distill flags present):
teacher = TeacherWrapper(...)
distill_loss_fn = ChunkedDistillLoss(...)

# Per-step (replaces the chunked_ce_fn branch when distill is active):
teacher_output = teacher(input_ids)           # no_grad, fp16
out = model(input_ids)                        # student
if distill_loss_fn is not None:
    loss_dict = distill_loss_fn(
        student_hlow=out, student_lm_weight=model.lm_head.weight,
        teacher_output=teacher_output, targets=targets,
    )
    loss = loss_dict["loss"]
    # log loss_dict for visibility
```

No changes to the student model. `use_chunked_ce=True` stays on.

### 4.2 Tests

`scripts/test_distill.py` — 6 tests:

1. **Loss parity vs manual reference.** 4 seq × 8 tokens, d=64, V=128,
   compute loss both via `ChunkedDistillLoss` and via a reference
   numpy implementation. Relative error < 1e-3.

2. **KL reduces to CE when teacher = one-hot ground truth.** Set
   teacher logits such that softmax produces the target token with
   probability 0.999. `mode=kl` loss should match `cross_entropy`
   loss on the same targets within fp16 noise.

3. **Top-k matches full KL at k→V.** With k = vocab_size, V3 should
   equal V1 on the same teacher + student within 1e-4.

4. **Chunked vs unchunked parity.** Same inputs through
   `ChunkedDistillLoss(chunk_size=512)` and a one-shot full-tensor
   reference should match within fp16 chunked-CE floor (1e-3).

5. **GradScaler compat.** Run 5 steps with fp16 AMP scaler active;
   verify no `inf`/`nan` emerges from the distill path specifically.

6. **No-teacher fallthrough.** `--distill-teacher` absent → trainer
   takes the exact same `ChunkedLinearCrossEntropyLoss` path as
   today (byte-identical loss trajectory on a fixed seed).

### 4.3 Smoke integration

`scripts/test_distill_smoke.py` — 50-step smoke test on babylm with
the V2 (kl_ce) path, verifies:
- No NaN
- Loss descent parity with non-distilled baseline within noise
- Teacher memory overhead within budget

Run locally + on Machine A before DDP launch.

## 5. Experimental protocol

### Stage 0 — infra smoke (1 day)

```bash
bash sync_remote.sh
bash run_remote.sh "cd ~/Desktop/ai_lab/autokernel-halo-strix && \
    python scripts/test_distill.py && \
    python scripts/test_distill_smoke.py"
```

Pass criteria: all 6 unit tests green, 50-step smoke descends without
NaN, memory overhead within 2 GB of baseline.

### Stage 1 — wikitext-103 A/B (3 days)

Three DDP runs, each 1 epoch on wikitext-103 (1869 opt steps, ~68
min + 65% distill overhead ≈ 112 min each):

| Run | Config | Checkpoints |
|---|---|---|
| **Control** | OdinHalo no distill (already exists) | `checkpoints/odin-halo-wikitext-ddp/step_1869.pt` |
| **V1** | `--distill-mode kl --distill-temperature 2.0` | `checkpoints/odin-halo-distill-v1/` |
| **V2** | `--distill-mode kl_ce --distill-alpha 0.5 --distill-temperature 2.0` | `checkpoints/odin-halo-distill-v2/` |
| **V3** | `--distill-mode topk --distill-topk 32 --distill-temperature 2.0` | `checkpoints/odin-halo-distill-v3/` |

Control already exists; only V1/V2/V3 need new runs. Total new
compute: ~5.6 hours DDP wall clock.

Launch each run with `--auto-eval` so scorecards auto-populate at
step 500, 1000, 1500, 1869.

All three runs use identical student config:

```
--model models/odin_halo.py --class-name OdinHalo
--dataset datasets/wikitext-103-odin32k.bin --epochs 1
--block-size 256 --batch-size 16 --accum-steps 8
--compile --no-muon --lr 8e-4 --backend gloo
--z-loss 1e-4 --attn-softcap 50.0 --activation-monitor
--max-grad-norm 0.8 --chunked-ce --auto-eval
```

plus the distill flags.

Teacher: `checkpoints/odin-flat-wikitext-ddp/step_1869.pt` —
already present on both machines.

### Stage 2 — evaluation

Scorecard comparison at step 1869:

| Metric | Control | V1 | V2 | V3 | Gate |
|---|---:|---:|---:|---:|---|
| wikitext_val BPB | 1.888 | ? | ? | ? | ≤ 1.835 (50% gap closure) |
| gpt_small_val BPB | ? | ? | ? | ? | ≤ control |
| stem_crawl_val BPB | ? | ? | ? | ? | ≤ control |
| dolma_val BPB | ? | ? | ? | ? | ≤ control |
| distinct_2 | 0.990 | ? | ? | ? | ≥ 0.85 (no mode collapse) |
| self_ppl | 14.12 | ? | ? | ? | ≤ control + 10% |
| tok/s aggregate | 29,957 | ? | ? | ? | ≥ 17,500 (≤1.7× cost) |

**Primary gate**: ≥1 variant achieves wikitext_val BPB ≤ 1.835 AND
the throughput gate AND no held-out domain regresses by >3%.

### Stage 3 — go/no-go decision

- **Go** → promote V2 (or winning variant) to Sprint 3 dolma-10B.
  Uses same teacher (OdinFlat wikitext-103 ckpt), target BPB gap
  closure on dolma's validation slice.
- **No-go** → document findings in a results doc, park with
  `kill_date: 2026-09-01`, consider (a) longer teacher training, (b)
  larger teacher (if we ever train an OdinFlat on dolma), (c) archive.

## 6. Risk register

| Risk | Severity | Likelihood | Mitigation |
|---|---:|---:|---|
| Teacher compiled-cache collision with student | Low | Med | Load teacher before student; separate compile scope. If collision, disable teacher compile and accept the 5% throughput hit. |
| Teacher fp16 logits underflow on rare tokens | Med | Low | Temperature T=2.0 + fp32 promotion of log-softmax in KL. V3 avoids the issue entirely. |
| DDP sync cost doubles (two forwards per step) | Med | Med | Teacher runs independently per rank (unified memory makes this cheap); no gradient sync on teacher. Expected <10% DDP overhead increase. |
| KL + CE mix destabilizes logit softcap | Med | Low | Apply softcap to student logits before both KL and CE terms. Test 2 validates KL=CE identity under this path. |
| Activation monitor picks up teacher's activations as "layer 0" | Low | Med | Explicitly scope activation hooks to `model` (student), not teacher. Already the default since hooks go on `model.shared_layers`. |
| Teacher was trained with different block_size than student run | Low | Low | Both are block=256 in our existing checkpoints. |
| Tokenizer mismatch | Low | Low | Both use vidar-32k (checked). Verify in Stage 0 smoke. |
| Student drift from teacher's fp16-specific artifacts | Low | Low | V2 (KL+CE mix) grounds the student on ground truth; drift bounded. |
| Memory runs out at B=16 | Med | Low | Fallback: accum 16 at B=8. Budget analysis suggests 7.9 GB peak, which is safe. |
| Throughput gate fails even with V3 | Med | Med | Accept and proceed to dolma validation anyway IF quality gain justifies; or ship V3 only. |
| Distilled model regresses on held-out domains (stem, dolma) | Med | Med | Primary gate includes no-regression clause; if regressed, archive and document. |

## 7. Deliverables

At completion of Stage 3:

1. **3 new modules** in `halo_training/`: `distill.py`, `distill_loss.py`,
   `trainer.py` edits + `cli.py` edits.
2. **1 test file**: `scripts/test_distill.py` + smoke.
3. **3 training-run directories**: `checkpoints/odin-halo-distill-{v1,v2,v3}/`.
4. **Scorecards** at step 500, 1000, 1500, 1869 for each variant.
5. **1 results doc**: `knowledge/training/self_distillation_results_2026.md`
   with the full table, decision rationale, recommended default.
6. **Updated STATUS.md entry** summarizing outcome.
7. **If GO**: `launch_ddp.sh` gains `DISTILL_TEACHER` env var, and
   AGENTS.md training-commands section adds a distillation example.

## 8. Timeline

| Stage | Duration | Gating artifact |
|---|---|---|
| Infra implementation (modules + tests) | 3–4 days | 6/6 tests green + smoke |
| Stage 0 (infra smoke on remote) | 1 day | Smoke passes on Machine A |
| Stage 1 (three DDP runs) | 3 days | All three runs complete; scorecards present |
| Stage 2 (evaluation + writeup) | 1 day | Results doc drafted |
| Stage 3 (decision + publish) | 1 day | STATUS.md updated, AGENTS.md updated |
| **Total** | **~2 weeks** | — |

## 9. Open decisions deferred to post-V2

- **EMA-self-distillation** (teacher = EMA of student, updated per
  step) — explicitly out of scope for E1. If V2 succeeds, consider
  as a follow-up experiment under phase v3.2.
- **Feature matching / hint matching** (intermediate layer
  distillation) — requires layer correspondence between flat and
  looped, non-trivial, deferred.
- **Teacher switching mid-training** — if OdinHalo ever surpasses
  OdinFlat, switch teacher — interesting but premature.
- **Distillation during the SFT stage** — when SFT infra lands,
  revisit this spec's infrastructure as a foundation.

## 10. Kill conditions

- **Stage 0 fail**: smoke memory exceeds 10 GB per rank, OR smoke
  throughput below 12K tok/s per node. Investigate; if not fixable
  in 2 days, abandon.
- **Stage 1 fail**: all three variants regress vs control on wikitext
  BPB. Rare but possible if student has already consumed the
  teacher's information content; archive.
- **Stage 2 partial pass**: one variant passes BPB but throughput gate
  fails. Ship that variant as opt-in `--distill-teacher` flag; do not
  promote to default.

## Related docs

- `knowledge/architectures/v3_speculative_directions_2026.md` — §E1
  catalogue entry (one-line pointer here).
- `docs/superpowers/plans/2026-05-07-v3-experiment-roadmap.md` —
  where this spec sits in phase v3.1.
- `knowledge/training/fp16_stability_gfx1151.md` — stability stack
  this spec must coexist with.
- `kernels/hip/chunked_linear_cross_entropy.py` — pattern for the
  `ChunkedDistillLoss` implementation.
- `STATUS.md` (2026-05-05 OdinFlat section) — source of the 1.79
  teacher baseline number.

---
title: "fp16 Stability on gfx1151: Prevention, Forensics, Response"
domain: training
type: reference
status: active
tags: [fp16, gradscaler, stability, nan, overflow, looped-models, gfx1151, strix-halo, dolma]
related:
  - imu1_recipe_2026.md
  - normuon_throughput_gfx1151.md
  - ../../CONSTRAINTS.md
  - ../../AGENTS.md
---

# fp16 Stability on gfx1151

Practical notes for surviving fp16 training on Strix Halo, focused on
**long-horizon** runs (multi-epoch, resumed, or dolma-scale) where
overflow becomes a real risk. Consolidates what `scripts/train_ddp.py`
has learned since Sprint 1; written after a 2-epoch OdinHalo dolma-10B
run produced NaN loss + NaN gradients mid-training.

## Why fp16 overflow happens on long runs

gfx1151 has no bf16 support that's stable for our workload (compile
crashes, 24% throughput hit — see CONSTRAINTS.md). fp16's dynamic range
is [-65504, 65504] for the largest magnitude. Four mechanisms push us
out of that range during long training:

1. **GradScaler runaway.** `torch.amp.GradScaler` halves its scale on
   overflow, doubles after `growth_interval` consecutive good steps.
   The default 2000-step interval is too aggressive for long runs — over
   20K+ steps the scale can climb to 2^16 or higher, amplifying any
   outlier gradient beyond fp16 max during unscale.

2. **iter_scales drift in looped models.** OdinHalo / VidarHalo learn
   a per-Parcae-iteration scalar. Without a clamp, these can drift above
   ~5 during long training; multiplied across 3 iterations, activations
   grow geometrically (≈125×) per forward.

3. **Pre-softmax attention score overflow.** `Q @ K.T` at long context
   (T ≥ 512) with unbounded QK magnitudes can produce individual score
   elements exceeding 65504 before softmax normalizes. QK-norm mitigates
   but doesn't eliminate this.

4. **Weight magnitude creep.** As the model learns, `||W||` grows.
   Combined with large activations, matmul outputs start landing at fp16
   boundaries. A single outlier microbatch with 3σ-larger-than-usual
   features can trip the first NaN.

## Prevention (Ring 1; shipped 2026-05-07)

### `--z-loss <w>` auxiliary regularization

Applied when the model forward returns logits (dict path or bare tensor).
Adds `z_weight * logsumexp(logits)^2.mean()` to the main CE. This
penalizes drift of the log-partition function — equivalent to keeping
logit magnitudes tame.

Recommended defaults:

```
--z-loss 1e-4 --z-loss-fraction 0.4
```

Active across the first 40% of training, then tapered off. Throughput
cost <2%. For looped/long-horizon runs use this by default; for short
(<3K step) runs it's optional.

### `iter_scales.clamp(-4, 4)` in forward (OdinHalo, VidarHalo)

Source-level clamp in `_apply_iter_norm`. Parameter is *never* mutated —
only the value used for this forward is bounded. Checkpoint-compatible
both ways (old checkpoints load; new checkpoints work on old code via
the gradient not being affected by clamp).

### GradScaler `growth_interval=500` (was 2000)

Per `torch.amp.GradScaler`, scale doubles after this many consecutive
non-overflow steps. At 500 the scale recovers more slowly after a
backoff, staying bounded away from fp16 max.

### `--max-grad-norm` defaults to 0.8 on resumed runs

When `--resume-from <ckpt>` is set, the default grad-norm clip tightens
from 1.0 → 0.8. Resumed models have larger weight magnitudes so grads
are correspondingly larger in absolute terms. User-specified
`--max-grad-norm` always wins.

### `--attn-softcap <c>` pre-softmax score softcap

Applies `scores = c * tanh(scores / c)` before softmax in every
`Attention`, `CodaAttention`, and `NoPECodaAttention` instance. Default
off (`--attn-softcap 0.0`) uses `F.scaled_dot_product_attention` —
no regression.

When enabled, attention falls back to a manual path (~3% throughput
hit) with scores bounded to `[-c, c]`. Recommended `c=50.0` for
long-context looped runs; leaves normal attention patterns untouched
but caps any pathological outlier.

### `--activation-monitor` opt-in maxabs tracker

Registers forward hooks on each layer, samples `maxabs` every N steps
(default 100), writes JSONL to `$CKPT_DIR/activation_stats.jsonl`.

Schema:
```json
{"step": N, "layer": "layers.5", "maxabs": 1234.5, "fp16_headroom": 53.1,
 "dtype": "float16", "shape": [16, 512, 768]}
```

`fp16_headroom` = 65504 / maxabs. Values <2 indicate pending overflow;
values 5-50 are healthy. Use this to set a Phase-specific softcap or
retrain with different LR if a particular layer consistently runs hot.

Cost when off: zero. Cost when on: ~1 reduction per sampled step.

## Forensics (Ring 2 R1)

When `StabilityGuard` detects a NaN trigger, **before** rollback, we
save `$CKPT_DIR/nan_dump_step_{step}.pt` with:

```python
{
    "step": int,                      # optimizer step
    "global_step": int,
    "trigger": "nan_loss" | "loss_spike" | "param_nan_or_unknown" | "grad_skips",
    "loss_val": float,                # what tripped check_loss
    "microbatch_idx": int,            # position in the accum cycle
    "input_ids_cpu": Tensor,          # the offending batch — rerunnable
    "targets_cpu": Tensor,
    "doc_ids_cpu": Tensor | None,
    "scaler_state": {
        "scale": float,               # current scale at NaN
        "growth_tracker": int,        # steps since last backoff
        "growth_interval": int,
        "init_scale": float,
    },
    "weight_maxabs": {name: float},   # per-param max |w|
    "grad_norm_history": [(step, norm), ...],  # last 50 steps
    "activation_stats": {layer: {maxabs, dtype, shape}} | None,
    "consecutive_grad_skips": int,
}
```

Read back with:
```python
dump = torch.load("checkpoints/run/nan_dump_step_12345.pt",
                  weights_only=False)
print(dump["trigger"], dump["scaler_state"]["scale"])
sorted(dump["weight_maxabs"].items(), key=lambda x: -x[1])[:10]  # top-10 hot params
```

Diagnostic checklist:
- Is `scaler_state.scale` >= 16384? → scale runaway was proximate cause.
  Retrain with `--resume-from` + `--max-grad-norm 0.5` and fewer
  growth_interval (default 500 already).
- Is any `weight_maxabs` >= 100? → that param has drifted. Check whether
  it's in the 2D Muon group (usually) or a 1D scalar (usually a gate or
  scale).
- Is any `activation_stats.*.maxabs` close to 65504? → that layer was
  the overflow point. Consider `--attn-softcap 50.0` (if attention) or
  lowering LR.
- Is `grad_norm_history` showing a monotonic upward trend for ≥20
  steps? → training got into a bad curvature region. Rollback +
  LR-decay is the right move.

## Response (Ring 2 R3 + R5)

### StabilityGuard.rollback tightens GradScaler too

When rollback fires, the scaler's `growth_interval` is halved (floor
100). This makes post-rollback recovery more conservative. Together
with the LR-halve this reduces the chance of a second overflow at the
same operating point.

Rollback ladder:
1. LR halves (lr_decay_on_rollback=0.5)
2. Scaler `growth_interval` halves (floor 100)
3. Reload weights from last saved checkpoint
4. Clear loss EMA
5. After 5 consecutive rollbacks, abort with `RuntimeError`

### scaler.scale in log

Every periodic log line now prints `scale=1.0e+03`. Also appended to
`train_log.jsonl` as `scaler_scale`. Warning emitted when scale exceeds
16384, debounced at one message per 1000 steps.

## Recommended flags for long-horizon runs

```bash
EXTRA_FLAGS="--z-loss 1e-4 --z-loss-fraction 0.4 \
             --attn-softcap 50.0 \
             --activation-monitor \
             --max-grad-norm 0.8" \
bash scripts/launch_ddp.sh
```

For *resumed* runs from an existing checkpoint, add `--resume-from
<path.pt>`. The grad-norm auto-tightening triggers automatically.

For *multi-epoch dolma-10B* specifically, the combination above is the
tested default. If NaN still occurs, the `nan_dump_step_N.pt` artifact
tells you which of the four overflow paths caused it, and you can
tighten the specific knob.

## What we explicitly do NOT have

- **No bf16 path.** bf16 would solve fp16 overflow by construction
  (same exponent range as fp32), but on gfx1151 / ROCm 7.12 it costs
  ~24% throughput and compile-crashes with `max-autotune`. Per
  CONSTRAINTS.md, fp16 only. Revisit only if the bf16 situation
  improves on the ROCm stack.

- **No data-pointer advance on rollback.** Rollback reloads weights but
  retries from the next batch in sequence. If the exact same offending
  batch is hit again, the guard fires again; after 5 rollbacks we abort.
  The forensics dump captures `input_ids_cpu` so we can inspect whether
  it was data-specific.

- **No automatic bf16 escalation** when rollbacks accumulate. If we had
  a bf16 path this would be the natural response at rollback #3+. Since
  we don't, the rollback ladder ends at abort.

## Related

- `knowledge/training/imu1_recipe_2026.md` — IMU-1 / NorMuon recipe,
  references this doc for fp16 stability around NorMuon updates.
- `knowledge/training/normuon_throughput_gfx1151.md` — NorMuon fp16 NS
  matmul path. Orthogonal to this doc but same "fp16 on gfx1151" theme.
- `CONSTRAINTS.md` §fp16 — the hardware-level constraint list.
- `AGENTS.md` §Training gotchas — quick-reference for flag defaults.
- `scripts/test_fp16_stability.py` — regression tests for every knob
  documented above.

# Flat vs Looped Odin: Throughput Comparison (2026-05-05)

## Summary

| Model | Unique Params | Effective Params | Layers | Steady-state tok/s | Memory | Delta |
|-------|-------------:|----------------:|-------:|-------------------:|-------:|------:|
| **OdinFlat** | 121.7M | 121.7M | 14 (12 conv + 2 GQA) | **19,400** | 6.1 GB | **+27.5%** |
| **OdinHalo (looped)** | 57.6M | ~157M | 6 shared × 3 iters = 18 | **15,220** | 5.9 GB | baseline |

**Flat model is 27.5% faster in raw tok/s** at identical batch/block/compile settings.

## Configuration

Both models run under identical conditions:
- `TORCH_COMPILE_MODE=max-autotune-no-cudagraphs`
- `--compile --block-size 256 --max-steps 500 --epochs 1 --dataset babylm`
- Batch=16, accum=4, effective batch=64
- Machine A (gfx1151, ROCm 7.12)
- Throughput measured at steady-state (steps 200-500, after compilation stabilizes)

## Throughput trajectory

### OdinFlat (121.7M)
```
[step  100] tok/s=15,810  (still warming)
[step  200] tok/s=19,475  (stabilized)
[step  300] tok/s=19,417
[step  400] tok/s=19,405
[step  500] tok/s=19,386
```
Steady-state: **~19,400 tok/s** (±0.5% variance steps 200-500)

### OdinHalo looped (57.6M unique / ~157M effective)
```
[step  100] tok/s=12,196  (still warming)
[step  200] tok/s=15,235  (stabilized)
[step  300] tok/s=15,223
[step  400] tok/s=15,229
[step  500] tok/s=15,212
```
Steady-state: **~15,220 tok/s** (±0.1% variance steps 200-500)

## Why flat is faster

1. **Fewer forward passes per step**: 14 layer calls (flat) vs 18 (6 shared × 3 iterations). That's 22% less compute, accounting for most of the 27.5% throughput delta.

2. **No iteration overhead**: The looped model pays per-iteration costs:
   - `SimpleParcaeInjection` (linear projection + gate)
   - `iter_norm` (RMSNorm + scale + position embedding addition)
   - Skip gates (sigmoid + gated addition)
   - MoDA depth_kv_proj + depth_kvs routing in GQA block

3. **Better Inductor fusion opportunities**: Flat model's forward is a simple sequential loop over 14 independent layers. The looped model's `_forward_unrolled` has Python control flow (for-loop with list manipulation for depth_kv_buffer) that Inductor must trace through.

## Why looped may still win at scale

- **3× parameter efficiency**: 57.6M unique params achieve 157M effective compute depth. At fixed memory budget (e.g., GPU VRAM limit), looped models fit more effective capacity.
- **Regularization effect**: Weight sharing across iterations acts as implicit regularization — potentially better generalization per parameter.
- **MoDA cross-iteration attention**: Depth KVs enable information flow between iterations that flat models lack (each flat layer only sees its own output).

## Architecture diff

| Feature | OdinFlat | OdinHalo (looped) |
|---------|----------|-------------------|
| Layer count | 14 unique | 6 shared × 3 iters |
| Total forward passes | 14 | 18 |
| Weight sharing | None | Full (6 layers reused) |
| Parcae injection | No | Yes (re-injects input embed) |
| Iteration norm/scales | No | Yes |
| Loop position embeds | No | Yes |
| Skip gates | No | Yes (sigmoid-gated residual) |
| MoDA depth KVs | No | Yes (cross-iter attention) |
| Logit softcap | Yes (30) | Yes (30) |
| NoPE GQA (XSA) | Yes | Yes |
| HyPE (RoPE conv gate) | Yes | Yes |
| Factorized embed/head | Yes | Yes |
| SwiGLU FFN | Yes | Yes |

## Memory comparison

| Model | Peak VRAM | Optimizer state (est.) |
|-------|----------:|----------------------:|
| OdinFlat (121.7M) | 6.1 GB | ~974M floats (2× params × 4 bytes) |
| OdinHalo (57.6M) | 5.9 GB | ~461M floats |

Flat uses 3.4% more VRAM at runtime but 2.1× more optimizer state memory. For larger models or tighter GPU budgets, the looped architecture's memory efficiency becomes decisive.

## Smoke test results (200 steps, block=512, max-autotune)

| Model | Loss decrease | NaN/Inf | Memory | Throughput (200-step avg) |
|-------|:-------------|:--------|:-------|:--------------------------|
| OdinFlat | 10.26 → 4.20 PASS | 0 PASS | 9.2 GB PASS | 18,488 tok/s PASS |
| OdinHalo | 10.40 → 4.89 PASS | 0 PASS | 10.2 GB PASS | 12,719 tok/s (incl. warmup) |

Note: 200-step smoke tests include compilation warmup in the average, which suppresses throughput numbers below steady-state. 500-step runs above show true steady-state.

## Conclusion

For throughput-sensitive workloads where model size can accommodate unique parameters, OdinFlat delivers **+27.5% tok/s**. For memory-constrained or parameter-efficient scenarios, OdinHalo's 3× parameter reuse is the better architecture. The choice depends on the constraint:

- **Throughput-bound** (want max tok/s, have VRAM headroom): OdinFlat
- **Memory-bound** (want max effective capacity per GB): OdinHalo
- **Generalization-bound** (want best loss per training token): TBD (requires full-epoch comparison)

## DDP Training Results (2-node Thunderbolt 4)

Both OdinFlat and OdinHalo were trained for one full epoch on wikitext-103
(~123M tokens) over 2 Strix Halo machines via TB4 gloo, using identical config:

```
Config: batch=16 x accum=8 x 2 nodes = 256 effective batch
        block=256, lr=8e-4, cosine schedule, warmup=300
        TORCH_COMPILE_MODE=max-autotune-no-cudagraphs
        Backend: gloo over thunderbolt0 (10.77.0.x)
        Optimizer: fused AdamW (no Muon)
```

### Side-by-side comparison

| Metric | OdinFlat | OdinHalo | OdinFlat advantage |
|--------|--------:|--------:|:--:|
| Params (unique) | 121.7M | 57.6M | +2.11x |
| Params (effective) | 121.7M | ~157M | −23% |
| Layer forward passes per step | 14 | 18 (6 shared x 3 iters) | −22% compute |
| **Aggregate tok/s** | **39,110** | 29,957 | **+30.6%** |
| Per-node tok/s | ~19,555 | ~14,979 | +30.5% |
| **Final loss** | **4.4698** | 4.7121 | **−0.24 lower** |
| Final BPB | 1.791 | 1.888 | −5.1% |
| Checkpoint size | 1.46 GB | 691 MB | +2.11x (tracks params) |
| Per-node memory | 6.6 GB | 6.2 GB | +6.5% |
| Wall time (1 epoch) | 52 min | 68 min | −31% |
| MFU | 24.0% | 8.8% | +15.2 pts |
| DDP scaling efficiency | 100.8% | 99.0% | comparable |

### Loss trajectory comparison

| Step | OdinFlat loss | OdinHalo loss | Gap |
|-----:|-------------:|-------------:|----:|
| 500 | 4.93 | 5.49 | +0.56 |
| 1000 | 4.69 | 4.99 | +0.30 |
| 1500 | 4.54 | 4.78 | +0.24 |
| **1869 (final)** | **4.47** | **4.71** | **+0.24** |

Gap narrowed from +0.56 at step 500 to +0.24 by epoch end — OdinHalo's learning
rate is marginally higher in the middle of training (steeper descent per step)
but OdinFlat's 2.11x unique-parameter advantage still dominates at this training
budget (~1x params in tokens, well below Chinchilla-optimal 20x).

### MFU difference explained

OdinFlat achieves 24% MFU vs OdinHalo's 8.8%, despite OdinHalo doing MORE total
compute per step (18 layer passes vs 14). The MFU formula is:

  `MFU = (6 * n_params * tokens) / (time * peak_flops)`

Where `n_params` is the **parameter count**, not the compute count. Since MFU
credits each parameter as if it were used once per step, weight-sharing models
like OdinHalo are systematically underrated: the 6 shared layers are counted
6x in compute but only 1x in `n_params`. Adjusted MFU (using effective params):
OdinHalo at ~157M effective params → ~24% MFU. Apples-to-apples.

### DDP setup notes

- `scripts/launch_ddp.sh` orchestrates both machines from Machine A via SSH
- **Parameterizable**: `MODEL=models/odin_halo.py CLASS=OdinHalo \
  CKPT_DIR=checkpoints/odin-halo-wikitext-ddp bash scripts/launch_ddp.sh`
- **Fully detached** via `setsid nohup ... < /dev/null`; torchrun PPID=1 after launch
- Training survives SSH disconnect
- TB4 TCP connectivity reliable for the rendezvous window
- `LD_PRELOAD` of custom RCCL is unnecessary for gloo backend (warning can be ignored)
- Machine B uses `~/Desktop/comfyui-rocm7.12/autokernel-halo-strix/` (different venv path)

### Practical guidance

**Choose OdinFlat when:**
- Throughput matters (30%+ faster)
- Have VRAM headroom for 2.11x larger checkpoint + optimizer state
- Training budget is limited (≤ 1x params in tokens) — gets better loss faster
- Shipping inference at scale (fewer sequential layer calls = lower latency)

**Choose OdinHalo when:**
- Memory-constrained (smaller checkpoints, less optimizer state)
- Training budget is Chinchilla-optimal or larger (20x+ tokens/params) — expected
  to surpass OdinFlat at high compute budgets where parameter sharing acts as
  effective regularization
- Want to probe emergent depth capabilities (18 effective layers with shared weights)

## Extended training: gpt-training-small (296M tokens, resumed from wikitext)

Ran both models for 1 additional epoch on gpt-training-small, resuming from
their respective wikitext-trained step_1869 checkpoints. Same DDP / TB4 /
gloo infrastructure, matched config where possible.

| Metric | OdinFlat | OdinHalo |
|--------|--------:|--------:|
| Resume from | wikitext step_1869 (loss 4.47) | wikitext step_1869 (loss 4.71) |
| Block size | 512 | 512 |
| Effective batch (sequences) | 512 (16x16x2) | 256 (16x8x2) |
| Optimizer steps | 1,128 | 2,257 |
| Warmup | 500 | 500 |
| Peak LR | 6e-4 | 6e-4 |
| **Final loss on gpt-small** | **5.0677** | **5.0977** |
| Final BPB | 2.031 | 2.043 |
| Aggregate tok/s | 39,164 | 29,898 |
| Wall time | 2h 5min | 2h 45min |
| Per-node memory | ~10 GB | 10.8 GB |
| Checkpoint size | 1.46 GB | 691 MB |

**Loss gap has narrowed dramatically from the wikitext run:**

| Dataset | OdinFlat loss | OdinHalo loss | Gap |
|---------|-------------:|-------------:|----:|
| wikitext (1 epoch fresh) | 4.47 | 4.71 | **+0.24** |
| gpt-small (1 epoch resumed) | 5.07 | 5.10 | **+0.03** |

Gap collapsed from 0.24 → 0.03 (87% reduction) after one additional epoch on
different data. Consistent with the weight-sharing regularization hypothesis:
OdinHalo's 3× parameter reuse acts as implicit regularization that pays off more
at higher total training budgets. At extrapolated Chinchilla-optimal scale
(~20× tokens/params), OdinHalo may surpass OdinFlat on held-out loss.

Throughput gap unchanged: OdinFlat remains +31% faster on the same hardware.
For throughput-sensitive workloads, flat is still the pragmatic choice.

## Inference Sampling Ablation (step_1869 checkpoints)

Ran the 3-stage narrowing sweep `scripts/ablate_odin_flat_sampling.py` on both
checkpoints with identical protocol: 2 prompts × 3 samples × 15 configs.
Metrics: distinct-2 (higher = more diverse), self-PPL (lower = more confident).

### Winning configurations

| Parameter | OdinFlat | OdinHalo |
|-----------|:--------:|:--------:|
| `temperature` | 0.60 | 0.60 |
| `repetition_penalty` | **1.00** | **1.15** |
| `top_p` | **1.00** (disabled) | **0.95** |
| `top_k` | **0** (disabled) | **40** |
| distinct-2 | 0.765 | 0.990 |
| self-PPL | 9.84 | 14.11 |

### Key finding: OdinHalo needs MORE sampling constraints

OdinFlat converged to unconstrained sampling (no top_p, no top_k, no rep_pen) —
its distribution is tight enough that filtering hurts diversity without
improving quality. OdinHalo needed all three knobs active:

- **rep_pen=1.15** — OdinHalo repeats more at rep_pen=1.0 (dist2=0.797); a
  1.15 penalty pushes dist2 up to 0.990 at modest PPL cost (11.6 → 14.1)
- **top_p=0.95 + top_k=40** — OdinHalo's logit tail contains more low-quality
  tokens; restricting to the top 40 / 95th percentile prevents sampling them

This matches the loss gap: OdinHalo's final loss is 4.71 vs OdinFlat's 4.47.
The 0.24 loss difference manifests as a less-confident output distribution,
which benefits from tail clipping.

### Sample quality

Both produce Wikipedia-style English fragments with topical drift within ~30
tokens, characteristic of a ~1× params-in-tokens training budget (well below
Chinchilla-optimal ~20×).

**OdinFlat sample (temp=0.6, prompt="The history of the Roman Empire"):**
> "The history of the Roman Empire, by the British Revolution. The first Roman
> Empire of the King John of the province was the pope @-@ century, with the
> monarchy led by the throne of the region of the French and the Netherlands
> in 1630s, the Commonwealth of the reign of the Welsh and the Empire..."

**OdinHalo sample (same prompt, its winning config):**
> "The history of the Roman Empire, and in the Roman Empire. The Roman Empire's
> reign is a large role was the Holy Roman Catholic Church at the Kingdom and
> the first to be seen by a Romanian, as well known as the Royalist or a
> 'very good manor.'"

OdinHalo tends to emit `<|endoftext|>` earlier (shorter generations) while
OdinFlat keeps generating through 120 tokens. This tracks with OdinHalo's
tendency to overpredict EOS in the wikitext corpus (short article fragments).

### Diagnostic value

The inference ablation is a sensitive detector of model quality at matched
training budgets: the winning sampling config reveals the distribution shape,
not just its mean. OdinFlat's "unconstrained sampling wins" is the signature
of a well-trained-for-its-budget model. OdinHalo's "needs all filters on" is
the signature of a looser distribution that benefits from restriction.

## Artifacts

- `models/odin_flat.py` — OdinFlat model implementation
- `scripts/launch_ddp.sh` — one-command DDP launcher over TB4
- `scripts/tcp_test.py` — TB4 connectivity diagnostic
- `datasets/wikitext-103-odin32k.bin` — pretokenized wikitext-103 (odin-32k vocab)
- `checkpoints/odin-flat-wikitext-ddp/` — training checkpoints + logs
- This report: `docs/perf/flat-vs-looped-odin-2026-05-05.md`

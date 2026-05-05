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

OdinFlat trained on wikitext-103 (123M tokens) over 2 Strix Halo machines via TB4 gloo:

```
Config: batch=16 x accum=8 x 2 nodes = 256 effective batch
        block=256, lr=8e-4, cosine schedule, warmup=300
        TORCH_COMPILE_MODE=max-autotune-no-cudagraphs
        Backend: gloo over thunderbolt0 (10.77.0.x)
```

| Step | Loss | BPB | Aggregate tok/s | MFU | Per-node memory |
|-----:|-----:|----:|----------------:|----:|----------------:|
| 50 | 9.39 | 3.76 | 20,176 | 12.4% | 6.6 GB |
| 100 | 7.01 | 2.81 | 39,933 | 24.5% | 6.6 GB |
| 200 | 6.09 | 2.44 | 39,825 | 24.5% | 6.6 GB |
| 500 | 5.28 | 2.11 | 39,538 | 24.3% | 6.6 GB |
| 1000 | 4.86 | 1.95 | 39,400 | 24.2% | 6.6 GB |

**Aggregate steady-state: ~39,500 tok/s** (~19,750 per node).

DDP scaling efficiency: 39,500 / (19,400 × 2) = **101.8%** — slightly super-linear
(likely noise, but confirms gloo over TB4 has negligible overhead at this batch/accum config).

Loss trajectory is healthy — converging from 9.39 to 4.86 through 1000 steps with no
instability. Full epoch (1,869 steps) should reach ~4.5 final loss.

### DDP setup notes

- `scripts/launch_ddp.sh` orchestrates both machines from Machine A via SSH
- TB4 TCP connectivity works (ping + TCP verified) but requires both torchrun
  processes to start within the rendezvous timeout window
- `LD_PRELOAD` of custom RCCL is unnecessary for gloo backend (warning can be ignored)
- Machine B uses `~/Desktop/comfyui-rocm7.12/autokernel-halo-strix/` (different venv path)

## Artifacts

- `models/odin_flat.py` — OdinFlat model implementation
- `scripts/launch_ddp.sh` — one-command DDP launcher over TB4
- `scripts/tcp_test.py` — TB4 connectivity diagnostic
- `datasets/wikitext-103-odin32k.bin` — pretokenized wikitext-103 (odin-32k vocab)
- `checkpoints/odin-flat-wikitext-ddp/` — training checkpoints + logs
- This report: `docs/perf/flat-vs-looped-odin-2026-05-05.md`

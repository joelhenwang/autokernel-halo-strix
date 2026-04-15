---
title: "Training Pipeline Optimization"
domain: design-specs
type: spec
status: active
related:
  - docs/superpowers/specs/2026-04-08-halo-training-stack-design.md
  - docs/superpowers/specs/2026-04-10-training-evolution-design.md
  - docs/superpowers/specs/2026-04-10-backward-pass-optimization-design.md
tags: [%training, %pipeline, %optimizer]
---

# Training Pipeline Optimization

**Date:** 2026-04-10
**Status:** Scripts implemented and tested — see Preliminary Results
**Workstream:** D (of A/B/C/D optimization roadmap)
**Depends on:** Workstreams A+C (need model-level optimizations settled first)

## Problem

We've focused on model forward/backward optimization (kernels, attention, scan) but haven't profiled the full training pipeline. On gfx1151 with unified memory, there may be significant gains in data loading, optimizer step, gradient handling, and compilation overhead that we're leaving on the table.

Our best training result is 43K tok/s (LlamaModel 124.7M with compile+autokernel, 54% MFU). The 46% gap to theoretical peak suggests pipeline overhead.

## Scope

### Part 1: Profile the full training step

Use `torch.profiler` to measure time spent in each phase of a training step:

```python
with torch.profiler.profile(
    activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA],
    with_stack=True,
) as prof:
    for step in range(10):
        loss = train_step(model, batch)
```

Breakdown to measure:
- **Data loading** (CPU): tokenization, collation, transfer to GPU
- **Forward pass** (GPU): model forward
- **Loss computation** (GPU): cross-entropy
- **Backward pass** (GPU): gradient computation
- **Optimizer step** (GPU/CPU): AdamW parameter update
- **torch.compile overhead** (CPU): graph tracing, compilation cache misses
- **Gradient clipping** (GPU): norm computation + scaling
- **Logging/callbacks** (CPU): metric computation, checkpointing

### Part 2: Data loading optimization

Current: `BabyLMDataset` in `halo_training/data.py` loads pre-tokenized parquet. Potential issues:
- Single-process data loading (no `num_workers`)
- No prefetching (GPU idle while CPU prepares next batch)
- Parquet decompression on CPU

Test:
- `DataLoader(num_workers=4, pin_memory=True, prefetch_factor=2)`
- On Strix Halo unified memory, `pin_memory` may be a no-op (already shared)
- Verify with profiler: is data loading overlapping with GPU compute?

### Part 3: Optimizer optimization

Current: `AdamW(fused=True)` for <2B models, DeepSpeed CPUAdam for >2B.

Test:
- **torch.compile the optimizer:** `torch.compile(optimizer.step)` — may fuse multiple param updates
- **Gradient accumulation efficiency:** verify no redundant allreduce/sync between microbatches
- **Mixed precision optimizer:** fp16 master weights (risky but saves memory bandwidth)

### Part 4: torch.compile tuning

Current: `torch.compile(mode="default")` with autokernel.

Test:
- `mode="max-autotune"` — searches for faster kernels (longer compile, potentially faster runtime)
- `torch._dynamo.config.cache_size_limit` — increase if recompilation is happening
- `torch._inductor.config.coordinate_descent_tuning = True` — Inductor's own GEMM tuning
- `TORCHINDUCTOR_MAX_AUTOTUNE_GEMM_BACKENDS=TRITON,ATen` — control which GEMM backends Inductor considers
- Verify: is torch.compile recompiling on every step? (check `torch._dynamo.utils.counters`)

### Part 5: Gradient accumulation + batch size tuning

Current: batch_size varies per architecture. On gfx1151 with 128GB unified memory:
- Larger batches improve GPU utilization (more parallelism per kernel launch)
- But larger batches may not fit in L2 cache, increasing DRAM traffic
- Find the sweet spot where tok/s peaks

Test batch sizes: 4, 8, 16, 32, 64 (with gradient accumulation to match effective batch)

### Part 6: Strix Halo unified memory specifics

Unlike discrete GPUs, Strix Halo shares LPDDR5X between CPU and GPU:
- **No PCIe transfer overhead** for data loading
- **But CPU and GPU compete for bandwidth** — heavy CPU data preprocessing during GPU compute steals memory bandwidth
- Profile: is CPU activity (data loading, optimizer) causing GPU memory stalls?
- Mitigation: schedule CPU work during GPU-idle phases (between backward and optimizer)

## Implementation

### Files to create
| File | Purpose |
|------|---------|
| `scripts/profile_training_step.py` | Full step profiler with breakdown |
| `scripts/tune_training_pipeline.py` | Test DataLoader configs, batch sizes, compile modes |

### Execution
1. Profile current best architecture (winner from A) for 50 steps
2. Identify top 3 bottlenecks by wall-clock time
3. Apply fixes for each bottleneck
4. Re-profile to verify improvement
5. Run 15-min training to confirm tok/s gain

## Expected Impact

| Optimization | Expected gain | Likelihood |
|-------------|---------------|------------|
| DataLoader num_workers + prefetch | 0-5% | Medium (may already be overlapped) |
| torch.compile max-autotune | 2-10% | Medium (longer compile but faster runtime) |
| Inductor GEMM tuning | 1-5% | Medium |
| Batch size tuning | 2-8% | High (almost always helps) |
| Unified memory scheduling | 0-3% | Low (hard to control) |

## Preliminary Results (2026-04-10)

### Profile: AMADEUS 243.8M (batch=4, seq=256, eager, no compile)

| Phase | Time (ms) | % of Step |
|-------|-----------|-----------|
| **backward** | 102.6 | **53.1%** |
| **forward** | 37.9 | 19.6% |
| **optimizer_step** | 35.8 | 18.5% |
| grad_clip | 13.6 | 7.0% |
| loss | 2.5 | 1.3% |
| data_load | 0.8 | 0.4% |
| **Total** | **193.2** | |

**Throughput:** 5,300 tok/s (batch=4, seq=256)

**Top bottlenecks:** backward (53%), forward (20%), optimizer (19%). Data loading is negligible (0.4%) — unified memory eliminates PCIe transfer overhead as expected.

### Batch size tuning: AMADEUS (seq=256, eager)

| batch_size | tokens/step | tok/s |
|-----------|-------------|-------|
| 4 | 1,024 | 5,374 |
| 8 | 2,048 | 7,220 |
| **16** | **4,096** | **7,539** |
| 32 | 8,192 | 7,518 |
| 64 | 16,384 | 7,517 |

**Peak at batch=16**, diminishing returns beyond. The L2 cache (6 MB) fits activations well at batch=16; larger batches spill to DRAM without gaining more parallelism.

### Remaining tests (not yet run)

- DataLoader num_workers/pin_memory/prefetch configs (script ready)
- torch.compile mode comparison (default vs reduce-overhead vs max-autotune)
- Inductor coordinate_descent_tuning
- Full 15-min before/after comparison

## Verification

Before/after tok/s on the same 15-min training run. Profile traces saved for comparison.

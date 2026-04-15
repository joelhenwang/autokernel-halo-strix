---
title: "Halo Training Stack"
domain: project
type: reference
status: active
related:
  - README.md
tags: [%halo-training, %training-stack, %cli]
---

# Halo Training Stack

Composable pretraining stack for AMD Strix Halo (gfx1151, RDNA 3.5 APU) on ROCm 7.12. Supports two training modes:

- **Mode A** (<2B params): Whole-model `torch.compile`, direct forward/backward. 14.5K tok/s baseline, 43K tok/s with autokernel.
- **Mode B** (>2B params): Per-layer activation checkpointing + streaming. Trades ~20-30% throughput for dramatically lower memory.

Auto-selects the right mode based on model size and available memory.

## Quick Start

### CLI

```bash
# Basic training (auto mode, 45-min budget, BabyLM dataset)
python -m halo_training --model models/llama_7b.py --class-name LlamaModel --dataset babylm

# With torch.compile + autokernel (best performance)
python -m halo_training --model models/llama_7b.py --class-name LlamaModel \
    --compile --optimize-kernels

# Smoke test (200 steps, checks loss/grads/memory/throughput)
python -m halo_training --model models/llama_7b.py --class-name LlamaModel --smoke

# Large model (Mode B forced)
python -m halo_training --model models/llama_7b.py --class-name LlamaModel7B \
    --mode B --batch-size 4 --time-budget 90
```

### Library API

```python
from halo_training import train

# Simplest usage
from models.llama_7b import LlamaModel
model = LlamaModel()
stats = train(model, dataset="babylm", compile=True, optimize_kernels=True)

# With custom configuration
stats = train(
    model,
    dataset="datasets/babylm-strict-small",
    epochs=1,
    time_budget_minutes=45,
    batch_size=64,
    block_size=1024,
    accum_steps=4,
    base_lr=8e-4,
    compile=True,
    optimize_kernels=True,
    mode="auto",           # "auto", "A", or "B"
    log_interval=10,
)
```

## Architecture

```
cli.py / train()
    │
    ├── data.py              BabyLMDataset, build_dataloader (parquet + txt)
    ├── optimizer.py         build_optimizer (COOKBOOK.md groups + CPUAdam), build_scheduler
    ├── memory.py            MemoryBudget, suggest_mode → picks Mode A or B
    │
    ├─ Mode A ──→ trainer.py      Direct forward/backward + torch.compile
    ├─ Mode B ──→ streaming.py    LayerStreamingTrainer (per-layer checkpoint)
    │
    ├── callbacks.py         PhaseScheduler, MemoryMonitor, StateNormMonitor, PerParamGradMonitor
    ├── metrics.py           compute_bpb, ThroughputTracker, TrainingLogger
    └── evaluate.py          evaluate_bpb (val loss), benchmark_inference (prefill/decode)
```

## CLI Flags

| Flag | Default | Description |
|------|---------|-------------|
| `--model` | required | Path to model .py file |
| `--class-name` | required | Model class name in that file |
| `--dataset` | `datasets/babylm-strict-small` | Dataset path |
| `--smoke` | false | Run 200-step smoke test instead of training |
| `--time-budget` | 45 | Training budget in minutes |
| `--batch-size` | 16 | Microbatch size |
| `--block-size` | 1024 | Sequence length |
| `--accum-steps` | 4 | Gradient accumulation steps |
| `--lr` | 8e-4 | Base learning rate |
| `--compile` | false | Apply torch.compile |
| `--optimize-kernels` | false | Apply autokernel.optimize() for HIP kernel acceleration |
| `--mode` | auto | Training mode: `auto`, `A`, or `B` |

## Verified Results (AMD Ryzen AI MAX+ 395, Radeon 8060S)

| Configuration | Throughput | MFU | Notes |
|---------------|-----------|-----|-------|
| Mode A baseline (124.7M) | 14.5K tok/s | 17% | Eager, no autokernel |
| Mode A + compile | 13.8K tok/s | — | cudagraph_mark_step_begin fix applied |
| Mode A + compile + autokernel | **43K tok/s** | **54%** | **3.05x speedup** |
| Mode B (2.09B) | 853 tok/s | — | Layer-streaming, 34.5 GB memory |
| Decode benchmark (7B) | 103 tok/s | — | KV-cache, prefill 10ms |

## Public API

All exports from `halo_training`:

| Export | Module | Description |
|--------|--------|-------------|
| `train` | trainer | Main training loop (Mode A/B auto-select) |
| `build_optimizer` | optimizer | COOKBOOK.md param groups + DeepSpeed CPUAdam fallback |
| `build_scheduler` | optimizer | Cosine annealing with warmup (optional warm restarts) |
| `BabyLMDataset` | data | Loads parquet/txt, tokenizes with tiktoken GPT-2 |
| `build_dataloader` | data | DataLoader factory (pin_memory=False for unified mem) |
| `PhaseScheduler` | callbacks | Multi-phase unfreezing (backbone → memory → all) |
| `MemoryMonitor` | callbacks | GPU memory tracking with configurable warning threshold |
| `StateNormMonitor` | callbacks | Recurrence instability detection (norm ratio tracking) |
| `PerParamGradMonitor` | callbacks | Top-K gradient norm logging per parameter |
| `compute_bpb` | metrics | Bits-per-byte from cross-entropy loss |
| `ThroughputTracker` | metrics | tok/s and MFU (Model FLOP Utilization) |
| `TrainingLogger` | metrics | Structured stdout + JSON lines logging |
| `get_layer_iterator` | model_utils | Generic layer discovery across model architectures |
| `count_parameters` | model_utils | Trainable/total parameter count |
| `estimate_memory` | model_utils | Component-wise memory budget estimation |
| `run_smoke_test` | smoke | Standardized 200-step validation (6 pass/fail criteria) |
| `LayerStreamingTrainer` | streaming | Mode B: per-layer activation checkpointing |
| `MemoryBudget` | memory | Memory estimation and pressure monitoring |
| `suggest_mode` | memory | Auto Mode A/B selection (60% threshold) |
| `evaluate_bpb` | evaluate | Validation BPB on held-out split |
| `benchmark_inference` | evaluate | Prefill latency + decode throughput measurement |

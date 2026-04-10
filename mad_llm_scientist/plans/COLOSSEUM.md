# COLOSSEUM

**Unified Memory Training Arena — MegaTrain Techniques Adapted for Strix Halo**

## Hypothesis

MegaTrain (2604.05091) trains 100B+ models on single GPUs by streaming layers from CPU RAM through GPU. Their bottleneck: PCIe (32-128 GB/s). Our advantage: Strix Halo has UNIFIED MEMORY (128 GB (~116 GB GPU-visible), 240 GB/s). No PCIe. No streaming. Zero-copy access. We take their 3 best ideas and skip the 5 that exist only for PCIe.

**Works with ANY of our architectures.** It's infrastructure, not a model change.

**Source:** github.com/DLYuanGod/MegaTrain (Apache-2.0)

---

## What We Take vs Skip

| MegaTrain Technique | Take? | Strix Halo Rationale |
|---|---|---|
| CPU-side optimizer (AVX) | **YES** | Frees GPU memory for larger batches |
| Block-wise activation recomputation | **YES** | Saves activation memory, bounded by checkpoint interval |
| Stateless execution (no autograd graph) | **YES** | Reduces PyTorch overhead |
| Double-buffered H2D streaming | NO | No PCIe — unified memory is zero-copy |
| Pinned slab recycling | NO | No pinning needed |
| K-slab gradient offloading | NO | Gradients in same memory |
| Layer-contiguous tiling | SIMPLIFIED | One address space, no staging |

---

## Technique 1: CPU-Side Optimizer

Optimizer states (fp32 momentum + variance) live in CPU-preferred memory. GPU does forward/backward only. CPU updates weights after backward.

```python
# Option A: Manual CPU AdamW
class UnifiedCPUOptimizer:
    def __init__(self, model, lr=8e-4, betas=(0.9, 0.999), wd=0.1):
        self.lr, self.betas, self.wd = lr, betas, wd
        self.m = {n: torch.zeros_like(p, device='cpu') for n, p in model.named_parameters()}
        self.v = {n: torch.zeros_like(p, device='cpu') for n, p in model.named_parameters()}
        self.step_count = 0

    def step(self, model):
        self.step_count += 1
        b1, b2 = self.betas
        bc1 = 1 - b1**self.step_count  # bias correction
        bc2 = 1 - b2**self.step_count
        for name, param in model.named_parameters():
            if param.grad is None: continue
            g = param.grad.float()  # unified memory: may be zero-copy
            # Weight decay
            param.data.mul_(1 - self.lr * self.wd)
            # Momentum update (CPU)
            self.m[name].mul_(b1).add_(g, alpha=1-b1)
            self.v[name].mul_(b2).addcmul_(g, g, value=1-b2)
            # Parameter update
            update = (self.m[name] / bc1) / ((self.v[name] / bc2).sqrt() + 1e-8)
            param.data.add_(update.to(param.device), alpha=-self.lr)

# Option B: DeepSpeed CPUAdam (preferred if available on ROCm)
# from deepspeed.ops.adam import DeepSpeedCPUAdam
# optimizer = DeepSpeedCPUAdam(model.parameters(), lr=8e-4)
```

**Memory savings:**

| Component | Standard (GPU) | COLOSSEUM (CPU-side) |
|---|---|---|
| BF16 weights | 500 MB (GPU) | 500 MB (unified, GPU-accessible) |
| FP32 optimizer m, v | 2 GB (GPU) | 2 GB (CPU-preferred, not GPU-cached) |
| **GPU memory freed** | 0 | **~2 GB** |

## Technique 2: Block-wise Activation Recomputation

```python
# Instead of storing all 16 layers' activations:
# Checkpoint every K=4 layers → store only 4 checkpoints
# Recompute 3 layers from each checkpoint during backward

# PyTorch native:
from torch.utils.checkpoint import checkpoint_sequential
output = checkpoint_sequential(model.layers, segments=4, input=h)

# Memory: O(4 × batch × seq × d) instead of O(16 × batch × seq × d)
# For batch=48, seq=512, d=1024: saves ~1.5 GB
```

## Technique 3: Stateless Execution

Replace persistent autograd graphs with layer templates (from MegaTrain Section 4.4):

```python
# Instead of PyTorch's autograd building a full graph:
# Use manual forward/backward with pre-allocated buffers

# Simplified version: torch.compile handles most of this
model = torch.compile(model, mode="reduce-overhead")
# mode="reduce-overhead" uses CUDA graphs which are essentially "templates"
```

For maximum benefit, combine with `torch.amp.autocast` and `GradScaler`:

```python
scaler = torch.amp.GradScaler("cuda")
with torch.amp.autocast("cuda", dtype=torch.float16):
    output = model(input_ids)
    loss = criterion(output, targets) / accum_steps
scaler.scale(loss).backward()
if (step + 1) % accum_steps == 0:
    scaler.step(optimizer)  # CPU-side optimizer
    scaler.update()
    optimizer.zero_grad(set_to_none=True)
```

---

## Combined Impact

| Config | Batch | GPU Mem Used | Est. tok/s | 15-min Tokens |
|---|---|---|---|---|
| Standard | 48 | ~6 GB | ~10K | ~9M |
| + CPU optimizer | 96 | ~4 GB | ~12K | ~10.8M |
| + Activation checkpointing | 96 | ~2.5 GB | ~11K | ~9.9M |
| + torch.compile | 96 | ~2.5 GB | **~20-25K** | **~18-22.5M** |

**2-2.5x more tokens in the same 15 minutes.** This is pure infrastructure — no architectural changes.

---

## Compatibility

| Architecture/Strategy | Compatible | Notes |
|---|---|---|
| ALL 17+ architectures | YES | Pure training infrastructure |
| Self-Curriculum | YES | Orthogonal (data sampling vs compute) |
| Lottery Forge | YES | Overparameterized phase benefits most (330M) |
| Any future architecture | YES | No model assumptions |

---

## Risks & Mitigations

| Risk | Severity | Mitigation |
|------|----------|------------|
| CPU optimizer slower than GPU optimizer | MEDIUM | DeepSpeed CPUAdam is C++/AVX optimized. For 250M params, CPU update < 50ms (dwarfed by forward/backward). |
| Activation recomputation slows training ~30% | MEDIUM | Batch size 2x compensates. Net: more tokens per wall-clock second. |
| Unified memory contention (CPU+GPU accessing same) | LOW | GPU accesses are sequential (one layer at a time). No contention pattern. |

## Success Criteria

1. Batch size doubles (48→96 or higher) without OOM
2. Total tokens in 15 min increases > 50% over standard training
3. Loss matches standard training at equal tokens seen
4. Peak GPU memory < 4 GB (leaving room for larger models)

## Implementation Roadmap

1. Implement UnifiedCPUOptimizer (or integrate DeepSpeed CPUAdam)
2. Add activation checkpointing (`checkpoint_sequential`, K=4)
3. Benchmark: standard vs COLOSSEUM on GPT-2 124M baseline
4. Verify: larger batch fits, more tokens/min, same loss trajectory
5. Apply to first architecture build (Caveman LFM or AMADEUS)

---

## Hardware Optimization Notes (Strix Halo gfx1151)

> Added 2026-04-09 based on AMADEUS implementation findings.

### halo_training Already Implements This
The `halo_training/` package already implements unified memory training:
- **Mode A** (<2B): whole-model torch.compile, 43K tok/s with autokernel (124M transformer)
- **Mode B** (>2B): per-layer activation checkpointing via `LayerStreamingTrainer`, 853 tok/s (2.09B)
- **CPUAdam:** DeepSpeed CPUAdam works on ROCm with monkey-patch (see CLAUDE.md)
- **Auto mode selection** via `suggest_mode()` at 60% GPU memory threshold

### Verified Baselines
- 243M SSM (AMADEUS): 6.4K tok/s, 12.7 GB memory
- 124M transformer: 14.5K tok/s (eager) / 43K tok/s (compile+autokernel)
- 2.09B transformer: 853 tok/s, 34.5 GB (Mode B)

Much of this plan is already implemented. Focus integration effort on novel architectures, not reimplementing the training loop.

### External Kernel Integration (verified 2026-04-10)

Updated baselines should include external kernel speedups: causal-conv1d (10x), mamba-ssm scan (5.6x), hybrid_attention (8.9% vs SDPA). All verified on gfx1151 with full backward.

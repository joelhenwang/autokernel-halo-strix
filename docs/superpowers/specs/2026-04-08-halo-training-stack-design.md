# HALO-TRAINING-STACK Design

> _Readable version: key constraints, decisions, and callouts are emphasized with simple inline styling._

**Date:** 2026-04-08  
**Status:** **All Phases Complete** — Mode A: 42.9K tok/s (with autokernel), Mode B: 853 tok/s (2B), 3.05x training speedup  
**Scope:** _Composable training stack for AMD Ryzen AI MAX+ 395 (Strix Halo), supporting models from 100M to 7B+ on a single machine with unified memory._

> **At a glance**
> - **Primary goal:** one training stack for both custom LLMs and standard HuggingFace models
> - **Simple path:** direct training for models under ~2B parameters
> - **Scale path:** layer-streaming for 7B+ when memory gets tight
> - **Key enabler:** unified memory removes PCIe offload complexity

---

## 1. Motivation

> **Why this matters:** the repo already has strong parts; the design is about composing them cleanly.

**Goal:** compose the existing systems into one training stack without forcing them into a transformer-only shape.

Three systems exist independently in this repo:

- **mad_llm_scientist** — 13 novel LLM architectures (<250M params) with non-standard components
  - _Griffin recurrence_
  - _Engram hash tables_
  - _DHO_
  - _gated convolutions_
  - _Sinkhorn mHC_
- **autokernel** — HIP kernel fusions for inference optimization (**1.6-16x** per op on `gfx1151`)
- **MegaTrain** (upstream) — CPU-offloaded training for 100B+ models on single GPU, designed for PCIe-connected NVIDIA hardware

> **Key tension:** these systems were built for different assumptions, so the integration has to stay flexible.

This design composes them into a unified training stack that:

1. Trains any `nn.Module` (not just transformers) on Strix Halo
2. Works simply at <2B params (standard PyTorch loop + optimizations)
3. Scales to 7B with layer-streaming when memory gets tight
4. Exploits unified memory to eliminate MegaTrain's PCIe streaming complexity

---

## 2. Hardware Reference (Confirmed)

> **Confirmed target:** AMD Ryzen AI MAX+ 395, the top Strix Halo SKU.

**Target:** AMD Ryzen AI MAX+ 395 *(top Strix Halo SKU)*

| **Parameter** | **Value** |
|---|---|
| **CPU** | 16 Zen 5 cores / 32 threads, 3.0-5.1 GHz, 16 MB L2 + 64 MB L3, AVX-512 (double-pumped on 256-bit datapath) |
| **GPU** | Radeon 8060S — 40 CUs (20 WGPs), 2.9 GHz, gfx1151 RDNA 3.5, wave32 |
| **FP16 / FP32** | ~59.4 / ~29.7 TFLOPS |
| **Memory** | **128 GB** soldered LPDDR5X, 256-bit bus, ~240 GB/s, unified CPU+GPU |
| **TDP** | 45-120W configurable |
| **NPU** | XDNA 2, 50 TOPS INT8 |
| **Key constraint** | _No MFMA matrix cores — memory-bound for nearly all ops_ |

> **What to remember:** memory capacity is the big unlock; bandwidth is the main bottleneck.

These values supersede estimates in `knowledge/amd_rdna35_strix_halo.md` (~120 GB/s, ~50 TFLOPS FP16, 28-54W TDP).

---

## 3. Architecture Overview

> **Hierarchy:** hardware → optimization layer → training engine → model families.

> **Read this top-down:** hardware → kernel layer → training engine → model families.

```
+-------------------------------------------------+
|  **mad_llm_scientist** architectures *(100-200M)* |
|  + _HuggingFace models_ *(7B)*                    |
+-------------------------------------------------+
|  **Training Engine**                            |
|  +- **Mode A** *(<2B)*: standard PyTorch loop    |
|  +- **Mode B** *(7B+)*: layer-streaming + checkpointing |
|  + DeepSpeed CPUAdam *(5-7x optimizer speedup)*  |
+-------------------------------------------------+
|  Kernel Optimization Layer                      |
|  +- `autokernel.optimize()` *(HIP fusions)*      |
|  +- `flash_attn` *(ROCm build, attention models)* |
|  +- `torch.compile` *(model only, reduce-overhead)* |
+-------------------------------------------------+
|  Hardware: 40 CUs, 128 GB unified LPDDR5X       |
|  ~240 GB/s BW, ~59.4 TFLOPS FP16, gfx1151      |
+-------------------------------------------------+
```

---

## 4. Two-Mode Training Engine

> **Rule of thumb:** stay simple under ~2B, switch to streaming only when memory pressure forces it.

### Design Decision: **No Stateless Templates**

> _This is a structural choice, not just an implementation preference._

The `mad_llm_scientist` architectures use non-standard components *(Griffin recurrence, DHO, Engram, Sinkhorn mHC)* that don't fit this template. On unified memory, the stateless template optimization *(eliminating persistent autograd graphs to save GPU memory)* is unnecessary — the autograd graph for 7B is ~2-4 GB out of 128 GB available.

**Decision:** Ignore stateless templates and create a **generic layer iterator** that works with any `nn.Module` block.

### Mode A: **Direct Training** `(<2B params)`

> **Best fit:** custom 100M-200M models and most moderate-size experiments.

For `mad_llm_scientist` architectures *(100-200M)* and moderate models up to ~2B. The entire model + optimizer states + activations fit in **128 GB unified memory**.

```python
model = build_model()                               # any nn.Module
model = autokernel.optimize(model)                  # HIP kernel fusions
optimizer = DeepSpeedCPUAdam(model.parameters())     # 5-7x faster than AdamW
model = torch.compile(model, mode="reduce-overhead") # graph-level fusion

for batch in dataloader:
    output = model(batch)
    loss = loss_fn(output, batch)                   # custom loss (MTP, etc.)
    loss.backward()
    optimizer.step()                                # CPU-side, fp32, AVX-512
    optimizer.zero_grad()
```

MegaTrain contributes almost nothing here — just the DeepSpeed CPUAdam integration and potentially data loading. The value is that the same codebase handles both modes.

**Memory budget** at 100-200M:

| **Component** | **100M** | **200M** |
|---|---:|---:|
| Params *(fp16)* | 0.2 GB | 0.4 GB |
| Optimizer *(fp32 m, v, master)* | 1.2 GB | 2.4 GB |
| Activations + gradients | ~1 GB | ~2 GB |
| **Total** | **~2.4 GB** | **~4.8 GB** |

> _Small models barely touch unified memory; optimization should focus on clean throughput._

### Mode B: **Layer-Streaming Training** `(2B-7B+)`

> **Best fit:** large models where optimizer state begins to dominate memory.

For large models where optimizer states approach memory limits. _This is the “memory-tight” path._

**Memory budget** at 7B:

| **Component** | **Size** |
|---|---:|
| Params *(fp16)* | 14 GB |
| Optimizer *(fp32 m, v, master)* | 84 GB |
| Activations *(with checkpointing)* | 5-15 GB |
| **Total** | **~103-113 GB** of **128 GB** |

> _This is why 7B is still possible on a single machine: the memory budget is tight, but not impossible._

The layer-streaming pipeline, simplified from MegaTrain for unified memory:

```python
# Forward
for i, layer in enumerate(model.layers):
    if should_checkpoint(i):
        h = torch.utils.checkpoint.checkpoint(layer, h)
    else:
        h = layer(h)

# Backward: standard autograd with recomputation from checkpoints

# Optimizer: **DeepSpeed CPUAdam** — reads grads from unified memory, writes updated weights
optimizer.step()
```

### What We Keep from MegaTrain

> **Keep:** the pieces that are still valuable on unified memory.

_The useful pieces survive; the transformer-specific assumptions do not._

- Layer-by-layer execution with activation checkpointing
- CPU-side optimizer integration (DeepSpeed CPUAdam)
- Memory pressure monitoring
- Gradient accumulation across microbatches
- Data loading pipeline

### What We Drop

> **Drop:** assumptions that only exist because of PCIe-era offload.

_These are the PCIe-era abstractions that unified memory makes unnecessary._

- **3-stream pipeline** *(S_H2D, S_comp, S_D2H)* — no PCIe transfers on unified memory
- **Double-buffered weight staging** — weights are already GPU-accessible
- **Pinned slab recycling** — no pinning needed on unified memory
- **Stateless template pool** — replaced by **generic layer iterator** *(any `nn.Module`)*
- **Layer-contiguous memory tiling** — no DMA burst optimization needed

### Mode Crossover Point (~2B)

> **Transition zone:** still feasible in direct mode, but checkpointing becomes increasingly attractive.

At ~2B params, optimizer states (3x model size in fp32) reach ~24 GB. Combined with params and activations, total approaches ~34 GB — still well within 128 GB but large enough that activation checkpointing and memory monitoring become worthwhile.

### 7B Escape Valves (if memory is tight)

> **Order of operations:** try the cheapest memory win first.

Ordered by impact and risk:

1. **Reduce microbatch size** — trivial, costs throughput
2. **Increase checkpoint frequency** — trade recompute for memory
3. **8-bit Adam** (bitsandbytes) — optimizer states 84 GB -> 42 GB, proven stable
4. **Gradient accumulation** — smaller activations per step
5. **Selective layer offload** — NUMA hints on unified memory for cold layers

---

## 5. Precision Strategy

> **Core policy:** fp16 for compute, fp32 for optimizer state.

**fp16 forward/backward + fp32 optimizer states on CPU.**

### Why fp16 for compute

| **Metric** | **fp32** | **fp16** | **Benefit** |
|---|---:|---:|---:|
| Compute throughput | ~29.7 TFLOPS | ~59.4 TFLOPS | 2x |
| Bytes per parameter | 4 B | 2 B | 2x less bandwidth |
| 7B model size | 28 GB | 14 GB | 2x more fits in cache |

> _On Strix Halo, bandwidth matters more than raw arithmetic._

On Strix Halo, the bandwidth halving matters more than the TFLOPS doubling — most ops are memory-bound at 240 GB/s.

### Why fp32 for optimizer states

> **Reason:** small updates disappear too easily in fp16.

fp16 has ~3 decimal digits of precision. Weight updates are often 1/10,000th the weight magnitude — they vanish in fp16, causing training to stall. Adam's variance accumulator (v) is even worse, tracking squared gradients that underflow in fp16.

DeepSpeed CPUAdam keeps m, v, and master weights in fp32 on the CPU side. The optimizer step runs on CPU cores (AVX-512), not GPU CUs — no GPU bandwidth wasted.

```
GPU (fp16):  forward -> loss -> backward -> gradients (fp16)
                                                |
CPU (fp32):  read grads -> Adam update (fp32 m, v, master) -> write updated fp16 weights
                                                |
GPU (fp16):  next forward pass with updated weights
```

On unified memory, "CPU" and "GPU" share the same 128 GB — the optimizer reads/writes are just memory operations, no transfers.

---

## 6. Kernel Optimization Layer

> **Layered speedup:** graph fusion, kernel fusion, and attention-specific acceleration each solve a different bottleneck.

Three complementary acceleration paths applied at different stages.

### 6a. `autokernel.optimize()` — HIP Kernel Fusions

> **Role:** remove memory passes by fusing common operator chains.

Applied to the model before training begins. Fuses sequences of PyTorch ops into single HIP kernels, reducing memory passes during both forward and backward.

```python
model = autokernel.optimize(model)  # applied once before training
```

Relevant fusions for mad_llm_scientist architectures:

| Pattern | Speedup | Used by |
|---------|---------|---------|
| Fused residual + RMSNorm | 6.6x | All architectures (every layer) |
| RMSNorm | 3.3x | All architectures |
| SwiGLU (silu_gate_mul) | 1.6x | All SwiGLU FFN blocks |
| Rotary embedding | 3.7x | Architectures with attention |
| Fused bias + SiLU | 1.9x | Gated convolution blocks |
| Cross entropy (online) | 1.8x | Loss computation |

**Training compatibility:** Fused kernels must work with PyTorch autograd. The `_torch_ops.py` custom op registrations (via `torch.library`) enable this — custom ops are visible to both autograd and the Inductor compiler.

### 6b. `flash_attn` — For Attention-Based Architectures

> **Role:** only matters when the model actually uses attention.

For architectures that use attention (or hybrid designs with occasional attention layers):

- ROCm build already available on the training machine
- MegaTrain already depends on `flash_attn.flash_attn_func`
- Memory-efficient (O(N) instead of O(N^2) for sequence length)
- Avoids the 0.05x penalty of naive attention on gfx1151

For SSM/recurrence architectures (CAVEMAN-LFM, SPECTRAL-HYDRA, etc.), this layer is unused.

### 6c. `torch.compile` — Graph-Level Fusion

> **Role:** catch the parts autokernel does not cover.

Complements autokernel for ops without custom HIP kernels.

```python
# Mode A (<2B): compile the whole model
model = torch.compile(model, mode="reduce-overhead")

# Mode B (7B+): compile per-layer for layer-streaming compatibility
for layer in model.layers:
    layer = torch.compile(layer, mode="reduce-overhead")
```

Benchmarked at 1.16-1.28x speedup from Inductor backend. Fuses ops that autokernel doesn't cover (embedding lookups, tensor transposes, custom recurrence internals).

**IMPORTANT: Only compile the model, never the optimizer.** Compiling the optimizer takes extremely long and can crash. DeepSpeed CPUAdam runs on CPU anyway — there is no benefit.

**Composability:** `torch.library` custom ops from autokernel are visible to the Inductor compiler. `autokernel.optimize()` + `torch.compile()` is the target path, with fallback to autokernel-only if issues arise.

### Acceleration Stack Summary

> **Stack order:** `torch.compile` above `autokernel`, with `flash_attn` used only where relevant.

```
+-------------------------------------------+
| torch.compile (graph-level fusion) 1.16-1.28x
| +---------------------------------------+ |
| | autokernel (HIP kernel fusions) 1.6-6.6x per op
| | +-----------------------------------+ | |
| | | flash_attn (if attention used)    | | |
| | +-----------------------------------+ | |
| +---------------------------------------+ |
|        DeepSpeed CPUAdam            5-7x   |
+-------------------------------------------+
```

---

## 7. DeepSpeed CPUAdam Integration

> **Why it belongs here:** it is the simplest high-impact optimizer win for this hardware.

### Why CPUAdam

| **Optimizer** | **Runs on** | **Relative speed** |
|---|---|---:|
| PyTorch AdamW | GPU | 1x baseline |
| DeepSpeed CPUAdam | CPU (AVX-512 + OpenMP) | 5-7x |

> _CPUAdam is useful because the CPU is strong enough to keep optimizer step cost low._

CPUAdam is pure CPU code — no GPU dependency. Zen 5 has AVX-512 (double-pumped on mobile, still faster than AVX2). 16 cores with OpenMP parallelism makes the optimizer step negligible relative to forward/backward.

### Installation on Strix Halo

```bash
# 1. Verify ROCm environment
export ROCM_HOME=/opt/rocm
python -c "import torch; print(torch.version.hip)"
rocminfo | grep gfx1151

# 2. Install dependencies
pip install py-cpuinfo    # AVX-512 detection for Zen 5

# 3. Install DeepSpeed with pre-built CPU Adam
DS_BUILD_CPU_ADAM=1 pip install deepspeed

# 4. Verify
ds_report
python -c "
from deepspeed.ops.adam import DeepSpeedCPUAdam
import torch
p = torch.randn(1024, requires_grad=True)
opt = DeepSpeedCPUAdam([p], lr=1e-3)
p.grad = torch.randn(1024)
opt.step()
print('CPUAdam OK')
"
```

### Known Issues

| Issue | Risk | Mitigation |
|-------|------|------------|
| "cuda is missing" warning (DeepSpeed #4768) | Low — false positive on ROCm | `DS_BUILD_CPU_ADAM=1` pre-build avoids it |
| gfx1151 untested upstream | Low — CPUAdam is CPU-only code | Gradient sync uses standard PyTorch `.to(device)` |
| `ds_report` shows ops as NOT_COMPATIBLE | Low | Pre-build with `DS_BUILD_CPU_ADAM=1` |

### What We Use vs. Skip from DeepSpeed

> **Use:** CPUAdam.  
> **Skip:** the transformer-era distributed pieces that unified memory makes unnecessary.

| Feature | Use? | Why |
|---------|------|-----|
| CPUAdam | **Yes** | 5-7x optimizer speedup |
| ZeRO-Offload | **No** | Designed for PCIe, unnecessary on unified memory |
| ZeRO stages 1-3 | **No** | Single-GPU, no sharding needed |
| Pipeline parallelism | **Maybe (future)** | TB4 interconnect to other Strix Halo machines has enough bandwidth for activation passing (~4 MB/transfer). Enables 14B+ across 2 machines. |
| Activation checkpointing | **Maybe** | PyTorch native `torch.utils.checkpoint` may suffice |
| FP16 optimizer | **No** | fp32 optimizer states for stability |

### Fallback

If CPUAdam has issues on gfx1151, fall back to `torch.optim.AdamW(fused=True)`. Slower but zero-dependency.

---

## 8. Integration with `mad_llm_scientist`

> **Compatibility requirement:** the engine must accept unusual layers without forcing them into a transformer mold.

### Non-Standard Components

The training engine must handle these without special-casing:

| **Component** | **Standard?** | **Training concern** |
|---|---|---|
| RMSNorm | Yes | autokernel fuses it |
| SwiGLU FFN | Yes | autokernel fuses it |
| Rotary embedding | Yes | autokernel fuses it |
| Griffin gated recurrence | No | Stateful forward — activation checkpointing must preserve state |
| Gated short conv (k=3) | No | Causal padding, works with standard autograd |
| Engram hash tables | No | Separate param group, 5x learning rate |
| DHO recurrence | No | Complex-valued state, needs fp32 accumulation |
| Sinkhorn mHC | No | 4-branch residual, iterative normalization |
| Meta tokens | No | Learned prefix, prepended to input |
| MTP heads | No | Multi-token prediction, weighted loss from 4 heads |
| TernaryLinear | No | Clipped STE, needs special gradient handling |

### Requirements on Training Engine

> **Checklist:** these are the non-negotiables for a usable engine.

1. **Multiple parameter groups** — Engram tables need 5x LR, TernaryLinear needs lower LR. DeepSpeed CPUAdam supports param groups natively.
2. **Custom loss computation** — MTP heads produce 4 loss terms with different weights. Training loop needs a `loss_fn` hook.
3. **Phase training** — COOKBOOK.md specifies multi-phase unfreezing. Training loop needs a callback for `requires_grad` toggling.
4. **Stateful recurrence** — Griffin/DHO maintain hidden states. Activation checkpointing must preserve state correctly.
5. **No assumption about layer structure** — generic layer iterator, don't inspect internals.

### Proposed Training API

```python
from halo_training import train

train(
    model=model,                          # any nn.Module
    dataset="babylm",                     # or path to data
    epochs=1,                             # auto-calculated if time_budget set
    time_budget_minutes=15,               # alternative to epochs
    param_groups=model.param_groups(),    # custom LR/WD per group
    loss_fn=model.compute_loss,           # custom loss (MTP, etc.)
    callbacks=[PhaseUnfreezing(schedule)],# phase training
    compile=True,                         # torch.compile model (NOT optimizer)
    optimize_kernels=True,                # autokernel.optimize()
)
```

---

## 9. Model Size & Throughput Targets

> **Expectation setting:** small models should be fast; large models should remain practical.

### Target: 100-200M params

> **Primary bakeoff zone:** this is where most iteration will happen.

For 15-minute bakeoffs on BabyLM (16M tokens).

Throughput estimates at 25% MFU (59.4 TFLOPS FP16 peak):

| **Params** | **tok/s** | **Time (1 epoch, 16M tok)** | **Time (3 epochs)** |
|---|---:|---:|---:|
| 100M | ~25K | ~11 min | ~32 min |
| 125M | ~20K | ~13.5 min | ~40 min |
| 150M | ~17K | ~16 min | ~49 min |
| 200M | ~12K | ~22 min | ~65 min |

> _These numbers are about iteration speed, not just raw model throughput._

For the 15-minute bakeoff: 1 full epoch at 100-130M, partial epoch at 200M.

### Scaling to 7B

> **Reality check:** 7B is slower, but still viable on one box.

| Params | tok/s | Time (1 epoch) | Mode |
|--------|-------|----------------|------|
| 1B | 2K-5K | 53-133 min | A (direct) |
| 2B | 1K-2.5K | 107-267 min | A (direct) |
| 7B | 200-500 | 9-22 hours | B (layer-streaming) |

### Multi-Machine Scaling (Future)

> **Future note:** this section is intentionally optional for the first implementation.

Multiple identical Strix Halo machines connected via Thunderbolt 4 (~5 GB/s, no RDMA).

| Parallelism | Data crossing wire | Transfer time | Viable? |
|-------------|-------------------|---------------|---------|
| Pipeline | Activations (~4 MB/microbatch) | ~1 ms | **Yes** |
| Data | Full gradient allreduce (~14 GB at 7B) | ~4 seconds/step | **No** |

Pipeline parallelism over TB4:

| Machines | Total memory | Max model |
|----------|-------------|-----------|
| 1 | 128 GB | 7B |
| 2 | 256 GB | 14-20B |
| 3 | 384 GB | 25-30B |
| 4 | 512 GB | 40B+ |

---

## 10. Testing Plan

> **Test philosophy:** validate environment first, then correctness, then speed, then scale.

### Phase 1: Environment

> **Goal:** prove the machine, compiler, and core packages are healthy.

| # | Test | Pass criteria |
|---|------|---------------|
| 1 | ROCm + gfx1151 detection | `rocminfo \| grep gfx1151` finds GPU |
| 2 | PyTorch HIP backend | `torch.cuda.is_available()` returns True |
| 3 | DeepSpeed CPUAdam install | `ds_report` shows cpu_adam [OKAY] |
| 4 | CPUAdam smoke test | 10 optimizer steps, weights finite |
| 5 | flash_attn import | `from flash_attn import flash_attn_func` succeeds |
| 6 | autokernel import | `import autokernel` succeeds |

### Phase 2: Single-Machine Training (Mode A)

> **Goal:** prove the simple path works before trying the complex one.

| # | Test | Pass criteria |
|---|------|---------------|
| 7 | Forward pass (1M dummy) | Output tensor, correct shape |
| 8 | Backward + CPUAdam step (1M) | Finite gradients, weights update |
| 9 | autokernel + forward (10M, RMSNorm+SwiGLU) | Fusions applied, output matches within atol=0.01 |
| 10 | torch.compile + forward (10M) | Compiles without crash, output correct |
| 11 | 100-step training (25M, BabyLM subset) | Loss decreases monotonically |
| 12 | Full bakeoff dry run (100M, 15 min) | Completes, BPB reported, no OOM |
| 13 | Custom architecture (CAVEMAN-LFM ~100M) | Trains with Griffin + Engram + phase unfreezing, loss decreases |

### Phase 3: Performance

> **Goal:** verify the speedups are real and that copies are not sneaking in.

| # | Test | Pass criteria |
|---|------|---------------|
| 14 | tok/s at 100M | >= 10K tok/s |
| 15 | CPUAdam vs AdamW | >= 3x faster optimizer step |
| 16 | autokernel forward speedup | >= 1.15x with fusions |
| 17 | Memory at 100M | < 5 GB total |
| 18 | No unnecessary copies | rocprof shows zero H2D/D2H hipMemcpy |

### Phase 4: Scaling (Mode B)

> **Goal:** confirm the memory-tight path remains stable at 7B.

| # | Test | Pass criteria |
|---|------|---------------|
| 19 | 7B model load | Fits in 128 GB |
| 20 | 7B forward + backward (1 batch) | Finite loss and gradients |
| 21 | 7B 10-step training | Loss decreases, memory stable |
| 22 | 7B memory high-water mark | < 120 GB |
| 23 | 7B throughput | >= 10 TFLOPS |

### Phase 5: Multi-Machine (Future)

> **Goal:** keep the future path visible without making it a blocker.

| # | Test | Pass criteria |
|---|------|---------------|
| 24 | TB4 bandwidth measurement | >= 3 GB/s sustained |
| 25 | 2-machine 14B pipeline | Both stages compute, activations transfer |
| 26 | 2-machine convergence | Loss within 5% of single-machine |

---

## 11. Deliverables

> **Output:** this design should lead directly to a concrete package and a few documentation updates.

### This Design Creates

> **Near-term artifact:** the spec itself, plus the implementation path.

| File | Purpose |
|------|---------|
| This spec | Architecture and integration design |

### Implementation Will Update

> **Docs to revise:** the hardware reference, the porting plan, and the repo guidance.

| File | Changes |
|------|---------|
| `knowledge/amd_rdna35_strix_halo.md` | Correct to confirmed specs (240 GB/s, 59.4 TFLOPS, 128 GB, 45-120W) |
| `mad_llm_scientist/plans/MEGATRAIN-HALO.md` | Add DeepSpeed CPUAdam, flash_attn, correct specs, drop stateless templates, add pipeline parallelism future note |
| `mad_llm_scientist/CLAUDE.md` | Correct memory (128 GB) and bandwidth (~240 GB/s) |

### Implementation Will Create

> **Code to add:** a training package and, eventually, an adapted MegaTrain fork.

| Artifact | Purpose |
|----------|---------|
| `halo_training/` package | The training stack (Mode A + B, API from Section 8) |
| MegaTrain fork (adapted) | Upstream fork with unified memory simplification |

---

## 12. Risks

> **Risk posture:** most issues are integration risks, not hardware limits.

| Risk | Severity | Mitigation |
|------|----------|------------|
| autokernel fusions lack backward pass for training | Medium | Verify `torch.library` custom ops have autograd support; fall back to eager for ops that don't |
| DeepSpeed CPUAdam fails on gfx1151 | Low | CPUAdam is CPU-only code; fall back to `torch.optim.AdamW(fused=True)` |
| Activation checkpointing breaks stateful recurrence (Griffin/DHO) | Medium | Test early with dummy recurrence module; may need custom checkpoint wrapper |
| 7B doesn't fit in 128 GB with full optimizer states | Low | 8-bit Adam reduces optimizer 84 GB -> 42 GB; gradient accumulation reduces activation peak |
| torch.compile crashes on novel architecture ops | Medium | Compile per-layer, mark problematic ops with `torch.compiler.disable`; fall back to autokernel-only |
| TB4 latency too high for pipeline parallelism | Low | Only affects future multi-machine path; single-machine is primary |
| MegaTrain upstream assumes transformer internals beyond templates | Medium | Start with Mode A (no MegaTrain dependency); adopt MegaTrain components incrementally |

---

## 13. References

> **Pointers:** these are the source documents that should stay in sync with this design.

- **MegaTrain paper:** arxiv.org/abs/2604.05091 — CPU-offloaded training for 100B+ on single GPU
- **MegaTrain repo:** github.com/DLYuanGod/MegaTrain (Apache-2.0)
- **Porting guide:** `mad_llm_scientist/plans/MEGATRAIN-HALO.md`
- **Architecture cookbook:** `mad_llm_scientist/COOKBOOK.md`
- **Hardware reference:** `knowledge/amd_rdna35_strix_halo.md`
- **autokernel API:** `autokernel/__init__.py`
- **DeepSpeed CPUAdam:** DeepSpeed issue #4768 (ROCm false warning)

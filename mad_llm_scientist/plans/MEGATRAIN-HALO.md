# MEGATRAIN-HALO

**Porting Guide: Adapt MegaTrain for AMD Strix Halo (gfx1151, ROCm 7.12)**

## Purpose

This is NOT an architecture or training strategy. It's a **porting guide** for a specialized agent to fork `github.com/DLYuanGod/MegaTrain` (Apache-2.0) and make it work on AMD Strix Halo with ROCm.

**Key context:** Strix Halo (Ryzen AI MAX+ 395) has UNIFIED MEMORY (128 GB, ~240 GB/s). MegaTrain's core complexity exists to manage PCIe streaming. On unified memory, most of that complexity can be REMOVED, making the port a SIMPLIFICATION.

**Training machine has:** ROCm 7.12, PyTorch 2.10.0+rocm7.12.0, `hipcc` for gfx1151, DeepSpeed 0.17.5.

**Design doc:** `docs/superpowers/specs/2026-04-08-halo-training-stack-design.md` — covers the full training stack architecture.
**Implementation:** `halo_training/` package — Mode A (direct, <2B) already working at 13.8K tok/s.

### Key Design Decisions (from HALO-TRAINING-STACK design)

1. **Ignore MegaTrain's stateless templates** — they assume Attention+MLP structure. mad_llm_scientist architectures use Griffin recurrence, Engram, DHO, etc. Create a generic layer iterator instead.
2. **DeepSpeed CPUAdam** for CPU-side optimizer (5-7x faster). Requires monkey-patch on ROCm (see `halo_training/optimizer.py`). For Mode A (<2B), use `torch.optim.AdamW(fused=True)` since params stay on GPU.
3. **fp16 forward/backward + fp32 optimizer on CPU** — full 59.4 TFLOPS utilization.
4. **torch.compile(model, mode="reduce-overhead")** — 1.3-1.8x throughput. NEVER compile the optimizer (crashes on ROCm).
5. **Gradient checkpointing** — only when needed (approaching memory limits). Not worth the ~20-30% slowdown when memory is ample.
6. **Pipeline parallelism** (future) — multiple Strix Halo machines via Thunderbolt 4 (~5 GB/s). Viable for activation passing (~4 MB/microbatch), not for gradient allreduce.

---

## Repo Structure Assessment

```
MegaTrain/infinity/
├── cuda_pipeline/     ← CUDA-specific. HIGH effort.
├── memory/            ← PCIe-centric. HIGH effort (rewrite for unified memory).
├── runtime/           ← CUDA streams. MEDIUM effort.
├── scheduler/         ← Likely GPU-agnostic. LOW effort.
├── ops/               ← Custom operators. MEDIUM effort.
├── model/             ← HuggingFace loading. GPU-AGNOSTIC.
├── adapters/          ← LoRA etc. GPU-AGNOSTIC.
├── config/            ← Configuration. GPU-AGNOSTIC.
├── data/              ← Data loading. GPU-AGNOSTIC.
├── csrc/              ← C++/CUDA source. MEDIUM effort (hipify).
├── optimizer.py       ← CPU-side optimizer. LOW effort.
├── true_cpu_offloading.py  ← Core innovation. MEDIUM effort.
├── profiler.py        ← CUDA profiling. LOW effort.
└── simple_profiler.py ← Simple timing. GPU-AGNOSTIC.
```

---

## Porting Plan by Priority

### Priority 1: GPU-Agnostic (NO CHANGES needed)

| Module | Why No Changes |
|--------|---------------|
| `config/` | YAML parsing, no GPU code |
| `data/` | DataLoader, tokenization |
| `model/` | HuggingFace `AutoModelForCausalLM` |
| `adapters/` | LoRA/PEFT wrappers |
| `simple_profiler.py` | Python time.time() |

**Action:** Verify with `import infinity` on ROCm. Fix any import-time CUDA assumptions.

### Priority 2: Trivial Adaptations (LOW effort)

**`optimizer.py`** — CPU-side AdamW. Uses numpy/torch CPU tensors. Should work as-is. Verify DeepSpeed CPUAdam availability on ROCm (`pip install deepspeed` with ROCm).

**`profiler.py`** — Replace any NVTX calls with ROCTX or remove. `torch.cuda.Event` works on ROCm via HIP backend.

**`flash-attn` dependency in `ops/`** — Already available on the training machine. Verify import path matches MegaTrain's expectations.

### Priority 3: HIP Porting (MEDIUM effort)

**`csrc/` — C++/CUDA source files:**

```bash
# Step 1: Auto-convert CUDA → HIP
hipify-perl -inplace csrc/*.cu csrc/*.cuh

# Step 2: Manual review for:
# - __shfl_*() intrinsics: Strix Halo uses WARP_SIZE=32 (not 64)
# - Shared memory sizes: verify <= 64 KB per CU
# - Launch bounds: verify <= 1024 threads per block
# - Any hardcoded warp size constants

# Step 3: Compile
hipcc --offload-arch=gfx1151 -O3 csrc/*.hip -shared -o libmegatrain.so
```

**`runtime/` — Execution engine:**

| CUDA API | HIP Equivalent | Notes |
|----------|---------------|-------|
| `cudaStream_t` | `hipStream_t` | Same semantics |
| `cudaEvent_t` | `hipEvent_t` | Same semantics |
| `cudaStreamCreate` | `hipStreamCreate` | Same API |
| `cudaEventRecord` | `hipEventRecord` | Same API |
| `cudaStreamWaitEvent` | `hipStreamWaitEvent` | Same API |
| `cudaMemcpyAsync` | `hipMemcpyAsync` | Same API |
| `cudaDeviceSynchronize` | `hipDeviceSynchronize` | Same API |

Most stream/event code works with simple find-replace. PyTorch's `torch.cuda.*` already maps to HIP on ROCm.

### Priority 4: Unified Memory Simplification (HIGH effort, HIGH impact)

**`memory/` — Memory management:**

MegaTrain uses:
- Pinned slab recycling (for PCIe DMA)
- Layer-contiguous tiling (for burst transfers)
- JIT packing from pageable → pinned memory
- Staging buffer pools

**On Strix Halo, REPLACE ALL OF THIS with:**

```python
class UnifiedMemoryManager:
    """Simplified memory manager for unified memory (no PCIe)."""
    
    def __init__(self, model):
        # All parameters already GPU-accessible via unified memory
        # No pinning, no staging, no transfers needed
        self.param_store = {}
        for name, param in model.named_parameters():
            # Keep fp32 optimizer states in CPU-preferred allocation
            self.param_store[name] = {
                'param': param,  # unified memory, GPU-accessible
                'm': torch.zeros_like(param, device='cpu'),  # optimizer momentum
                'v': torch.zeros_like(param, device='cpu'),  # optimizer variance
            }
    
    def get_layer_params(self, layer_idx):
        # No transfer needed — just return the params (already GPU-accessible)
        return self.param_store[layer_idx]
    
    def update_grads(self, layer_idx, grads):
        # No D2H transfer — grads are in unified memory
        # CPU optimizer reads them directly
        pass
```

**`cuda_pipeline/` — Double-buffered streaming:**

MegaTrain's 3-stream pipeline (H2D, Compute, D2H) exists for PCIe.

**On Strix Halo, REPLACE with single-stream compute:**

```python
class UnifiedPipeline:
    """No streaming needed — unified memory makes transfers obsolete."""
    
    def __init__(self):
        self.compute_stream = torch.cuda.Stream()
    
    def forward_layer(self, layer, h):
        with torch.cuda.stream(self.compute_stream):
            return layer(h)
    
    # No H2D stream. No D2H stream. No double buffering.
    # Parameters are already where they need to be.
```

**`true_cpu_offloading.py` — CPU offloading:**

- Keep: CPU-side optimizer update logic (the valuable part)
- Remove: explicit `tensor.cpu()` / `tensor.cuda()` transfers
- Replace with: direct unified memory access
- Keep: OpenMP parallelization for CPU-side AdamW
- Remove: `cudaMemcpyAsync` calls for gradient evacuation

---

## ROCm-Specific Issues to Watch

| Issue | Where to Check | Fix |
|---|---|---|
| WARP_SIZE=32 (RDNA 3.5) vs MegaTrain assumes 64 | `csrc/`, any `__shfl_*` calls | Grep for `WARP_SIZE`, `warpSize`, `__shfl`. Set to 32. |
| No MFMA (matrix cores) | `ops/` custom matmul kernels | Remove custom matmul opts, rely on rocBLAS + flash-attn |
| `expandable_segments` allocator | Memory fragmentation control | Set `PYTORCH_HIP_ALLOC_CONF=expandable_segments:True` |
| hipcc compilation time | `csrc/` build | Pre-compile ALL .hip files before first run (~100s each) |
| `torch.cuda.is_available()` checks | Throughout codebase | Works on ROCm (HIP is mapped to CUDA API) |
| `torch.cuda.get_device_properties()` | Device detection | May report as "AMD" not "NVIDIA" — check string comparisons |

---

## Testing Plan (sequential — each must pass before next)

| # | Test | Command | Pass Criteria |
|---|---|---|---|
| 1 | Package import | `python -c "import infinity"` | No errors |
| 2 | Config loading | `python -c "from infinity.config import *"` | No errors |
| 3 | 125M model forward | Script: load GPT-2 small, forward 1 batch | Output tensor, correct shape |
| 4 | 125M backward | Same + loss.backward() | Finite gradients |
| 5 | CPU optimizer | Step with CPU AdamW | Weights change, all finite |
| 6 | 100-step train loop | Full loop on 125M, BabyLM | Loss decreases |
| 7 | 1B model load | Load 1B config | Fits in 128 GB |
| 8 | 1B training (10 steps) | Forward + backward + optimizer | No OOM, loss finite |
| 9 | No unnecessary copies | Profile with `rocprof` | Zero H2D/D2H transfers |
| 10 | Throughput benchmark | Train 1B for 100 steps, measure TFLOPS | > 5 TFLOPS |

---

## Post-Port Optimizations

1. **Remove all transfer code paths** — verify with profiler that zero `hipMemcpy*` calls occur
2. **Increase batch size** — unified memory allows larger batches since no staging buffers
3. **Use `torch.compile`** — `mode="reduce-overhead"` for small models
4. **Benchmark vs native PyTorch** — MegaTrain-Halo should be faster for >250M models due to CPU optimizer
5. **Integrate with autokernel** — `autokernel.optimize(model, compile=True)` for HIP kernel replacements

---

## Deliverable

A forked repo `MegaTrain-Halo/` that:
1. Installs and imports on ROCm 7.12
2. Trains models up to 7B on Strix Halo (128 GB unified)
3. Uses simplified unified memory pipeline (no PCIe streaming)
4. Maintains CPU-side optimizer for memory efficiency
5. Passes all 10 tests in the testing plan
6. Documents all changes from upstream in CHANGELOG.md

---

## Risks

| Risk | Severity | Mitigation |
|------|----------|------------|
| csrc/ CUDA kernels too complex to hipify | MEDIUM | Start with Python-only fallbacks. Custom C++ can be added incrementally. |
| MegaTrain's model/ layer assumes NVIDIA internals | LOW | Uses HuggingFace standard API — should be GPU-agnostic. |
| Unified memory slower than expected | LOW | ~240 GB/s is fast. Benchmark and compare to standard PyTorch. |
| flash-attn ROCm build incompatible with MegaTrain's API | MEDIUM | Check import paths. MegaTrain uses `flash_attn.flash_attn_func`. |

---

## Hardware Optimization Notes (Strix Halo gfx1151)

> Added 2026-04-09 based on AMADEUS implementation findings.

### halo_training Already Covers Most of This
The `halo_training/` package is operational:
- Mode A/B auto-selection, DeepSpeed CPUAdam, gradient checkpointing
- `autokernel.optimize(model, training=True)` for HIP kernel replacements
- 4 custom ops with autograd backward (rmsnorm, rotary_emb, silu_gate_mul, fused_res_rmsnorm)

### Key Finding: Scan Implementation
For any SSM/recurrence porting, use **chunked linear recurrence** (chunk_size=64). Both sequential loops and `torch.associative_scan` yield only 1.3K tok/s (4% MFU). Chunked gives 6.4K tok/s (16% MFU) — 5x improvement.

### Reference Implementation
`models/amadeus.py` — complete AMADEUS model with chunked scan, verified on gfx1151.

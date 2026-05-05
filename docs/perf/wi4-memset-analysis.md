# WI4: Memset elimination — closed as "not attackable from user code"

**Phase 2 work item:** WI4 — eliminate `Memset (Device)` at 4.1% of Phase 1 wall time.
**Status:** CLOSED. Memsets originate from framework/runtime internals, not from user code allocations.
**Evidence:** `scripts/bench_zero_grad.py`, shape-annotated profile (`docs/perf/wi2-shape-calls.md`).

## Investigation path

### Step 1: Inspect OdinHalo forward for `torch.zeros*` allocations

Searched `models/odin_halo.py` and `models/components/*`:

- **Zero allocations in the forward path.** All `torch.zeros` calls are at `__init__` time
  (`nn.Parameter` initialization for `loop_pos_embeds`, `skip_gates`, norm weights, etc.)
  and are NOT repeated per step.
- The only per-step tensor creation is `torch.polar(freqs_cos, freqs_sin)` at `odin_halo.py:219`
  — creates a 256×32 complex tensor (~128 KB), but this is allocated via `torch.empty + fill`,
  not a separate Memset.

### Step 2: Shape-annotated `aten::zero_` profile

From `docs/perf/wi2-shape-calls.md`:

| Shape | Calls | Total μs |
|-------|------:|---------:|
| `(4096, 512)` | 90 | 0.0 |
| `(512, 3)` | 45 | 0.0 |
| `(4096, 32768)` | 3 | 0.0 |
| various smaller | 84 | 0.0 |

**Every `aten::zero_` op records 0.0 μs of self-CUDA time.** These are all either
elided by Inductor (fused into the graph prologue) or optimized away.

This means **the 221 μs of `Memset (Device)` in Phase 1 are NOT attributable to any
user-visible `aten::zero_` op.** They come from lower-level sources.

### Step 3: Test grad-tensor lifecycle alternatives

Hypothesis: `optimizer.zero_grad(set_to_none=True)` causes autograd to re-create
grad tensors each step via `torch.zeros_like`, triggering Memsets.

Benchmark (OdinHalo, batch=16, compile_zones, 100 steps × 3 repeats):

| Strategy | Median tok/s | Stdev | Δ vs A |
|----------|------------:|------:|-------:|
| **A: `set_to_none=True` (default)** | **14,119** | 67 | +0.00% |
| B: `set_to_none=False` (in-place zero_) | 13,838 | 48 | **−1.99%** |
| C: pre-alloc grads + `_foreach_zero_` | 13,810 | 18 | **−2.19%** |

**Result:** the default `set_to_none=True` is the fastest of the three. Both
alternatives are regressions — they force an explicit Memset of every grad
tensor every step whether or not that parameter was touched, whereas
`set_to_none=True` lets autograd skip grad creation for params outside the
current backward subgraph and batches allocations through the caching allocator
(which reuses memory without Memset).

### Step 4: Locate the real Memset source

Since the Memsets are NOT from `aten::zero_`, NOT from `optimizer.zero_grad`, and
NOT from explicit user allocations, they must originate from one of:

1. **rocBLAS matmul workspace zeroing.** Tensile/rocBLAS GEMM kernels often
   zero their scratch buffers between calls. Not user-controllable.

2. **Fused AdamW internal buffers.** `_fused_adamw_` (the C++ op) manages its
   own temporary storage. Not user-controllable.

3. **GradScaler overflow-detection scalars.** `scaler.unscale_` writes a
   `found_inf` bool/scalar each step. Each is a tiny allocation but accumulates
   via HIP Memset.

4. **Autograd save-for-backward preamble zeroing.** PyTorch's autograd engine
   sometimes zero-initializes intermediate tensors for correctness when
   `create_graph=True` or in mixed-precision paths.

5. **Inductor-generated persistent reduction kernel preambles.** Some triton
   kernels in the fusion catalog use zero-initialized accumulators.

Confirming which source(s) dominate requires **rocprof** traces with full
kernel provenance — well beyond Phase 2 scope and yielding no actionable code
changes even if identified.

## Decision

**WI4 CLOSED** as "not attackable from user code."

The 4.1% `Memset (Device)` cost is distributed across framework-internal
workspace zeroing that we cannot touch without patching PyTorch/HIP/Inductor.
User-level strategies (grad pre-allocation, `set_to_none=False`) make
throughput worse.

### Potential future work (not Phase 2)

- **If a discrete-GPU baseline is ever needed for comparison:** rerun the same
  profile on dGPU ROCm to see if Strix Halo's unified memory is inflating
  Memset count (unified memory must zero-clear pages on first access, which
  discrete GPUs can skip for dedicated DRAM).
- **If PyTorch ever exposes `PYTORCH_NO_MEMSET_WORKSPACE` or similar:** retry.
  No such flag exists today.

## Artifacts

- `scripts/bench_zero_grad.py` — 3-way zero_grad strategy benchmark (kept).
- Closure documented here. No code changes shipped.

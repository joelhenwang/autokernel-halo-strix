# Design: torch.compile + Custom Op Registration + Weight Pipelining

**Date:** 2026-04-08
**Status:** Approved for implementation
**Target:** AMD Strix Halo gfx1151 (RDNA 3.5, 170 GB/s, no MFMA)

## Problem

We have two optimization paths that are currently mutually exclusive:

1. **torch.compile (Inductor)**: 1.16x prefill speedup on LlamaModel7B by fusing PyTorch-native operations (embedding, attention softmax, transposes, residual adds)
2. **HIP kernel replacements**: 1.053x prefill speedup from 4 hand-tuned kernels (rmsnorm 3.3x, fused_residual_add_rmsnorm 6.6x, silu_gate_mul 1.6x, rotary_embedding 3.7x)

torch.compile fails when it encounters our dynamically-loaded HIP modules because it can't trace through `load_inline()` extensions. Registering our kernels as `torch.library.custom_op` makes them visible to the compiler as opaque nodes, allowing Inductor to fuse everything BETWEEN them.

**Expected combined speedup:** ~1.22x prefill (1.16x Inductor fusion + ~1.05x kernel wins).

## Part 1: Custom Op Registration

### New file: `kernels/hip/_torch_ops.py`

Registers each HIP kernel as a `torch.library.custom_op`:

```python
import torch
from typing import Tuple

# --- RMSNorm ---
@torch.library.custom_op("autokernel::rmsnorm", mutates_args=())
def rmsnorm_op(x: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    from kernels.hip.rmsnorm import kernel_fn
    return kernel_fn(x, weight)

@rmsnorm_op.register_fake
def _(x: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    return x.new_empty(x.shape)

# --- Rotary Embedding (fp32 intermediate) ---
@torch.library.custom_op("autokernel::rotary_emb_fp32", mutates_args=())
def rotary_emb_fp32_op(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    from kernels.hip.rotary_embedding import kernel_fn_fp32
    return kernel_fn_fp32(x, cos, sin)

@rotary_emb_fp32_op.register_fake
def _(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    return x.new_empty(x.shape)

# --- SiLU Gate Multiply (SwiGLU) ---
@torch.library.custom_op("autokernel::silu_gate_mul", mutates_args=())
def silu_gate_mul_op(gate: torch.Tensor, up: torch.Tensor) -> torch.Tensor:
    from kernels.hip.silu_gate_mul import kernel_fn
    return kernel_fn(gate, up)

@silu_gate_mul_op.register_fake
def _(gate: torch.Tensor, up: torch.Tensor) -> torch.Tensor:
    return gate.new_empty(gate.shape)

# --- Fused Residual Add + RMSNorm (dual output) ---
@torch.library.custom_op("autokernel::fused_res_rmsnorm", mutates_args=())
def fused_res_rmsnorm_op(
    x: torch.Tensor, residual: torch.Tensor, weight: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor]:
    from kernels.hip.fused_residual_add_rmsnorm import kernel_fn_dual
    return kernel_fn_dual(x, residual, weight)

@fused_res_rmsnorm_op.register_fake
def _(
    x: torch.Tensor, residual: torch.Tensor, weight: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor]:
    return x.new_empty(x.shape), x.new_empty(x.shape)
```

### Model patching: `_CompiledOptimizedLlamaModel`

Instead of wrapping modules (which breaks torch.compile tracing), create a new forward function that calls custom ops directly:

```python
def _patch_llama_for_compile(model):
    """Replace LlamaModel ops with custom op calls, then compile."""
    import kernels.hip._torch_ops  # register custom ops

    # Patch RMSNorm.forward to use custom op
    original_rmsnorm_forward = model.norm.__class__.forward
    def rmsnorm_forward(self, x):
        if x.dtype == torch.float16:
            return torch.ops.autokernel.rmsnorm(x, self.weight)
        return original_rmsnorm_forward(self, x)

    # Patch FeedForward.forward for SwiGLU fusion
    # Patch Attention.forward for rotary + fused QKV
    # Patch TransformerBlock.forward for fused residual + norm
    ...

    return torch.compile(model, backend="inductor")
```

The key insight: patching `forward` methods on the classes (not instances) allows torch.compile to trace through the full graph while hitting our custom ops at the right points.

### verify.py integration

New flag: `--compile-with-kernels`

```bash
python verify.py --model models/llama_7b.py --class-name LlamaModel7B \
    --input-shape 1,512 --compile-with-kernels
```

This:
1. Loads the model
2. Patches forward methods to call custom ops
3. Applies `torch.compile(model, backend="inductor")`
4. Runs benchmark (reference uses the compiled model, so correctness is vs uncompiled)

## Part 2: Weight Pipelining

### Phase A: `mode="reduce-overhead"`

torch.compile's `reduce-overhead` mode uses CUDA graphs internally, which can overlap memory transfers with compute. Try:

```python
model = torch.compile(model, backend="inductor", mode="reduce-overhead")
```

This requires static shapes (no dynamic KV-cache growth). For prefill benchmarks (fixed input shape), this works directly. For decode, pre-allocate max-length KV-cache buffers.

### Phase B: Manual CUDA Graph Capture (if Phase A insufficient)

Capture the decode loop as a CUDA graph:

```python
# Pre-allocate static buffers
kv_cache = allocate_kv_cache(max_seq_len=2048, ...)
input_buf = torch.empty(1, 1, dim, device="cuda", dtype=torch.float16)

# Capture
g = torch.cuda.CUDAGraph()
with torch.cuda.graph(g):
    output = model.decode_step(input_buf, kv_cache, pos=0)

# Replay (much faster — no Python overhead, enables pipelining)
for i in range(n_tokens):
    input_buf.copy_(next_token_embedding)
    g.replay()
```

CUDA graphs eliminate Python overhead (~0.5ms/step) and allow the GPU scheduler to overlap weight loads with compute across layers.

## Files Modified

1. **`kernels/hip/_torch_ops.py`** (NEW) — custom op registrations
2. **`verify.py`** — `--compile-with-kernels` flag, model patching logic
3. No changes to existing kernel files

## Success Criteria

- Custom ops register without error
- `torch.compile` traces the full model graph without graph breaks
- Prefill speedup: >=1.15x (matching or exceeding torch.compile alone)
- Correctness: max_abs_error < 0.01 vs uncompiled model
- Decode with reduce-overhead: measurable tok/s improvement

## Risks

- ROCm Inductor may not handle tuple returns from custom ops (fused_res_rmsnorm returns 2 tensors)
- `register_fake` for dynamic shapes (sequence length varies) may cause recompilation
- `reduce-overhead` mode may not support ROCm CUDA graphs on gfx1151
- Custom ops are opaque to Inductor — it can't fuse INTO them, only AROUND them

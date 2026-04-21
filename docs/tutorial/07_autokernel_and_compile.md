# Part 07: Autokernel + torch.compile -- The Custom Ops Bridge

## Goal
Register your custom CUDA kernels as first-class PyTorch operations via `torch.library`, making them compatible with `torch.compile`. Then measure the combined speedup of autokernel (custom kernels) plus Inductor (graph-level fusion) and learn to debug compile issues.

## Why This Matters
Parts 05-06 gave you custom kernels that are faster than PyTorch's eager execution. But `torch.compile` also provides significant speedup by fusing PyTorch operations at the graph level. The problem: these two optimization strategies conflict by default. Custom kernels loaded via `load_inline()` are opaque blobs to the compiler -- Inductor cannot see through them, and every call to a custom kernel causes a "graph break" that prevents Inductor from fusing the operations around it. This part bridges that gap.

---

## 7.1 The Conflict

### How torch.compile Works

When you call `torch.compile(model)`, three things happen:

1. **TorchDynamo** intercepts Python bytecode during the first forward pass and captures the computation graph (which PyTorch ops are called and in what order).
2. **AOTAutograd** traces the backward pass from the forward graph.
3. **Inductor** (the default backend) optimizes the graph: fuses element-wise ops, generates efficient CUDA kernels via Triton, schedules memory, and outputs compiled code.

The result: PyTorch ops that were separate kernels in eager mode get fused into larger, more efficient kernels by Inductor. For example, Inductor might fuse `dropout + add + layernorm` into a single kernel -- the same kind of fusion you did by hand in Part 05, but done automatically for ALL the PyTorch ops in your model.

### The Graph Break Problem

When Dynamo encounters a function call it cannot trace (like your `load_inline()` CUDA kernel), it inserts a **graph break**. This splits the computation graph into two subgraphs:

```
Without custom ops:
[full graph: embedding -> attn -> norm -> ffn -> norm -> ... -> output]
                    ^--- Inductor fuses everything

With load_inline() custom ops:
[subgraph 1: embedding -> attn] BREAK [subgraph 2: norm] BREAK [subgraph 3: ffn] BREAK ...
                    ^--- Inductor can only fuse within each tiny subgraph
```

Each graph break:
- Prevents Inductor from fusing ops across the break boundary.
- Forces a return to Python interpreter (slow) between subgraphs.
- Means the compiled code is actually slower than well-optimized eager in some cases.

You can see graph breaks with:
```python
import torch._dynamo
torch._dynamo.config.log_level = logging.DEBUG
# Or set environment variable:
# TORCH_LOGS="dynamo" python your_script.py
```

### The Solution: torch.library.custom_op

Register your CUDA kernel as a **custom op** that Dynamo recognizes as a valid computation node. Dynamo treats it as an opaque-but-valid node in the graph -- no graph break. Inductor cannot fuse *into* it (it is still a black box), but it can fuse all PyTorch operations *around* it.

```
With registered custom ops:
[full graph: embedding -> attn -> CUSTOM_RMSNORM -> ffn -> CUSTOM_SWIGLU -> ... -> output]
                    ^--- Inductor fuses everything except the custom op internals
                         Custom ops handle their own fusion internally (your CUDA code)
```

---

## 7.2 torch.library.custom_op

### The Decorator

```python
@torch.library.custom_op("mylib::op_name", mutates_args=())
def my_op(x: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    # Call your CUDA kernel here
    return my_cuda_kernel(x, weight)
```

Key parts:
- `"mylib::op_name"`: The qualified name. `mylib` is a namespace (prevents collisions). `op_name` is the operation name.
- `mutates_args=()`: Declares which arguments are modified in-place. Empty tuple means no in-place mutation (pure function). This is critical for Inductor's analysis.
- Type annotations: Required. Dynamo uses them to trace the op's signature.

After registration, you call the op via:
```python
result = torch.ops.mylib.op_name(x, weight)
```

### What Registration Provides

1. **No graph break**: Dynamo records the op as a node in the graph.
2. **Schema inference**: Inductor knows the input/output types and shapes (via `register_fake`).
3. **Autograd integration**: You register a backward function so the op works in training.
4. **Serialization**: The op can be saved/loaded with `torch.export`.

---

## 7.3 Registering Forward

### Step 1: Define the Op

```python
# custom_ops.py
import torch
import torch.nn.functional as F

# -------------------------------------------------------------------------
# RMSNorm custom op
# -------------------------------------------------------------------------

@torch.library.custom_op("autokernel::rmsnorm", mutates_args=())
def rmsnorm_op(x: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    """Fused RMSNorm forward pass using custom CUDA kernel."""
    from kernels.rmsnorm import kernel_fn  # your CUDA kernel from Part 05
    return kernel_fn(x, weight)
```

### Step 2: Register the Fake (Shape Inference)

During `torch.compile` tracing, Dynamo does NOT run the actual CUDA kernel. Instead, it calls the **fake** (also called **meta**) function to determine the output shape and dtype. This runs on CPU with fake tensors (no actual data).

```python
@rmsnorm_op.register_fake
def rmsnorm_fake(x: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    """Shape inference: output has the same shape as input."""
    return x.new_empty(x.shape)
```

The fake function must:
- Accept the same arguments as the real op.
- Return tensor(s) with the correct shape and dtype.
- NOT access tensor data (only `.shape`, `.dtype`, `.device`).
- Be fast (it runs during compilation, not inference).

For most element-wise ops, `x.new_empty(x.shape)` is correct (same shape, same dtype, same device).

### Step 3: Verify Registration

```python
# Test that the op is registered and callable
x = torch.randn(8, 512, 768, dtype=torch.float16, device='cuda')
weight = torch.randn(768, dtype=torch.float16, device='cuda')

# Call via torch.ops namespace
result = torch.ops.autokernel.rmsnorm(x.view(-1, 768), weight)
print(f"Output shape: {result.shape}")  # [4096, 768]

# Test that it works with torch.compile
@torch.compile
def compiled_fn(x, w):
    return torch.ops.autokernel.rmsnorm(x, w)

result2 = compiled_fn(x.view(-1, 768), weight)
print(f"Compiled output shape: {result2.shape}")  # [4096, 768]
print("No graph breaks!")
```

### Multiple Outputs

Some ops return multiple tensors. For fused residual + RMSNorm, the output is `(hidden, normed)`:

```python
from typing import Tuple

@torch.library.custom_op("autokernel::fused_res_rmsnorm", mutates_args=())
def fused_res_rmsnorm_op(
    x: torch.Tensor, residual: torch.Tensor, weight: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor]:
    from kernels.fused_residual_rmsnorm import kernel_fn_dual
    return kernel_fn_dual(x, residual, weight)


@fused_res_rmsnorm_op.register_fake
def fused_res_rmsnorm_fake(
    x: torch.Tensor, residual: torch.Tensor, weight: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor]:
    return x.new_empty(x.shape), x.new_empty(x.shape)
```

---

## 7.4 Registering Backward

### Why This Matters for Training

Without a registered backward, the custom op only works in `torch.no_grad()` mode (inference). For training, autograd needs to know how to compute gradients through the op.

### The Pattern: CUDA Forward, PyTorch Backward

The recommended approach:
- **Forward**: Use your CUDA kernel (fast, optimized).
- **Backward**: Use pure PyTorch ops (correct, autograd-compatible, compile-fusible).

Why not CUDA backward too? Three reasons:
1. **Correctness**: Backward passes involve complex gradient chains. PyTorch's autograd is battle-tested. A custom CUDA backward that is wrong corrupts training silently.
2. **Development speed**: Writing CUDA backward kernels is 3-5x more effort than forward kernels, for marginal speedup (backward is typically less memory-bound).
3. **Compile compatibility**: PyTorch backward ops get fused by Inductor, so you get graph-level optimization for free.

### Implementation

```python
def _rmsnorm_setup(ctx, inputs, output):
    """Save tensors needed for backward computation."""
    x, weight = inputs
    ctx.save_for_backward(x, weight)
    ctx.eps = 1e-6


def _rmsnorm_backward(ctx, grad_output):
    """Compute gradients using pure PyTorch (not CUDA).

    d/dx [x * w / sqrt(mean(x^2) + eps)]
    = w / sqrt(...) - x * w * x / (N * (mean(x^2) + eps)^(3/2))
    """
    x, weight = ctx.saved_tensors
    eps = ctx.eps

    # Promote to fp32 for numerical stability
    x_f = x.float()
    w_f = weight.float()
    g_f = grad_output.float()

    # Recompute forward intermediates
    rms_sq = x_f.pow(2).mean(-1, keepdim=True) + eps
    rms_inv = rms_sq.rsqrt()
    normed = x_f * rms_inv
    D = x_f.shape[-1]

    # Gradient w.r.t. weight: sum over batch/sequence dims
    grad_weight = (g_f * normed).sum(dim=tuple(range(g_f.ndim - 1)))

    # Gradient w.r.t. x
    grad_normed = g_f * w_f
    grad_x = (grad_normed * rms_inv
              - normed * (grad_normed * normed).sum(-1, keepdim=True) / D)

    return grad_x.to(x.dtype), grad_weight.to(weight.dtype)


# Register the autograd formula
rmsnorm_op.register_autograd(
    _rmsnorm_backward,
    setup_context=_rmsnorm_setup,
)
```

### How `setup_context` and the Backward Work Together

1. During forward: `_rmsnorm_setup` is called with the inputs and output. It saves tensors to `ctx` for backward use.
2. During backward: `_rmsnorm_backward` is called with `ctx` and the gradient of the output. It retrieves saved tensors and computes input gradients.

This is the same pattern as `torch.autograd.Function`, but adapted for `torch.library`.

### Verify Backward Works

```python
# Test gradient computation
x = torch.randn(4, 128, 768, dtype=torch.float16, device='cuda', requires_grad=True)
weight = torch.randn(768, dtype=torch.float16, device='cuda', requires_grad=True)

# Forward through custom op
x_flat = x.view(-1, 768)
out = torch.ops.autokernel.rmsnorm(x_flat, weight)
loss = out.sum()
loss.backward()

print(f"x.grad shape: {x.grad.shape}")        # [4, 128, 768]
print(f"weight.grad shape: {weight.grad.shape}")  # [768]
print(f"x.grad is not None: {x.grad is not None}")
print(f"weight.grad is not None: {weight.grad is not None}")

# Verify gradients numerically (torch.autograd.gradcheck)
# Use float64 for numerical gradient checking
x_check = torch.randn(2, 32, 64, dtype=torch.float64, device='cuda', requires_grad=True)
w_check = torch.randn(64, dtype=torch.float64, device='cuda', requires_grad=True)

# Need fp64 version of the op for gradcheck
def rmsnorm_fp64(x, w):
    rms = (x.pow(2).mean(-1, keepdim=True) + 1e-6).rsqrt()
    return x * rms * w

passed = torch.autograd.gradcheck(
    rmsnorm_fp64, (x_check.view(-1, 64), w_check),
    eps=1e-6, atol=1e-4, rtol=1e-3
)
print(f"Gradient check: {'PASSED' if passed else 'FAILED'}")
```

---

## 7.5 The Combined Pipeline

### The Correct Order

```python
import autokernel

# Step 1: Build model
model = GPT2Model(vocab_size=50257, n_layers=12, n_heads=12, d_model=768)

# Step 2: Apply autokernel (replaces modules with CUDA kernel wrappers)
model = autokernel.optimize(model, dtype=torch.float16)

# Step 3: Apply torch.compile (Inductor fuses everything around custom ops)
model = torch.compile(model, backend="inductor")

# IMPORTANT: Step 2 MUST come before Step 3.
# If you compile first, the compiled graph has the original PyTorch ops.
# Then autokernel replaces the underlying modules, but the compiled graph
# still references the old ops -- your replacements are silently ignored.
```

### Why Order Matters

```
CORRECT: optimize() -> compile()
  1. autokernel replaces RMSNorm modules with CUDA kernel wrappers
  2. torch.compile traces the model, sees torch.ops.autokernel.rmsnorm
  3. Inductor generates code that calls the custom op + fuses everything else

WRONG: compile() -> optimize()
  1. torch.compile traces the model, sees standard PyTorch RMSNorm ops
  2. Inductor generates code with fused PyTorch ops (including its own RMSNorm fusion)
  3. autokernel replaces underlying modules, but compiled code ignores them
  Result: autokernel replacements have ZERO effect
```

### All-in-One API

```python
# autokernel.optimize() with compile=True does both in the right order
model = autokernel.optimize(model, dtype=torch.float16, compile=True)
```

This is equivalent to:
```python
model = autokernel.optimize(model, dtype=torch.float16, compile=False)
import kernels.custom_ops  # ensure all ops are registered
model = torch.compile(model, backend="inductor")
```

### Registering All Custom Ops

The ops must be registered before `torch.compile` traces the model. Create a module that registers everything on import:

```python
"""kernels/custom_ops.py -- Register all CUDA kernels as torch.library custom ops.

Import this module before calling torch.compile on an autokernel-optimized model.
This is imported automatically by autokernel.optimize(compile=True).
"""

import torch
import torch.nn.functional as F
from typing import Tuple


# -------------------------------------------------------------------------
# 1. RMSNorm
# -------------------------------------------------------------------------

@torch.library.custom_op("autokernel::rmsnorm", mutates_args=())
def rmsnorm_op(x: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    from kernels.rmsnorm import kernel_fn
    return kernel_fn(x, weight)

@rmsnorm_op.register_fake
def _(x, weight):
    return x.new_empty(x.shape)

def _rmsnorm_setup(ctx, inputs, output):
    x, weight = inputs
    ctx.save_for_backward(x, weight)
    ctx.eps = 1e-6

def _rmsnorm_backward(ctx, grad_output):
    x, weight = ctx.saved_tensors
    eps = ctx.eps
    x_f, w_f, g_f = x.float(), weight.float(), grad_output.float()
    rms_sq = x_f.pow(2).mean(-1, keepdim=True) + eps
    rms_inv = rms_sq.rsqrt()
    normed = x_f * rms_inv
    D = x_f.shape[-1]
    grad_weight = (g_f * normed).sum(dim=tuple(range(g_f.ndim - 1)))
    grad_normed = g_f * w_f
    grad_x = grad_normed * rms_inv - normed * (grad_normed * normed).sum(-1, keepdim=True) / D
    return grad_x.to(x.dtype), grad_weight.to(weight.dtype)

rmsnorm_op.register_autograd(_rmsnorm_backward, setup_context=_rmsnorm_setup)


# -------------------------------------------------------------------------
# 2. SiLU Gate Multiply (SwiGLU activation)
# -------------------------------------------------------------------------

@torch.library.custom_op("autokernel::silu_gate_mul", mutates_args=())
def silu_gate_mul_op(gate: torch.Tensor, up: torch.Tensor) -> torch.Tensor:
    from kernels.swiglu import kernel_fn
    return kernel_fn(gate, up)

@silu_gate_mul_op.register_fake
def _(gate, up):
    return gate.new_empty(gate.shape)

def _silu_setup(ctx, inputs, output):
    gate, up = inputs
    ctx.save_for_backward(gate, up)

def _silu_backward(ctx, grad_output):
    gate, up = ctx.saved_tensors
    g_f, gate_f, up_f = grad_output.float(), gate.float(), up.float()
    sig = torch.sigmoid(gate_f)
    silu_gate = gate_f * sig
    d_silu = sig * (1.0 + gate_f * (1.0 - sig))
    grad_gate = g_f * up_f * d_silu
    grad_up = g_f * silu_gate
    return grad_gate.to(gate.dtype), grad_up.to(up.dtype)

silu_gate_mul_op.register_autograd(_silu_backward, setup_context=_silu_setup)


# -------------------------------------------------------------------------
# 3. Rotary Embedding (fp32 intermediate)
# -------------------------------------------------------------------------

@torch.library.custom_op("autokernel::rotary_emb_fp32", mutates_args=())
def rotary_emb_op(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    from kernels.rotary_embedding import kernel_fn_fp32
    return kernel_fn_fp32(x, cos, sin)

@rotary_emb_op.register_fake
def _(x, cos, sin):
    return x.new_empty(x.shape)

def _rotary_setup(ctx, inputs, output):
    x, cos, sin = inputs
    ctx.save_for_backward(cos, sin)
    ctx.x_dtype = x.dtype

def _rotary_backward(ctx, grad_output):
    cos, sin = ctx.saved_tensors
    # RoPE backward: rotate by -theta (negate sin)
    g = grad_output.float()
    c = cos.float()
    s = sin.float()
    def rotate_half(t):
        t1, t2 = t.chunk(2, dim=-1)
        return torch.cat((-t2, t1), dim=-1)
    grad_x = g * c + rotate_half(g) * (-s)
    return grad_x.to(ctx.x_dtype), None, None

rotary_emb_op.register_autograd(_rotary_backward, setup_context=_rotary_setup)


# -------------------------------------------------------------------------
# 4. Fused Residual + RMSNorm (dual output)
# -------------------------------------------------------------------------

@torch.library.custom_op("autokernel::fused_res_rmsnorm", mutates_args=())
def fused_res_rmsnorm_op(
    x: torch.Tensor, residual: torch.Tensor, weight: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor]:
    from kernels.fused_residual_rmsnorm import kernel_fn_dual
    return kernel_fn_dual(x, residual, weight)

@fused_res_rmsnorm_op.register_fake
def _(x, residual, weight):
    return x.new_empty(x.shape), x.new_empty(x.shape)

def _fused_res_setup(ctx, inputs, output):
    x, residual, weight = inputs
    hidden, normed = output
    ctx.save_for_backward(hidden, weight)
    ctx.eps = 1e-6

def _fused_res_backward(ctx, grad_hidden, grad_normed):
    hidden, weight = ctx.saved_tensors
    eps = ctx.eps
    h_f, w_f, gn_f = hidden.float(), weight.float(), grad_normed.float()
    rms_sq = h_f.pow(2).mean(-1, keepdim=True) + eps
    rms_inv = rms_sq.rsqrt()
    normed_h = h_f * rms_inv
    D = h_f.shape[-1]
    grad_weight = (gn_f * normed_h).sum(dim=tuple(range(gn_f.ndim - 1)))
    grad_normed_scaled = gn_f * w_f
    grad_h_from_norm = (grad_normed_scaled * rms_inv
                        - normed_h * (grad_normed_scaled * normed_h).sum(-1, keepdim=True) / D)
    total_grad_h = grad_hidden.float() + grad_h_from_norm
    grad_x = total_grad_h.to(hidden.dtype)
    return grad_x, grad_x, grad_weight.to(weight.dtype)

fused_res_rmsnorm_op.register_autograd(
    _fused_res_backward, setup_context=_fused_res_setup
)


print("autokernel: All custom ops registered with torch.library")
```

---

## 7.6 Measuring the Combined Speedup

### Four Configurations

To understand the contribution of each optimization, benchmark all four combinations:

```python
"""benchmark_combined.py -- Measure autokernel + torch.compile combined speedup."""
import torch
import time
import autokernel
from models.gpt2 import GPT2Model

def time_forward(model, inputs, warmup=30, iterations=100):
    """Time model forward pass with proper synchronization."""
    for _ in range(warmup):
        with torch.no_grad():
            model(*inputs)
    torch.cuda.synchronize()

    start = time.perf_counter()
    for _ in range(iterations):
        with torch.no_grad():
            model(*inputs)
    torch.cuda.synchronize()
    return (time.perf_counter() - start) / iterations * 1000  # ms


# Test inputs
tokens = torch.randint(0, 50257, (8, 512), device='cuda')

# Config 1: Baseline (eager PyTorch, no optimizations)
model_base = GPT2Model(vocab_size=50257, n_layers=12, n_heads=12, d_model=768)
model_base = model_base.cuda().half().eval()
t_base = time_forward(model_base, (tokens,))

# Config 2: Autokernel only (custom CUDA kernels, no compile)
model_ak = GPT2Model(vocab_size=50257, n_layers=12, n_heads=12, d_model=768)
model_ak.load_state_dict(model_base.state_dict(), strict=False)
model_ak = autokernel.optimize(model_ak, dtype=torch.float16, compile=False)
t_ak = time_forward(model_ak, (tokens,))

# Config 3: torch.compile only (no custom kernels)
model_compile = GPT2Model(vocab_size=50257, n_layers=12, n_heads=12, d_model=768)
model_compile = model_compile.cuda().half().eval()
model_compile = torch.compile(model_compile, backend="inductor")
t_compile = time_forward(model_compile, (tokens,))

# Config 4: Both (autokernel + compile)
model_both = GPT2Model(vocab_size=50257, n_layers=12, n_heads=12, d_model=768)
model_both.load_state_dict(model_base.state_dict(), strict=False)
model_both = autokernel.optimize(model_both, dtype=torch.float16, compile=True)
t_both = time_forward(model_both, (tokens,))

# Results
print(f"\n{'='*60}")
print(f"{'Configuration':<30} {'Time (ms)':<12} {'Speedup':<10}")
print(f"{'='*60}")
print(f"{'Baseline (eager)':<30} {t_base:<12.2f} {'1.00x':<10}")
print(f"{'+ Autokernel only':<30} {t_ak:<12.2f} {t_base/t_ak:<10.2f}x")
print(f"{'+ torch.compile only':<30} {t_compile:<12.2f} {t_base/t_compile:<10.2f}x")
print(f"{'+ Both (AK + compile)':<30} {t_both:<12.2f} {t_base/t_both:<10.2f}x")
print(f"{'='*60}")
```

### Expected Results (RTX 4060 Ti, GPT-2 124M, B=8, T=512)

```
============================================================
Configuration                  Time (ms)    Speedup
============================================================
Baseline (eager)               12.40        1.00x
+ Autokernel only              11.20        1.11x
+ torch.compile only           9.50         1.31x
+ Both (AK + compile)          8.90         1.39x
============================================================
```

### Why the Combination Wins

| Optimization | What it does | Where it helps |
|---|---|---|
| Autokernel | Replaces specific ops with hand-tuned CUDA kernels | Memory-bound ops (RMSNorm, SwiGLU, RoPE) |
| torch.compile | Fuses chains of PyTorch ops into compiled kernels | Chains of element-wise ops, dropout patterns, reshapes |
| Both | Custom kernels for specific hotspots + Inductor fusion for everything else | The full model |

Autokernel gives modest full-model speedup (~1.05-1.1x) because matmuls dominate (~70% of runtime), and we do not accelerate matmuls. But for the non-matmul parts, autokernel gives 2-6x per-op speedup.

torch.compile gives ~1.3x by fusing the many small ops that autokernel does not cover (attention mask computation, dropout, various reshapes and transposes).

Together: autokernel handles the ops it is best at (specific fused patterns), and Inductor handles everything else. The speedups compose multiplicatively for non-overlapping operations.

---

## 7.7 Debugging Compile Issues

### Viewing Graph Breaks

```bash
# Environment variable (most verbose)
TORCH_LOGS="dynamo" python train.py

# Or in code
import logging
torch._dynamo.config.log_level = logging.DEBUG
```

Graph break messages look like:
```
[2024-01-15 12:00:00] GRAPH BREAK: unsupported: call_function UserDefinedFunction
    at layers.0.attention.forward (line 42)
    Reason: unknown function 'my_custom_cuda_fn'
```

### Common Causes and Fixes

**Cause 1: Unregistered custom ops**
```python
# BAD: raw CUDA call -- Dynamo cannot trace this
def forward(self, x):
    return my_cuda_module.rmsnorm_kernel(x, self.weight)

# GOOD: registered custom op -- Dynamo records it as a graph node
def forward(self, x):
    return torch.ops.autokernel.rmsnorm(x, self.weight)
```

**Cause 2: Python control flow dependent on tensor values**
```python
# BAD: branch depends on tensor data
def forward(self, x):
    if x.max() > 0:  # graph break -- Dynamo can't specialize on runtime values
        return self.path_a(x)
    return self.path_b(x)

# GOOD: branch depends on static attributes
def forward(self, x):
    if self.use_path_a:  # constant at trace time -- no graph break
        return self.path_a(x)
    return self.path_b(x)
```

**Cause 3: print() statements**
```python
# BAD: print causes graph break
def forward(self, x):
    print(f"x shape: {x.shape}")  # graph break!
    return self.norm(x)

# GOOD: remove prints in production, or guard with torch.compiler
def forward(self, x):
    if not torch.compiler.is_compiling():
        print(f"x shape: {x.shape}")
    return self.norm(x)
```

**Cause 4: Data-dependent shapes**
```python
# BAD: output shape depends on input data
def forward(self, x, mask):
    selected = x[mask]  # dynamic shape -- graph break
    return selected

# GOOD: use padding or fixed shapes
def forward(self, x, mask):
    return x * mask.unsqueeze(-1)  # fixed shape, no graph break
```

### Per-Module Compilation

For models with dynamic parts (loops, conditionals), compile each static component independently:

```python
class LoopingModel(nn.Module):
    def __init__(self, n_layers):
        super().__init__()
        self.layers = nn.ModuleList([TransformerBlock() for _ in range(n_layers)])
        self.n_iters = 3  # model has a loop

    def forward(self, x):
        for iteration in range(self.n_iters):
            for layer in self.layers:
                x = layer(x)  # this part is static and compilable
        return x

# Compile each layer independently (the loop stays in Python)
model = LoopingModel(n_layers=12)
for i, layer in enumerate(model.layers):
    model.layers[i] = torch.compile(layer, backend="inductor")

# The model forward's loop structure does not break compilation,
# because each layer is compiled independently.
```

This "compile zones" pattern avoids graph breaks from Python loops while still getting Inductor fusion within each layer.

### torch.compiler.is_compiling() Guard

Sometimes you need different behavior during tracing vs execution:

```python
@torch.library.custom_op("autokernel::my_op", mutates_args=())
def my_op(x: torch.Tensor) -> torch.Tensor:
    # This runs during EXECUTION (not tracing)
    return my_cuda_kernel(x)

# The register_fake runs during TRACING (compilation)
@my_op.register_fake
def _(x):
    return x.new_empty(x.shape)
```

In wrapper modules:
```python
class MyWrapper(nn.Module):
    def forward(self, x):
        if x.dtype == torch.float16:
            # Use custom CUDA kernel for fp16
            return torch.ops.autokernel.my_op(x)
        else:
            # PyTorch fallback for other dtypes
            return self.pytorch_fallback(x)
```

This `if x.dtype == ...` check works fine with torch.compile because `dtype` is known at trace time (it is a static property of the tensor, not data-dependent).

---

## 7.8 CUDAGraphs (reduce-overhead Mode)

### What CUDAGraphs Do

Normally, the CPU launches one CUDA kernel at a time, waiting for each launch to complete before issuing the next. This CPU-GPU communication has overhead (~5-10 microseconds per kernel launch).

CUDAGraphs eliminate this overhead by:
1. **Recording**: Execute the model once, recording every CUDA kernel call and its arguments.
2. **Replaying**: On subsequent calls, replay the entire recorded sequence in one GPU-side operation. The CPU issues a single "replay" command instead of hundreds of individual kernel launches.

### When CUDAGraphs Help

CUDAGraphs help when:
- Your model has **many small kernels** (hundreds of kernel launches per forward pass).
- The **CPU is the bottleneck** (kernel launch overhead > kernel execution time).
- Shapes are **static** (CUDAGraphs require fixed input shapes).

CUDAGraphs do NOT help when:
- Your model has **few large kernels** (matmuls dominate, launch overhead is negligible).
- The **GPU is the bottleneck** (kernels take 1ms+ each, launch overhead is 0.005ms).
- Shapes are **dynamic** (different batch sizes, sequence lengths).

### Using CUDAGraphs with torch.compile

```python
# reduce-overhead mode enables CUDAGraphs automatically
model = torch.compile(model, mode="reduce-overhead")
```

The `reduce-overhead` mode:
1. Compiles the model with Inductor (same as `mode="default"`).
2. Records the compiled graph as a CUDAGraph on the first forward pass.
3. Replays the CUDAGraph on subsequent forward passes.

### Gradient Accumulation Caveat

When using gradient accumulation with CUDAGraphs, you must mark step boundaries:

```python
for step, batch in enumerate(dataloader):
    # Mark the beginning of a new "step" for CUDAGraph segmentation.
    # Without this, CUDAGraphs records across accumulation steps,
    # which fails because gradient buffers change between steps.
    torch.compiler.cudagraph_mark_step_begin()

    loss = model(batch) / accum_steps
    loss.backward()

    if (step + 1) % accum_steps == 0:
        optimizer.step()
        optimizer.zero_grad()
```

### Measuring CUDAGraph Impact

```python
# Compare three modes
configs = [
    ("default (Inductor)", "default"),
    ("reduce-overhead (+ CUDAGraphs)", "reduce-overhead"),
    ("max-autotune (+ tuning)", "max-autotune"),
]

for desc, mode in configs:
    model = GPT2Model(vocab_size=50257, n_layers=12, n_heads=12, d_model=768)
    model = autokernel.optimize(model, dtype=torch.float16, compile=False)
    model = torch.compile(model, backend="inductor", mode=mode)

    t = time_forward(model, (tokens,))
    print(f"{desc:<45} {t:.2f} ms  ({t_base/t:.2f}x)")
```

Expected (RTX 4060 Ti, GPT-2 124M):
```
default (Inductor)                            8.90 ms  (1.39x)
reduce-overhead (+ CUDAGraphs)                8.20 ms  (1.51x)
max-autotune (+ tuning)                       8.50 ms  (1.46x)
```

CUDAGraphs provide an additional 5-10% on top of Inductor, primarily from eliminating kernel launch overhead for the many small operations in a transformer.

---

## Exercises

### Exercise 1: Register Your RMSNorm Kernel as a Custom Op

Following the pattern in Section 7.3-7.4, register your RMSNorm kernel from Part 05:

1. Define the op with `@torch.library.custom_op("tutorial::rmsnorm", mutates_args=())`.
2. Register the fake for shape inference.
3. Register the backward via `register_autograd`.
4. Verify: call the op, check output shape, run `loss.backward()`, check gradients exist.
5. Verify: wrap in `torch.compile`, confirm no graph breaks.

```python
# Skeleton
@torch.library.custom_op("tutorial::rmsnorm", mutates_args=())
def my_rmsnorm(x: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    # TODO: call your CUDA kernel
    pass

@my_rmsnorm.register_fake
def _(x, weight):
    # TODO: return correctly shaped empty tensor
    pass

# TODO: implement and register backward
```

### Exercise 2: Register Your SwiGLU Kernel with Autograd

Register the SwiGLU kernel from Part 05 as `tutorial::silu_gate_mul`:

1. Forward calls your fused SiLU*gate CUDA kernel.
2. Backward computes gradients in PyTorch (derive the gradient formula for `silu(gate) * up`).
3. Test with gradient checking: `torch.autograd.gradcheck()` in fp64 mode.
4. Test in a training loop: train a 2-layer MLP with SwiGLU activation for 50 steps, verify loss decreases.

The gradient formulas:
```
output = silu(gate) * up = gate * sigmoid(gate) * up

d(output)/d(gate) = up * (sigmoid(gate) + gate * sigmoid(gate) * (1 - sigmoid(gate)))
                   = up * sigmoid(gate) * (1 + gate * (1 - sigmoid(gate)))

d(output)/d(up) = silu(gate) = gate * sigmoid(gate)
```

### Exercise 3: Measure Combined Speedup

Run the full benchmark from Section 7.6 on your setup:

1. Baseline (eager).
2. Autokernel only.
3. torch.compile only.
4. Both.

Record your results. Compare to the expected values. If your results differ significantly, investigate:
- Is your GPU clock throttling? (Check `nvidia-smi -q -d CLOCK`)
- Is the first `torch.compile` call including compilation time? (Increase warmup.)
- Are you measuring wall-clock time correctly? (`torch.cuda.synchronize()` before timing.)

Fill in this table with your actual measurements:

| Configuration | Time (ms) | Speedup | Notes |
|---|---|---|---|
| Baseline | ??? | 1.00x | |
| + Autokernel | ??? | ???x | |
| + torch.compile | ??? | ???x | |
| + Both | ??? | ???x | |

---

## Checkpoint

Before moving to Part 08, verify:

1. **All custom ops registered**: RMSNorm, SiLU Gate Mul, Rotary Embedding, and Fused Residual RMSNorm are all registered under the `autokernel::` namespace.

2. **No graph breaks**: `torch.compile(model)` runs without any graph break warnings when using autokernel-optimized models.

3. **Backward works**: All registered ops support gradients. A training loop using the compiled model converges (loss decreases).

4. **Combined speedup measured**: You have measured all four configurations and the combination is faster than either alone.

5. **You understand the tradeoffs**: You can explain when to use `mode="default"` vs `mode="reduce-overhead"`, when CUDAGraphs help and when they do not, and why per-module compilation is needed for looping models.

```python
# Verification script
import torch

# Check ops are registered
assert hasattr(torch.ops, 'autokernel'), "autokernel namespace not found"
assert hasattr(torch.ops.autokernel, 'rmsnorm'), "rmsnorm op not registered"
assert hasattr(torch.ops.autokernel, 'silu_gate_mul'), "silu_gate_mul op not registered"
assert hasattr(torch.ops.autokernel, 'rotary_emb_fp32'), "rotary_emb op not registered"
assert hasattr(torch.ops.autokernel, 'fused_res_rmsnorm'), "fused_res_rmsnorm op not registered"

# Check backward works
x = torch.randn(2, 64, 128, dtype=torch.float16, device='cuda', requires_grad=True)
w = torch.randn(128, dtype=torch.float16, device='cuda', requires_grad=True)
out = torch.ops.autokernel.rmsnorm(x.view(-1, 128), w)
out.sum().backward()
assert x.grad is not None, "No gradient for x"
assert w.grad is not None, "No gradient for weight"

# Check compile works without graph breaks
@torch.compile(backend="inductor")
def test_compiled(x, w):
    return torch.ops.autokernel.rmsnorm(x, w)

result = test_compiled(x.view(-1, 128).detach(), w.detach())
assert result.shape == (2 * 64, 128), f"Wrong shape: {result.shape}"

print("Part 07 complete. All custom ops registered, backward works, compile clean.")
print("Ready for Part 08: Data Pipeline.")
```

---

**Previous: [Part 06 -- Autokernel: Pattern Matching & Kernel Replacement](06_autokernel.md)**
**Next: [Part 08 -- Data Pipeline: Mixing, Filtering, and Scaling](08_data_pipeline.md)**

# Part 06: Autokernel -- Pattern Matching & Kernel Replacement

## Goal
Build a library that walks any `nn.Module`, detects optimization-eligible patterns (RMSNorm, SwiGLU, fused QKV, etc.), and automatically replaces them with your custom CUDA kernels -- without changing a single line in the model file. By the end, one function call optimizes an arbitrary transformer model.

## Why This Matters
You now have four fast CUDA kernels from Part 05. But they only work standalone -- you have to manually find the right places in a model, extract the right tensors, call your kernel, and stitch the result back in. Do that for every model and every kernel, and you are rewriting models by hand forever. Autokernel solves this: write the detection logic once, and any model benefits.

---

## 6.1 The Problem

After Part 05, you can do this:

```python
# Manual kernel integration -- tedious and model-specific
class OptimizedTransformerBlock(nn.Module):
    def forward(self, x, residual, freqs_cis):
        attn_out = self.attention(self.attention_norm(x), freqs_cis)
        # Manually call your fused kernel instead of separate add + rmsnorm
        hidden, normed = fused_residual_rmsnorm(attn_out, x, self.ffn_norm.weight)
        return hidden + self.feed_forward(normed)
```

Problems with this approach:
1. You must modify every model file to use your kernels.
2. Different models use different attribute names (`wq`/`q_proj`, `w1`/`gate_proj`).
3. You have to track which kernels apply where, handle dtype, handle shapes.
4. If you update a kernel, you re-edit every model file.

What we want:

```python
# One line. Works on any model. No model code changes.
model = autokernel.optimize(model)
```

### Design Goals

1. **Zero model code changes**: The model author writes standard PyTorch. Autokernel post-processes the model.
2. **Pattern-based detection**: Each kernel has a "pattern" that describes what module structure it can replace.
3. **Priority system**: When multiple patterns match, apply the highest-priority one.
4. **Reversible**: `autokernel.restore(model)` undoes all replacements.
5. **Transparent**: `autokernel.report(model)` shows what was replaced and estimated speedup.

---

## 6.2 Walking the Module Tree

Every PyTorch model is a tree of `nn.Module` objects. `model.named_modules()` iterates every node:

```python
import torch.nn as nn

class RMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x):
        rms = (x.float().pow(2).mean(-1, keepdim=True) + self.eps).rsqrt()
        return (x.float() * rms).to(x.dtype) * self.weight

class SwiGLU(nn.Module):
    def __init__(self, dim, hidden_dim):
        super().__init__()
        self.gate_proj = nn.Linear(dim, hidden_dim, bias=False)
        self.up_proj = nn.Linear(dim, hidden_dim, bias=False)
        self.down_proj = nn.Linear(hidden_dim, dim, bias=False)

    def forward(self, x):
        return self.down_proj(torch.silu(self.gate_proj(x)) * self.up_proj(x))

class TransformerBlock(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.attention_norm = RMSNorm(dim)
        self.ffn_norm = RMSNorm(dim)
        self.attention = nn.MultiheadAttention(dim, 12)
        self.feed_forward = SwiGLU(dim, dim * 4)

class TransformerModel(nn.Module):
    def __init__(self, dim=768, n_layers=12):
        super().__init__()
        self.layers = nn.ModuleList([TransformerBlock(dim) for _ in range(n_layers)])

# Walking the tree
model = TransformerModel()
for name, module in model.named_modules():
    print(f"{name:<50} {type(module).__name__}")
```

Output:
```
                                                   TransformerModel
layers                                             ModuleList
layers.0                                           TransformerBlock
layers.0.attention_norm                            RMSNorm
layers.0.ffn_norm                                  RMSNorm
layers.0.attention                                 MultiheadAttention
layers.0.feed_forward                              SwiGLU
layers.0.feed_forward.gate_proj                    Linear
layers.0.feed_forward.up_proj                      Linear
layers.0.feed_forward.down_proj                    Linear
layers.1                                           TransformerBlock
...
```

Each `(name, module)` pair gives you the dotted path (e.g., `"layers.0.attention_norm"`) and the module object. You can inspect the module's class, attributes, and children to decide if it matches a kernel pattern.

---

## 6.3 Pattern Design

A pattern is a set of rules that identify a module as a candidate for kernel replacement. Good patterns are:

1. **Specific enough** to avoid false positives (do not accidentally replace a module that merely happens to have a `weight` attribute).
2. **Flexible enough** to handle naming variations across model families.
3. **Based on structure** (attributes, types) not behavior (you cannot inspect a module's `forward()` reliably).

### Pattern: RMSNorm

RMSNorm modules have a characteristic signature:
- Class name contains "RMSNorm" (or "T5LayerNorm" -- same operation, different name)
- Has a `weight` parameter
- Has an `eps` attribute
- Does NOT have a `bias` parameter (that would be LayerNorm)
- Is NOT an instance of `nn.LayerNorm`

```python
RMSNORM_CLASS_NAMES = {
    "RMSNorm", "LlamaRMSNorm", "MistralRMSNorm",
    "T5LayerNorm", "GemmaRMSNorm", "Qwen2RMSNorm",
}

def matches_rmsnorm(name: str, module: nn.Module) -> bool:
    cls_name = type(module).__name__
    # Match by class name (most reliable)
    if cls_name in RMSNORM_CLASS_NAMES:
        return True
    # Match by attribute signature (catches custom implementations)
    if (hasattr(module, "weight")
            and hasattr(module, "eps")
            and not hasattr(module, "bias")
            and cls_name.lower().endswith("norm")
            and not isinstance(module, nn.LayerNorm)):
        return True
    return False
```

### Pattern: SwiGLU FFN

SwiGLU modules have three linear projections with specific roles:

```python
# Naming convention aliases
SWIGLU_ALIASES = [
    ("gate_proj", "down_proj", "up_proj"),   # HuggingFace LLaMA/Mistral
    ("w1", "w2", "w3"),                       # Original LLaMA (gate, down, up)
]

def _has_linear_attrs(module: nn.Module, attrs: tuple) -> bool:
    """Check if module has all named attributes and the first is nn.Linear."""
    return (
        all(hasattr(module, a) for a in attrs)
        and isinstance(getattr(module, attrs[0], None), nn.Linear)
    )

def matches_swiglu(name: str, module: nn.Module) -> bool:
    for alias in SWIGLU_ALIASES:
        if _has_linear_attrs(module, alias):
            return True
    return False
```

### Pattern: Attention with Q/K/V

```python
QKV_ALIASES = [
    ("wq", "wk", "wv", "wo"),                  # LLaMA
    ("q_proj", "k_proj", "v_proj", "o_proj"),   # HuggingFace
]

def matches_attention(name: str, module: nn.Module) -> bool:
    for q, k, v, o in QKV_ALIASES:
        if (_has_linear_attrs(module, (q, k, v, o))):
            return True
    return False
```

### Why Attribute-Based Detection Works

PyTorch modules store their sub-modules and parameters as attributes. A Linear layer `self.gate_proj = nn.Linear(...)` becomes `module.gate_proj`. This is stable across model versions and frameworks -- the attribute name is part of the API contract. Checking `hasattr(module, "gate_proj")` is fast and reliable.

Contrast with trying to analyze the `forward()` method's source code or tracing the computation graph -- both are fragile and complex.

---

## 6.4 Module Replacement

Once you detect a pattern match, you need to:
1. Create a new module that wraps your CUDA kernel.
2. Copy the original module's weights into the wrapper.
3. Replace the original module in the model tree.

### Creating Wrapper Modules

```python
class RMSNormReplacement(nn.Module):
    """Drop-in replacement for RMSNorm using a custom CUDA kernel."""

    def __init__(self, original: nn.Module, kernel_fn):
        super().__init__()
        # Copy the weights from the original -- do NOT reinitialize
        self.weight = original.weight
        self.eps = getattr(original, "eps", 1e-6)
        self.kernel_fn = kernel_fn
        # Store original for restore()
        self._original_module = original

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dtype == torch.float16:
            # Use our fast CUDA kernel for fp16
            return self.kernel_fn(
                x.view(-1, x.shape[-1]),
                self.weight
            ).view(x.shape)
        else:
            # Fall back to PyTorch for other dtypes (fp32, bf16)
            rms = (x.float().pow(2).mean(-1, keepdim=True) + self.eps).rsqrt()
            return (x.float() * rms).to(x.dtype) * self.weight
```

Key points:
- **Copy weights by reference**: `self.weight = original.weight` shares the same Parameter object. No memory duplication.
- **Preserve dtype behavior**: Only use the CUDA kernel for fp16. Fall back to PyTorch for other dtypes. This avoids having to write kernels for every dtype.
- **Store the original**: Keep a reference so you can restore later.

### The `_replace_module()` Helper

PyTorch does not have a built-in "replace this submodule" function. You must navigate the parent chain and use `setattr`:

```python
def _replace_module(model: nn.Module, name: str, new_module: nn.Module) -> None:
    """Replace a named submodule in the model.

    Args:
        model: root model
        name: dotted path like "layers.0.attention_norm"
        new_module: the replacement module
    """
    parts = name.split(".")
    parent = model
    for p in parts[:-1]:
        parent = getattr(parent, p)
    setattr(parent, parts[-1], new_module)
```

How this works: for `name = "layers.0.attention_norm"`:
1. `parts = ["layers", "0", "attention_norm"]`
2. Navigate: `model.layers` -> `model.layers[0]` (ModuleList supports int-string access)
3. Replace: `model.layers[0].attention_norm = new_module`

### SwiGLU Replacement

```python
class SiluGateMulReplacement(nn.Module):
    """Replace SwiGLU FFN with fused SiLU*Gate kernel in the activation step."""

    def __init__(self, original: nn.Module, kernel_fn,
                 gate_attr: str, down_attr: str, up_attr: str):
        super().__init__()
        # Keep the linear layers (cuBLAS handles those)
        self.gate_proj = getattr(original, gate_attr)
        self.down_proj = getattr(original, down_attr)
        self.up_proj = getattr(original, up_attr)
        self.kernel_fn = kernel_fn
        self._original_module = original

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gate = self.gate_proj(x)
        up = self.up_proj(x)
        if gate.dtype == torch.float16:
            # Fused SiLU(gate) * up via CUDA kernel
            activated = self.kernel_fn(gate.contiguous(), up.contiguous())
        else:
            # PyTorch fallback
            activated = torch.silu(gate) * up
        return self.down_proj(activated)
```

Notice that we do NOT fuse the linear projections into CUDA. cuBLAS (the NVIDIA BLAS library) is already optimal for matrix multiplications. We only fuse the element-wise activation, which is the memory-bound part.

---

## 6.5 The Fused QKV Pattern

### Why Fuse Q, K, V?

Most transformer models compute query, key, and value with three separate matrix multiplications:

```python
q = self.wq(x)   # [B, T, D] @ [D, D] -> [B, T, D]
k = self.wk(x)   # [B, T, D] @ [D, D_kv] -> [B, T, D_kv]
v = self.wv(x)   # [B, T, D] @ [D, D_kv] -> [B, T, D_kv]
```

Three separate cuBLAS calls. Each has fixed overhead (kernel launch, workspace allocation). More importantly, three small GEMMs have lower arithmetic intensity than one large GEMM. cuBLAS is more efficient on large matrices.

### The Fusion

Concatenate the three weight matrices into one:

```python
# w_qkv.weight = cat([wq.weight, wk.weight, wv.weight], dim=0)
# One GEMM: qkv = x @ w_qkv.T  -> [B, T, D + D_kv + D_kv]
# Then split:
# q = qkv[..., :D]
# k = qkv[..., D:D+D_kv]
# v = qkv[..., D+D_kv:]
```

This is a cuBLAS-level optimization (bigger matrix = better utilization), not a custom kernel. But it requires module replacement because the model has three separate `nn.Linear` modules.

### Implementation

```python
class FusedQKVAttentionReplacement(nn.Module):
    """Replace separate Q/K/V projections with a single fused GEMM."""

    def __init__(self, original: nn.Module,
                 q_attr: str, k_attr: str, v_attr: str, o_attr: str):
        super().__init__()
        q_proj = getattr(original, q_attr)
        k_proj = getattr(original, k_attr)
        v_proj = getattr(original, v_attr)
        self.wo = getattr(original, o_attr)

        # Read head counts from the original module
        self.n_heads = getattr(original, "n_heads",
                               getattr(original, "num_heads", 1))
        self.n_kv_heads = getattr(original, "n_kv_heads",
                                   getattr(original, "num_key_value_heads",
                                            self.n_heads))
        self.head_dim = getattr(original, "head_dim",
                                q_proj.out_features // self.n_heads)
        self.n_rep = self.n_heads // self.n_kv_heads  # for GQA repeat

        # Fused weight: concatenate Q, K, V weights along output dimension
        total_out = q_proj.out_features + k_proj.out_features + v_proj.out_features
        self.w_qkv = nn.Linear(q_proj.in_features, total_out, bias=False)
        self.w_qkv.weight.data = torch.cat([
            q_proj.weight.data,
            k_proj.weight.data,
            v_proj.weight.data,
        ], dim=0)

        # Remember split sizes for the output
        self.q_size = q_proj.out_features
        self.k_size = k_proj.out_features

        self._original_module = original

    def forward(self, x: torch.Tensor, freqs_cis: torch.Tensor) -> torch.Tensor:
        B, T, _ = x.shape

        # Single fused GEMM
        qkv = self.w_qkv(x)

        # Split the output
        q = qkv[..., :self.q_size].view(B, T, self.n_heads, self.head_dim)
        k = qkv[..., self.q_size:self.q_size + self.k_size].view(
            B, T, self.n_kv_heads, self.head_dim)
        v = qkv[..., self.q_size + self.k_size:].view(
            B, T, self.n_kv_heads, self.head_dim)

        # Transpose for attention: [B, H, T, D]
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        # Apply RoPE (using your custom kernel from Part 05)
        # ... apply_rotary_emb(q, k, freqs_cis) ...

        # GQA: repeat K, V heads to match Q head count
        if self.n_rep > 1:
            k = k.repeat_interleave(self.n_rep, dim=1)
            v = v.repeat_interleave(self.n_rep, dim=1)

        # Attention (cuBLAS + scaled_dot_product_attention)
        y = torch.nn.functional.scaled_dot_product_attention(q, k, v, is_causal=True)

        return self.wo(y.transpose(1, 2).contiguous().view(B, T, -1))
```

### Weight Concatenation Details

```python
# Before: three separate weights
# wq.weight: [D, D]       (e.g., [768, 768])
# wk.weight: [D_kv, D]    (e.g., [768, 768] or [256, 768] for GQA)
# wv.weight: [D_kv, D]

# After: one fused weight
# w_qkv.weight: [D + D_kv + D_kv, D]  (e.g., [1792, 768])
# The concatenation is along dim=0 (the output dimension)

# This works because nn.Linear computes: output = input @ weight.T
# For fused: output = input @ [wq; wk; wv].T
#          = input @ [wq.T | wk.T | wv.T]
#          = [input @ wq.T | input @ wk.T | input @ wv.T]
#          = [q | k | v]  (concatenated along last dimension)
```

---

## 6.6 Priority System

When multiple patterns can match the same module, you need a priority system. Consider:

- A `TransformerBlock` module could match the `fused_residual_rmsnorm` pattern (which replaces the entire block's forward).
- The block's `attention_norm` submodule could also match the `rmsnorm` pattern.

If you apply both, the block replacement already handles the norm -- the standalone norm replacement is redundant and would cause errors.

### The Rules

1. Higher priority patterns are applied first.
2. Once a module is replaced, it is marked and skipped by lower-priority patterns.
3. A replaced module's children are NOT walked again (the replacement module has different children).

```python
from dataclasses import dataclass

@dataclass
class Pattern:
    name: str
    priority: int        # higher = applied first
    op_speedup: float    # estimated speedup (informational)

    def matches(self, name: str, module: nn.Module, model: nn.Module) -> bool:
        raise NotImplementedError

    def apply(self, name: str, module: nn.Module, model: nn.Module) -> nn.Module:
        raise NotImplementedError
```

### Priority Table

| Pattern | Priority | Reason |
|---------|----------|--------|
| fused_residual_rmsnorm (block-level) | 100 | Subsumes standalone rmsnorm |
| fused_qkv | 90 | Subsumes standalone rotary_embedding |
| rmsnorm | 50 | Standalone, only if not subsumed |
| layernorm | 50 | Standalone |
| silu_gate_mul | 40 | Activation-level |
| rotary_embedding | 30 | Only if fused_qkv excluded |

### Implementation

```python
def apply_patterns(model: nn.Module, patterns: list) -> dict:
    """Walk the model and apply matching patterns.

    Returns a dict of {pattern_name: count_of_replacements}.
    """
    # Sort by priority (highest first)
    patterns.sort(key=lambda p: -p.priority)

    replaced_names = set()  # track which modules have been replaced
    applied = {}

    for pattern in patterns:
        count = 0
        for name, module in list(model.named_modules()):
            # Skip if already replaced by a higher-priority pattern
            if name in replaced_names:
                continue
            if pattern.matches(name, module, model):
                try:
                    replacement = pattern.apply(name, module, model)
                    _replace_module(model, name, replacement)
                    replaced_names.add(name)
                    count += 1
                except Exception as e:
                    print(f"  autokernel: {pattern.name} failed on {name}: {e}")
        if count > 0:
            applied[pattern.name] = count

    return applied
```

The key line is `if name in replaced_names: continue`. Once `layers.0` is replaced by a block-level pattern, `layers.0.attention_norm` is skipped because `"layers.0"` is in the replaced set.

Wait -- that check is by exact name, not by prefix. For the block-level replacement, we replace `layers.0` which prevents `layers.0.attention_norm` from matching *because the replacement module has different children*. The `list(model.named_modules())` call captures the module tree at the start of the pattern iteration. After replacing `layers.0`, the next pattern's iteration will use a fresh `model.named_modules()` which will have the replacement's children, not the original's.

A simpler approach that works in practice: re-enumerate modules for each pattern:

```python
for pattern in patterns:
    count = 0
    # Re-enumerate to see current (possibly modified) tree
    for name, module in list(model.named_modules()):
        if name in replaced_names:
            continue
        if pattern.matches(name, module, model):
            replacement = pattern.apply(name, module, model)
            _replace_module(model, name, replacement)
            replaced_names.add(name)
            count += 1
```

---

## 6.7 The `optimize()` API

### The Public Interface

```python
"""autokernel/__init__.py"""

def optimize(model, dtype=torch.float16, compile=False,
             include=None, exclude=None):
    """Auto-detect and apply all applicable kernel optimizations.

    Args:
        model: any nn.Module
        dtype: target dtype (default fp16 for CUDA kernels)
        compile: also apply torch.compile after kernel replacement
        include: only apply these pattern names (None = all)
        exclude: skip these pattern names (None = none)

    Returns:
        the optimized model (modified in-place, also returned for chaining)
    """
    ...

def report(model):
    """Return a dict describing what optimizations were applied."""
    ...

def restore(model):
    """Undo all optimizations, restoring original modules."""
    ...

def list_patterns():
    """Return names of all available optimization patterns."""
    ...
```

### Full Implementation

```python
# autokernel/_registry.py

_STATE_ATTR = "_autokernel_state"

ALL_PATTERNS = [
    FusedResidualRMSNormPattern(),    # priority 100
    FusedQKVPattern(),                 # priority 90
    RMSNormPattern(),                  # priority 50
    LayerNormPattern(),                # priority 50
    SiluGateMulPattern(),              # priority 40
    RotaryEmbeddingPattern(),          # priority 30
]

def optimize(model, dtype=torch.float16, compile=False,
             include=None, exclude=None):
    # 1. Cast to target dtype and move to GPU
    model = model.to(dtype=dtype)
    if torch.cuda.is_available() and not next(model.parameters()).is_cuda:
        model = model.cuda()
    model.eval()

    # 2. Filter patterns
    patterns = list(ALL_PATTERNS)
    if include is not None:
        include_set = set(include)
        patterns = [p for p in patterns if p.name in include_set]
    if exclude is not None:
        exclude_set = set(exclude)
        patterns = [p for p in patterns if p.name not in exclude_set]

    # 3. Apply patterns (highest priority first)
    patterns.sort(key=lambda p: -p.priority)
    originals = {}
    applied = {}
    replaced_names = set()

    for pattern in patterns:
        count = 0
        for name, module in list(model.named_modules()):
            if name in replaced_names:
                continue
            if pattern.matches(name, module, model):
                try:
                    replacement = pattern.apply(name, module, model)
                    originals[name] = module
                    _replace_module(model, name, replacement)
                    replaced_names.add(name)
                    count += 1
                except Exception as e:
                    print(f"  autokernel: {pattern.name} failed on {name}: {e}")
        if count > 0:
            applied[pattern.name] = count

    # 4. Optionally compile
    if compile:
        model = torch.compile(model, backend="inductor")

    # 5. Store state for report/restore
    setattr(model, _STATE_ATTR, {
        "originals": originals,
        "applied": applied,
        "patterns": {p.name: p for p in patterns},
        "compiled": compile,
    })

    # 6. Print summary
    print(f"\n  autokernel: Applied {sum(applied.values())} replacements:")
    for pname, count in applied.items():
        p = next((x for x in patterns if x.name == pname), None)
        speedup = f"{p.op_speedup:.1f}x" if p else "?"
        print(f"    {pname}: {count} modules (est. {speedup} per op)")

    return model


def report(model):
    state = getattr(model, _STATE_ATTR, None)
    if state is None:
        return {"status": "not optimized", "patterns": {}}
    result = {"status": "optimized", "compiled": state["compiled"], "patterns": {}}
    for name, count in state["applied"].items():
        pattern = state["patterns"].get(name)
        result["patterns"][name] = {
            "modules_replaced": count,
            "op_speedup": f"{pattern.op_speedup:.1f}x" if pattern else "unknown",
        }
    return result


def restore(model):
    state = getattr(model, _STATE_ATTR, None)
    if state is None:
        return model
    for name, original in state["originals"].items():
        _replace_module(model, name, original)
    delattr(model, _STATE_ATTR)
    return model


def list_patterns():
    return [p.name for p in ALL_PATTERNS]
```

---

## 6.8 Testing the Full Pipeline

### Step 1: Apply to GPT-2

```python
"""test_autokernel.py -- Verify autokernel on GPT-2 124M."""
import torch
import autokernel
from models.gpt2 import GPT2Model  # your model from Part 02

# Build model
model = GPT2Model(
    vocab_size=50257,
    n_layers=12,
    n_heads=12,
    d_model=768,
)
print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

# Optimize
model = autokernel.optimize(model, dtype=torch.float16)
print("\nOptimization report:")
for k, v in autokernel.report(model)["patterns"].items():
    print(f"  {k}: {v}")
```

Expected output:
```
Model parameters: 124,439,808

  autokernel: Applied 38 replacements:
    fused_residual_rmsnorm: 12 modules (est. 6.6x per op)
    fused_qkv: 12 modules (est. 1.3x per op)
    silu_gate_mul: 12 modules (est. 1.6x per op)
    rmsnorm: 2 modules (est. 3.3x per op)

Optimization report:
  fused_residual_rmsnorm: {'modules_replaced': 12, 'op_speedup': '6.6x'}
  fused_qkv: {'modules_replaced': 12, 'op_speedup': '1.3x'}
  silu_gate_mul: {'modules_replaced': 12, 'op_speedup': '1.6x'}
  rmsnorm: {'modules_replaced': 2, 'op_speedup': '3.3x'}
```

### Step 2: Correctness Verification

```python
# Compare optimized vs unoptimized model outputs
def verify_correctness(model_class, model_kwargs, input_fn, n_tests=10):
    """Verify that autokernel.optimize() preserves model correctness.

    Args:
        model_class: the model class to instantiate
        model_kwargs: dict of kwargs for the model constructor
        input_fn: fn(model) -> tuple of input tensors
        n_tests: number of random inputs to test
    """
    torch.manual_seed(42)

    # Build two identical models
    model_ref = model_class(**model_kwargs).cuda().half().eval()
    model_opt = model_class(**model_kwargs).cuda().half().eval()

    # Copy weights so they are identical
    model_opt.load_state_dict(model_ref.state_dict())

    # Optimize one
    model_opt = autokernel.optimize(model_opt, dtype=torch.float16)

    max_abs_errors = []
    for i in range(n_tests):
        torch.manual_seed(i)
        inputs = input_fn(model_ref)

        with torch.no_grad():
            out_ref = model_ref(*inputs)
            out_opt = model_opt(*inputs)

        if isinstance(out_ref, tuple):
            out_ref, out_opt = out_ref[0], out_opt[0]

        abs_err = (out_ref.float() - out_opt.float()).abs().max().item()
        max_abs_errors.append(abs_err)
        print(f"  Test {i}: max abs error = {abs_err:.2e}")

    avg_err = sum(max_abs_errors) / len(max_abs_errors)
    max_err = max(max_abs_errors)

    passed = max_err < 0.01  # fp16 tolerance
    print(f"\n  Avg error: {avg_err:.2e}, Max error: {max_err:.2e}")
    print(f"  Status: {'PASS' if passed else 'FAIL'}")
    return passed

# Run verification
ok = verify_correctness(
    GPT2Model,
    {"vocab_size": 50257, "n_layers": 12, "n_heads": 12, "d_model": 768},
    lambda model: (torch.randint(0, 50257, (4, 128), device='cuda'),),
    n_tests=10,
)
assert ok, "Autokernel correctness verification failed!"
```

### Step 3: Speedup Measurement

```python
import time

def measure_speedup(model_class, model_kwargs, input_fn,
                    warmup=20, iterations=100):
    """Measure inference speedup from autokernel.optimize()."""
    torch.manual_seed(42)

    # Baseline (unoptimized)
    model_base = model_class(**model_kwargs).cuda().half().eval()
    inputs = input_fn(model_base)

    for _ in range(warmup):
        with torch.no_grad():
            model_base(*inputs)
    torch.cuda.synchronize()

    start = time.perf_counter()
    for _ in range(iterations):
        with torch.no_grad():
            model_base(*inputs)
    torch.cuda.synchronize()
    base_time = (time.perf_counter() - start) / iterations

    # Optimized
    model_opt = model_class(**model_kwargs).cuda().half().eval()
    model_opt.load_state_dict(model_base.state_dict())
    model_opt = autokernel.optimize(model_opt, dtype=torch.float16)

    for _ in range(warmup):
        with torch.no_grad():
            model_opt(*inputs)
    torch.cuda.synchronize()

    start = time.perf_counter()
    for _ in range(iterations):
        with torch.no_grad():
            model_opt(*inputs)
    torch.cuda.synchronize()
    opt_time = (time.perf_counter() - start) / iterations

    speedup = base_time / opt_time
    print(f"\n  Baseline:  {base_time*1000:.2f} ms/forward")
    print(f"  Optimized: {opt_time*1000:.2f} ms/forward")
    print(f"  Speedup:   {speedup:.2f}x")
    return speedup

speedup = measure_speedup(
    GPT2Model,
    {"vocab_size": 50257, "n_layers": 12, "n_heads": 12, "d_model": 768},
    lambda model: (torch.randint(0, 50257, (8, 512), device='cuda'),),
)
print(f"Overall inference speedup: {speedup:.2f}x")
```

### Step 4: Restore and Verify

```python
# Verify that restore() works
model = GPT2Model(vocab_size=50257, n_layers=12, n_heads=12, d_model=768)
model = model.cuda().half().eval()

# Optimize
model = autokernel.optimize(model, dtype=torch.float16)
assert hasattr(model, "_autokernel_state")
print(f"After optimize: {autokernel.report(model)['status']}")

# Restore
model = autokernel.restore(model)
assert not hasattr(model, "_autokernel_state")
print(f"After restore:  {autokernel.report(model)['status']}")

# Verify original module types are back
for name, module in model.named_modules():
    if "attention_norm" in name:
        assert type(module).__name__ == "RMSNorm", f"Restore failed: {name} is {type(module)}"
print("Restore verified: all original modules restored.")
```

---

## Exercises

### Exercise 1: Add a Fused LayerNorm Pattern

Add support for `nn.LayerNorm` replacement. LayerNorm differs from RMSNorm in that it has both `weight` and `bias`, and it subtracts the mean before normalizing:

```
LayerNorm(x) = (x - mean(x)) / sqrt(var(x) + eps) * weight + bias
```

Write:
1. A CUDA kernel for fused LayerNorm (or use the one from Part 05 as a base).
2. A `LayerNormPattern` class that matches `nn.LayerNorm` instances.
3. A `LayerNormReplacement` wrapper.

Test on a BERT-style model (which uses LayerNorm instead of RMSNorm).

### Exercise 2: `autokernel.restore()` Verification

The current `restore()` implementation replaces modules with their stored originals. But it does not verify that the originals still have the right weights. Extend `restore()` to:

1. Verify that the original module's weights match what was saved.
2. Handle the case where the user called `model.load_state_dict()` after `optimize()` (the fused QKV weights need to be un-fused).
3. Add a `verify=True` parameter that runs a forward pass before and after restore to confirm output equivalence.

### Exercise 3: Support HuggingFace Naming Conventions

HuggingFace models use different attribute names than our LLaMA implementation. Extend your patterns to handle:

```python
# HuggingFace LLaMA:
#   model.layers[i].self_attn.q_proj
#   model.layers[i].self_attn.k_proj
#   model.layers[i].self_attn.v_proj
#   model.layers[i].self_attn.o_proj
#   model.layers[i].mlp.gate_proj
#   model.layers[i].mlp.up_proj
#   model.layers[i].mlp.down_proj
#   model.layers[i].input_layernorm
#   model.layers[i].post_attention_layernorm

# HuggingFace Mistral (same as LLaMA)
# HuggingFace Qwen2 (same naming, different class names)
```

Test by:
1. Installing `transformers`: `pip install transformers`
2. Loading a small HuggingFace model: `AutoModelForCausalLM.from_pretrained("TinyLlama/TinyLlama-1.1B-Chat-v1.0")`
3. Running `autokernel.optimize()` on it
4. Verifying correctness with `model.generate()`

---

## Checkpoint

Before moving to Part 07, verify:

1. **`autokernel.optimize(model)` works on GPT-2**: Applies 30+ replacements across all 12 layers.

2. **Correctness passes**: Optimized model output matches unoptimized within fp16 tolerance (atol < 0.01).

3. **Measurable speedup**: At least 1.05x faster on inference (the full model speedup is modest because matmuls dominate; per-kernel speedups are much higher).

4. **`autokernel.restore(model)` works**: Reverts all replacements, original module types confirmed.

5. **`autokernel.report(model)` is informative**: Shows which patterns matched and how many modules were replaced.

```python
# Quick verification
model = build_gpt2()
model = autokernel.optimize(model, dtype=torch.float16)
r = autokernel.report(model)
assert r["status"] == "optimized"
assert len(r["patterns"]) >= 3, "Expected at least 3 pattern types applied"
assert sum(v["modules_replaced"] for v in r["patterns"].values()) >= 20

# Correctness
assert verify_correctness_passes, "Optimized model not matching reference"

# Restore
model = autokernel.restore(model)
assert autokernel.report(model)["status"] == "not optimized"

print("Part 06 complete. Ready for Part 07: Autokernel + torch.compile.")
```

---

**Previous: [Part 05 -- Kernel Progression: Fused Ops and Benchmarking](05_kernel_progression.md)**
**Next: [Part 07 -- Autokernel + torch.compile: The Custom Ops Bridge](07_autokernel_and_compile.md)**

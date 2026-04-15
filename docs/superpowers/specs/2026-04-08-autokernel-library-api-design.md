---
title: "AutoKernel Library API"
domain: design-specs
type: spec
status: active
related:
  - docs/superpowers/specs/2026-04-08-torch-compile-custom-ops-design.md
  - docs/superpowers/specs/2026-04-10-optimization-libraries-design.md
tags: [%autokernel, %api, %pattern-matching]
---

# Design: AutoKernel Library API

**Date:** 2026-04-08
**Status:** Draft
**Goal:** `model = autokernel.optimize(model)` — one-liner GPU kernel optimization for any PyTorch model on AMD ROCm.

## Problem

AutoKernel's optimization pipeline is currently spread across verify.py (wrappers), kernels/hip/ (HIP kernels), and bench.py (evaluation). Using the optimizations requires:
- Running verify.py with many flags (`--incremental --fused-qkv --compile-with-kernels`)
- Knowledge of which kernels apply to which model architecture
- Manual configuration of kernel order and composition

Users should be able to write `model = autokernel.optimize(model)` and get the best available optimization automatically.

## API Surface

```python
import autokernel

# Simple: auto-detect architecture, apply all applicable optimizations
model = autokernel.optimize(model, dtype=torch.float16)

# With torch.compile integration
model = autokernel.optimize(model, dtype=torch.float16, compile=True)

# Selective
model = autokernel.optimize(
    model, 
    dtype=torch.float16,
    include=["rmsnorm", "silu_gate_mul"],   # only these
    exclude=["rotary_embedding"],            # skip this one
    compile=True,
    compile_mode="default",
)

# Inspect
report = autokernel.report(model)  
# Returns dict: {pattern_name: {count, modules_replaced, op_speedup}}

# List available patterns
autokernel.list_patterns()
# → ['rmsnorm', 'layernorm', 'silu_gate_mul', 'fused_residual_rmsnorm', 
#     'rotary_embedding', 'fused_qkv', 'fused_bias_gelu', 'fused_bias_silu']

# Restore original model
autokernel.restore(model)
```

## Architecture

### Pattern Registry

Each optimization is a `Pattern` — a self-contained unit that detects and replaces matching modules:

```python
@dataclass
class Pattern:
    name: str                   # "rmsnorm"
    priority: int               # higher = applied first (block-level before op-level)
    op_speedup: float           # benchmark speedup vs PyTorch (for reporting)
    
    def matches(self, name: str, module: nn.Module, model: nn.Module) -> bool:
        """Return True if this pattern applies to the given module."""
        ...
    
    def apply(self, name: str, module: nn.Module, model: nn.Module) -> Tuple[nn.Module, int]:
        """Replace module, return (wrapper, count_replaced)."""
        ...
```

### Module Detection Strategy

Patterns use multiple signals to detect applicable modules across different model architectures:

| Signal | Example | Covers |
|--------|---------|--------|
| Class name | `"RMSNorm"`, `"LayerNorm"` | Any framework |
| `isinstance` | `nn.LayerNorm` | Standard PyTorch |
| Attribute names | `wq/wk/wv/wo` OR `q_proj/k_proj/v_proj/o_proj` | LLaMA, HuggingFace |
| Child module pattern | Module has `w1 + w2 + w3` Linear children | SwiGLU detection |
| Weight shapes | 3 Linears with same input dim | QKV fusion candidates |

### Patterns Implemented (AMD HIP backend)

| Pattern | Priority | Detection | Wrapper | Op Speedup |
|---------|----------|-----------|---------|------------|
| `fused_residual_rmsnorm` | 100 | `attention + attention_norm + feed_forward + ffn_norm` children | `_FusedResidualRMSNormBlockWrapper` | 6.6x |
| `fused_qkv` | 90 | `wq + wk + wv + wo` OR `q_proj + k_proj + v_proj + o_proj` Linear children | `_FusedQKVAttentionWrapper` | 1.3x (GEMM) |
| `rmsnorm` | 50 | Class name contains "RMSNorm" | `_RMSNormWrapper` | 3.3x |
| `layernorm` | 50 | `isinstance(m, nn.LayerNorm)` | `_LayerNormWrapper` | 1.06x |
| `silu_gate_mul` | 40 | Module has `w1 + w2 + w3` OR `gate_proj + up_proj + down_proj` | `_SiluGateMulWrapper` | 1.6x |
| `rotary_embedding` | 30 | Attention module with RoPE (wq/wk/wv/wo + model has freqs_cis) | `_RotaryAttentionWrapper` | 3.7x |
| `fused_bias_gelu` | 20 | `nn.Linear` followed by GELU activation | Inline replacement | 1.9x |
| `fused_bias_silu` | 20 | `nn.Linear` followed by SiLU activation | Inline replacement | 1.9x |

Priority order ensures block-level patterns (which subsume op-level ones) are applied first. Lower-priority patterns skip modules already wrapped by higher-priority ones.

### torch.compile Integration

When `compile=True`:
1. Register custom ops via `kernels/hip/_torch_ops.py`
2. Patch model forward methods to call `torch.ops.autokernel.*` instead of raw PyTorch ops
3. Apply `torch.compile(model, backend="inductor", mode=compile_mode)`
4. Handle `profile.py` sys.path conflict automatically

### Complex Buffer Preservation

`autokernel.optimize()` handles the `model.to(dtype=float16)` complex buffer destruction issue automatically — saves complex buffers before dtype cast, restores after. This is transparent to the user.

## File Structure (new files only)

```
autokernel/
    __init__.py      # optimize(), report(), restore(), list_patterns()
    _registry.py     # PatternRegistry class, optimize engine, module walking
    _patterns.py     # All Pattern implementations (imports wrappers from verify.py)
```

Total: ~300 lines across 3 files. All wrapper classes and kernel functions are imported from existing code.

## What Stays Unchanged

- `kernels/hip/` — all HIP kernels untouched
- `kernels/hip/_compile.py` — compilation utility untouched  
- `kernels/hip/_torch_ops.py` — custom op registrations untouched
- `verify.py` — all existing flags and wrappers untouched (autokernel/ imports FROM verify.py)
- `bench.py` — benchmark harness untouched
- `reference.py` — reference implementations untouched

## Supported Models (Day 1)

| Model | Architecture | Applicable Patterns |
|-------|-------------|-------------------|
| LlamaModel (ours) | RoPE + SwiGLU + RMSNorm + GQA | All 8 patterns |
| LlamaModel7B (ours) | Same | All 8 patterns |
| HuggingFace LLaMA | `q_proj/k_proj/v_proj/o_proj` naming | All 8 (with alias detection) |
| HuggingFace Mistral | Same as LLaMA | All 8 |
| GPT-2 (ours) | LayerNorm + GELU + `c_attn` fused QKV | layernorm, fused_bias_gelu |
| BERT (ours) | LayerNorm + GELU | layernorm, fused_bias_gelu |

## Verification

```bash
# Existing verify.py tests still work
python verify.py --model models/llama_7b.py --class-name LlamaModel --input-shape 1,512 --incremental

# New library API test
python -c "
import torch
from models.llama_7b import LlamaModel
import autokernel

model = LlamaModel().cuda().half()
optimized = autokernel.optimize(model, compile=True)
print(autokernel.report(optimized))

# Quick correctness check
ids = torch.randint(0, 32000, (1, 128), device='cuda')
with torch.no_grad():
    out = optimized(ids)
print(f'Output shape: {out.shape}')
"
```

## Success Criteria

- `autokernel.optimize(LlamaModel())` works with zero configuration
- `autokernel.optimize(LlamaModel7B())` achieves >= 1.15x prefill speedup with `compile=True`
- `autokernel.report()` accurately lists applied optimizations
- `autokernel.restore()` returns model to original state
- No regressions on existing verify.py / bench.py workflows
- GPT-2 and BERT models also optimizable (LayerNorm + bias+GELU patterns)

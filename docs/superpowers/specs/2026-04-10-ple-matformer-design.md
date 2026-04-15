---
title: "PLE + MatFormer: Composable Modules for Elastic Architecture Design"
domain: design-specs
type: spec
status: active
related:
  - docs/superpowers/specs/2026-04-10-update-hypotheses-design.md
  - docs/superpowers/specs/2026-04-10-training-evolution-design.md
tags: [%ple, %matformer, %ablation]
---

# PLE + MatFormer: Composable Modules for Elastic Architecture Design

**Date:** 2026-04-10
**Status:** Design approved, pending implementation
**Supersedes:** VIRTUOSO.md (simple PLE on AMADEUS)

## Problem

Our 250M-param architectures train well but deploy as fixed-size models. Google's Gemma 4 demonstrated two techniques that improve both quality and deployment flexibility:

1. **Per-Layer Embeddings (PLE)**: Each decoder layer gets its own token embedding, solving the "frontloading problem" where standard transformers pack all token info into one initial embedding that degrades with depth.

2. **MatFormer**: Train one model containing nested submodels as strict weight prefixes. Extract smaller models at inference by slicing weight matrices — zero post-training cost.

We want both as composable modules the scientist can plug into any base architecture.

## Design

### Principle: Separate Modules, Independent Testing

Three independent modules, each testable in isolation:
- `PLEModule` — per-layer embeddings (3 modes for ablation)
- `MatFormerSwiGLU` — nested FFN training (drop-in SwiGLU replacement)
- `MatFormerAttention` — nested attention heads (optional, for speculative decoding)

A `Virtuoso` composer wraps any base model and attaches these modules without modifying base code.

---

## PLEModule (`models/ple.py`)

### Config

```python
@dataclass
class PLEConfig:
    vocab_size: int = 50257
    d_model: int = 1024
    n_layers: int = 16
    ple_mode: str = "a+b"      # "a" | "b" | "a+b"
    ple_dim: int = 64           # Path A bottleneck
    ple_table_rank: int = 32    # Path B shared table rank
    ple_table_dim: int = 64     # Path B per-layer dim
    inject_point: str = "before"  # "before" or "after" block
```

### Path A: Context-Aware Projection

Each layer projects the current hidden state through a gated bottleneck:

```
h → context_down[i](Linear d_model→ple_dim) → GELU → context_up[i](Linear ple_dim→d_model)
```

No embedding table. The per-layer signal comes from the model's own hidden state. Each layer learns a unique "lens" on the current representation.

**Parameters:** 16 layers × (1024×64 + 64×1024) + 16 norms = **~2.1M**

### Path B: Factored Token-Identity

One shared embedding table with per-layer learned mixing:

```
shared_table[token_ids]  →  (B, T, 32)
@ layer_mixing[i]        →  (B, T, 64)     # einsum('btr,rp->btp')
→ table_up[i](Linear 64→d_model)
```

No dependence on hidden state. Each layer gets a specialized view of the original token identity.

**Parameters:** shared_table(50257×32) + mixing(16×32×64) + table_up(16×64×1024) + norms = **~2.7M**

### Path A+B: Combined

```
combined = context_aware_out + token_identity_out
output = ple_norm[i](combined)
→ add to residual stream
```

**Parameters: ~4.8M** (sum of A + B)

### Forward Signature

```python
def forward(self, h: Tensor, token_ids: Tensor, layer_idx: int) -> Tensor:
    """Returns (B, T, d_model) to ADD to residual stream."""
```

### Design Decisions

- **Injection point "before" (default):** Injects before the block so the mixer (Griffin, Mamba, attention) sees the PLE-augmented hidden state. Matches the existing engram injection pattern (`models/engram.py`).
- **Factored table (Path B):** One shared table across all layers instead of 16 separate tables. At vocab=50257, 16 separate tables at dim=32 would cost 25.7M params (too expensive). The factored design costs only 2.7M.
- **GELU activation (Path A):** Matches Gemma 4's implementation. GELU is smoother than SiLU for the gating function.
- **No gating between paths A and B in combined mode:** Simple sum. The RMSNorm handles scale. Adding a learned gate would be premature — the ablation will tell us if both paths help.

---

## MatFormerSwiGLU (`models/matformer.py`)

### Config

```python
@dataclass
class MatFormerConfig:
    granularities: Tuple[float, ...] = (0.125, 0.25, 0.5, 1.0)
    mode: str = "train"        # "train" | "eval" | "extract"
    fixed_granularity: Optional[float] = None
```

### How Nesting Works

Existing SwiGLU layout (fused gate+up):
- `w_gate_up`: `Linear(d_model, 2 * ffn_inner)` — first half is gate, second half is up
- `w_down`: `Linear(ffn_inner, d_model)`

For granularity `g` with `m = round_to_128(g * ffn_inner)`:
- `F.linear(x, w_gate_up.weight[:2*m, :])` → chunk into gate(m) and up(m)
- `F.silu(gate) * up` → hidden of size m
- `F.linear(hidden, w_down.weight[:, :m])` → output

### Slice Widths (ffn_inner=2560)

| Granularity | m | w_gate_up slice | w_down slice | FLOPs |
|-------------|---|-----------------|--------------|-------|
| 1/8 | 384 | (768, 1024) | (1024, 384) | 15% |
| 1/4 | 640 | (1280, 1024) | (1024, 640) | 25% |
| 1/2 | 1280 | (2560, 1024) | (1024, 1280) | 50% |
| 1 | 2560 | (5120, 1024) | (1024, 2560) | 100% |

All slice widths are multiples of 128 (aligned for Tensile tiles on gfx1151).

### Training

Each step: sample ONE random granularity index (uniform), apply to ALL layers. Standard CE loss. The key insight from the MatFormer paper: shared parameters (smaller granularities' neurons) receive more gradient updates since they're part of every configuration. This gradient asymmetry is what makes extracted submodels outperform independently trained models.

### Extraction

```python
def extract_submodel(self, granularity: float) -> SwiGLU:
    """Returns a plain SwiGLU with sliced weights. Zero overhead at inference."""
    m = round_to_128(granularity * self.ffn_inner)
    ffn = SwiGLU(self.d_model, m)
    ffn.w_gate_up.weight.data = self.w_gate_up.weight.data[:2*m, :].clone()
    ffn.w_down.weight.data = self.w_down.weight.data[:, :m].clone()
    return ffn
```

### Parameters: Zero Additional

MatFormer adds no parameters. It trains the same SwiGLU weights with nested sampling.

---

## MatFormerAttention (optional)

Wraps `GQAAttentionLayer` with nested query head counts. KV heads stay fixed (2).

| Granularity | Active Q heads | KV heads |
|-------------|---------------|----------|
| 0.25 | 2 | 2 |
| 0.5 | 4 | 2 |
| 1.0 | 8 | 2 |

During training: randomly sample head count. During inference: use fixed head count.

**Shared KV cache:** Draft model (2 Q heads) and verifier (8 Q heads) share the same KV cache since the draft's KV heads are a prefix of the verifier's. This enables efficient speculative decoding without duplicating KV computation.

---

## Virtuoso Composer (`models/virtuoso.py`)

```python
class Virtuoso(nn.Module):
    def __init__(
        self,
        base_model_cls,       # Tempest, Amadeus, Prometheus, or any nn.Module
        use_ple: bool = True,
        ple_mode: str = "a+b",
        use_matformer: bool = True,
        **base_kwargs,
    ):
        self.base = base_model_cls(**base_kwargs)
        if use_ple:
            self.ple = PLEModule(PLEConfig(ple_mode=ple_mode, ...))
        if use_matformer:
            self._wrap_ffn_layers()  # in-place SwiGLU → MatFormerSwiGLU

    def _wrap_ffn_layers(self):
        """Module surgery: replace every SwiGLU in base with MatFormerSwiGLU."""
        for name, module in self.base.named_modules():
            if isinstance(module, SwiGLU):
                parent = _get_parent(self.base, name)
                wrapped = MatFormerSwiGLU.from_swiglu(module, MatFormerConfig())
                setattr(parent, name.split('.')[-1], wrapped)

    def forward(self, input_ids, targets=None):
        h = self.base.tok_embeddings(input_ids)
        # ... dispatch per-architecture forward with PLE injection at each layer
```

**Key design:** PLE injection in the outer loop, MatFormer via module surgery. Base model code is never modified.

---

## Kernel Optimization

### New: `kernels/hip/fused_ple_gate.py`

Fuses PLE Path A: `Linear(1024→64) → GELU → elementwise_mul → Linear(64→1024) → RMSNorm`.

At ple_dim=64, this is purely memory-bound. A single kernel avoids materializing the 64-dim intermediate to global memory, saving 2 round-trips. Template: `kernels/hip/silu_gate_mul.py`.

### Existing Kernels (no changes needed)

- `fused_residual_add_rmsnorm`: wires directly for `h = h + ple_out` before block
- `silu_gate_mul`: MatFormer's sliced SwiGLU uses this kernel with smaller `m`
- All existing kernels handle variable tensor sizes via grid tiling

---

## Parameter Budget

| Component | Params |
|-----------|--------|
| Base (Tempest 16L, d=1024, ffn=2560) | ~239M |
| + PLE (mode=a+b) | +4.8M |
| + MatFormer | +0M |
| **Total** | **~244M** |

Under 250M budget with room for the factored table.

---

## Experimental Matrix (8 runs)

### PLE Ablation (runs 1-4)
| # | Config | Measures |
|---|--------|----------|
| 1 | Base alone | Control BPB, tok/s |
| 2 | Base + PLE(mode="a") | Context-aware alone (+2.1M) |
| 3 | Base + PLE(mode="b") | Token-identity alone (+2.7M) |
| 4 | Base + PLE(mode="a+b") | Combined (+4.8M). Is A+B > max(A,B)? |

### MatFormer (runs 5-6)
| # | Config | Measures |
|---|--------|----------|
| 5 | Base + MatFormer | Training overhead. BPB at each granularity {1/8, 1/4, 1/2, 1} |
| 6 | Base + best PLE + MatFormer | Does PLE compensate MatFormer quality loss? |

### Advanced (runs 7-8)
| # | Config | Measures |
|---|--------|----------|
| 7 | + adaptive compute | Route easy/hard tokens through different granularities |
| 8 | Prometheus + PLE + MatFormer + nested attn | Speculative decode speedup |

Protocol: BabyLM, 45 min budget, log every 10 steps.

---

## Hardware Notes (Strix Halo gfx1151)

- **PLE Path B table** (50257×32 = ~200KB fp16): fits entirely in L2 cache (6 MB). Repeated lookups across layers are near-free.
- **Per-layer mixing matrices** (16×32×64 = ~130KB fp16): trivially fits L2.
- **MatFormer slicing** uses `F.linear` with sliced weights — rocBLAS handles the smaller GEMM shapes. All slice widths (384, 640, 1280, 2560) are multiples of 128 for Tensile tile alignment.
- **Unified memory**: no PCIe penalty for PLE tables. Gemma offloads PLE to CPU RAM; on Strix Halo, CPU and GPU share the same LPDDR5X at 240 GB/s.
- **autokernel.optimize** patterns (RMSNorm 3.3x, SwiGLU 1.6x, fused_residual_add_rmsnorm 6.6x) apply directly to the base model inside Virtuoso.
- **Chunked linear recurrence** mandatory for any SSM-based base (AMADEUS).

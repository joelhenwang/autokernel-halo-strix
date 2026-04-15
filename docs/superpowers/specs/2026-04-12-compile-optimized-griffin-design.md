---
title: "Compile-Optimized Griffin Block"
domain: design-specs
type: spec
status: active
related:
  - docs/superpowers/specs/2026-04-08-torch-compile-custom-ops-design.md
  - docs/superpowers/specs/2026-04-10-training-pipeline-optimization-design.md
  - docs/halo-strix-apu/02_training_performance_analysis.md
tags: [%torch-compile, %griffin, %fused-block, %tempest]
---

# Compile-Optimized Griffin Block

**Date:** 2026-04-12
**Goal:** Close the 2.4x throughput gap between Griffin models and LlamaModel by making Griffin blocks compile-friendly.

## Problem Statement

At 124M params, LlamaModel reaches 49.3K tok/s while Tempest reaches 20.2K tok/s — a 2.44x gap. In eager mode, the gap is only 1.27x. The difference comes entirely from torch.compile effectiveness:

| Factor | LlamaModel | Tempest | Gap |
|--------|-----------|---------|-----|
| Compile boost | 3.2x | 1.7x | 1.88x |
| Block-level fusion | FusedResidualRMSNorm matches | No pattern match | ~30% |
| Scan opacity | SDPA = 1 kernel | Chunked scan = ~15 ops | ~40% |
| Compile region size | 1-2 large regions/block | Many small fragments | ~20% |

**Target:** 90% of LlamaModel = ~44K tok/s at 124M params. Currently 20.2K. Need ~2.2x improvement.

## Design

### Phase B: Custom Op + Fused Block (target: 30-40% improvement → ~28K tok/s)

#### B1: Griffin Scan Custom Op

Register the Griffin linear recurrence as a `torch.library` custom op in `kernels/hip/_torch_ops.py`, following the exact pattern used for `selective_scan`.

```python
@torch.library.custom_op("autokernel::griffin_scan", mutates_args=())
def griffin_scan_op(decay: torch.Tensor, value: torch.Tensor) -> torch.Tensor:
    """Linear recurrence: h[t] = decay[t] * h[t-1] + value[t]"""
    # Uses vectorized chunked scan (existing implementation from tempest.py)
    ...

@griffin_scan_op.register_fake
def _(decay, value):
    return decay.new_empty(decay.shape)

def _griffin_scan_backward(ctx, grad_output):
    """Reverse linear recurrence for gradients."""
    # grad_state[t] = grad_output[t] + decay[t+1] * grad_state[t+1]
    # grad_decay[t] = grad_state[t] * h[t-1]
    # grad_value[t] = grad_state[t]
    ...

griffin_scan_op.register_autograd(_griffin_scan_backward, setup_context=...)
```

**What this achieves:** torch.compile sees the entire scan as ONE opaque node. All element-wise ops before and after the scan become fuseable.

**Backward implementation:** Two options, try in order:
1. **FLA HGRN backend** — if `fla` is installed, use its Triton-based recurrence for both forward and backward (0.40ms, full autograd). Wire as priority 0.
2. **Vectorized chunked scan** — our existing implementation, no Python loops. Forward recomputes states for backward (same pattern as selective_scan). Registered as autograd.
3. **Fallback** — if neither available, skip custom op registration, use inline chunked scan (current behavior).

#### B2: Compile-Safe Fused Griffin Block Pattern

Add a new autokernel pattern that replaces TempestBlock with a compile-friendly wrapper.

**File:** `autokernel/_patterns.py`

The replacement forward becomes:
```python
def forward(self, x, velocity):
    # Pre-norm (matched by RMSNormPattern → HIP kernel)
    x_norm = self.pre_norm(x)
    
    # Parallel mixer: conv ‖ griffin_scan
    conv_out = self.conv(x_norm)
    aiv = self.griffin_w_aiv(x_norm)
    a_proj, i_proj, v = aiv.split(self.d_rec, dim=-1)
    a = torch.sigmoid(a_proj + self.decay_bias)
    i_gate = torch.sigmoid(i_proj)
    input_signal = torch.sqrt((1 - a * a).clamp(min=1e-8)) * (i_gate * v)
    griffin_out = torch.ops.autokernel.griffin_scan(a, input_signal)  # ONE node
    
    # Out projection
    mixer_out = self.out_proj(torch.cat([conv_out, griffin_out], dim=-1))
    
    # Momentum residual
    velocity = self.beta * velocity + mixer_out
    x = x + velocity
    
    # Fused residual+norm for FFN (reuse kernel_fn_dual HIP kernel)
    # Actually: use torch.ops.autokernel.fused_res_rmsnorm for compile
    normed = torch.ops.autokernel.rmsnorm(x.view(-1, x.shape[-1]), self.ffn_norm_weight)
    normed = normed.view(x.shape)
    
    x = x + self.ffn(normed)
    return x, velocity
```

**Key difference from the failed v1 attempt:** 
- No lazy imports inside forward (caused `ModuleNotFoundError` during compile tracing)
- Uses registered `torch.ops.autokernel.*` custom ops (compile-safe by design)
- Griffin scan is an opaque custom op node, not inline Python

**Pattern matching:** Detect TempestBlock by attributes: `conv`, `griffin`, `out_proj`, `pre_norm`, `ffn`, `ffn_norm`, `momentum`. Already implemented in `_find_griffin_block_attrs()`.

#### B3: Wire FLA HGRN as Scan Backend

In `griffin_scan_op` forward, try FLA first:

```python
def griffin_scan_op(decay, value):
    # Priority 0: FLA HGRN (fastest, Triton, full backward)
    if _HAS_FLA:
        from fla.ops.hgrn import chunk_hgrn
        return chunk_hgrn(decay, value)  # adapt tensor layout
    
    # Priority 1: Vectorized chunked scan (our implementation)
    return _vectorized_chunked_scan(decay, value)
```

FLA's HGRN is measured at 0.40ms on gfx1151 — a single Triton kernel with full training backward. This alone could provide significant speedup over our ~15-op chunked scan.

### Phase C: Aggressive Block Restructuring (if B is promising)

Only pursue if Phase B achieves >25% improvement.

#### C1: Simplify Momentum Residual

Replace momentum residual (`velocity = beta * velocity + output; x = x + velocity`) with standard residual + learnable scaling:

```python
x = x + self.scale * mixer_out  # scale is a learnable scalar, init 1.0
```

This eliminates the velocity state from the block signature, making the block `forward(x) → x` instead of `forward(x, velocity) → (x, velocity)`. Simpler for compile.

**Quality tradeoff:** Momentum residual was measured as +0.3% improvement in the PLE ablation. Minimal loss.

#### C2: Fuse Pre-Norm into Block

Instead of separate pre_norm → conv/griffin, register a single custom op:
```python
torch.ops.autokernel.fused_norm_griffin(x, norm_weight, conv_weight, w_aiv, decay_bias)
```
This makes the ENTIRE mixer path one node in the compile graph.

#### C3: Inline the FFN

If the block becomes `forward(x) → x` (no velocity), the FFN can be fused with the next block's pre_norm by the compiler, creating a single large region spanning two blocks.

## Expected Compile Graph (After Phase B)

**Before (current):**
```
Block: [norm][conv_proj][conv1d][sigmoid][sigmoid][sqrt][mul]
       [log][cumsum][exp][div][cumsum][mul][zeros][cat][cumsum]...
       [cat][out_proj][beta*vel+mixer][add][norm][gate_up][silu][mul][down]
```
~35+ separate ops per block.

**After Phase B:**
```
Block: [FUSED: norm → conv_proj → conv1d → sigmoid → sigmoid → sqrt → mul]
       [GRIFFIN_SCAN_OP]
       [FUSED: cat → out_proj → momentum → rmsnorm → gate_up → silu → mul → down]
```
3 regions: two large fused + one opaque custom op.

## Success Criteria

| Metric | Current | Phase B Target | Phase C Target |
|--------|---------|---------------|---------------|
| Tempest124M tok/s (AK+compile) | 20,184 | **28,000+** | **35,000+** |
| Compile boost factor | 1.7x | 2.3x+ | 2.8x+ |
| % of LlamaModel throughput | 41% | 57%+ | 71%+ |
| Val loss on BabyLM (2 epochs) | 2.98 | ≤3.00 (no regression) | ≤3.05 |

## Implementation Order

1. Register `autokernel::griffin_scan` custom op with vectorized chunked scan forward + reverse scan backward
2. Wire FLA HGRN as priority 0 backend (if installed)
3. Update `GriffinRecurrence.forward()` in tempest.py to use `torch.ops.autokernel.griffin_scan`
4. Update `FusedGriffinBlockPattern` in `_patterns.py` to be compile-safe (no lazy imports, use registered ops)
5. Enable the pattern in `ALL_PATTERNS`
6. Benchmark: 124M Tempest with AK+compile → measure tok/s improvement
7. Verify: BabyLM 2-epoch training → confirm no quality regression
8. If >25% improvement: proceed to Phase C simplifications
9. If Phase C: remove momentum, simplify block signature, benchmark + verify again

## Files to Create/Modify

| File | Action |
|------|--------|
| `kernels/hip/_torch_ops.py` | Add `griffin_scan` custom op registration |
| `models/tempest.py` | Update `GriffinRecurrence.forward()` to use custom op |
| `models/tempest_124m.py` | Same update |
| `models/spectral_hydra.py` | Same update (uses same recurrence) |
| `autokernel/_patterns.py` | Fix `FusedGriffinBlockPattern` for compile safety, re-enable |
| `scripts/bench_124m_comparison.py` | Re-run after changes |

## Risks

1. **FLA HGRN tensor layout mismatch** — may need reshaping between our `(B, T, D)` and FLA's expected layout. Verify with correctness test.
2. **Backward numerical accuracy** — reverse scan in fp32 should match autograd. Test with `torch.autograd.gradcheck`.
3. **torch.compile still breaks** — if the custom op registration isn't compatible with Inductor tracing, the pattern won't help. Mitigation: test with `TORCH_COMPILE_DEBUG=1` to see graph structure.
4. **Phase C quality regression** — removing momentum residual may hurt quality. Gate on >25% throughput improvement before attempting.

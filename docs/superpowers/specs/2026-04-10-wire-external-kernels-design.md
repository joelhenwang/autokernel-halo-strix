---
title: "Wire External Kernel Speedups Into Architectures"
domain: design-specs
type: spec
status: active
related:
  - docs/superpowers/specs/2026-04-10-optimization-libraries-design.md
  - docs/superpowers/specs/2026-04-10-aiter-rocblas-optimization-design.md
tags: [%external-kernels, %causal-conv1d, %mamba-ssm]
---

# Wire External Kernel Speedups Into Architectures

**Date:** 2026-04-10
**Status:** Design approved, pending implementation
**Workstream:** A (of A/B/C/D optimization roadmap)

## Problem

We've verified 4 external kernel libraries on gfx1151 with significant speedups, but none are wired into our training architectures yet:

| Package | Speedup | Wired in? |
|---------|---------|-----------|
| causal-conv1d (10x vs nn.Conv1d) | Every GatedConv | No |
| mamba-ssm scan (5.6x vs HIP kernel) | AMADEUS Mamba3SISO | No |
| hybrid_attention (8.9% vs SDPA) | PROMETHEUS attention layers | No |
| FLA ops (GLA, Retention, HGRN) | New architecture options | No |

## Design

### Integration approach: try/except imports in each model file

No wrappers or abstraction layers. Each model file gets a try/except at the top that imports the fast backend if available, falls back gracefully. This is the simplest and most debuggable approach.

### AMADEUS (`models/amadeus.py`)

**Change 1: mamba-ssm selective scan**

In `_scan_dispatch()`, add mamba-ssm as highest-priority backend:

```python
def _scan_dispatch(x, dt, A_log, B, C, D, n_heads):
    # Priority: mamba-ssm > HIP kernel > chunked Python
    try:
        from mamba_ssm.ops.selective_scan_interface import selective_scan_fn
        return _mamba_ssm_scan(x, dt, A_log, B, C, D, n_heads, selective_scan_fn)
    except ImportError:
        pass
    # ... existing HIP and chunked fallbacks
```

The `_mamba_ssm_scan` wrapper reshapes tensors from our (B, T, D) layout to mamba-ssm's expected (B, D, T) layout and handles the A/B/C/D parameter format differences.

Key differences from our API:
- mamba-ssm expects `u` shape (B, D, L), our `x` is (B, L, D) → transpose
- mamba-ssm expects `delta` shape (B, D, L) → transpose `dt`
- mamba-ssm expects `A` shape (D, N) → reshape from our A_log
- mamba-ssm expects `B` shape (B, N, L) → transpose our B
- mamba-ssm `D` param is (D,) float32 → matches our D

**Change 2: causal-conv1d in GatedConv**

In `GatedConv.__init__` and `forward`, replace `nn.Conv1d` with `causal_conv1d_fn`:

```python
try:
    from causal_conv1d import causal_conv1d_fn
    _HAS_CAUSAL_CONV1D = True
except ImportError:
    _HAS_CAUSAL_CONV1D = False

class GatedConv(nn.Module):
    def __init__(self, d_model, d_conv, kernel_size=3):
        ...
        if _HAS_CAUSAL_CONV1D:
            # Store weight as (d_conv, kernel_size) for causal_conv1d_fn
            self.conv_weight = nn.Parameter(torch.randn(d_conv, kernel_size))
        else:
            self.conv = nn.Conv1d(d_conv, d_conv, kernel_size, padding=kernel_size-1, groups=d_conv, bias=True)

    def forward(self, x):
        ...
        if _HAS_CAUSAL_CONV1D:
            z = causal_conv1d_fn(y.transpose(1, 2), self.conv_weight).transpose(1, 2)
        else:
            z = self.conv(y.transpose(1, 2))[:, :, :T].transpose(1, 2)
```

Note: `causal_conv1d_fn` expects input (B, D, L) and weight (D, K). No bias support in the fast path — use `nn.Conv1d` fallback if bias is needed.

### PROMETHEUS (`models/prometheus.py`)

**Change 1: hybrid_attention**

In `_attention_forward()`, add hybrid as highest priority:

```python
def _attention_forward(q, k, v, causal=True):
    backend = _detect_attn_backend()
    if backend == "hybrid":
        from kernels.hip.hybrid_attention import hybrid_flash_sdpa_attention
        # hybrid expects (B, T, H, D), SDPA expects (B, H, T, D)
        # q is already (B, H, T, D) here — need to transpose
        return hybrid_flash_sdpa_attention(
            q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2), causal=causal
        ).transpose(1, 2)
    elif backend == "aule":
        ...
```

Update `_detect_attn_backend()` to check for hybrid first:
```python
try:
    from kernels.hip.hybrid_attention import hybrid_flash_sdpa_attention
    _ATTN_BACKEND = "hybrid"
except ImportError:
    ...
```

**Change 2: causal-conv1d in GatedConv**

Same as AMADEUS — `GatedConv` is imported from `models/tempest.py`, so the change is in tempest.py and propagates to both PROMETHEUS and TEMPEST.

### TEMPEST (`models/tempest.py`)

**Change 1: causal-conv1d in GatedConv**

This is the shared component. The change here propagates to PROMETHEUS (which imports from tempest) and can be reused by AMADEUS.

Since both AMADEUS and TEMPEST define their own `GatedConv`, the change needs to be applied in both files. PROMETHEUS imports from tempest, so it gets the fix for free.

### Shared: `GatedConv` causal-conv1d integration

The `GatedConv` class appears in:
- `models/tempest.py` (used by TEMPEST and PROMETHEUS)
- `models/amadeus.py` (its own copy)

Both need the same causal-conv1d integration. The conv weight format differs:
- `nn.Conv1d` groups=D uses weight shape (D, 1, K)
- `causal_conv1d_fn` expects weight shape (D, K)

We store weight as `(D, K)` and reshape for `nn.Conv1d` fallback.

## Training Protocol

1. **Smoke test** each architecture (200 steps) to verify no crashes
2. **15-min training runs** for all three on BabyLM:
   ```bash
   python -m halo_training --model models/amadeus.py --class-name Amadeus --dataset babylm --compile --optimize-kernels --time-budget 15
   python -m halo_training --model models/tempest.py --class-name Tempest --dataset babylm --compile --optimize-kernels --time-budget 15
   python -m halo_training --model models/prometheus.py --class-name Prometheus --dataset babylm --compile --optimize-kernels --time-budget 15
   ```
3. **Compare:** tok/s, loss, MFU, memory for each
4. **45-min run** for the winner

## Expected Results

| Architecture | Old tok/s | New backends | Expected tok/s |
|-------------|-----------|--------------|----------------|
| AMADEUS 243.8M | 10.4K | +mamba-ssm scan +causal-conv1d | 12-15K |
| TEMPEST ~245M | untrained | +causal-conv1d | 12-16K |
| PROMETHEUS ~216M | untrained | +hybrid_attn +causal-conv1d | 10-14K |

## Files to Modify

| File | Changes |
|------|---------|
| `models/tempest.py` | GatedConv: causal-conv1d integration |
| `models/amadeus.py` | GatedConv: causal-conv1d + _scan_dispatch: mamba-ssm |
| `models/prometheus.py` | _detect_attn_backend + _attention_forward: hybrid_attention |

## Verification

1. Import test: each model instantiates without error
2. Smoke test: 200 steps, loss decreases, no NaN
3. Fallback test: `_HAS_CAUSAL_CONV1D = False` — models still work with nn.Conv1d
4. Correctness: verify causal_conv1d output matches nn.Conv1d for same weights (already proved in bench_external_kernels.py)

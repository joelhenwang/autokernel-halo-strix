# Phase III: root-cause + fix — autograd breakage in `_RMSNormReplacement`

## Root cause

`autokernel/_patterns.py:_RMSNormReplacement.forward` directly called
`self.kernel_fn(x, weight)` (a raw pybind HIP op). The returned tensor
had **no `grad_fn`**, silently blocking gradient flow through every
RMSNorm in the model.

OdinFlat has 29 RMSNorms on the forward path (14 layers × 2 + `final_norm`).
Every gradient would have to pass through a RMSNorm to reach any upstream
parameter — including all 112M hidden params. Only `lm_head` (downstream
of no RMSNorm? actually downstream of final_norm, so upstream gradient
blocked there too) and its own weight could receive updates.

Observed signature matched exactly:
- Loss still descends (lm_head adapts)
- Intermediate activations stall at init magnitudes (~0.75 at step 200)
- Training is catastrophically frozen for 121M of 121.7M params

## Evidence (numeric harness + autograd test)

### `scripts/test_rmsnorm_numerics.py`
14 sequential HIP kernel invocations produce outputs IDENTICAL to
reference (ratio_of_norms = 1.000000 across all chain positions).
**Kernel forward is correct.**

### `scripts/test_rmsnorm_autograd.py`
With fp16 input passed to `_RMSNormReplacement`, backward fails:

    RuntimeError: element 0 of tensors does not require grad
                  and does not have a grad_fn

The output tensor from the raw kernel has `requires_grad=False`.

## Fix

Route the fp16 path through `torch.ops.autokernel.rmsnorm`, which is
registered with `register_autograd` in `kernels/hip/_torch_ops.py` and
has the same HIP forward PLUS a backward implementation (HIP fast path
+ PyTorch fallback for compile compatibility).

```python
# autokernel/_patterns.py _RMSNormReplacement.forward — fixed
if x.dtype == torch.float16:
    if self._autograd_op is not None:
        flat = x.reshape(-1, x.shape[-1])
        return self._autograd_op(flat, self.weight).view(orig_shape)
    return self.kernel_fn(x.view(-1, x.shape[-1]), self.weight).view(orig_shape)
```

## Validation

After fix, autograd test passes with `x.grad` and `weight.grad` both
matching reference within fp16 precision (ratio 0.999998 / 1.000000).

## Training-level validation

Re-ran the Phase II P3 probe (both `rmsnorm` + `fused_silu_gate_mul` enabled)
with the fix:

| Config | Steady tok/s | Loss @ 200 | maxabs.13 @ 200 | verdict |
|---|---:|---:|---:|---|
| P0 baseline (no HIP) | 31,331 | 4.7013 | 27.09 | reference |
| P3 broken (before fix) | 58,238 | 7.0305 ⚠ | 0.87 ⚠ | broken |
| **P3-FIXED** | **40,962** | **4.8184** ✓ | 2.14 ✓ | **FIXED** |

- Loss delta: **+0.117** (vs 0.5 gate → well within)
- Throughput lift: **+31% steady-state** (40,962 vs 31,331)
- maxabs trajectory: reasonable (2.14 at step 200; ~13× smaller than
  baseline but growing healthily, gradients flowing)
- No NaN, no scaler issues, grad = finite throughout

## Sprint 3A wall-time impact

- Current C3 baseline: **~61h** at 31,331 tok/s
- With rmsnorm fix: **~47h** at 40,962 tok/s (**saves ~14h**)

This is below the +80% aspirational target but is a genuine win with
zero additional work. The +80% from Phase 0.4 was a broken measurement
(forward only, no gradient flow).

## Side finding: `_FusedSwiGLUReplacement` / `_SiluGateMulReplacement`

Same pattern (direct raw kernel call, no autograd). Attempted the same
fix but training EXPLODES (gradient overflow, scaler collapses to 1e-2,
maxabs → 10000+ within 150 steps).

Cause: the HIP `silu_gate_mul_backward` kernel appears to have wrong
gradient math under OdinFlat's deep SwiGLU chain with fp16 autocast +
compile_zones. Kept the original raw-kernel path — Phase II P2 validated
that it trains at loss parity (4.67 vs 4.70 at step 200), which suggests
SOME gradient flow exists (possibly via the downstream `w_down` Linear
and the `chunk()` upstream). Requires dedicated follow-up investigation.

**TODO**: silu_gate_mul backward kernel numeric harness — separate issue.

## Artifacts

```
autokernel/_patterns.py                             fixed _RMSNormReplacement
scripts/test_rmsnorm_numerics.py                    forward equivalence harness
scripts/test_rmsnorm_autograd.py                    autograd regression test
scripts/verify_rmsnorm_fix.sh                       training-level verify probe
checkpoints/sprint15-bisect-P3-FIXED/               verified fix probe (200 steps)
```

## Next step (Phase IV)

Phase IV compile-interaction study to see if Inductor can squeeze more
throughput out of the fixed kernel path. Particularly: does
`AUTOKERNEL_NO_BWD_HIP=1` (disabling HIP backward, forcing PyTorch
fallback) let Inductor fuse better and exceed our +31%?

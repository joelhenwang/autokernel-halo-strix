# Triton Kernel Author Guide

**Repo location:** `autokernel/triton_base.py`, `autokernel/triton_autotune.py`,
`scripts/kernel_parity_harness.py`, `scripts/kernel_bench_harness.py`.

**Purpose:** provides a consistent, autograd-safe, autotuned interface
for adding Triton-backed replacements into the HALO training path.
Use this harness instead of writing a bare `torch.autograd.Function`
or raw `torch.library.custom_op` each time.

---

## Why this harness exists

The earlier HIP-kernel situation (see `docs/perf/autokernel-deep-analysis.md`)
taught us three rules:

1. Any kernel in a training path **must** route through PyTorch autograd.
   Raw pybind calls that return `grad_fn=None` silently freeze upstream
   parameters.
2. `torch.autograd.Function` is ~3× lower per-call overhead than
   `torch.library.custom_op`. For per-layer kernels invoked ~28 times per
   optimizer step, that's the difference between "net +0%" and "net +3%".
3. Autotune caches must be keyed on both shape AND git SHA, otherwise
   stale winners from a prior kernel revision will corrupt current runs.

This harness bakes those rules into base classes so you can't forget.

---

## Quick start: adding a new kernel

Minimum viable flow:

```python
# kernels/triton/my_kernel.py

import triton
import triton.language as tl
import torch
from autokernel.triton_base import TritonAutogradFunction


@triton.jit
def _fwd_kernel(
    OUT_ptr, A_ptr, B_ptr,
    N, stride_o, stride_a, stride_b,
    BLOCK: tl.constexpr,
):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < N
    a = tl.load(A_ptr + offs * stride_a, mask=mask)
    b = tl.load(B_ptr + offs * stride_b, mask=mask)
    tl.store(OUT_ptr + offs * stride_o, a + b, mask=mask)


@triton.jit
def _bwd_kernel(
    GA_ptr, GB_ptr, GO_ptr,
    N, stride, BLOCK: tl.constexpr,
):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < N
    go = tl.load(GO_ptr + offs * stride, mask=mask)
    tl.store(GA_ptr + offs * stride, go, mask=mask)
    tl.store(GB_ptr + offs * stride, go, mask=mask)


class MyAddTritonFn(TritonAutogradFunction):
    """a + b, Triton-fused."""

    @staticmethod
    def forward(ctx, a, b):
        assert a.is_contiguous() and b.is_contiguous()
        out = torch.empty_like(a)
        N = a.numel()
        grid = (triton.cdiv(N, 1024),)
        _fwd_kernel[grid](out, a, b, N,
                          out.stride(0) if out.stride() else 1,
                          a.stride(0) if a.stride() else 1,
                          b.stride(0) if b.stride() else 1,
                          BLOCK=1024)
        return out

    @staticmethod
    def backward(ctx, grad_out):
        grad_a = torch.empty_like(grad_out)
        grad_b = torch.empty_like(grad_out)
        N = grad_out.numel()
        grid = (triton.cdiv(N, 1024),)
        _bwd_kernel[grid](grad_a, grad_b, grad_out, N,
                          grad_out.stride(0) if grad_out.stride() else 1,
                          BLOCK=1024)
        return grad_a, grad_b


# Invoke: y = MyAddTritonFn.apply(a, b)
```

---

## Required tests for every new kernel

Before merging, your kernel must have:

### 1. Parity test (forward + backward)

```python
# scripts/test_triton_my_kernel.py
from scripts.kernel_parity_harness import compare_forward_backward
from kernels.triton.my_kernel import MyAddTritonFn


def test_parity():
    result = compare_forward_backward(
        name="my_add",
        reference_fn=lambda a, b: a + b,
        triton_fn=lambda a, b: MyAddTritonFn.apply(a, b),
        shapes=[(128,), (4, 512), (16, 32, 128)],
        dtypes=[torch.float16, torch.bfloat16],
        input_count=2,
        n_trials=5,
    )
    assert result["all_pass"], result["failures"]
```

### 2. Isolated throughput bench

```python
from scripts.kernel_bench_harness import bench_kernel_fwd_bwd

result = bench_kernel_fwd_bwd(
    name="my_add",
    triton_fn=lambda a, b: MyAddTritonFn.apply(a, b),
    reference_fn=lambda a, b: a + b,
    shape=(16, 512, 2048),
    dtype=torch.float16,
    input_count=2,
)
# Ship gate: result["speedup_fwd_bwd"] >= 1.05
```

### 3. End-to-end model probe

200-step training probe on OdinFlat/OdinHalo with the kernel enabled,
comparing loss@200 to baseline. Loss parity within ±0.05.

### 4. Phase E autograd-safety CI test

Any kernel invoked in the training path should pass the general CI smoke
test (`scripts/test_autokernel_autograd_safety.py`). This is currently
run as part of the autokernel audit suite.

---

## Autotune integration

For kernels whose optimal tile size depends on shape, wrap the launcher
function with `cached_autotune`:

```python
from autokernel.triton_autotune import cached_autotune


@cached_autotune(
    kernel_name="my_add_fwd",
    configs=[
        {"BLOCK": 512},
        {"BLOCK": 1024},
        {"BLOCK": 2048},
    ],
    key_shape=lambda args: (args[1].shape[-1], str(args[1].dtype)),
)
def _launch_fwd(out, a, b, BLOCK):
    N = a.numel()
    grid = (triton.cdiv(N, BLOCK),)
    _fwd_kernel[grid](out, a, b, N, ..., BLOCK=BLOCK)
```

First call with a new shape times every config and persists the winner
to `~/.cache/autokernel/triton_autotune/<git-sha>/my_add_fwd.json`.
Subsequent calls hit the cache.

Clear caches with `python -c "from autokernel.triton_autotune import clear_cache; clear_cache()"`.

---

## Ship gate for Phase D kernels

For a Triton kernel to land in the training path:

| Criterion | Threshold | Measurement |
|---|---|---|
| Numerical parity fwd | `rel_err < 2e-3` at fp16 | `kernel_parity_harness` |
| Numerical parity bwd | `rel_err < 5e-3` at fp16 | `kernel_parity_harness` |
| Isolated throughput | `speedup_fwd_bwd >= 1.05x` | `kernel_bench_harness` |
| End-to-end wall | ≥5% total step improvement | 200-step training probe |
| Loss parity | `|Δloss| <= 0.05` at step 200 | same probe |

Miss any and the kernel stays behind an opt-in flag or is reverted.

---

## Current Phase D roadmap

- **D.A: Harness** (this doc + the 4 modules listed above). SHIPPED.
- **D.B: Fused SwiGLU** — `kernels/triton/fused_swiglu.py`. Target: fuse
  `w_gate_up → chunk → silu*gate → w_down` into one Triton dispatch.
  Estimated 1-2 weeks.
- **D.C: Fused CE + z-loss** — only if Phase B's fix leaves ≥3% of wall
  time in CE still measurable on Machine A.
- **D.D: Batched NorMuon Newton-Schulz** — amortize NS launch overhead
  across all 2D params. Estimated 2 weeks.

---

## Non-goals

- Replacing the whole `autokernel/_patterns.py` with Triton-only paths.
  HIP kernels with `torch.library.custom_op + register_autograd` are
  fine where they beat Triton (rmsnorm fp16 currently).
- Supporting fp32 Triton kernels. The fp32 path should eager-fall back
  to PyTorch (Inductor fuses fp32 well).
- Multi-device / TP-aware Triton kernels. Single-GPU is the scope.

---

## Debugging checklist

- **Segfault in forward**: missing `a.is_contiguous()` check. Triton
  kernels assume contiguous layouts.
- **Grad mismatch at shape (N, 1)**: broadcasting. Triton kernels need
  explicit shape handling. Test at `(1, 1)`, `(N, 1)`, and `(N, K)`.
- **Autotune winner is consistent "0-us"**: your timing loop isn't
  syncing; add `torch.cuda.synchronize()` to `_bench`.
- **`grad_fn = None` on output**: you forgot to use `.apply`. `TritonAutogradFunction.apply(...)` — not direct call.

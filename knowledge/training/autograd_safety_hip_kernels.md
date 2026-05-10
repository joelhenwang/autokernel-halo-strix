# Autograd Safety for HIP / Triton Kernels on gfx1151

**Status:** Required reading for anyone adding or modifying kernels in
`kernels/hip/` or `kernels/triton/`.

**Root-cause incident:** `docs/perf/autokernel-deep-analysis.md` (2026-05-10).
A raw pybind11 C++ call to a HIP kernel severed gradient flow to 44M
parameters in OdinFlat's FFN. Forward was +30 % faster but training
quality degraded (+0.65 loss regression at step 2000) because 23 % of
the parameter population was silently frozen at initialization.

---

## The rule

**Any kernel whose output feeds into a gradient-carrying op MUST be
routed through PyTorch autograd.** There are exactly three legal ways
to do this:

### 1. `torch.library.custom_op` + `register_autograd`

Best for C++ kernels with a native backward kernel. Highest overhead
(~5 μs per dispatch) but integrates cleanly with `torch.compile`.

```python
# kernels/hip/_torch_ops.py

@torch.library.custom_op("autokernel::silu_gate_mul", mutates_args=())
def silu_gate_mul_op(gate: torch.Tensor, up: torch.Tensor) -> torch.Tensor:
    from kernels.hip.silu_gate_mul import kernel_fn
    return kernel_fn(gate, up)

def _silu_gate_mul_setup(ctx, inputs, output):
    gate, up = inputs
    ctx.save_for_backward(gate, up)

def _silu_gate_mul_backward(ctx, grad_output):
    gate, up = ctx.saved_tensors
    from kernels.hip.silu_gate_mul_backward import kernel_fn as bwd
    return bwd(gate, up, grad_output)

silu_gate_mul_op.register_autograd(
    _silu_gate_mul_backward, setup_context=_silu_gate_mul_setup
)
```

In autokernel Replacements, invoke as `torch.ops.autokernel.silu_gate_mul(...)`.

### 2. `torch.autograd.Function`

Best for Triton kernels or Python-wrapped backward. Lower overhead
(~1-2 μs per dispatch) but more manual setup. Use for high-frequency
kernels where `custom_op` overhead measurably hurts.

```python
class _FusedSwiGLUTritonFn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, gate, up):
        assert gate.is_contiguous() and up.is_contiguous()
        out = torch.empty_like(gate)
        _fwd_kernel[grid](out, gate, up, ...)
        ctx.save_for_backward(gate, up)
        return out

    @staticmethod
    def backward(ctx, grad_out):
        gate, up = ctx.saved_tensors
        grad_gate = torch.empty_like(gate)
        grad_up = torch.empty_like(up)
        _bwd_kernel[grid](grad_gate, grad_up, gate, up, grad_out, ...)
        return grad_gate, grad_up


# Invoke:
y = _FusedSwiGLUTritonFn.apply(gate.contiguous(), up.contiguous())
```

Prefer the `autokernel.triton_base.TritonAutogradFunction` base which
handles dtype/device validation uniformly. See
`knowledge/kernels/triton_author_guide.md`.

### 3. Eager PyTorch fallback

If neither a working custom op nor an autograd.Function is available,
use plain PyTorch ops. Inductor can fuse them under `torch.compile` to
within 10-15 % of a hand-written kernel for simple patterns.

```python
def forward(self, x):
    # Autograd-safe because F.silu and elementwise mul have ATen autograd
    return F.silu(gate) * up
```

This is the Phase B.3 `_LayerNormReplacement` path — no HIP LayerNorm
backward exists, so the fp16 path uses `F.layer_norm`.

---

## What silently breaks

**A raw pybind11 call from a C++ extension**:

```python
# UNSAFE: self.kernel_fn is a pybind11-wrapped C++ function.
# The returned tensor has requires_grad=False and grad_fn=None.
def forward(self, x):
    return self.kernel_fn(x, self.weight)
```

**The failure mode:**

1. The pybind11 C++ function creates a `torch::Tensor` via `torch::empty`
   or similar. That tensor's `requires_grad` is False by default and
   `grad_fn` is None. PyTorch dispatcher is not involved.
2. If the caller's input had `requires_grad=True`, autograd *would*
   normally propagate. But since the output tensor has no `grad_fn`,
   autograd's backward traversal stops at the output.
3. Downstream ops that use this output still compute their own
   parameters' gradients correctly (e.g. `w_down(activated)` still
   gets `w_down.weight.grad = grad_out.T @ activated`). So the
   immediate failure is invisible.
4. Upstream parameters (e.g. `w_gate_up.weight`) receive no gradient
   contribution. They stay at init forever. 23 % of OdinFlat's params
   silently frozen = observed.

**Visible symptoms:**
- Throughput is mysteriously higher than expected (backward skipped for
  the frozen params).
- Memory is mysteriously lower (no backward state saved for them).
- Training loss *still descends* because the remaining 77 % of params
  can absorb gradient signal.
- Quality gap opens over long horizons: +0.05 loss at step 200,
  +0.3 at step 500, +0.65 at step 2000 for the OdinFlat incident.

---

## Detection: how to prove your kernel is safe

### Static audit (pre-merge, fast, no GPU)

```bash
python scripts/audit_autokernel_replacements.py
```

AST-scans every `_*Replacement` class in `autokernel/_patterns.py`.
Any verdict other than `SAFE` or `CONDITIONAL-SAFE` blocks merge.

### Per-replacement unit tests (CUDA)

```bash
python -m pytest scripts/test_phase_b_autograd_safety.py
```

For each Replacement, constructs a minimal nn.Module, wraps it with
the Replacement, runs forward+backward, asserts every leaf parameter
has a finite non-None grad.

### CI smoke test (CUDA)

```bash
python -m pytest scripts/test_autokernel_autograd_safety.py
```

One-step training on OdinFlatMini with `--optimize-kernels`, asserts
every requires_grad parameter received a gradient. Hard CI gate.

### Runtime preflight (training launch)

`scripts/train_ddp.py` with `--optimize-kernels` auto-dispatches a dummy
batch before the main training loop. If any parameter fails the grad
check, training aborts with an actionable error message pointing back to
this document.

### Direct diagnosis (any time)

Run a 50-step probe with `--diag-frozen-params checkpoints/<name>/diag.jsonl`.
JSONL contains per-step, per-parameter `.grad.norm()`. Analyze with
`scripts/analyze_diag_frozen_params.py` to produce a blast-radius table.
This is what caught the original OdinFlat bug.

---

## Author workflow for a new kernel

1. **Decide dispatch mechanism** — `custom_op` (C++ backward exists) vs
   `autograd.Function` (Triton or Python backward).
2. **Write the kernel** in `kernels/hip/<name>.py` (C++/HIP) or
   `kernels/triton/<name>.py` (Triton).
3. **Register autograd** — add a `@torch.library.custom_op` decorator
   to `kernels/hip/_torch_ops.py`, OR subclass
   `autokernel.triton_base.TritonAutogradFunction`.
4. **Wire into `autokernel/_patterns.py`** — construct a Replacement
   class whose `forward()` calls `torch.ops.autokernel.<name>(...)`
   or `MyTritonFn.apply(...)`. Do NOT call the raw `kernel_fn(...)`.
5. **Write the 5 required tests**:
   - Parity (forward + backward) via
     `scripts/kernel_parity_harness.compare_forward_backward`
   - Autograd safety via per-replacement test in
     `scripts/test_phase_b_autograd_safety.py`
   - Isolated bench via `scripts/kernel_bench_harness.bench_kernel_fwd_bwd`
   - E2E 200-step training probe
   - Static audit passes
6. **Run `python scripts/audit_autokernel_replacements.py`** — verdict
   for your new Replacement must be `SAFE` or `CONDITIONAL-SAFE`.
7. **Submit PR** — CI runs `test_autokernel_autograd_safety.py` as a
   gate. Preflight at training launch catches any remaining leaks.

---

## History

| Event | Date | Ref |
|---|---|---|
| Phase III fix: `_RMSNormReplacement` autograd | 2026-05-09 | `docs/perf/odinflat-rmsnorm-fix.md` |
| Phase V deep dive: silu HIP silent freeze | 2026-05-10 | `docs/perf/odinflat-throughput-final.md` |
| Track 3.A blast-radius diagnostic | 2026-05-10 | `docs/perf/autokernel-frozen-blast-radius.md` |
| Deep-analysis synthesis | 2026-05-10 | `docs/perf/autokernel-deep-analysis.md` |
| Phase B: wire autograd for all UNSAFE replacements | 2026-05-11 | commit `5ebe594` |
| Phase B.5: fused z-loss in `_CrossEntropyHIP` | 2026-05-11 | commit `f24d8dd` |
| Phase D.A: Triton kernel harness shipped | 2026-05-11 | commit TBD |
| Phase E: runtime guardrails + CI test | 2026-05-11 | commit `404b140` |

---

## Non-goals of this document

- Does not cover Triton kernel authoring beyond autograd safety; see
  `knowledge/kernels/triton_author_guide.md`.
- Does not cover performance tuning (tile sizes, autotune config); see
  the author guide.
- Does not cover ROCm/HIP-specific debugging; see
  `knowledge/hardware/amd_rdna35_strix_halo.md`.

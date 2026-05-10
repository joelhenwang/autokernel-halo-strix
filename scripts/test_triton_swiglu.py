"""Parity + sanity tests for Triton fused SwiGLU (Phase D.B).

CUDA-only (requires triton + HIP GPU). Runs on Machine A or B post-sync.

Plan: Phase D.B validation tests per
knowledge/kernels/triton_author_guide.md "Required tests" section.
"""

from __future__ import annotations

import os
import sys
import unittest

import torch
import torch.nn.functional as F


REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

_CUDA_AVAILABLE = torch.cuda.is_available()

try:
    from kernels.triton.fused_swiglu import (
        fused_swiglu, TritonFusedSwiGLUModule, _TRITON_AVAILABLE,
    )
except Exception as exc:  # noqa: BLE001
    _TRITON_AVAILABLE = False
    _import_err = exc
    fused_swiglu = None  # type: ignore


_REQUIRE = _CUDA_AVAILABLE and _TRITON_AVAILABLE


@unittest.skipUnless(_REQUIRE, "Triton + CUDA required")
class TritonSwiGLUParity(unittest.TestCase):

    def _reference(self, gate, up):
        return F.silu(gate) * up

    def test_fwd_parity_small_shapes(self):
        torch.manual_seed(0)
        for shape in [(4, 8, 128), (16, 32, 256), (2, 512, 2048)]:
            gate = torch.randn(*shape, device="cuda", dtype=torch.float16)
            up = torch.randn(*shape, device="cuda", dtype=torch.float16)
            ref = self._reference(gate.clone(), up.clone())
            out = fused_swiglu(gate.clone(), up.clone())
            abs_err = (ref.float() - out.float()).abs().max().item()
            rel_err = abs_err / (ref.float().abs().max().item() + 1e-8)
            self.assertLess(rel_err, 5e-3,
                            f"fwd {shape}: abs={abs_err:.3e} rel={rel_err:.3%}")

    def test_bwd_parity(self):
        torch.manual_seed(1)
        shape = (4, 32, 1024)

        gate_ref = torch.randn(*shape, device="cuda", dtype=torch.float16, requires_grad=True)
        up_ref = torch.randn(*shape, device="cuda", dtype=torch.float16, requires_grad=True)
        gate_trit = gate_ref.detach().clone().requires_grad_(True)
        up_trit = up_ref.detach().clone().requires_grad_(True)

        ref = self._reference(gate_ref, up_ref)
        out = fused_swiglu(gate_trit, up_trit)

        grad_out = torch.randn_like(ref)
        ref.backward(grad_out)
        out.backward(grad_out)

        for name, r, t in [("gate", gate_ref, gate_trit), ("up", up_ref, up_trit)]:
            abs_err = (r.grad.float() - t.grad.float()).abs().max().item()
            rel_err = abs_err / (r.grad.float().abs().max().item() + 1e-8)
            self.assertLess(rel_err, 5e-3,
                            f"bwd {name}: abs={abs_err:.3e} rel={rel_err:.3%}")

    def test_fp32_falls_back_eager(self):
        """fp32 tensor should bypass Triton and use eager path."""
        gate = torch.randn(4, 8, 128, device="cuda", dtype=torch.float32, requires_grad=True)
        up = torch.randn(4, 8, 128, device="cuda", dtype=torch.float32, requires_grad=True)
        out = fused_swiglu(gate, up)
        # With eager fallback, out.grad_fn should refer to plain mul/SiluBackward, not TritonFn.
        # Just assert grad flows (concrete proof of autograd safety).
        out.sum().backward()
        self.assertIsNotNone(gate.grad)
        self.assertIsNotNone(up.grad)

    def test_module_matches_reference(self):
        """TritonFusedSwiGLUModule.forward should match an eager-SwiGLU reference."""
        torch.manual_seed(2)
        dim, hidden = 128, 512
        mod = TritonFusedSwiGLUModule(dim, hidden).cuda().half()
        # Eager reference: same weights
        class RefSwiGLU(torch.nn.Module):
            def __init__(self, wgu, wd):
                super().__init__()
                self.w_gate_up = wgu
                self.w_down = wd

            def forward(self, x):
                g, u = self.w_gate_up(x).chunk(2, dim=-1)
                return self.w_down(F.silu(g) * u)

        ref = RefSwiGLU(mod.w_gate_up, mod.w_down)
        x = torch.randn(4, 16, dim, device="cuda", dtype=torch.float16)
        out_t = mod(x)
        out_r = ref(x)
        abs_err = (out_t.float() - out_r.float()).abs().max().item()
        rel_err = abs_err / (out_r.float().abs().max().item() + 1e-8)
        self.assertLess(rel_err, 5e-3,
                        f"module vs eager: abs={abs_err:.3e} rel={rel_err:.3%}")


@unittest.skipUnless(_REQUIRE, "Triton + CUDA required")
class TritonSwiGLUBench(unittest.TestCase):
    """Opt-in benchmark sanity. Actual ship-gate decision uses
    scripts/kernel_bench_harness.py at a production-realistic shape."""

    def test_smoke_bench(self):
        from scripts.kernel_bench_harness import bench_kernel_fwd_bwd, print_result

        result = bench_kernel_fwd_bwd(
            name="fused_swiglu_smoke",
            triton_fn=lambda g, u: fused_swiglu(g, u),
            reference_fn=lambda g, u: F.silu(g) * u,
            shape=(4, 256, 2048),
            dtype=torch.float16,
            input_count=2,
            warmup=10,
            iters=50,
        )
        print_result(result)
        # No assertion — this is a smoke test. Ship-gate thresholds are
        # measured in the isolated harness at production-realistic shapes.


if __name__ == "__main__":
    unittest.main(verbosity=2)

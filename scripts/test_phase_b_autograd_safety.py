"""Phase B.6: autograd-flow regression tests for the 5 fixed replacements.

For each _*Replacement class patched in Phase B, constructs a minimal
nn.Module, applies the replacement, runs forward+backward on a small
input, and asserts:
  1. All leaf parameters receive a non-None .grad
  2. No parameter has .grad all zeros (modulo the initial-step exceptions
     documented in Track 3.A: v_res_scale is all_none when no v_prev)

These tests are CUDA-only (HIP kernels require GPU). CI smoke test in
test_autokernel_autograd_safety.py runs a similar assertion on a whole
model (Phase E).

Plan: Phase B.6 of master remediation plan.
"""

from __future__ import annotations

import os
import sys
import unittest

import torch
import torch.nn as nn
import torch.nn.functional as F

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)


_CUDA_AVAILABLE = torch.cuda.is_available()


def _fwd_bwd(module, inputs):
    """Run forward+backward on the given module + list of inputs, return
    the set of parameters with zero or None grad."""
    module.zero_grad()
    out = module(*inputs)
    # Collapse tuple outputs (Griffin returns (x, velocity))
    if isinstance(out, (tuple, list)):
        out = out[0]
    loss = out.sum()
    loss.backward()
    zero_or_none = []
    for name, p in module.named_parameters():
        if p.grad is None:
            zero_or_none.append((name, "none"))
        elif float(p.grad.abs().sum().item()) == 0.0:
            zero_or_none.append((name, "zero"))
    return zero_or_none


@unittest.skipUnless(_CUDA_AVAILABLE, "CUDA required for HIP kernel tests")
class SwiGLUAutogradTests(unittest.TestCase):

    def test_fused_swiglu_replacement_grad_flows(self):
        from autokernel._patterns import _FusedSwiGLUReplacement
        from kernels.hip.silu_gate_mul import kernel_fn

        class Orig(nn.Module):
            def __init__(self, dim=64, hidden=128):
                super().__init__()
                self.w_gate_up = nn.Linear(dim, 2 * hidden, bias=False)
                self.w_down = nn.Linear(hidden, dim, bias=False)

            def forward(self, x):
                gate, up = self.w_gate_up(x).chunk(2, dim=-1)
                return self.w_down(F.silu(gate) * up)

        orig = Orig().cuda().half()
        replaced = _FusedSwiGLUReplacement(orig, kernel_fn).cuda().half()
        x = torch.randn(4, 8, 64, device="cuda", dtype=torch.float16, requires_grad=True)
        problems = _fwd_bwd(replaced, [x])
        self.assertEqual(problems, [],
                         f"Params with zero/none grad post-backward: {problems}")

    def test_silu_gate_mul_replacement_grad_flows(self):
        from autokernel._patterns import _SiluGateMulReplacement
        from kernels.hip.silu_gate_mul import kernel_fn

        class Orig(nn.Module):
            def __init__(self, dim=64, hidden=128):
                super().__init__()
                self.w1 = nn.Linear(dim, hidden, bias=False)
                self.w2 = nn.Linear(hidden, dim, bias=False)
                self.w3 = nn.Linear(dim, hidden, bias=False)

            def forward(self, x):
                return self.w2(F.silu(self.w1(x)) * self.w3(x))

        orig = Orig().cuda().half()
        replaced = _SiluGateMulReplacement(orig, kernel_fn,
                                             "w1", "w2", "w3").cuda().half()
        x = torch.randn(4, 8, 64, device="cuda", dtype=torch.float16, requires_grad=True)
        problems = _fwd_bwd(replaced, [x])
        self.assertEqual(problems, [],
                         f"Params with zero/none grad post-backward: {problems}")


@unittest.skipUnless(_CUDA_AVAILABLE, "CUDA required for HIP kernel tests")
class LayerNormAutogradTests(unittest.TestCase):

    def test_layernorm_replacement_grad_flows(self):
        from autokernel._patterns import _LayerNormReplacement
        from kernels.hip.layernorm import kernel_fn

        orig = nn.LayerNorm(64).cuda().half()
        replaced = _LayerNormReplacement(orig, kernel_fn).cuda().half()
        x = torch.randn(4, 8, 64, device="cuda", dtype=torch.float16, requires_grad=True)
        problems = _fwd_bwd(replaced, [x])
        self.assertEqual(problems, [],
                         f"Params with zero/none grad post-backward: {problems}")


@unittest.skipUnless(_CUDA_AVAILABLE, "CUDA required for HIP kernel tests")
class CrossEntropyZLossTests(unittest.TestCase):

    def test_ce_full_zloss_grad_parity_vs_eager(self):
        """When z_loss_weight>0, kernel.ce_full should produce the same
        gradient on logits as the eager equivalent:
          loss = F.cross_entropy(logits, targets) + z_w * logsumexp(logits)^2.mean()
        """
        import kernel as _ce_k
        torch.manual_seed(0)
        B, V = 64, 1024
        logits_eager = torch.randn(B, V, device="cuda", dtype=torch.float16, requires_grad=True)
        logits_fused = logits_eager.detach().clone().requires_grad_(True)
        targets = torch.randint(0, V, (B,), device="cuda")

        # Eager reference (fp32 for numerical determinism)
        le = logits_eager.float()
        loss_e = F.cross_entropy(le, targets) + 1e-4 * le.logsumexp(dim=-1).pow(2).mean()
        loss_e.backward()
        grad_eager = logits_eager.grad.detach()

        # Fused path via ce_full
        loss_f = _ce_k.ce_full(logits_fused, targets,
                                softcap=0.0, ignore_index=-100,
                                label_smoothing=0.0,
                                mode="tiny", z_loss_weight=1e-4)
        loss_f.backward()
        grad_fused = logits_fused.grad.detach()

        # Grad should match within fp16 precision. tolerance is generous because
        # fp16 rounding on 1024-wide softmax can accumulate.
        abs_err = (grad_eager.float() - grad_fused.float()).abs().max().item()
        rel_err = abs_err / (grad_eager.float().abs().max().item() + 1e-8)
        self.assertLess(rel_err, 0.05,
                        f"CE+zloss grad mismatch: abs={abs_err:.3e}, rel={rel_err:.3%}")

    def test_ce_full_zloss_value_matches_eager(self):
        """The scalar loss value should include the z_loss contribution."""
        import kernel as _ce_k
        torch.manual_seed(1)
        B, V = 32, 512
        logits = torch.randn(B, V, device="cuda", dtype=torch.float16)
        targets = torch.randint(0, V, (B,), device="cuda")

        loss_no_z = _ce_k.ce_full(logits.clone(), targets, mode="tiny", z_loss_weight=0.0)
        loss_with_z = _ce_k.ce_full(logits.clone(), targets, mode="tiny", z_loss_weight=1e-3)

        # Z-loss contribution should be positive (lse > 0 generally).
        diff = float((loss_with_z - loss_no_z).item())
        self.assertGreater(diff, 0.0,
                           f"z_loss did not add to loss (diff={diff:.3e})")

    def test_ce_full_zloss_disabled_matches_no_z(self):
        """z_loss_weight=0 must produce identical loss to the pre-Phase-B.5 path."""
        import kernel as _ce_k
        torch.manual_seed(2)
        B, V = 32, 512
        logits = torch.randn(B, V, device="cuda", dtype=torch.float16)
        targets = torch.randint(0, V, (B,), device="cuda")
        loss_a = _ce_k.ce_full(logits.clone(), targets, mode="tiny", z_loss_weight=0.0)
        loss_b = _ce_k.ce_full(logits.clone(), targets, mode="tiny")  # default z_loss_weight
        self.assertAlmostEqual(float(loss_a.item()), float(loss_b.item()), places=4)


class StaticAuditTests(unittest.TestCase):
    """CPU-only: verify the static audit classifier agrees that post-Phase-B
    no Replacement class is UNSAFE."""

    def test_no_unsafe_replacements(self):
        import json
        audit_json = os.path.join(REPO_ROOT, "docs", "perf", "autokernel-static-audit.json")
        if not os.path.exists(audit_json):
            self.skipTest("static audit JSON not present; run scripts/audit_autokernel_replacements.py")
        with open(audit_json, "r") as f:
            data = json.load(f)
        unsafe = [c for c in data["class_infos"] if c["overall_verdict"] == "UNSAFE"]
        self.assertEqual(unsafe, [],
                         f"Static audit found UNSAFE replacements after Phase B: "
                         f"{[c['class'] for c in unsafe]}")


if __name__ == "__main__":
    unittest.main(verbosity=2)

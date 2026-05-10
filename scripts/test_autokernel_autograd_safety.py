"""Phase E.2 CI smoke test: autograd safety of autokernel.

Runs a minimal training step on OdinFlatMini with --optimize-kernels and
asserts every trainable parameter receives a non-None finite gradient.
Designed to run in CI (Machine B) as a hard gate before any autokernel
change merges.

Complements:
  - scripts/audit_autokernel_replacements.py (static audit, CPU, fast)
  - scripts/test_phase_b_autograd_safety.py (per-replacement unit tests)
  - Phase E.3 preflight in train_ddp.py (runtime hard-error at launch)

Plan: Phase E.2 of master remediation plan.
"""

from __future__ import annotations

import os
import sys
import unittest

import torch

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

_CUDA_AVAILABLE = torch.cuda.is_available()


@unittest.skipUnless(_CUDA_AVAILABLE, "CUDA required for autokernel smoke test")
class AutokernelAutogradSafetyCI(unittest.TestCase):
    """One-step training smoke test on OdinFlatMini."""

    ALLOWED_ZERO_PATTERNS = {
        "v_res_scale",   # first-layer, no v_prev
        "head_gate",     # only active when caller passes head_gate_active=True
    }

    def _check_grads(self, model, label: str):
        """Assert every trainable param received a finite non-None grad.

        Returns list of offending param names; empty list = all clean.
        """
        offenders = []
        for name, p in model.named_parameters():
            if not p.requires_grad:
                continue
            if p.grad is None:
                if any(x in name for x in self.ALLOWED_ZERO_PATTERNS):
                    continue
                offenders.append(f"{label}/{name}: grad is None")
            elif not torch.isfinite(p.grad).all():
                offenders.append(f"{label}/{name}: grad has non-finite values")
            elif float(p.grad.detach().float().abs().sum().item()) == 0.0:
                if any(x in name for x in self.ALLOWED_ZERO_PATTERNS):
                    continue
                offenders.append(f"{label}/{name}: grad all zero")
        return offenders

    def _run_one_step(self, use_optimize_kernels: bool, label: str) -> list[str]:
        from models.odin_flat import OdinFlatMini
        model = OdinFlatMini().cuda()

        if use_optimize_kernels:
            import autokernel
            model = autokernel.optimize(model, training=True)

        model.train()
        vocab = 100
        for _name, mod in model.named_modules():
            if hasattr(mod, "num_embeddings"):
                vocab = int(mod.num_embeddings)
                break
        x = torch.randint(0, vocab, (2, 64), device="cuda", dtype=torch.long)
        t = torch.randint(0, vocab, (2, 64), device="cuda", dtype=torch.long)

        for p in model.parameters():
            if p.grad is not None:
                p.grad.zero_()

        with torch.amp.autocast("cuda", dtype=torch.float16):
            out = model(x, targets=t)
        if isinstance(out, torch.Tensor) and out.dim() == 0:
            loss = out
        elif isinstance(out, dict) and "logits" in out:
            logits = out["logits"]
            loss = torch.nn.functional.cross_entropy(
                logits.view(-1, logits.size(-1)).float(), t.view(-1),
            )
        elif isinstance(out, torch.Tensor):
            loss = torch.nn.functional.cross_entropy(
                out.view(-1, out.size(-1)).float(), t.view(-1),
            )
        else:
            self.fail(f"unknown model output type: {type(out)}")

        loss.backward()
        return self._check_grads(model, label)

    def test_baseline_one_step_clean(self):
        """Without --optimize-kernels, every param must get a grad."""
        offenders = self._run_one_step(use_optimize_kernels=False, label="V0")
        self.assertEqual(offenders, [],
                         f"BASELINE one-step has grad issues (should never happen): {offenders}")

    def test_optimize_kernels_one_step_clean(self):
        """With --optimize-kernels and Phase B fixes, every param must
        still receive a grad. This is the core CI gate."""
        offenders = self._run_one_step(use_optimize_kernels=True, label="V1")
        self.assertEqual(offenders, [],
                         f"--optimize-kernels one-step has autograd-severed params:\n  "
                         + "\n  ".join(offenders) +
                         "\nThis indicates a Replacement class is calling raw pybind. "
                         "See docs/perf/autokernel-deep-analysis.md and run "
                         "scripts/audit_autokernel_replacements.py to isolate.")


if __name__ == "__main__":
    unittest.main(verbosity=2)

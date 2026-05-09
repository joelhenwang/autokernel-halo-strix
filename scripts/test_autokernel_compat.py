"""Regression test for autokernel pattern-matcher compatibility.

Phase 0 (2026-05-08): prior to this test, autokernel.optimize() would
silently wrap blocks whose forward signature is incompatible with
`_FusedResidualRMSNormBlockReplacement.forward(self, x, freqs_cis)`.

Sprint 1 added kwargs (`doc_mask`, `v_prev`, `head_gate_active`,
`return_v`) and OdinHalo has `depth_kvs` (MoDA). The wrapper does not
forward any of these, so wrapping crashes the forward pass with a
``TypeError: forward() got an unexpected keyword argument ...``.

Fix (Phase 0): blocks that carry Sprint 1 kwargs or MoDA kwargs declare
``_skip_autokernel = True`` as a class attribute. The pattern matcher
(``autokernel._patterns._find_block_attrs``) already honors this flag.

This test exists to prevent silent regression: if someone removes
``_skip_autokernel`` in the future, or adds a new block with incompatible
kwargs, the next CI run fails loudly.

Scope covered:
  - OdinHalo    (looped, MoDA + HyPE)
  - OdinFlat    (flat, Sprint 1 kwargs)
  - OdinHaloMini (tiny smoke variant)
  - OdinFlatMini (tiny smoke variant)

For each model:
  1. Instantiate on CUDA (skipped on CPU-only hosts)
  2. Run a forward pass (baseline)
  3. Apply ``autokernel.optimize(model, training=True)``
  4. Run another forward pass; must NOT raise
  5. Verify outputs are finite

Run directly::

    python scripts/test_autokernel_compat.py
"""

from __future__ import annotations

import sys
import unittest
from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path

import torch

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))


def _load_model_class(model_path: Path, class_name: str):
    spec = spec_from_file_location(f"user_model_{class_name}", model_path)
    mod = module_from_spec(spec)
    sys.modules[f"user_model_{class_name}"] = mod
    spec.loader.exec_module(mod)
    return getattr(mod, class_name)


class AutokernelCompatibilityTests(unittest.TestCase):
    """Verify autokernel.optimize() doesn't crash Sprint 1+ block signatures."""

    @classmethod
    def setUpClass(cls):
        if not torch.cuda.is_available():
            raise unittest.SkipTest(
                "autokernel_compat tests require CUDA (autokernel uses HIP kernels)")
        try:
            import autokernel  # noqa: F401
        except ImportError:
            raise unittest.SkipTest("autokernel package not importable")

    def _forward_and_check(self, model, batch_size=1, seq_len=32):
        """Run a tiny forward pass and verify outputs are finite."""
        vocab_size = getattr(model, "vocab_size", 32768)
        x = torch.randint(
            0, min(vocab_size, 100), (batch_size, seq_len),
            device="cuda", dtype=torch.long)
        with torch.no_grad(), torch.amp.autocast("cuda", dtype=torch.float16):
            out = model(x)
        # Output could be (logits,) or logits or a dict
        if isinstance(out, tuple):
            out = out[0]
        if isinstance(out, dict):
            out = next(iter(out.values()))
        self.assertTrue(torch.is_tensor(out), "Forward pass must return a tensor")
        self.assertTrue(torch.isfinite(out).all(), "Forward output has NaN/Inf")
        return out

    def _round_trip(self, model_path: str, class_name: str):
        """Load + baseline forward + optimize + optimized forward."""
        import autokernel

        cls = _load_model_class(_REPO_ROOT / model_path, class_name)
        model = cls().cuda().eval()

        # Baseline: forward must succeed pre-optimization
        self._forward_and_check(model)

        # Apply autokernel. This should NOT raise. If _skip_autokernel is
        # missing on a Sprint 1 block, the optimized forward call will
        # raise TypeError and this test fails.
        model = autokernel.optimize(model, training=True)

        # Post-optimization forward pass must succeed too
        self._forward_and_check(model)

    # -------------------------- individual tests --------------------------

    def test_autokernel_optimize_odinhalo(self):
        """OdinHalo has NoPEMoDAGQABlock with depth_kvs kwarg."""
        self._round_trip("models/odin_halo.py", "OdinHalo")

    def test_autokernel_optimize_odinhalo_mini(self):
        """OdinHaloMini: tiny smoke variant with same block types."""
        self._round_trip("models/odin_halo.py", "OdinHaloMini")

    def test_autokernel_optimize_odinflat(self):
        """OdinFlat has NoPEGQABlock with doc_mask / v_prev / head_gate_active."""
        self._round_trip("models/odin_flat.py", "OdinFlat")

    def test_autokernel_optimize_odinflat_mini(self):
        """OdinFlatMini: tiny smoke variant with same block types."""
        self._round_trip("models/odin_flat.py", "OdinFlatMini")

    def test_skip_autokernel_attribute_on_moda_block(self):
        """Regression: NoPEMoDAGQABlock.* must declare _skip_autokernel."""
        cls = _load_model_class(_REPO_ROOT / "models/odin_halo.py", "OdinHalo")
        # Find a NoPEMoDAGQABlock by walking the instantiated model
        model = cls()
        found_moda = False
        for name, mod in model.named_modules():
            klass = type(mod).__name__
            if klass == "NoPEMoDAGQABlock":
                found_moda = True
                self.assertTrue(
                    getattr(mod, "_skip_autokernel", False),
                    f"{name} ({klass}) is missing _skip_autokernel = True; "
                    "autokernel will wrap it and crash on depth_kvs kwarg."
                )
        self.assertTrue(found_moda, "OdinHalo should contain at least one NoPEMoDAGQABlock")

    def test_skip_autokernel_attribute_on_gqa_block(self):
        """Regression: NoPEGQABlock (OdinFlat) must declare _skip_autokernel."""
        cls = _load_model_class(_REPO_ROOT / "models/odin_flat.py", "OdinFlat")
        model = cls()
        found_gqa = False
        for name, mod in model.named_modules():
            klass = type(mod).__name__
            if klass == "NoPEGQABlock":
                found_gqa = True
                self.assertTrue(
                    getattr(mod, "_skip_autokernel", False),
                    f"{name} ({klass}) is missing _skip_autokernel = True; "
                    "autokernel will wrap it and crash on Sprint 1 kwargs."
                )
        self.assertTrue(found_gqa, "OdinFlat should contain at least one NoPEGQABlock")


if __name__ == "__main__":
    unittest.main(verbosity=2)

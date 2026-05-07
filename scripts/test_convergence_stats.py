"""Unit tests for Phase A' convergence_stats evaluator.

Covers:
  - Import + callable run()
  - _resolve_layers detects looped (shared_layers) and flat (layers) models
  - _effective_rank returns sensible values for controlled inputs
  - _pick_split preference order
  - run() returns None on models without layers
  - run() returns expected keys for looped and flat models
  - Looped model reports iter_k_cos_to_final with length mean_recurrence - 1
  - Looped model's iter_transition_cos has length mean_recurrence - 1
  - Registry in scripts/eval_checkpoint.py includes convergence_stats

Run directly:
    python scripts/test_convergence_stats.py
"""

from __future__ import annotations

import os
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from halo_training.eval import convergence_stats  # noqa: E402
from halo_training.eval.convergence_stats import (  # noqa: E402
    _effective_rank,
    _pick_split,
    _resolve_layers,
)


class _SyntheticModelNoLayers(torch.nn.Module):
    """Bare model: no .layers, no .shared_layers."""

    def __init__(self):
        super().__init__()
        self.lin = torch.nn.Linear(8, 8)

    def forward(self, x):
        return self.lin(x.float())


class TestModuleBasics(unittest.TestCase):
    def test_import_and_run_callable(self):
        self.assertTrue(callable(convergence_stats.run))

    def test_resolve_layers_returns_none_when_absent(self):
        m = _SyntheticModelNoLayers()
        layers, is_looped = _resolve_layers(m)
        self.assertIsNone(layers)
        self.assertFalse(is_looped)

    def test_resolve_layers_detects_flat(self):
        class FlatModel(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.layers = torch.nn.ModuleList(
                    [torch.nn.Linear(4, 4) for _ in range(3)]
                )

        m = FlatModel()
        layers, is_looped = _resolve_layers(m)
        self.assertIsNotNone(layers)
        self.assertFalse(is_looped)
        self.assertEqual(len(layers), 3)

    def test_resolve_layers_prefers_shared_layers(self):
        class LoopedModel(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.shared_layers = torch.nn.ModuleList(
                    [torch.nn.Linear(4, 4) for _ in range(2)]
                )
                self.layers = torch.nn.ModuleList(
                    [torch.nn.Linear(4, 4)]  # should be ignored
                )

        m = LoopedModel()
        layers, is_looped = _resolve_layers(m)
        self.assertTrue(is_looped)
        self.assertEqual(len(layers), 2)

    def test_pick_split_preference_order(self):
        # wikitext_val wins if present
        splits = {
            "wikitext_val": ("/tmp/a.bin", 0, 100),
            "gpt_small_val": ("/tmp/b.bin", 0, 100),
        }
        split, name = _pick_split(splits)
        self.assertEqual(name, "wikitext_val")

        # Falls through in order
        splits = {
            "wikitext_val": None,
            "gpt_small_val": ("/tmp/b.bin", 0, 100),
            "dolma_val": ("/tmp/c.bin", 0, 100),
        }
        split, name = _pick_split(splits)
        self.assertEqual(name, "gpt_small_val")

        # None returned when all unavailable
        splits = {"wikitext_val": None, "dolma_val": None}
        split, name = _pick_split(splits)
        self.assertIsNone(split)
        self.assertIsNone(name)


class TestEffectiveRank(unittest.TestCase):
    def test_rank_1_matrix_has_stable_rank_1(self):
        # Rank-1 matrix: u @ v.T, effective rank should be 1.0
        u = torch.randn(100, 1)
        v = torch.randn(1, 16)
        h = u @ v
        self.assertAlmostEqual(_effective_rank(h), 1.0, places=3)

    def test_isotropic_noise_has_high_rank(self):
        # Random normal → stable rank near full (16 for [100, 16] with 100 rows)
        torch.manual_seed(0)
        h = torch.randn(200, 16)
        r = _effective_rank(h)
        # Allow wide tolerance — stable rank of random normal [N, D] is roughly D
        self.assertGreater(r, 10)
        self.assertLess(r, 17)

    def test_subsample_threshold(self):
        # Large matrix gets subsampled to max_rows without crashing
        h = torch.randn(10000, 8)
        r = _effective_rank(h, max_rows=256)
        self.assertFalse(np.isnan(r))
        self.assertGreater(r, 1.0)


class TestRunOnModels(unittest.TestCase):
    """Run the evaluator end-to-end on tiny real models with a synthetic bin."""

    @classmethod
    def setUpClass(cls):
        # Build a temporary `.bin` file with token ids in range [0, 1000)
        cls.tmp_dir = tempfile.mkdtemp(prefix="conv_stats_test_")
        cls.bin_path = os.path.join(cls.tmp_dir, "tiny.bin")
        rng = np.random.default_rng(7)
        tokens = rng.integers(0, 900, size=4096).astype(np.uint16)
        tokens.tofile(cls.bin_path)

        file_size = os.path.getsize(cls.bin_path)
        cls.splits = {
            "wikitext_val": (cls.bin_path, 0, file_size),
            "gpt_small_val": None,
            "stem_crawl_val": None,
            "dolma_val": None,
        }

    @classmethod
    def tearDownClass(cls):
        import shutil

        shutil.rmtree(cls.tmp_dir, ignore_errors=True)

    def _check_common_keys(self, result, expected_layers, expected_looped, expected_mr):
        self.assertIsNotNone(result)
        self.assertEqual(result["is_looped"], expected_looped)
        self.assertEqual(result["mean_recurrence"], expected_mr)
        self.assertEqual(result["num_layers"], expected_layers)
        self.assertEqual(len(result["per_layer_cos_to_final"]), expected_layers)
        self.assertEqual(
            len(result["per_layer_cos_to_final_frac_high"]), expected_layers)
        self.assertEqual(
            len(result["inter_layer_transition_cos"]), expected_layers - 1)
        # Last layer's cos-to-final should be ~1.0
        last_cos = result["per_layer_cos_to_final"][-1]
        self.assertAlmostEqual(last_cos, 1.0, places=2)
        # Effective rank should be finite and positive
        self.assertIsNotNone(result["effective_rank_final"])
        self.assertGreater(result["effective_rank_final"], 0.0)

    def test_run_on_odinflat_mini(self):
        # Import lazily — pulls torch / CUDA
        from importlib.util import module_from_spec, spec_from_file_location

        spec = spec_from_file_location(
            "odin_flat_test", _REPO_ROOT / "models" / "odin_flat.py")
        mod = module_from_spec(spec)
        spec.loader.exec_module(mod)
        model = mod.OdinFlatMini().eval()
        # Force CPU for CI portability if no CUDA; the evaluator uses
        # next(model.parameters()).device
        device = "cuda" if torch.cuda.is_available() else "cpu"
        if device == "cpu":
            self.skipTest("convergence_stats requires CUDA (autocast)")
        model = model.to(device).half()

        result = convergence_stats.run(
            model, tokenizer=None, validation_splits=self.splits,
            num_batches=1, seq_len=64, batch_size=1)
        self._check_common_keys(result, expected_layers=6,
                                expected_looped=False, expected_mr=1)
        self.assertIsNone(result["iter_k_cos_to_final"])
        self.assertIsNone(result["iter_transition_cos"])

    def test_run_on_odinhalo_mini(self):
        from importlib.util import module_from_spec, spec_from_file_location

        spec = spec_from_file_location(
            "odin_halo_test", _REPO_ROOT / "models" / "odin_halo.py")
        mod = module_from_spec(spec)
        spec.loader.exec_module(mod)
        model = mod.OdinHaloMini().eval()
        device = "cuda" if torch.cuda.is_available() else "cpu"
        if device == "cpu":
            self.skipTest("convergence_stats requires CUDA (autocast)")
        model = model.to(device).half()

        result = convergence_stats.run(
            model, tokenizer=None, validation_splits=self.splits,
            num_batches=1, seq_len=64, batch_size=1)
        # OdinHaloMini: n_shared_layers=6, mean_recurrence=2
        self._check_common_keys(result, expected_layers=6,
                                expected_looped=True, expected_mr=2)
        self.assertEqual(len(result["iter_k_cos_to_final"]), 1)
        self.assertEqual(len(result["iter_k_cos_to_final_frac_high"]), 1)
        self.assertEqual(len(result["iter_transition_cos"]), 1)
        # Cos values in [-1, 1]
        for v in result["iter_k_cos_to_final"]:
            self.assertGreaterEqual(v, -1.0)
            self.assertLessEqual(v, 1.0)

    def test_run_returns_none_when_model_lacks_layers(self):
        model = _SyntheticModelNoLayers().eval()
        result = convergence_stats.run(
            model, tokenizer=None, validation_splits=self.splits,
            num_batches=1, seq_len=64, batch_size=1)
        self.assertIsNone(result)


class TestRegistry(unittest.TestCase):
    """Ensure scripts/eval_checkpoint.py registers convergence_stats."""

    def test_registry_includes_convergence_stats(self):
        # Parse the file as text to avoid executing the CLI main
        path = _REPO_ROOT / "scripts" / "eval_checkpoint.py"
        src = path.read_text(encoding="utf-8")
        self.assertIn('"convergence_stats"', src,
                      "convergence_stats not in EVALUATORS registry")
        self.assertIn("halo_training.eval.convergence_stats", src)


if __name__ == "__main__":
    unittest.main(verbosity=2)

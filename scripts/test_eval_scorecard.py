"""Unit tests for Sprint 2 evaluation scorecard (Phase 1 scaffolding).

These tests verify infrastructure only — no model/checkpoint required. Later
phases add tests that load real checkpoints.

Run directly:
    python scripts/test_eval_scorecard.py

Or with pytest if installed:
    pytest scripts/test_eval_scorecard.py -v
"""

from __future__ import annotations

import json
import os
import sys
import tempfile
import unittest
from pathlib import Path

# Ensure repo root is on sys.path for halo_training imports
_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from halo_training.eval.common import (  # noqa: E402
    checkpoint_basename,
    current_timestamp_utc,
    default_validation_splits,
    resolve_eval_machine,
)
from halo_training.eval.scorecard import (  # noqa: E402
    SCHEMA_VERSION,
    _compute_avg_bpb,
    _safe_get,
    append_jsonl_index,
    assemble_scorecard,
    write_scorecard_json,
)


class _FakeModel:
    """Minimal stand-in so summarise_model() works without torch for these tests."""

    def __init__(self, n_params=1_000_000, d_model=768, n_layers=14):
        import torch

        # Create a single parameter so sum(p.numel()) returns something sensible
        self._p = torch.nn.Parameter(torch.zeros(n_params))
        self.d_model = d_model
        self.n_layers = n_layers

    def parameters(self):
        return [self._p]


class CommonHelpersTests(unittest.TestCase):
    def test_checkpoint_basename(self):
        cases = [
            (
                "checkpoints/odin-flat-wikitext-ddp/step_1869.pt",
                "odin-flat-wikitext-ddp-step-1869",
            ),
            (
                "checkpoints/odin-flat-stem-crawl-ddp/step_4046.pt",
                "odin-flat-stem-crawl-ddp-step-4046",
            ),
            (
                "/abs/path/to/run-name/step_0_fallback.pt",
                "run-name-step-0-fallback",
            ),
        ]
        for path, expected in cases:
            with self.subTest(path=path):
                self.assertEqual(checkpoint_basename(path), expected)

    def test_timestamp_format(self):
        ts = current_timestamp_utc()
        # e.g. "2026-05-06T14:22:01Z"
        self.assertRegex(ts, r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}Z$")

    def test_resolve_eval_machine_priority(self):
        # CLI overrides everything
        self.assertEqual(resolve_eval_machine("a"), "a")
        # env var if no CLI
        old = os.environ.get("EVAL_MACHINE")
        os.environ["EVAL_MACHINE"] = "test-b"
        try:
            self.assertEqual(resolve_eval_machine(None), "test-b")
        finally:
            if old is None:
                del os.environ["EVAL_MACHINE"]
            else:
                os.environ["EVAL_MACHINE"] = old
        # hostname fallback is covered by integration only (non-deterministic value)

    def test_default_validation_splits_handles_missing(self):
        # On the Windows dev box the datasets/*.bin files likely don't exist.
        # Result should be all None, not a crash.
        result = default_validation_splits()
        expected_keys = {
            "wikitext_val", "gpt_small_val", "stem_crawl_val", "dolma_val",
        }
        self.assertEqual(set(result.keys()), expected_keys)
        # Every entry is either None or (path, offset, length)
        for key, val in result.items():
            if val is None:
                continue
            self.assertIsInstance(val, tuple)
            self.assertEqual(len(val), 3)
            path, offset, length = val
            self.assertTrue(os.path.exists(path), f"{key} path must exist")
            self.assertGreater(offset, 0)
            self.assertGreater(length, 0)
            self.assertEqual(length % 2, 0, "must be uint16-aligned")


class ScorecardSchemaTests(unittest.TestCase):
    def _make_scorecard(self, extra_results=None):
        model = _FakeModel()
        results = {
            "per_domain_bpb": {
                "wikitext_val": 1.79,
                "gpt_small_val": 2.03,
                "stem_crawl_val": None,
                "dolma_val": None,
            },
            "sampling": {
                "distinct_2": 0.765,
                "self_ppl": 9.84,
                "winning_config": {"temperature": 0.6},
            },
            "inference_profile": {
                "tok_s_seq256_bs1": 142.3,
                "tok_s_seq512_bs1": 119.7,
                "tok_s_seq1024_bs1": 88.4,
                "peak_mem_gb_seq256": 0.91,
                "peak_mem_gb_seq512": 1.44,
                "peak_mem_gb_seq1024": 2.81,
            },
            "sample_pack": None,
            "activation_stats": None,
        }
        if extra_results:
            results.update(extra_results)
        return assemble_scorecard(
            checkpoint_path="checkpoints/test-run/step_123.pt",
            model_file="models/odin_flat.py",
            class_name="OdinFlat",
            model=model,
            step=123,
            eval_machine="test",
            eval_duration_s=12.5,
            eval_config={"max_tokens_per_domain": 50_000},
            results=results,
        )

    def test_scorecard_required_keys(self):
        sc = self._make_scorecard()
        required = {
            "schema_version", "checkpoint", "checkpoint_name", "step",
            "model", "eval_machine", "eval_timestamp", "eval_duration_s",
            "eval_config", "per_domain_bpb", "sampling",
            "inference_profile", "sample_pack", "activation_stats",
        }
        self.assertTrue(required.issubset(sc.keys()),
                        f"missing keys: {required - sc.keys()}")
        self.assertEqual(sc["schema_version"], SCHEMA_VERSION)

    def test_scorecard_json_serializable(self):
        sc = self._make_scorecard()
        # Must round-trip through JSON without raising
        s = json.dumps(sc, indent=2)
        restored = json.loads(s)
        self.assertEqual(restored["checkpoint_name"], "test-run-step-123")
        self.assertEqual(restored["model"]["class"], "OdinFlat")

    def test_write_scorecard_json_roundtrip(self):
        sc = self._make_scorecard()
        with tempfile.TemporaryDirectory() as tmp:
            path = write_scorecard_json(sc, tmp)
            self.assertTrue(os.path.exists(path))
            with open(path, encoding="utf-8") as f:
                restored = json.load(f)
            self.assertEqual(restored, sc)


class JsonlIndexTests(unittest.TestCase):
    def test_append_preserves_prior_lines(self):
        import copy

        scorecard_a = self._build("run-a-step-100", avg_bpb_override={"wikitext_val": 1.8})
        scorecard_b = self._build("run-b-step-200", avg_bpb_override={"wikitext_val": 1.7})

        with tempfile.TemporaryDirectory() as tmp:
            jsonl = os.path.join(tmp, "index.jsonl")
            append_jsonl_index(scorecard_a, jsonl)
            append_jsonl_index(scorecard_b, jsonl)

            with open(jsonl, encoding="utf-8") as f:
                lines = f.readlines()
            self.assertEqual(len(lines), 2)

            a = json.loads(lines[0])
            b = json.loads(lines[1])
            self.assertEqual(a["ckpt"], "run-a-step-100")
            self.assertEqual(b["ckpt"], "run-b-step-200")
            self.assertAlmostEqual(a["avg_bpb"], 1.8, places=4)
            self.assertAlmostEqual(b["avg_bpb"], 1.7, places=4)

    def test_append_tolerates_missing_sections(self):
        """JSONL appender must not crash if evaluators returned None."""
        sc = {
            "eval_timestamp": current_timestamp_utc(),
            "checkpoint_name": "sparse-step-1",
            "step": 1,
            "eval_machine": "test",
            "per_domain_bpb": None,
            "sampling": None,
            "inference_profile": None,
            "sample_pack": None,
        }
        with tempfile.TemporaryDirectory() as tmp:
            jsonl = os.path.join(tmp, "index.jsonl")
            append_jsonl_index(sc, jsonl)
            with open(jsonl, encoding="utf-8") as f:
                parsed = json.loads(f.read().strip())
            self.assertIsNone(parsed["avg_bpb"])
            self.assertIsNone(parsed["distinct_2"])
            self.assertIsNone(parsed["tok_s_512"])

    def _build(self, checkpoint_name: str, *, avg_bpb_override=None):
        """Build a minimal scorecard-shaped dict for JSONL tests."""
        bpb = {"wikitext_val": 1.8}
        if avg_bpb_override:
            bpb = avg_bpb_override
        return {
            "schema_version": SCHEMA_VERSION,
            "checkpoint": f"checkpoints/{checkpoint_name.rsplit('-', 2)[0]}/step_*.pt",
            "checkpoint_name": checkpoint_name,
            "step": 1,
            "eval_machine": "test",
            "eval_timestamp": current_timestamp_utc(),
            "eval_duration_s": 1.0,
            "per_domain_bpb": bpb,
            "sampling": {"distinct_2": 0.5, "self_ppl": 10.0},
            "inference_profile": {"tok_s_seq512_bs1": 100.0, "peak_mem_gb_seq512": 1.0},
        }


class EvaluatorImportTests(unittest.TestCase):
    """Phase 2+ evaluators must be importable and expose run()."""

    def test_per_domain_bpb_importable(self):
        from halo_training.eval import per_domain_bpb

        self.assertTrue(callable(per_domain_bpb.run))

    def test_sampling_importable(self):
        from halo_training.eval import sampling

        self.assertTrue(callable(sampling.run))

    def test_inference_profile_importable(self):
        from halo_training.eval import inference_profile

        self.assertTrue(callable(inference_profile.run))

    def test_sample_pack_importable(self):
        from halo_training.eval import sample_pack

        self.assertTrue(callable(sample_pack.run))

    def test_activation_stats_importable(self):
        from halo_training.eval import activation_stats

        self.assertTrue(callable(activation_stats.run))

    def test_sample_pack_v1_has_20_prompts(self):
        """Frozen sample pack must have exactly 20 non-empty prompts."""
        path = _REPO_ROOT / "evals" / "sample_pack_v1.txt"
        self.assertTrue(path.exists(), f"missing {path}")
        with open(path, encoding="utf-8") as f:
            prompts = [line.strip() for line in f if line.strip()]
        self.assertEqual(len(prompts), 20, f"expected 20 prompts, got {len(prompts)}")
        for i, p in enumerate(prompts):
            self.assertGreater(len(p), 0, f"prompt {i} empty")
            self.assertLess(len(p), 500, f"prompt {i} too long: {len(p)} chars")

    def test_sample_pack_canonicalize_deterministic(self):
        """_canonicalize must produce identical output for identical input."""
        from halo_training.eval.sample_pack import _canonicalize

        samples = ["hello world", "foo\nbar", "baz"]
        self.assertEqual(_canonicalize(samples), _canonicalize(samples))
        # Different input → different output
        self.assertNotEqual(_canonicalize(samples), _canonicalize(samples + ["extra"]))

    def test_sample_pack_auto_detect_prior(self):
        """Auto-detect prior: picks highest step_*.pt less than current."""
        import tempfile
        from halo_training.eval.sample_pack import _auto_detect_prior

        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            # Create fake sibling checkpoints
            for step in [100, 500, 1000, 1500]:
                (tmp_path / f"step_{step}.pt").write_bytes(b"")
            current = tmp_path / "step_1500.pt"
            prior = _auto_detect_prior(str(current))
            self.assertIsNotNone(prior)
            self.assertTrue(prior.endswith("step_1000.pt"))
            # Oldest file has no prior
            oldest = tmp_path / "step_100.pt"
            self.assertIsNone(_auto_detect_prior(str(oldest)))

    def test_ablation_harness_exports_library_api(self):
        """Sprint 2 sampling evaluator depends on these names. Don't rename."""
        from scripts.ablate_odin_flat_sampling import (
            run_ablation,
            select_winning_config,
        )

        self.assertTrue(callable(run_ablation))
        self.assertTrue(callable(select_winning_config))

    def test_train_ddp_exposes_auto_eval(self):
        """Sprint 2 Phase 5: --auto-eval flag and spawn_auto_eval must exist."""
        import importlib

        # Import train_ddp as a module to inspect its argparse + helpers
        spec = importlib.util.spec_from_file_location(
            "train_ddp_phase5_test",
            _REPO_ROOT / "scripts" / "train_ddp.py",
        )
        module = importlib.util.module_from_spec(spec)
        # Don't actually execute (would try to set up distributed) — just
        # read the source for the required markers.
        source = (_REPO_ROOT / "scripts" / "train_ddp.py").read_text(encoding="utf-8")
        self.assertIn("--auto-eval", source, "--auto-eval flag must be registered")
        self.assertIn("def spawn_auto_eval(", source,
                      "spawn_auto_eval helper must be defined")
        self.assertIn("start_new_session=True", source,
                      "auto-eval subprocess must be detached")


class InternalHelperTests(unittest.TestCase):
    def test_compute_avg_bpb_ignores_null(self):
        self.assertAlmostEqual(
            _compute_avg_bpb({"a": 1.0, "b": 2.0, "c": None}), 1.5, places=4,
        )
        self.assertIsNone(_compute_avg_bpb({"a": None, "b": None}))
        self.assertIsNone(_compute_avg_bpb(None))
        self.assertIsNone(_compute_avg_bpb({}))

    def test_compute_avg_bpb_ignores_diagnostic_keys(self):
        """Regression: don't average underscore-prefixed keys like _block_size."""
        per_domain = {
            "wikitext_val": 2.0,
            "gpt_small_val": 3.0,
            "stem_crawl_val": None,
            "_raw_ce": {"wikitext_val": 5.0, "gpt_small_val": 7.0},
            "_block_size": 512,
            "_batch_size": 8,
            "_max_tokens_per_domain": 20000,
            "_bpb_formula": "ce/ln(2)/3.6",
        }
        # Expected: mean(2.0, 3.0) = 2.5, not mean(2, 3, 512, 8, 20000) = 4105
        self.assertAlmostEqual(_compute_avg_bpb(per_domain), 2.5, places=4)

    def test_safe_get_tolerates_none(self):
        sc = {"sampling": {"distinct_2": 0.8}, "inference_profile": None}
        self.assertEqual(_safe_get(sc, "sampling", "distinct_2"), 0.8)
        self.assertIsNone(_safe_get(sc, "sampling", "missing"))
        self.assertIsNone(_safe_get(sc, "inference_profile", "tok_s_seq512_bs1"))
        self.assertIsNone(_safe_get(sc, "absent_section", "anything"))


if __name__ == "__main__":
    unittest.main(verbosity=2)

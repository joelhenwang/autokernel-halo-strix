"""Sprint 1.5 Phase A unit tests (SPECTRA + μP).

Consolidates the per-task smoke scripts into a single regression suite.

Run directly:
    python scripts/test_sprint1_5.py

Covers:
  A.1 SPECTRA apply_post_clip bounds spectral norm
  A.2 NorMuon with spectra_post=True clips single-step delta
  A.3 mup.apply_mup_init rescales hidden/readout, preserves embedding
  A.3 mup.build_mup_param_groups returns 3 groups with correct LRs
  A.4 build_imu1_optimizer(use_mup=True) yields 4+ groups
  A.4 build_imu1_optimizer(use_mup=False) preserves Sprint 1 behavior (2 groups)
  A.5 OdinFlat(use_mup=True) applies μP init after construction
  A.6 OdinFlat30M has ~33M params, head_dim=64
  A.7 train_ddp.py exposes all Sprint 1.5 CLI flags
  A.7 load_model_from_file forwards μP kwargs via ctor-signature filter
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


def _load(path, name):
    spec = spec_from_file_location(path.stem, path)
    mod = module_from_spec(spec)
    spec.loader.exec_module(mod)
    return getattr(mod, name)


class SpectraTests(unittest.TestCase):
    def test_apply_post_clip_bounds_spectral_norm(self):
        from halo_training.spectra import apply_post_clip

        torch.manual_seed(0)
        M = torch.randn(64, 128)
        clipped = apply_post_clip(M, clip_norm=1.0)
        sigma1 = torch.svd(clipped).S[0].item()
        self.assertLessEqual(sigma1, 1.01,
                             f"clipped sigma1={sigma1} breaches 1.01")

    def test_apply_post_clip_identity_small(self):
        from halo_training.spectra import apply_post_clip

        M = torch.randn(16, 16) * 0.01
        clipped = apply_post_clip(M, clip_norm=1.0)
        self.assertTrue(torch.allclose(clipped, M))

    def test_apply_post_clip_non_2d_noop(self):
        from halo_training.spectra import apply_post_clip

        v = torch.randn(8)
        out = apply_post_clip(v, clip_norm=1.0)
        self.assertIs(out, v)

    def test_apply_post_clip_invalid_clip(self):
        from halo_training.spectra import apply_post_clip

        M = torch.randn(4, 4)
        with self.assertRaises(ValueError):
            apply_post_clip(M, clip_norm=-1.0)


class NorMuonSpectraTests(unittest.TestCase):
    def test_normuon_post_clip_bounds_single_step_delta(self):
        from halo_training.normuon import NorMuon

        torch.manual_seed(99)
        model = torch.nn.Linear(32, 64, bias=False)
        opt = NorMuon(
            muon_params=[model.weight], lr=1.0,
            spectra_post=True, spectra_clip_norm=1.0,
            cautious_wd=False, neuron_wise_norm=False,
        )
        w_before = model.weight.data.clone()
        (model(torch.randn(4, 32)).sum()).backward()
        opt.step()
        delta = model.weight.data - w_before
        sigma1 = torch.svd(delta.to(torch.float32)).S[0].item()
        self.assertLessEqual(sigma1, 1.01)


class MupInitTests(unittest.TestCase):
    def test_apply_mup_init_rescales_hidden_only(self):
        from halo_training.mup import apply_mup_init

        # Build a toy model with one "tok_embeddings" and one hidden Linear.
        class _Toy(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.tok_embeddings = torch.nn.Linear(512, 32, bias=False)
                self.hidden = torch.nn.Linear(512, 512, bias=False)

        torch.manual_seed(0)
        m = _Toy()
        emb_before = m.tok_embeddings.weight.detach().norm().item()
        hid_before = m.hidden.weight.detach().norm().item()

        apply_mup_init(m, d_base=256)
        emb_after = m.tok_embeddings.weight.detach().norm().item()
        hid_after = m.hidden.weight.detach().norm().item()

        self.assertAlmostEqual(emb_before, emb_after, places=5)
        # Hidden rescaled by sqrt(256 / 512) = ~0.7071
        expected = hid_before * (256.0 / 512.0) ** 0.5
        self.assertAlmostEqual(hid_after, expected, places=4)


class MupParamGroupsTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.OdinFlat = _load(_REPO_ROOT / "models/odin_flat.py", "OdinFlat")

    def test_three_groups_with_correct_lrs(self):
        from halo_training.mup import build_mup_param_groups

        model = self.OdinFlat()
        groups = build_mup_param_groups(model, base_lr=0.0235, d_base=256)
        self.assertEqual(len(groups), 3)
        tags = [g["_mup_group"] for g in groups]
        self.assertEqual(tags, ["embedding", "hidden", "readout"])
        d_ratio = model.d_model / 256
        self.assertAlmostEqual(groups[0]["lr"], 0.0235, places=9)
        self.assertAlmostEqual(groups[1]["lr"], 0.0235 / d_ratio, places=9)
        self.assertAlmostEqual(groups[2]["lr"], 0.0235 / d_ratio ** 2, places=9)

    def test_no_param_in_two_groups(self):
        from halo_training.mup import build_mup_param_groups

        model = self.OdinFlat()
        groups = build_mup_param_groups(model, base_lr=0.0235, d_base=256)
        seen = set()
        for g in groups:
            for p in g["params"]:
                self.assertNotIn(id(p), seen,
                                 "param appears in multiple μP groups")
                seen.add(id(p))


class BuildImu1OptimizerTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.OdinFlat = _load(_REPO_ROOT / "models/odin_flat.py", "OdinFlat")

    def test_use_mup_true_has_4_groups(self):
        from halo_training.optimizer import build_imu1_optimizer

        model = self.OdinFlat()
        opt = build_imu1_optimizer(
            model, lr_2d=0.0235, lr_1d=0.007,
            use_mup=True, mup_base_width=256,
        )
        non_empty = [g for g in opt.param_groups if len(g["params"]) > 0]
        self.assertGreaterEqual(len(non_empty), 4)

    def test_use_mup_false_preserves_sprint1(self):
        from halo_training.optimizer import build_imu1_optimizer

        model = self.OdinFlat()
        opt = build_imu1_optimizer(
            model, lr_2d=0.0235, lr_1d=0.007, use_mup=False,
        )
        non_empty = [g for g in opt.param_groups if len(g["params"]) > 0]
        self.assertEqual(len(non_empty), 2)


class OdinFlatMupInitTests(unittest.TestCase):
    def test_use_mup_flag_rescales_hidden(self):
        OdinFlat = _load(_REPO_ROOT / "models/odin_flat.py", "OdinFlat")

        torch.manual_seed(42)
        a = OdinFlat(use_mup=False)
        torch.manual_seed(42)
        b = OdinFlat(use_mup=True, mup_base_width=256)

        # At least one hidden param should be smaller with μP.
        any_rescaled = False
        for (na, pa), (nb, pb) in zip(a.named_parameters(), b.named_parameters()):
            if pa.dim() < 2:
                continue
            if "tok_embeddings" in na:
                continue  # embedding unchanged
            na_norm = pa.detach().norm().item()
            nb_norm = pb.detach().norm().item()
            if nb_norm < na_norm * 0.99:
                any_rescaled = True
                break
        self.assertTrue(any_rescaled, "μP init did not reduce any hidden norm")


class OdinFlat30MTests(unittest.TestCase):
    def test_param_count_and_head_dim(self):
        cls = _load(_REPO_ROOT / "models/odin_flat_30m.py", "OdinFlat30M")
        model = cls()
        n = sum(p.numel() for p in model.parameters())
        self.assertGreaterEqual(n, 25e6)
        self.assertLessEqual(n, 36e6)
        self.assertEqual(model.head_dim, 64)


class CliFlagsTests(unittest.TestCase):
    def test_train_ddp_exposes_sprint15_flags(self):
        src = (_REPO_ROOT / "scripts" / "train_ddp.py").read_text(encoding="utf-8")
        for flag in (
            "--spectra-post", "--spectra-clip-norm", "--spectra-ns-iter",
            "--spectra-pre",
            "--mup", "--mup-base-width", "--mup-attn",
        ):
            self.assertIn(flag, src, f"missing CLI flag {flag}")

    def test_load_model_forwards_mup_kwargs(self):
        from scripts.train_ddp import load_model_from_file

        model = load_model_from_file(
            str(_REPO_ROOT / "models/odin_flat.py"), "OdinFlat",
            use_mup=True, mup_base_width=256)
        self.assertTrue(getattr(model, "use_mup", False))
        self.assertEqual(getattr(model, "mup_base_width", None), 256)

    def test_load_model_ignores_unknown_kwargs(self):
        """A kwarg the ctor doesn't accept should be filtered, not raise."""
        from scripts.train_ddp import load_model_from_file

        model = load_model_from_file(
            str(_REPO_ROOT / "models/odin_flat.py"), "OdinFlat",
            use_mup=True, mup_base_width=256,
            nonexistent_flag_for_filter_test=42)
        self.assertTrue(getattr(model, "use_mup", False))


if __name__ == "__main__":
    unittest.main(verbosity=2)

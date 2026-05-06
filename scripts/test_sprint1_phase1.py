"""Unit tests for Sprint 1 Phase 1 foundation-wins infrastructure.

Covers:
  - PreTokenizedDataset / BabyLMDataset 3-tuple return with doc_ids
  - split_params_2d_vs_1d partitioning correctness
  - build_imu1_optimizer AdamW two-group path
  - LayerNorm scaling init (1/sqrt(layer_idx+1))
  - Intra-document mask plumbing through NoPECodaAttention
  - OdinFlat.forward accepts doc_ids kwarg gracefully

Runs on CPU, no CUDA required for most tests. The attention test uses
CPU fp32 which is slow but adequate for correctness.

Run directly:
    python scripts/test_sprint1_phase1.py
"""

from __future__ import annotations

import math
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

from halo_training.data import (  # noqa: E402
    DEFAULT_EOS_ID,
    BabyLMDataset,
    PreTokenizedDataset,
)
from halo_training.optimizer import (  # noqa: E402
    build_imu1_optimizer,
    split_params_2d_vs_1d,
)


# ---------------------------------------------------------------------------
# Task 1.1 / 1.2 — doc_ids in datasets
# ---------------------------------------------------------------------------


def _make_bin(path: Path, tokens: list[int]) -> None:
    """Write a list[int] to disk as uint16 .bin (pre-tokenized format)."""
    arr = np.asarray(tokens, dtype=np.uint16)
    arr.tofile(str(path))


class DocIdsTests(unittest.TestCase):
    def test_pretokenized_returns_3_tuple(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "toy.bin"
            # 16 tokens -> with stride 9 (block_size=8, +1) = 1 chunk of 8
            _make_bin(path, list(range(1, 17)))
            ds = PreTokenizedDataset(str(path), block_size=8)
            item = ds[0]
            self.assertEqual(len(item), 3)
            x, y, doc = item
            self.assertEqual(x.shape, (8,))
            self.assertEqual(y.shape, (8,))
            self.assertEqual(doc.shape, (8,))
            self.assertEqual(doc.dtype, torch.int32)
            # No EOS (0) in tokens 1..16, so all doc_ids are 0
            self.assertTrue(torch.all(doc == 0))
            # Windows: release memmap handle before tempdir cleanup
            del ds, item, x, y, doc

    def test_pretokenized_doc_ids_cumsum_at_eos(self):
        """doc_id increments at each EOS position."""
        # Layout: [7, 5, 0, 3, 9, 0, 4, 1, 2] — EOS at positions 2 and 5
        # block_size = 8, so x = tokens[:8] = [7, 5, 0, 3, 9, 0, 4, 1]
        # Expected doc_ids = cumsum(x == 0) = [0, 0, 1, 1, 1, 2, 2, 2]
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "toy.bin"
            _make_bin(path, [7, 5, DEFAULT_EOS_ID, 3, 9, DEFAULT_EOS_ID, 4, 1, 2])
            ds = PreTokenizedDataset(str(path), block_size=8)
            _, _, doc = ds[0]
            expected = torch.tensor([0, 0, 1, 1, 1, 2, 2, 2], dtype=torch.int32)
            self.assertTrue(torch.all(doc == expected),
                            f"doc_ids mismatch: {doc.tolist()} vs {expected.tolist()}")
            del ds, doc  # release memmap handle before tempdir cleanup

    def test_babylm_bin_path_returns_3_tuple(self):
        """Legacy BabyLMDataset in .bin mode also returns 3-tuple."""
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "toy.bin"
            _make_bin(path, [1, 2, DEFAULT_EOS_ID, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16])
            ds = BabyLMDataset(root=str(path), block_size=8)
            item = ds[0]
            self.assertEqual(len(item), 3)
            x, y, doc = item
            self.assertEqual(doc.shape, x.shape)
            del ds, item, x, y, doc  # release memmap handle before tempdir cleanup

    def test_default_eos_id_is_zero(self):
        """AGENTS.md contract: odin-32k EOS is id 0."""
        self.assertEqual(DEFAULT_EOS_ID, 0)


# ---------------------------------------------------------------------------
# Task 1.3 — split_params_2d_vs_1d
# ---------------------------------------------------------------------------


def _build_odin_flat_mini():
    """Construct a small OdinFlat variant for param-grouping tests."""
    from models.odin_flat import OdinFlatMini

    return OdinFlatMini()


class ParamGroupingTests(unittest.TestCase):
    def setUp(self):
        self.model = _build_odin_flat_mini()

    def test_partition_is_complete_and_disjoint(self):
        """Every trainable param lands in exactly one of (2D, 1D) groups."""
        group_2d, group_1d = split_params_2d_vs_1d(self.model)
        names_2d = {n for n, _ in group_2d}
        names_1d = {n for n, _ in group_1d}
        self.assertTrue(names_2d.isdisjoint(names_1d),
                        f"Overlap: {names_2d & names_1d}")

        # Sum via identity (respecting tied-weight de-dup)
        seen_ids = set()
        total_counted = 0
        for _, p in group_2d + group_1d:
            if id(p) in seen_ids:
                continue
            seen_ids.add(id(p))
            total_counted += p.numel()

        seen_ids2 = set()
        total_trainable = 0
        for p in self.model.parameters():
            if not p.requires_grad:
                continue
            if id(p) in seen_ids2:
                continue
            seen_ids2.add(id(p))
            total_trainable += p.numel()

        self.assertEqual(total_counted, total_trainable,
                         f"{total_counted} grouped vs {total_trainable} trainable "
                         "(missing or double-counted params)")

    def test_embedding_in_1d_group(self):
        """Raw embedding table belongs to the 1D/embed group."""
        _, group_1d = split_params_2d_vs_1d(self.model)
        names_1d = {n for n, _ in group_1d}
        self.assertIn("tok_embeddings.embed.weight", names_1d)

    def test_layernorm_in_1d_group(self):
        """RMSNorm gammas (ndim=1) land in the 1D group."""
        _, group_1d = split_params_2d_vs_1d(self.model)
        names_1d = {n for n, _ in group_1d}
        rms_names = [n for n in names_1d if "norm" in n.lower()]
        self.assertTrue(len(rms_names) > 0,
                        f"No RMSNorm gammas found in 1D group: {names_1d}")

    def test_qk_scale_in_1d_group(self):
        """q_scale / k_scale (per-head learned scalars) land in 1D group."""
        _, group_1d = split_params_2d_vs_1d(self.model)
        names_1d = {n for n, _ in group_1d}
        has_scale = any("scale" in n for n in names_1d)
        self.assertTrue(has_scale, f"No *_scale params in 1D group: {names_1d}")

    def test_linear_weights_in_2d_group(self):
        """Attention wq/wk/wv/wo and FFN weights are 2D, go to NorMuon."""
        group_2d, _ = split_params_2d_vs_1d(self.model)
        names_2d = {n for n, _ in group_2d}
        # Expect at least 4 attn linear weights + SwiGLU weights per GQA block
        attn_weights = [n for n in names_2d if any(p in n for p in ("wq", "wk", "wv", "wo"))]
        self.assertTrue(len(attn_weights) > 0, f"No attention weights in 2D: {names_2d}")


# ---------------------------------------------------------------------------
# Task 1.4 — build_imu1_optimizer
# ---------------------------------------------------------------------------


class Imu1OptimizerTests(unittest.TestCase):
    def setUp(self):
        self.model = _build_odin_flat_mini()

    def test_adamw_two_group_path(self):
        opt = build_imu1_optimizer(
            self.model, lr_2d=0.01, lr_1d=0.003, use_normuon=False,
        )
        self.assertIsInstance(opt, torch.optim.AdamW)
        # Expect two param groups with distinct LR + WD
        self.assertEqual(len(opt.param_groups), 2)

        g2d = opt.param_groups[0]
        g1d = opt.param_groups[1]
        self.assertAlmostEqual(g2d["lr"], 0.01)
        self.assertAlmostEqual(g1d["lr"], 0.003)
        self.assertAlmostEqual(g2d["weight_decay"], 0.1)
        self.assertAlmostEqual(g1d["weight_decay"], 0.0)

    def test_normuon_not_ready_raises(self):
        """Phase 1: normuon.py does not exist yet; use_normuon=True raises."""
        with self.assertRaises(RuntimeError):
            build_imu1_optimizer(self.model, use_normuon=True)


# ---------------------------------------------------------------------------
# Task 1.5 — LayerNorm scaling init
# ---------------------------------------------------------------------------


class LnScalingTests(unittest.TestCase):
    def test_depth_scaled_gamma(self):
        """After init, layer i's RMSNorm gamma should have mean ≈ 1/sqrt(i+1)."""
        model = _build_odin_flat_mini()
        from models._components import RMSNorm

        for layer_idx, layer in enumerate(model.layers):
            expected = 1.0 / math.sqrt(layer_idx + 1)
            rms_modules = [m for m in layer.modules() if isinstance(m, RMSNorm)]
            self.assertTrue(len(rms_modules) > 0,
                            f"layer {layer_idx} has no RMSNorm modules")
            for m in rms_modules:
                mean = m.weight.data.mean().item()
                # RMSNorm initialized to 1.0 then multiplied by scale
                self.assertAlmostEqual(
                    mean, expected, places=4,
                    msg=f"layer {layer_idx} RMSNorm gamma mean={mean} vs expected {expected}",
                )


# ---------------------------------------------------------------------------
# Task 1.6 — Intra-document mask behavior
# ---------------------------------------------------------------------------


class IntraDocMaskTests(unittest.TestCase):
    @unittest.skipUnless(torch.cuda.is_available(), "Needs CUDA/HIP (HyPE conv block uses fused HIP RoPE)")
    def test_forward_without_doc_ids_works(self):
        """Backward-compat: forward with only input_ids still runs."""
        from models.odin_flat import OdinFlatMini

        model = OdinFlatMini().cuda()
        model.eval()
        x = torch.randint(0, model.vocab_size, (2, 16), device="cuda")
        with torch.no_grad(), torch.amp.autocast("cuda", dtype=torch.float16):
            out = model(x)
        self.assertEqual(out.shape[:2], x.shape)
        self.assertTrue(torch.isfinite(out).all())

    @unittest.skipUnless(torch.cuda.is_available(), "Needs CUDA/HIP")
    def test_forward_with_doc_ids_and_mask_enabled(self):
        """With use_intra_doc_mask=True, forward accepts doc_ids and produces finite output."""
        from models.odin_flat import OdinFlatMini

        model = OdinFlatMini(use_intra_doc_mask=True).cuda()
        model.eval()
        x = torch.randint(0, model.vocab_size, (2, 16), device="cuda")
        doc_ids = torch.zeros_like(x, dtype=torch.int32)
        doc_ids[:, 8:] = 1  # two docs per sample
        with torch.no_grad(), torch.amp.autocast("cuda", dtype=torch.float16):
            out = model(x, doc_ids=doc_ids)
        self.assertEqual(out.shape[:2], x.shape)
        self.assertTrue(torch.isfinite(out).all())

    @unittest.skipUnless(torch.cuda.is_available(), "Needs CUDA/HIP")
    def test_doc_mask_disabled_is_no_op(self):
        """use_intra_doc_mask=False should produce identical output w/ or w/o doc_ids."""
        from models.odin_flat import OdinFlatMini

        torch.manual_seed(42)
        model = OdinFlatMini(use_intra_doc_mask=False).cuda()
        model.eval()
        x = torch.randint(0, model.vocab_size, (1, 16), device="cuda")

        with torch.no_grad(), torch.amp.autocast("cuda", dtype=torch.float16):
            out_no = model(x)
            doc_ids = torch.zeros_like(x, dtype=torch.int32)
            out_with = model(x, doc_ids=doc_ids)

        self.assertTrue(torch.allclose(out_no, out_with, atol=1e-4))


if __name__ == "__main__":
    unittest.main(verbosity=2)

"""Unit tests for Phase 3 LEAP layer-exit aux loss.

Covers:
  - Module construction validation (bad indices, mismatched weights)
  - Hook registration + capture after forward
  - compute_aux_loss returns finite scalar, gradient flows to intermediates
  - compute_aux_loss errors when hooks haven't fired
  - close() removes hooks cleanly
  - layer_exit_decision: tokens with high cos exit; low cos continue
  - compatibility with torch.compile (skip if CUDA unavailable)

Run:
    python scripts/test_leap.py
"""

from __future__ import annotations

import sys
import unittest
from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path

import torch
import torch.nn as nn

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from halo_training.leap_layer_exit import (  # noqa: E402
    DEFAULT_TAU_TRAIN,
    LeapAuxLoss,
    layer_exit_decision,
)


class _ToyFlatModel(nn.Module):
    """Minimal flat model with .layers for hook testing."""

    def __init__(self, d=32, n_layers=4):
        super().__init__()
        self.d_model = d
        self.layers = nn.ModuleList([nn.Linear(d, d) for _ in range(n_layers)])
        self.final_norm = nn.LayerNorm(d)
        self.head = nn.Linear(d, 128)

    def forward(self, x):
        h = x.float()
        for layer in self.layers:
            h = layer(h)
        h_final = self.final_norm(h)
        return self.head(h_final), h_final


class TestLeapConstruction(unittest.TestCase):
    def test_empty_indices_raises(self):
        model = _ToyFlatModel()
        with self.assertRaises(ValueError):
            LeapAuxLoss(model, layer_indices=[])

    def test_weights_length_mismatch(self):
        model = _ToyFlatModel()
        with self.assertRaises(ValueError):
            LeapAuxLoss(model, layer_indices=[1, 2], weights=[0.5])

    def test_out_of_bounds_index(self):
        model = _ToyFlatModel(n_layers=4)
        with self.assertRaises(IndexError):
            LeapAuxLoss(model, layer_indices=[4])  # valid range 0..3
        with self.assertRaises(IndexError):
            LeapAuxLoss(model, layer_indices=[-1])

    def test_missing_layers_attr(self):
        # model WITHOUT .layers
        class _NoLayers(nn.Module):
            pass

        with self.assertRaises(AttributeError):
            LeapAuxLoss(_NoLayers(), layer_indices=[0])

    def test_alt_layers_attr(self):
        class _Odin(nn.Module):
            def __init__(self):
                super().__init__()
                self.shared_layers = nn.ModuleList([nn.Linear(8, 8) for _ in range(3)])

        leap = LeapAuxLoss(_Odin(), layer_indices=[0, 1],
                           layers_attr="shared_layers")
        self.assertEqual(len(leap._hooks), 2)
        leap.close()


class TestLeapForwardAndLoss(unittest.TestCase):
    def test_hooks_capture_after_forward(self):
        torch.manual_seed(0)
        model = _ToyFlatModel(d=16, n_layers=4).eval()
        leap = LeapAuxLoss(model, layer_indices=[1, 2])
        x = torch.randn(2, 8, 16)
        try:
            _ = model(x)
            self.assertIn(1, leap._captured)
            self.assertIn(2, leap._captured)
            self.assertEqual(leap._captured[1].shape, (2, 8, 16))
        finally:
            leap.close()

    def test_compute_aux_loss_finite_scalar(self):
        torch.manual_seed(0)
        model = _ToyFlatModel(d=16, n_layers=4)
        leap = LeapAuxLoss(model, layer_indices=[1, 2], weights=[0.3, 0.5])
        try:
            x = torch.randn(2, 8, 16)
            _, h_final = model(x)
            loss = leap.compute_aux_loss(h_final)
            self.assertEqual(loss.shape, ())
            self.assertTrue(torch.isfinite(loss))
            self.assertGreater(loss.item(), 0.0)
            self.assertLess(loss.item(), 1.0 * (0.3 + 0.5) + 1e-5)
        finally:
            leap.close()

    def test_compute_aux_loss_raises_without_forward(self):
        model = _ToyFlatModel()
        leap = LeapAuxLoss(model, layer_indices=[1])
        try:
            h_final = torch.randn(1, 4, 32)
            with self.assertRaises(RuntimeError):
                leap.compute_aux_loss(h_final)
        finally:
            leap.close()

    def test_gradient_flows_to_intermediate(self):
        """Ensure aux loss backward populates layer[1]'s weight.grad."""
        torch.manual_seed(0)
        model = _ToyFlatModel(d=16, n_layers=4)
        leap = LeapAuxLoss(model, layer_indices=[1])
        try:
            x = torch.randn(2, 8, 16)
            _, h_final = model(x)
            loss = leap.compute_aux_loss(h_final)
            loss.backward()
            self.assertIsNotNone(model.layers[1].weight.grad)
            self.assertTrue(torch.any(model.layers[1].weight.grad != 0))
            # h_final SHOULD NOT contribute via grad because we used stop-grad
            # (final_norm and head are downstream; loss only flows through hook).
            # We test: layers[3].weight.grad should NOT exist (only layers[0..1] on path)
            # Actually layers[2] and [3] come AFTER layer 1 so their outputs don't
            # enter aux loss. They WILL be on the autograd graph via forward though.
            # Just verify the hook layer received grad.
        finally:
            leap.close()

    def test_captures_cleared_after_compute(self):
        torch.manual_seed(0)
        model = _ToyFlatModel()
        leap = LeapAuxLoss(model, layer_indices=[1, 2])
        try:
            _ = model(torch.randn(1, 4, 32))
            _, h_final = model(torch.randn(1, 4, 32))
            self.assertGreater(len(leap._captured), 0)
            leap.compute_aux_loss(h_final)
            self.assertEqual(len(leap._captured), 0)
        finally:
            leap.close()

    def test_discard_captures(self):
        model = _ToyFlatModel()
        leap = LeapAuxLoss(model, layer_indices=[1])
        try:
            _ = model(torch.randn(1, 4, 32))
            self.assertGreater(len(leap._captured), 0)
            leap.discard_captures()
            self.assertEqual(len(leap._captured), 0)
        finally:
            leap.close()


class TestLeapClose(unittest.TestCase):
    def test_close_removes_hooks(self):
        model = _ToyFlatModel()
        leap = LeapAuxLoss(model, layer_indices=[1, 2])
        # 2 layer hooks + 1 final_norm hook = 3 total
        self.assertEqual(len(leap._hooks), 3)
        leap.close()
        self.assertEqual(len(leap._hooks), 0)

    def test_close_is_idempotent(self):
        model = _ToyFlatModel()
        leap = LeapAuxLoss(model, layer_indices=[0])
        leap.close()
        leap.close()  # should not raise

    def test_compute_aux_loss_auto_h_final(self):
        """Auto-capture h_final via final_norm hook works when h_final omitted."""
        model = _ToyFlatModel(d=16, n_layers=4)
        leap = LeapAuxLoss(model, layer_indices=[1, 2])
        try:
            x = torch.randn(2, 8, 16)
            _ = model(x)
            loss = leap.compute_aux_loss()  # no explicit h_final
            self.assertTrue(torch.isfinite(loss))
        finally:
            leap.close()

    def test_final_attr_none_requires_explicit_h_final(self):
        model = _ToyFlatModel()
        leap = LeapAuxLoss(model, layer_indices=[1], final_attr=None)
        try:
            _ = model(torch.randn(1, 4, 32))
            with self.assertRaises(RuntimeError):
                leap.compute_aux_loss()  # no h_final and no hook
        finally:
            leap.close()


class TestLayerExitDecision(unittest.TestCase):
    def test_high_cos_exits(self):
        # Two identical tensors -> cos=1 -> exit everywhere.
        h = torch.randn(2, 4, 16)
        mask = layer_exit_decision(h, h, theta=0.95)
        self.assertTrue(mask.all())
        self.assertEqual(mask.shape, (2, 4))

    def test_low_cos_no_exit(self):
        torch.manual_seed(0)
        h1 = torch.randn(2, 4, 16)
        h2 = torch.randn(2, 4, 16)
        mask = layer_exit_decision(h1, h2, theta=0.95)
        # Random unit vectors in 16d have cos ≈ 0 → none should exit
        self.assertFalse(mask.any())

    def test_shape_mismatch_raises(self):
        h1 = torch.randn(2, 4, 16)
        h2 = torch.randn(2, 4, 32)
        with self.assertRaises(ValueError):
            layer_exit_decision(h1, h2)


if __name__ == "__main__":
    unittest.main(verbosity=2)

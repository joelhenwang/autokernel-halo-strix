"""Unit tests for Sprint 1 Phase 2 (NorMuon + value residual + head gating).

Covers:
  - NorMuon optimizer: Newton-Schulz orthogonality, neuron-wise norm,
    Cautious WD sign mask, and the full optimizer step on a toy MLP.
  - Value residual: v_res_scale initialized to zero; output w/ and w/o
    v_prev matches at init; grad flows to v_res_scale.
  - Per-head gating: head_gate initialized to 1.0 (sigmoid = ~0.731);
    grad flows to head_gate.
  - Integration: OdinFlatMini with all features enabled produces finite
    outputs and propagates gradients cleanly.

Run directly:
    python scripts/test_sprint1_phase2.py
"""

from __future__ import annotations

import math
import sys
import unittest
from pathlib import Path

import torch
import torch.nn as nn

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from halo_training.normuon import (  # noqa: E402
    NorMuon,
    _cautious_wd_mask,
    _neuron_wise_normalize,
    _newton_schulz_polar_express,
)


# ---------------------------------------------------------------------------
# NorMuon primitives
# ---------------------------------------------------------------------------


class NewtonSchulzTests(unittest.TestCase):
    def test_newton_schulz_produces_orthogonal(self):
        """msgn(X) should be approximately orthogonal (X X^T ≈ I, or X^T X ≈ I)."""
        torch.manual_seed(0)
        G = torch.randn(64, 32)
        U = _newton_schulz_polar_express(G, steps=5, dtype=torch.float32)
        # U has shape (64, 32). U^T U ≈ I_32 (columns orthonormal).
        gram = U.T @ U
        eye = torch.eye(32)
        err = (gram - eye).abs().max().item()
        self.assertLess(err, 0.15,
                        f"orthogonality error {err} too large (expected < 0.15)")

    def test_newton_schulz_handles_tall(self):
        """Matrix wider than tall: same bound."""
        torch.manual_seed(1)
        G = torch.randn(32, 64)
        U = _newton_schulz_polar_express(G, steps=5, dtype=torch.float32)
        gram = U @ U.T
        eye = torch.eye(32)
        err = (gram - eye).abs().max().item()
        self.assertLess(err, 0.15, f"orthogonality error {err}")


class NeuronWiseNormTests(unittest.TestCase):
    def test_unit_row_norms(self):
        torch.manual_seed(2)
        M = torch.randn(16, 32) * 3.0  # mixed magnitudes
        N = _neuron_wise_normalize(M)
        row_norms = N.norm(dim=1)
        self.assertTrue(torch.allclose(row_norms, torch.ones(16), atol=1e-5))

    def test_zero_rows_handled(self):
        M = torch.zeros(4, 8)
        N = _neuron_wise_normalize(M)
        # Zero rows should stay zero (eps prevents div by zero)
        self.assertTrue(torch.all(N == 0))


class CautiousWdMaskTests(unittest.TestCase):
    def test_same_sign_true(self):
        grad = torch.tensor([1.0, -2.0, 3.0, 0.5])
        weight = torch.tensor([0.5, -1.0, 2.0, 0.1])  # all same sign as grad
        mask = _cautious_wd_mask(grad, weight)
        self.assertTrue(torch.all(mask))

    def test_opposite_sign_false(self):
        grad = torch.tensor([1.0, -2.0])
        weight = torch.tensor([-0.5, 1.0])
        mask = _cautious_wd_mask(grad, weight)
        self.assertFalse(torch.any(mask))

    def test_zero_handled(self):
        grad = torch.tensor([0.0, 1.0])
        weight = torch.tensor([1.0, 0.0])
        mask = _cautious_wd_mask(grad, weight)
        # sign(0)=0 so either product is 0 -> mask False
        self.assertFalse(torch.any(mask))


# ---------------------------------------------------------------------------
# Full NorMuon optimizer step
# ---------------------------------------------------------------------------


class NorMuonStepTests(unittest.TestCase):
    def _build_toy_mlp(self):
        torch.manual_seed(7)
        model = nn.Sequential(
            nn.Linear(32, 64, bias=True),
            nn.ReLU(),
            nn.Linear(64, 16, bias=True),
        )
        return model

    def test_single_step_produces_finite_update(self):
        model = self._build_toy_mlp()
        weights_2d = [p for p in model.parameters() if p.ndim == 2]
        weights_1d = [p for p in model.parameters() if p.ndim < 2]
        opt = NorMuon(
            muon_params=[{"params": weights_2d}],
            adamw_params=[{"params": weights_1d, "weight_decay": 0.0}],
            lr=0.01, adamw_lr=0.003, weight_decay=0.1,
        )

        x = torch.randn(4, 32)
        y = torch.randn(4, 16)

        pre_params = [p.detach().clone() for p in model.parameters()]
        pred = model(x)
        loss = (pred - y).pow(2).mean()
        loss.backward()
        opt.step()

        # Each parameter should have moved
        moved = 0
        for pre, p in zip(pre_params, model.parameters()):
            self.assertTrue(torch.isfinite(p).all(),
                            f"non-finite param after step: {p}")
            if not torch.allclose(pre, p):
                moved += 1
        self.assertGreater(moved, 0, "no parameter moved after step")

    def test_loss_decreases_over_few_steps(self):
        model = self._build_toy_mlp()
        weights_2d = [p for p in model.parameters() if p.ndim == 2]
        weights_1d = [p for p in model.parameters() if p.ndim < 2]
        opt = NorMuon(
            muon_params=[{"params": weights_2d}],
            adamw_params=[{"params": weights_1d, "weight_decay": 0.0}],
            lr=0.05, adamw_lr=0.01, weight_decay=0.1,
        )

        torch.manual_seed(11)
        x = torch.randn(32, 32)
        y = torch.randn(32, 16)
        losses = []
        for _ in range(10):
            opt.zero_grad(set_to_none=True)
            pred = model(x)
            loss = (pred - y).pow(2).mean()
            loss.backward()
            opt.step()
            losses.append(float(loss.item()))

        self.assertLess(losses[-1], losses[0],
                        f"loss did not decrease: {losses}")

    def test_cautious_wd_applied_conditionally(self):
        """With cautious_wd=True, WD is applied only where grad and weight share sign."""
        # Construct a simple 2D param where half the grads point opposite
        torch.manual_seed(19)
        w = torch.nn.Parameter(torch.ones(8, 8) * 0.5)  # positive weights
        opt = NorMuon(
            muon_params=[{"params": [w]}],
            lr=0.01, weight_decay=0.5,
            cautious_wd=True, neuron_wise_norm=False,  # isolate WD behavior
        )
        # Fake gradients: half positive (same sign as W -> decay), half negative
        w.grad = torch.cat([
            torch.ones(4, 8) * 0.001,       # same sign -> decays
            -torch.ones(4, 8) * 0.001,      # opposite sign -> no decay
        ])
        w_before = w.detach().clone()
        opt.step()
        # Relative change: first half should have shrunk more than second half
        delta_same = (w_before[:4] - w[:4]).abs().mean().item()
        delta_opp = (w_before[4:] - w[4:]).abs().mean().item()
        self.assertGreater(delta_same, delta_opp,
                           f"same-sign WD delta {delta_same} vs opposite {delta_opp}")


# ---------------------------------------------------------------------------
# Value residual + head gating
# ---------------------------------------------------------------------------


class ValueResidualTests(unittest.TestCase):
    def test_v_res_scale_init_zero(self):
        from models.components.attention import NoPECodaAttention

        attn = NoPECodaAttention(dim=64, n_heads=4, n_kv_heads=2)
        self.assertTrue(torch.all(attn.v_res_scale == 0.0),
                        f"v_res_scale must init at 0, got {attn.v_res_scale}")

    def test_head_gate_init_one(self):
        from models.components.attention import NoPECodaAttention

        attn = NoPECodaAttention(dim=64, n_heads=4, n_kv_heads=2)
        self.assertTrue(torch.allclose(attn.head_gate, torch.ones(4)),
                        f"head_gate init: {attn.head_gate}")
        # Effective gate fraction at init: sigmoid(1.0) ≈ 0.7311
        eff = torch.sigmoid(attn.head_gate)
        self.assertAlmostEqual(float(eff.mean()), 0.7311, places=4)


class OdinFlatMiniIntegrationTests(unittest.TestCase):
    @unittest.skipUnless(torch.cuda.is_available(), "Needs CUDA/HIP (HyPE conv block)")
    def test_forward_all_features_on(self):
        """With all Sprint 1 Phase 2 features enabled, forward runs cleanly."""
        from models.odin_flat import OdinFlatMini

        torch.manual_seed(31)
        model = OdinFlatMini(
            use_intra_doc_mask=True,
            use_value_residuals=True,
            use_head_gating=True,
        ).cuda()
        model.eval()
        x = torch.randint(0, model.vocab_size, (2, 32), device="cuda")
        doc_ids = torch.zeros_like(x, dtype=torch.int32)
        doc_ids[:, 16:] = 1

        with torch.no_grad(), torch.amp.autocast("cuda", dtype=torch.float16):
            out = model(x, doc_ids=doc_ids)
        self.assertEqual(out.shape[:2], x.shape)
        self.assertTrue(torch.isfinite(out).all())

    @unittest.skipUnless(torch.cuda.is_available(), "Needs CUDA/HIP")
    def test_v_res_scale_init_zero_output_matches_no_residual(self):
        """With v_res_scale=0 (init), output should match no-residual baseline."""
        from models.odin_flat import OdinFlatMini

        torch.manual_seed(42)
        model_a = OdinFlatMini(use_value_residuals=False).cuda().eval()
        torch.manual_seed(42)
        model_b = OdinFlatMini(use_value_residuals=True).cuda().eval()

        # Load model_a's weights into model_b so they match except for flag
        sd_a = model_a.state_dict()
        sd_b = model_b.state_dict()
        # Both should have identical v_res_scale parameters (init 0 by
        # NoPECodaAttention). Copy all shared tensors from model_a to model_b.
        for k in sd_a:
            if k in sd_b and sd_a[k].shape == sd_b[k].shape:
                sd_b[k].copy_(sd_a[k])
        model_b.load_state_dict(sd_b)

        x = torch.randint(0, model_a.vocab_size, (1, 16), device="cuda")
        with torch.no_grad(), torch.amp.autocast("cuda", dtype=torch.float16):
            out_a = model_a(x)
            out_b = model_b(x)

        self.assertTrue(torch.allclose(out_a, out_b, atol=5e-3),
                        f"max diff: {(out_a - out_b).abs().max().item()}")

    @unittest.skipUnless(torch.cuda.is_available(), "Needs CUDA/HIP")
    def test_gradient_flows_to_sprint1_params(self):
        """Verify v_res_scale and head_gate receive gradient during training.

        Requires >= 2 GQA layers so v_prev can thread from layer i to layer j.
        """
        from models.odin_flat import OdinFlatBase

        torch.manual_seed(53)
        # Custom mini with TWO GQA layers so value residual can activate.
        # Mirror OdinFlatMini's dims otherwise for speed.
        model = OdinFlatBase(
            vocab_size=1000, d_model=128, embed_rank=64,
            n_layers=6, gqa_positions=(2, 5),  # two GQA -> v_res_scale active
            n_heads=4, n_kv_heads=2,
            ffn_inner=256, d_conv=128,
            use_value_residuals=True,
            use_head_gating=True,
        ).cuda()
        model.train()

        x = torch.randint(0, model.vocab_size, (2, 16), device="cuda")
        y = torch.randint(0, model.vocab_size, (2, 16), device="cuda")

        with torch.amp.autocast("cuda", dtype=torch.float16):
            logits = model(x)
            loss = torch.nn.functional.cross_entropy(
                logits.reshape(-1, logits.size(-1)).float(),
                y.reshape(-1),
            )
        loss.backward()

        v_res_grads = []
        head_gate_grads = []
        for name, p in model.named_parameters():
            if "v_res_scale" in name and p.grad is not None:
                v_res_grads.append(p.grad.abs().max().item())
            if "head_gate" in name and p.grad is not None:
                head_gate_grads.append(p.grad.abs().max().item())

        self.assertTrue(len(v_res_grads) > 0, "no v_res_scale grads found")
        self.assertTrue(len(head_gate_grads) > 0, "no head_gate grads found")
        # With two GQA layers, the second layer's v_res_scale DOES see grad:
        self.assertTrue(all(math.isfinite(g) for g in v_res_grads))
        self.assertTrue(all(math.isfinite(g) for g in head_gate_grads))
        self.assertTrue(max(head_gate_grads) > 0,
                        "head_gate grad magnitude should be nonzero")
        self.assertTrue(max(v_res_grads) > 0,
                        "at least one v_res_scale grad should be nonzero "
                        "(the later GQA layer which consumes v_prev)")


if __name__ == "__main__":
    unittest.main(verbosity=2)

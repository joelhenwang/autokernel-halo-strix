"""Track 2: QKV-fusion tests.

4 tests per plan §4.2.4:
  1. test_forward_numerical_equivalence: old-style wq/wk/wv vs fused wqkv
     produce bitwise-equal outputs when fed the same inputs and weights.
  2. test_state_dict_migration: loading a pre-fusion checkpoint into the
     new fused class produces the correct wqkv.weight.
  3. test_wqkv_shape_conventions: for OdinFlat's 12/4/64 config, wqkv has
     shape (768+256+256, 768) = (1280, 768).
  4. test_split_reassembles: after wqkv(x).split(...), reassembling via
     torch.cat yields the original wqkv output exactly.

Plan: docs/superpowers/plans/2026-05-10-odinflat-throughput-investigation-plan.md
"""

from __future__ import annotations

import os
import sys

import pytest
import torch
import torch.nn as nn


REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from models.components.attention import Attention, CodaAttention, NoPECodaAttention


def _make_preFusion_wq_wk_wv(dim, n_heads, n_kv_heads):
    """Replicate the old pre-Track-2 Linear triple. Used by the migration
    and numerical-equivalence tests."""
    head_dim = dim // n_heads
    return (
        nn.Linear(dim, n_heads * head_dim, bias=False),
        nn.Linear(dim, n_kv_heads * head_dim, bias=False),
        nn.Linear(dim, n_kv_heads * head_dim, bias=False),
    )


def test_wqkv_shape_conventions():
    """OdinFlat's config: d=768, n_heads=12, n_kv_heads=4, head_dim=64.
    Expected wqkv.weight shape = (q_dim + 2*kv_dim, dim) = (1280, 768)."""
    attn = Attention(dim=768, n_heads=12, n_kv_heads=4, qk_norm=True)
    assert attn.wqkv.weight.shape == (1280, 768), (
        f"Expected (1280, 768), got {tuple(attn.wqkv.weight.shape)}"
    )
    # Also verify subclasses inherit the same layout.
    coda = CodaAttention(dim=768, n_heads=12, n_kv_heads=4, qk_norm=True)
    assert coda.wqkv.weight.shape == (1280, 768)
    nope = NoPECodaAttention(dim=768, n_heads=12, n_kv_heads=4)
    assert nope.wqkv.weight.shape == (1280, 768)


def test_split_reassembles():
    """After forward split, cat recovers the exact wqkv output."""
    torch.manual_seed(0)
    attn = Attention(dim=768, n_heads=12, n_kv_heads=4, qk_norm=False)
    x = torch.randn(2, 16, 768)
    qkv = attn.wqkv(x)
    q, k, v = qkv.split([attn._q_dim, attn._kv_dim, attn._kv_dim], dim=-1)
    qkv_reassembled = torch.cat([q, k, v], dim=-1)
    assert torch.equal(qkv, qkv_reassembled)
    # Also check shapes are right for the expected view() calls.
    B, T, _ = x.shape
    q_view = q.view(B, T, attn.n_heads, attn.head_dim)
    k_view = k.view(B, T, attn.n_kv_heads, attn.head_dim)
    v_view = v.view(B, T, attn.n_kv_heads, attn.head_dim)
    assert q_view.shape == (B, T, 12, 64)
    assert k_view.shape == (B, T, 4, 64)
    assert v_view.shape == (B, T, 4, 64)


def test_state_dict_migration():
    """Pre-fusion checkpoint keys (wq/wk/wv.weight) load correctly into
    the new fused class, producing the expected wqkv.weight."""
    torch.manual_seed(1)
    dim, n_heads, n_kv_heads = 768, 12, 4

    # Create a pre-fusion state dict.
    wq, wk, wv = _make_preFusion_wq_wk_wv(dim, n_heads, n_kv_heads)
    wo = nn.Linear(n_heads * (dim // n_heads), dim, bias=False)
    old_state = {
        "wq.weight": wq.weight.detach().clone(),
        "wk.weight": wk.weight.detach().clone(),
        "wv.weight": wv.weight.detach().clone(),
        "wo.weight": wo.weight.detach().clone(),
    }
    # Add QK-Norm params so strict=True still works.
    import math
    head_dim = dim // n_heads
    old_state["q_scale"] = torch.ones(n_heads, 1, 1) * math.sqrt(head_dim)
    old_state["k_scale"] = torch.ones(n_kv_heads, 1, 1) * math.sqrt(head_dim)

    # Load into fused model.
    fused = Attention(dim=dim, n_heads=n_heads, n_kv_heads=n_kv_heads, qk_norm=True)
    missing, unexpected = fused.load_state_dict(old_state, strict=False)
    # The migration hook should have consumed wq/wk/wv and produced wqkv.
    assert "wqkv.weight" not in missing, (
        f"wqkv.weight marked as missing (migration hook didn't fire): "
        f"missing={missing}, unexpected={unexpected}"
    )
    # Verify the fused tensor is the cat of the three.
    expected = torch.cat([old_state_b for old_state_b in [
        wq.weight.detach(), wk.weight.detach(), wv.weight.detach(),
    ]], dim=0)
    assert torch.equal(fused.wqkv.weight.detach(), expected), (
        "wqkv.weight does not match cat([wq, wk, wv], dim=0) after load"
    )


def test_forward_numerical_equivalence():
    """Compute forward with old-style split Linears and new fused layout;
    assert bit-identical outputs (same inputs, same weights, no RoPE)."""
    torch.manual_seed(2)
    dim, n_heads, n_kv_heads = 768, 12, 4
    head_dim = dim // n_heads
    B, T = 2, 16

    # Construct fused attention and extract the three weight slices that
    # would have been wq/wk/wv in the old layout. NB: wqkv.weight layout is
    # [q_dim, kv_dim, kv_dim] along dim 0 (the output-feature axis).
    fused = Attention(dim=dim, n_heads=n_heads, n_kv_heads=n_kv_heads, qk_norm=False)
    q_w, k_w, v_w = fused.wqkv.weight.split(
        [fused._q_dim, fused._kv_dim, fused._kv_dim], dim=0)
    assert q_w.shape == (n_heads * head_dim, dim)
    assert k_w.shape == (n_kv_heads * head_dim, dim)
    assert v_w.shape == (n_kv_heads * head_dim, dim)

    # Run the fused split path on a random input.
    x = torch.randn(B, T, dim)
    q_f, k_f, v_f = fused._split_qkv(x, B, T)

    # Run the old-style path manually with the same slices.
    q_old = torch.nn.functional.linear(x, q_w).view(B, T, n_heads, head_dim)
    k_old = torch.nn.functional.linear(x, k_w).view(B, T, n_kv_heads, head_dim)
    v_old = torch.nn.functional.linear(x, v_w).view(B, T, n_kv_heads, head_dim)

    # Assert bit-identity (no fp arithmetic difference; same GEMM inputs).
    assert torch.equal(q_f, q_old), (
        f"Q mismatch: max_abs_diff={(q_f - q_old).abs().max().item():.3e}"
    )
    assert torch.equal(k_f, k_old), (
        f"K mismatch: max_abs_diff={(k_f - k_old).abs().max().item():.3e}"
    )
    assert torch.equal(v_f, v_old), (
        f"V mismatch: max_abs_diff={(v_f - v_old).abs().max().item():.3e}"
    )


def test_forward_nope_coda_bypasses_autokernel():
    """NoPECodaAttention retains `_skip_autokernel = True` post-fusion.
    Ensures autokernel's FusedQKVPattern will not try to touch it even
    though wqkv pattern looks novel to the matcher."""
    attn = NoPECodaAttention(dim=768, n_heads=12, n_kv_heads=4)
    assert getattr(attn, "_skip_autokernel", False) is True

    # Also check the class level (not just instance level).
    assert getattr(NoPECodaAttention, "_skip_autokernel", False) is True


def test_forward_with_qk_norm_and_softcap():
    """Smoke test: Attention forward runs clean with qk_norm=True and
    softcap=50 — exercising the two non-default code paths in forward."""
    torch.manual_seed(3)
    attn = Attention(dim=768, n_heads=12, n_kv_heads=4, qk_norm=True,
                     attn_score_softcap=50.0)
    x = torch.randn(2, 16, 768)
    # RoPE cis table for T=16, head_dim=64 (half=32 freqs).
    import math
    freqs = torch.randn(16, 32, dtype=torch.complex64)
    out = attn(x, freqs)
    assert out.shape == (2, 16, 768)
    assert torch.isfinite(out).all()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

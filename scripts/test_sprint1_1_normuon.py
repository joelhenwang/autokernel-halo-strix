"""Sprint 1.1 Phase B: NorMuon + optimizer + CLI knob unit tests.

Covers:
- Task B.0: --ns-dtype / --neuron-norm-min-dim / --no-cautious-wd
  registered in scripts/train_ddp.py's argparser.
- Task B.1: fp16 NS executes without NaN on a toy MLP.
- Task B.2: neuron-wise normalization is gated by neuron_norm_min_dim.
- Task B.3: cautious_wd=False applies standard decoupled WD (no mask).
- build_imu1_optimizer accepts the new kwargs and forwards to NorMuon.

All tests use synthetic tiny models so they complete in <10s on CPU.
Machine B can run these in parallel while Machine A runs DDP ablations.
"""

from __future__ import annotations

import os
import subprocess
import sys

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, REPO_ROOT)

import torch
import torch.nn as nn

from halo_training.normuon import NorMuon, _neuron_wise_normalize
from halo_training.optimizer import build_imu1_optimizer


# ---------------------------------------------------------------------------
# Task B.0 — CLI flag registration
# ---------------------------------------------------------------------------

def _run_help(script_name: str) -> str:
    script_path = os.path.join(REPO_ROOT, "scripts", script_name)
    result = subprocess.run(
        [sys.executable, script_path, "--help"],
        capture_output=True, text=True, cwd=REPO_ROOT, timeout=30,
    )
    assert result.returncode == 0, (
        f"{script_name} --help failed:\n{result.stderr}"
    )
    return result.stdout


def test_new_flags_registered_in_train_ddp():
    help_text = _run_help("train_ddp.py")
    required = ["--ns-dtype", "--neuron-norm-min-dim", "--cautious-wd",
                "--no-cautious-wd"]
    missing = [f for f in required if f not in help_text]
    assert not missing, f"train_ddp.py missing flags: {missing}"
    print(f"  OK: all {len(required)} Phase B flags in train_ddp.py --help")


def test_new_flags_registered_in_profile_step():
    help_text = _run_help("profile_step.py")
    required = ["--ns-dtype", "--neuron-norm-min-dim", "--cautious-wd",
                "--no-cautious-wd"]
    missing = [f for f in required if f not in help_text]
    assert not missing, f"profile_step.py missing flags: {missing}"
    print(f"  OK: all {len(required)} Phase B flags in profile_step.py --help")


# ---------------------------------------------------------------------------
# Task B.1 — fp16 Newton-Schulz path executes
# ---------------------------------------------------------------------------

def test_ns_fp16_executes():
    """NorMuon with ns_dtype=torch.float16 runs one step without NaN/Inf."""
    torch.manual_seed(0)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = nn.Sequential(nn.Linear(128, 256), nn.ReLU(), nn.Linear(256, 64)).to(device)

    opt = NorMuon(
        muon_params=[{"params": [p for n, p in model.named_parameters()
                                 if p.ndim == 2]}],
        adamw_params=[{"params": [p for n, p in model.named_parameters()
                                  if p.ndim < 2],
                       "lr": 1e-4, "weight_decay": 0.0}],
        lr=1e-3, weight_decay=0.1,
        ns_dtype=torch.float16,
    )

    x = torch.randn(4, 128, device=device)
    loss = model(x).sum()
    loss.backward()
    opt.step()

    for n, p in model.named_parameters():
        assert torch.isfinite(p.data).all(), (
            f"fp16 NS produced non-finite param {n}: "
            f"has_nan={torch.isnan(p.data).any().item()} "
            f"has_inf={torch.isinf(p.data).any().item()}"
        )
    print("  OK: fp16 NS single step leaves all params finite")


# ---------------------------------------------------------------------------
# Task B.2 — neuron-wise normalization is size-gated
# ---------------------------------------------------------------------------

def test_neuron_norm_gated_skip():
    """neuron_norm_min_dim=128 skips normalization on a 64x64 2D param."""
    torch.manual_seed(1)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Two 2D params: small (64x64, min_dim=64<128 → skip) and large (512x512,
    # min_dim=512>=128 → apply).
    small = nn.Parameter(torch.randn(64, 64, device=device))
    large = nn.Parameter(torch.randn(512, 512, device=device))
    small.grad = torch.randn_like(small)
    large.grad = torch.randn_like(large)

    # Capture per-param pre-step data for delta computation
    s_before = small.data.clone()
    l_before = large.data.clone()

    opt = NorMuon(
        muon_params=[{"params": [small, large]}],
        lr=1e-2, weight_decay=0.0,  # no WD -> update direction is -lr * m_orth * scale
        momentum=0.0, nesterov=False,  # no momentum -> m = grad
        neuron_wise_norm=True,
        neuron_norm_min_dim=128,
    )
    opt.step()

    # With no momentum / no WD, update is -lr * scale * (maybe-normalized-m_orth).
    # For the *gated* path (small): m_orth rows have non-uniform norms.
    # For the *applied* path (large): m_orth rows have been normalized to 1
    # BEFORE being scaled by scale=sqrt(max_dim)*0.2.
    # We can check this by verifying row-norm statistics of the update delta.

    s_delta = (s_before - small.data) / 1e-2  # = scale * m_orth (for small, unnormalized)
    l_delta = (l_before - large.data) / 1e-2  # = scale * m_orth (for large, row-normalized)

    # Remove the fixed scale multiplier to inspect the raw m_orth rows
    s_scale = max(64, 64) ** 0.5 * 0.2
    l_scale = max(512, 512) ** 0.5 * 0.2

    s_rows = s_delta / s_scale   # raw m_orth (unnormalized)
    l_rows = l_delta / l_scale   # normalized m_orth

    s_row_norms = s_rows.norm(dim=1)
    l_row_norms = l_rows.norm(dim=1)

    # GATED (small): rows NOT normalized, so stddev of row norms > 0
    assert s_row_norms.std().item() > 1e-3, (
        f"small param was normalized but shouldn't have been "
        f"(row-norm std={s_row_norms.std().item()})"
    )
    # APPLIED (large): rows WERE normalized to ~1
    assert (l_row_norms - 1.0).abs().max().item() < 1e-3, (
        f"large param was NOT normalized; "
        f"row norms min={l_row_norms.min().item()} "
        f"max={l_row_norms.max().item()}"
    )
    print(f"  OK: neuron_norm_min_dim=128 gate: "
          f"small(64x64) skipped (row-norm std={s_row_norms.std().item():.3f}), "
          f"large(512x512) normalized (max |norm-1|={(l_row_norms - 1.0).abs().max().item():.2e})")


def test_neuron_norm_always_when_threshold_zero():
    """neuron_norm_min_dim=0 (default) applies normalization to all sizes."""
    torch.manual_seed(2)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    small = nn.Parameter(torch.randn(64, 64, device=device))
    small.grad = torch.randn_like(small)
    s_before = small.data.clone()

    opt = NorMuon(
        muon_params=[{"params": [small]}],
        lr=1e-2, weight_decay=0.0, momentum=0.0, nesterov=False,
        neuron_wise_norm=True, neuron_norm_min_dim=0,
    )
    opt.step()

    s_delta = (s_before - small.data) / 1e-2
    s_scale = max(64, 64) ** 0.5 * 0.2
    s_rows = s_delta / s_scale
    row_norms = s_rows.norm(dim=1)
    # With default gate (0), small param IS normalized
    assert (row_norms - 1.0).abs().max().item() < 1e-3, (
        f"neuron_norm_min_dim=0 should normalize all sizes; "
        f"max |norm-1|={(row_norms - 1.0).abs().max().item()}"
    )
    print("  OK: neuron_norm_min_dim=0 normalizes 64x64 param (default behavior)")


# ---------------------------------------------------------------------------
# Task B.3 — cautious_wd=False disables the mask
# ---------------------------------------------------------------------------

def test_cautious_wd_flag_disables():
    """cautious_wd=False applies unconditional decoupled WD."""
    torch.manual_seed(3)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Set up a 2D param where some entries have sign(grad)*sign(weight) > 0
    # (those WOULD get WD'd under cautious mode) and others have < 0 (those
    # would be SKIPPED under cautious mode).
    p1 = nn.Parameter(torch.tensor([[1.0, -1.0], [1.0, -1.0]], device=device))
    p1.grad = torch.tensor([[1.0, 1.0], [-1.0, -1.0]], device=device)
    # Cautious WD mask would be:
    #   sign(grad)*sign(weight) = sign([[1,-1],[-1,1]]) > 0 ? = [[T,F],[F,T]]
    # So entries (0,0) and (1,1) would get WD applied; (0,1) and (1,0) skipped.

    # With cautious_wd=False, ALL entries get WD applied uniformly.
    # Using a very large wd + small lr * big wd to make the WD effect dominant
    # and measurable vs the NS update.
    p_before = p1.data.clone()

    opt = NorMuon(
        muon_params=[{"params": [p1]}],
        lr=1e-3, weight_decay=0.5, momentum=0.0, nesterov=False,
        neuron_wise_norm=False,  # isolate WD effect
        cautious_wd=False,
    )
    opt.step()

    # Without cautious: p.data <- p.data * (1 - lr * wd) before NS update
    # So every entry gets multiplied by (1 - 1e-3 * 0.5) = 0.9995.
    # All entries should have the same multiplicative decay RATIO from WD.
    # The NS update adds a perturbation on top, but the WD ratio is uniform.
    # Hardest assertion: the ratio of decayed magnitude is identical across
    # positions where |p_before| is equal. For positions (0,0) and (1,0) both
    # have |p_before|=1; the WD-pre-update component is the same 0.9995.
    # The NS component differs, but we can check the *decay direction*:
    # In strict cautious mode, positions (0,1) and (1,0) would NOT be scaled
    # by (1-lr*wd). In non-cautious mode, ALL are scaled.

    # Simplest robust check: after zero-momentum no-neuron-norm step, the WD
    # pre-multiplier differs between the two modes. Run both and compare.
    p2 = nn.Parameter(torch.tensor([[1.0, -1.0], [1.0, -1.0]], device=device))
    p2.grad = torch.tensor([[1.0, 1.0], [-1.0, -1.0]], device=device)
    opt2 = NorMuon(
        muon_params=[{"params": [p2]}],
        lr=1e-3, weight_decay=0.5, momentum=0.0, nesterov=False,
        neuron_wise_norm=False,
        cautious_wd=True,
    )
    opt2.step()

    # Difference between cautious and non-cautious is the WD applied to
    # the "off-mask" entries (0,1) and (1,0). In non-cautious, they were
    # scaled by (1-5e-4) = 0.9995 before NS update. In cautious, they were
    # NOT. The NS contribution to these entries should be identical for
    # both (same grad, same m_orth from seed). So the delta between the two
    # final values on those entries equals the WD contribution.
    delta_off_mask_01 = abs(p1.data[0, 1].item() - p2.data[0, 1].item())
    delta_off_mask_10 = abs(p1.data[1, 0].item() - p2.data[1, 0].item())
    # At (0,0) and (1,1) both modes apply WD, so difference is ~zero.
    delta_on_mask_00 = abs(p1.data[0, 0].item() - p2.data[0, 0].item())
    delta_on_mask_11 = abs(p1.data[1, 1].item() - p2.data[1, 1].item())

    assert delta_off_mask_01 > 1e-5, (
        f"cautious=False vs True should differ on off-mask (0,1); "
        f"got delta={delta_off_mask_01}"
    )
    assert delta_off_mask_10 > 1e-5, (
        f"cautious=False vs True should differ on off-mask (1,0); "
        f"got delta={delta_off_mask_10}"
    )
    assert delta_on_mask_00 < 1e-6, (
        f"cautious=False vs True should agree on on-mask (0,0); "
        f"got delta={delta_on_mask_00}"
    )
    assert delta_on_mask_11 < 1e-6, (
        f"cautious=False vs True should agree on on-mask (1,1); "
        f"got delta={delta_on_mask_11}"
    )
    print(f"  OK: cautious_wd=False disables mask "
          f"(off-mask delta [0,1]={delta_off_mask_01:.2e} [1,0]={delta_off_mask_10:.2e}; "
          f"on-mask delta {delta_on_mask_00:.2e}, {delta_on_mask_11:.2e})")


# ---------------------------------------------------------------------------
# build_imu1_optimizer plumbing
# ---------------------------------------------------------------------------

def test_build_imu1_forwards_new_kwargs():
    """build_imu1_optimizer(use_normuon=True, ns_dtype=..., ...) plumbs through."""
    torch.manual_seed(4)
    # Tiny model so the optimizer has a 2D group to route
    m = nn.Sequential(nn.Linear(32, 64), nn.Linear(64, 16))

    opt = build_imu1_optimizer(
        m,
        lr_2d=1e-2, lr_1d=1e-3, weight_decay_2d=0.1,
        use_normuon=True,
        ns_dtype=torch.float16,
        neuron_norm_min_dim=128,
        cautious_wd=False,
    )
    from halo_training.normuon import NorMuon
    assert isinstance(opt, NorMuon)
    assert opt.ns_dtype == torch.float16
    assert opt.neuron_norm_min_dim == 128
    assert opt.cautious_wd is False
    print(f"  OK: build_imu1_optimizer forwards ns_dtype=fp16, "
          f"neuron_norm_min_dim=128, cautious_wd=False")


def test_build_imu1_defaults_stable():
    """build_imu1_optimizer(use_normuon=True) default kwargs match Phase 2 behavior."""
    m = nn.Sequential(nn.Linear(32, 64), nn.Linear(64, 16))
    opt = build_imu1_optimizer(m, use_normuon=True)
    from halo_training.normuon import NorMuon
    assert isinstance(opt, NorMuon)
    # Phase 2 defaults: fp32, always-apply neuron-norm, cautious WD
    assert opt.ns_dtype is None or opt.ns_dtype == torch.float32, opt.ns_dtype
    assert opt.neuron_norm_min_dim == 0, opt.neuron_norm_min_dim
    assert opt.cautious_wd is True
    assert opt.neuron_wise_norm is True
    print("  OK: build_imu1_optimizer use_normuon=True defaults "
          "match Phase 2 behavior (fp32, always-NN, cautious_wd=True)")


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------

def main():
    tests = [
        test_new_flags_registered_in_train_ddp,
        test_new_flags_registered_in_profile_step,
        test_ns_fp16_executes,
        test_neuron_norm_gated_skip,
        test_neuron_norm_always_when_threshold_zero,
        test_cautious_wd_flag_disables,
        test_build_imu1_forwards_new_kwargs,
        test_build_imu1_defaults_stable,
    ]
    for t in tests:
        print(f"[TEST] {t.__name__}")
        t()
    print(f"\nAll {len(tests)} tests passed")


if __name__ == "__main__":
    main()

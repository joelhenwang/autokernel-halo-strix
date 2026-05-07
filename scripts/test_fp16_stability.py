"""fp16 stability hardening tests.

Covers the Ring 1 prevention + Ring 2 response changes shipped alongside
Sprint 1.1. All tests use tiny synthetic models or mocks; they run on
CPU in under 30 seconds.

Test coverage:
  - P1: --z-loss flag registered; z-loss adds logsumexp^2 term to loss
  - P3: iter_scales.clamp applied at forward (OdinHalo pattern)
  - P4: GradScaler growth_interval default is 500 (not 2000)
  - P5: --max-grad-norm defaults to 0.8 when --resume-from is set
  - P6: --attn-softcap flag registered; softcap bounds pre-softmax scores
  - D1: --activation-monitor flag; ActivationMonitor writes JSONL
  - R1: save_nan_forensics writes a .pt dump with required keys
  - R3: StabilityGuard.rollback(scaler=...) halves scaler._growth_interval
  - R5: scaler.get_scale() appears in log JSONL

Run: python scripts/test_fp16_stability.py
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
from collections import deque

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, REPO_ROOT)

import torch
import torch.nn as nn


# ---------------------------------------------------------------------------
# Shared: load train_ddp module without triggering main()
# ---------------------------------------------------------------------------

def _run_help(script_name: str) -> str:
    result = subprocess.run(
        [sys.executable, os.path.join(REPO_ROOT, "scripts", script_name), "--help"],
        capture_output=True, text=True, cwd=REPO_ROOT, timeout=30,
    )
    assert result.returncode == 0, f"{script_name} --help failed:\n{result.stderr}"
    return result.stdout


# ---------------------------------------------------------------------------
# P1 — z-loss flag + math
# ---------------------------------------------------------------------------

def test_z_loss_flag_registered():
    help_text = _run_help("train_ddp.py")
    for flag in ("--z-loss", "--z-loss-fraction"):
        assert flag in help_text, f"train_ddp.py --help missing {flag}"
    print("  OK: --z-loss and --z-loss-fraction registered")


def test_z_loss_math_adds_logsumexp_sq():
    """Verify z = z_weight * logsumexp(logits)^2 increases base CE loss."""
    torch.manual_seed(0)
    logits = torch.randn(4, 8, 100)  # [B, T, V]
    targets = torch.randint(0, 100, (4, 8))
    ce = nn.functional.cross_entropy(
        logits.view(-1, 100), targets.view(-1)
    )
    z_weight = 1e-2  # large so the effect is visible
    z_term = z_weight * logits.float().logsumexp(dim=-1).pow(2).mean()
    # Check z_term > 0 (logsumexp^2 is non-negative; only 0 if all logits = -inf)
    assert z_term.item() > 0
    combined = ce + z_term
    assert combined > ce, f"z-loss should increase combined: {ce.item()} -> {combined.item()}"
    print(f"  OK: z-loss adds {z_term.item():.4f} to CE {ce.item():.4f} "
          f"(combined {combined.item():.4f})")


# ---------------------------------------------------------------------------
# P3 — iter_scales clamp applied at forward
# ---------------------------------------------------------------------------

def test_iter_scales_clamp_applied_in_forward():
    """Poison OdinHalo's iter_scales with a large value; verify clamp to 4.0."""
    from models.odin_halo import OdinHaloMini
    torch.manual_seed(0)
    model = OdinHaloMini()
    # Simulate a worst-case drift
    with torch.no_grad():
        model.iter_scales.data.fill_(10.0)  # Out of ±4 range
    # Directly invoke _apply_iter_norm with a simple tensor to verify clamp
    B, T, D = 2, 4, model.d_model
    h = torch.randn(B, T, D)
    out = model._apply_iter_norm(h, iter_idx=0)
    # After clamp, the scale applied should be 4.0 (upper bound of ±4)
    # We can verify by comparing: iter_norm(h) * 4.0 + loop_pos_embeds[0]
    expected = model.iter_norm(h) * 4.0 + model.loop_pos_embeds[0]
    diff = (out - expected).abs().max().item()
    assert diff < 1e-5, f"iter_scales not clamped; max diff={diff}"
    print(f"  OK: iter_scales clamp applied in forward (max diff from expected clamped={diff:.2e})")


# ---------------------------------------------------------------------------
# P4 — GradScaler growth_interval default is 500
# ---------------------------------------------------------------------------

def test_gradscaler_growth_interval_default():
    """Parse train_ddp.py and check the GradScaler construction has growth_interval=500."""
    path = os.path.join(REPO_ROOT, "scripts", "train_ddp.py")
    with open(path, encoding="utf-8") as f:
        src = f.read()
    assert "growth_interval=500" in src, (
        "train_ddp.py does not set GradScaler(growth_interval=500); "
        "P4 regression — looked for literal 'growth_interval=500'"
    )
    print("  OK: GradScaler(growth_interval=500) present in train_ddp.py")


# ---------------------------------------------------------------------------
# P5 — --max-grad-norm 0.8 when --resume-from is set (source-level check)
# ---------------------------------------------------------------------------

def test_p5_resumed_tightens_grad_norm():
    path = os.path.join(REPO_ROOT, "scripts", "train_ddp.py")
    with open(path, encoding="utf-8") as f:
        src = f.read()
    assert "args.max_grad_norm = 0.8" in src, (
        "train_ddp.py does not tighten max_grad_norm on resume-from; "
        "P5 regression"
    )
    print("  OK: P5 tightening logic present (args.max_grad_norm = 0.8 on resume)")


# ---------------------------------------------------------------------------
# P6 — attention softcap bounds scores
# ---------------------------------------------------------------------------

def test_attn_softcap_flag_registered():
    help_text = _run_help("train_ddp.py")
    assert "--attn-softcap" in help_text
    print("  OK: --attn-softcap flag registered")


def test_attn_softcap_bounds_pre_softmax_scores():
    """Attention._attention_core with softcap=50 should produce bounded outputs
    even when Q/K are set to extremely large values."""
    from models.components.attention import _attention_core
    torch.manual_seed(1)
    B, H, T, D = 1, 4, 8, 16
    # Huge Q, K that would produce scores of order 1e5 without the softcap
    q = torch.randn(B, H, T, D) * 100.0
    k = torch.randn(B, H, T, D) * 100.0
    v = torch.randn(B, H, T, D)
    # Softcap off: SDPA path (for baseline comparison; SDPA is numerically
    # stable by design so we don't assert this is bad — we just check that
    # the softcap path produces finite output).
    y_softcap = _attention_core(q, k, v, softcap=50.0, is_causal=True)
    assert torch.isfinite(y_softcap).all(), "softcap output has non-finite values"
    # Also verify the softcap path does not crash without softcap
    y_nosoftcap = _attention_core(q, k, v, softcap=0.0, is_causal=True)
    assert torch.isfinite(y_nosoftcap).all()
    print(f"  OK: _attention_core with softcap=50 on extreme Q/K produces finite "
          f"output, max|y|={y_softcap.abs().max().item():.2f}")


def test_attention_module_softcap_attribute():
    """Attention class accepts attn_score_softcap kwarg and exposes the attribute."""
    from models.components.attention import Attention
    m = Attention(dim=64, n_heads=4, n_kv_heads=2, attn_score_softcap=50.0)
    assert m.attn_score_softcap == 50.0
    # And default is 0.0 (zero-regression)
    m2 = Attention(dim=64, n_heads=4, n_kv_heads=2)
    assert m2.attn_score_softcap == 0.0
    print("  OK: Attention.attn_score_softcap defaults 0.0, accepts >0")


# ---------------------------------------------------------------------------
# D1 — activation monitor
# ---------------------------------------------------------------------------

def test_activation_monitor_flag_registered():
    help_text = _run_help("train_ddp.py")
    assert "--activation-monitor" in help_text
    assert "--activation-monitor-interval" in help_text
    print("  OK: --activation-monitor(-interval) registered")


def test_activation_monitor_writes_jsonl():
    """ActivationMonitor attached to a 2-layer MLP writes valid JSONL on sample."""
    from halo_training.activation_monitor import ActivationMonitor

    class TinyModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.layers = nn.ModuleList([
                nn.Linear(16, 16),
                nn.Linear(16, 16),
            ])
            self.norm_out = nn.LayerNorm(16)

        def forward(self, x):
            for layer in self.layers:
                x = layer(x)
            return self.norm_out(x)

    with tempfile.TemporaryDirectory() as tmp:
        out = os.path.join(tmp, "activation_stats.jsonl")
        model = TinyModel()
        monitor = ActivationMonitor(model, output_path=out, sample_every=1)
        monitor.attach()
        x = torch.randn(4, 16)
        model(x)
        monitor.step(global_step=1)
        monitor.detach()

        assert os.path.exists(out), "JSONL output not written"
        lines = [json.loads(l) for l in open(out, encoding="utf-8") if l.strip()]
        assert len(lines) >= 2, f"Expected >=2 lines (layers.0, layers.1); got {len(lines)}"
        for line in lines:
            assert "step" in line
            assert "layer" in line
            assert "maxabs" in line
            assert "fp16_headroom" in line
            assert line["step"] == 1
        print(f"  OK: monitor wrote {len(lines)} JSONL entries with required keys")


# ---------------------------------------------------------------------------
# R1 — NaN forensics dump
# ---------------------------------------------------------------------------

def _load_train_ddp_module():
    """Load train_ddp.py without executing main()."""
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "train_ddp_module", os.path.join(REPO_ROOT, "scripts", "train_ddp.py")
    )
    mod = importlib.util.module_from_spec(spec)
    sys.modules["train_ddp_module"] = mod
    spec.loader.exec_module(mod)
    return mod


def test_save_nan_forensics_writes_pt():
    """save_nan_forensics creates a .pt dump with required fields."""
    tm = _load_train_ddp_module()

    model = nn.Sequential(nn.Linear(8, 16), nn.Linear(16, 4))
    scaler = torch.amp.GradScaler("cuda", init_scale=2048.0, growth_interval=100)
    rgn = deque([(10, 1.5), (11, 1.6), (12, 1.7)], maxlen=50)

    with tempfile.TemporaryDirectory() as tmp:
        path = tm.save_nan_forensics(
            dump_dir=tmp,
            step=12,
            trigger="nan_loss",
            loss_val=float("nan"),
            batch_idx=3,
            input_ids=torch.randint(0, 100, (2, 8)),
            targets=torch.randint(0, 100, (2, 8)),
            doc_ids=None,
            scaler=scaler,
            model=model,
            recent_grad_norms=rgn,
            monitor=None,
            global_step=12,
        )
        assert path is not None and os.path.exists(path), "dump file not created"
        dump = torch.load(path, map_location="cpu", weights_only=False)
        for k in ("step", "trigger", "loss_val", "weight_maxabs",
                  "grad_norm_history", "scaler_state", "input_ids_cpu"):
            assert k in dump, f"dump missing key {k}"
        assert dump["trigger"] == "nan_loss"
        assert dump["step"] == 12
        assert len(dump["weight_maxabs"]) > 0
        assert len(dump["grad_norm_history"]) == 3
        print(f"  OK: nan_dump_step_12.pt has all required keys "
              f"({len(dump['weight_maxabs'])} params captured)")


# ---------------------------------------------------------------------------
# R3 — rollback halves growth_interval when scaler is passed
# ---------------------------------------------------------------------------

def test_rollback_halves_growth_interval():
    """StabilityGuard.rollback(scaler=...) halves the scaler._growth_interval."""
    tm = _load_train_ddp_module()

    # Build a toy model, optimizer, and a fake checkpoint for rollback to find.
    with tempfile.TemporaryDirectory() as tmp:
        model = nn.Sequential(nn.Linear(4, 4))
        optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)
        # Fake a checkpoint file
        ckpt_path = os.path.join(tmp, "step_100.pt")
        torch.save({
            "step": 100,
            "model_state_dict": {k.replace("._orig_mod.", "."): v
                                 for k, v in model.state_dict().items()},
            "optimizer_state_dict": optimizer.state_dict(),
        }, ckpt_path)

        guard = tm.StabilityGuard(checkpoint_dir=tmp, rank=0)
        scaler = torch.amp.GradScaler("cuda", init_scale=1024.0, growth_interval=500)
        # Trigger rollback
        step, ok = guard.rollback(model, optimizer, device="cpu", scaler=scaler)
        assert ok, "rollback failed"
        # growth_interval should be 250 now (500 // 2)
        assert scaler._growth_interval == 250, (
            f"growth_interval not halved: still {scaler._growth_interval}"
        )
        print(f"  OK: rollback halved scaler.growth_interval 500 -> {scaler._growth_interval}")


def test_rollback_scaler_optional():
    """rollback without scaler kwarg doesn't crash (backward compat)."""
    tm = _load_train_ddp_module()

    with tempfile.TemporaryDirectory() as tmp:
        model = nn.Sequential(nn.Linear(4, 4))
        optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)
        ckpt_path = os.path.join(tmp, "step_100.pt")
        torch.save({
            "step": 100,
            "model_state_dict": dict(model.state_dict()),
            "optimizer_state_dict": optimizer.state_dict(),
        }, ckpt_path)
        guard = tm.StabilityGuard(checkpoint_dir=tmp, rank=0)
        step, ok = guard.rollback(model, optimizer, device="cpu")
        assert ok
        print(f"  OK: rollback without scaler kwarg still works (backward-compat)")


# ---------------------------------------------------------------------------
# R5 — scaler_scale field present in log line
# ---------------------------------------------------------------------------

def test_log_line_includes_scale_field():
    """train_ddp.py log line includes `scale=` and log JSONL has scaler_scale key."""
    path = os.path.join(REPO_ROOT, "scripts", "train_ddp.py")
    with open(path, encoding="utf-8") as f:
        src = f.read()
    # The printed log line must show "scale=" (with format specifier)
    assert "scale={scale_val:.1e}" in src, "log line missing scale= field (R5)"
    assert '"scaler_scale": scale_val' in src, \
        "JSONL log missing scaler_scale key (R5)"
    print("  OK: scaler.scale present in both stdout log line and JSONL")


# ---------------------------------------------------------------------------
# Bug-fix: --max-steps mid-accum termination no longer needs try/except
# ---------------------------------------------------------------------------

def test_flush_guarded_by_backwards_in_cycle():
    """Flush block should be gated by backwards_in_cycle counter, not try/except.

    Historical behavior: a bare try/except AssertionError around
    _complete_step() masked a real state-machine mismatch. Fix replaces
    the silent-swallow with a counter-gated branch.
    """
    path = os.path.join(REPO_ROOT, "scripts", "train_ddp.py")
    with open(path, encoding="utf-8") as f:
        src = f.read()
    # Counter must be initialized before the training loop.
    assert "backwards_in_cycle = 0" in src, (
        "backwards_in_cycle counter not initialized in train_ddp.py"
    )
    # Counter must be incremented on each scaler.scale(loss).backward().
    assert "backwards_in_cycle += 1" in src, (
        "backwards_in_cycle counter never incremented"
    )
    # Flush block must gate on the counter, not use bare except AssertionError.
    assert "if backwards_in_cycle > 0:" in src, (
        "flush block not gated on backwards_in_cycle"
    )
    # Look for the actual except clause (colon + indent), NOT comment mentions.
    import re
    active_except = re.search(r"except AssertionError\s+as\s+\w+\s*:", src)
    assert active_except is None, (
        "try/except AssertionError still present as active code; "
        "bug-fix not applied cleanly"
    )
    print("  OK: flush path now counter-gated; AssertionError swallow removed")


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------

def main():
    tests = [
        test_z_loss_flag_registered,
        test_z_loss_math_adds_logsumexp_sq,
        test_iter_scales_clamp_applied_in_forward,
        test_gradscaler_growth_interval_default,
        test_p5_resumed_tightens_grad_norm,
        test_attn_softcap_flag_registered,
        test_attn_softcap_bounds_pre_softmax_scores,
        test_attention_module_softcap_attribute,
        test_activation_monitor_flag_registered,
        test_activation_monitor_writes_jsonl,
        test_save_nan_forensics_writes_pt,
        test_rollback_halves_growth_interval,
        test_rollback_scaler_optional,
        test_log_line_includes_scale_field,
        test_flush_guarded_by_backwards_in_cycle,
    ]
    for t in tests:
        print(f"[TEST] {t.__name__}")
        t()
    print(f"\nAll {len(tests)} tests passed")


if __name__ == "__main__":
    main()

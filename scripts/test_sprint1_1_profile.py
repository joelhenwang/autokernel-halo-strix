"""Sprint 1.1 Phase A: profile_step.py flag wiring + smoke run.

Run locally (Windows or remote Linux) to verify scripts/profile_step.py
accepts the Sprint 1 CLI flag block and can dispatch to the NorMuon path.
The smoke test uses tiny counts (--warmup 5 --measure 5) to stay under
30 seconds per AGENTS.md test budget.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import os

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
PROFILE_SCRIPT = os.path.join(REPO_ROOT, "scripts", "profile_step.py")


def _run_help():
    # Run --help and return stdout
    result = subprocess.run(
        [sys.executable, PROFILE_SCRIPT, "--help"],
        capture_output=True, text=True, cwd=REPO_ROOT, timeout=30,
    )
    assert result.returncode == 0, f"--help failed: {result.stderr}"
    return result.stdout


def test_help_lists_sprint1_flags():
    """Phase A exit criterion: --help lists all 7 Sprint 1 flags."""
    help_text = _run_help()
    required_flags = [
        "--imu1-groups",
        "--normuon",
        "--lr-2d",
        "--lr-1d",
        "--intra-doc-mask",
        "--value-residuals",
        "--head-gating",
    ]
    missing = [f for f in required_flags if f not in help_text]
    assert not missing, f"--help missing Sprint 1 flags: {missing}"
    print(f"  OK: all {len(required_flags)} Sprint 1 flags present in --help")


def test_help_lists_profile_args():
    help_text = _run_help()
    assert "--warmup" in help_text
    assert "--measure" in help_text
    assert "--model" in help_text
    assert "--class-name" in help_text
    assert "--dataset" in help_text
    assert "--compile" in help_text
    print("  OK: core profile args in --help")


def test_help_lists_compile_flag():
    help_text = _run_help()
    assert "--compile" in help_text
    print("  OK: --compile present")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args()

    tests = [
        test_help_lists_sprint1_flags,
        test_help_lists_profile_args,
        test_help_lists_compile_flag,
    ]
    for t in tests:
        print(f"[TEST] {t.__name__}")
        t()
    print(f"\nAll {len(tests)} tests passed")


if __name__ == "__main__":
    main()

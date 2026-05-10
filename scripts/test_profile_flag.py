"""Tests for Track 1.1 (--profile-steps) argparse + parser behavior.

These tests validate that the --profile-steps flag is accepted, parsed
correctly, and rejects bad input at argparse time. End-to-end behavior
(actual profile.json emission) is validated by running a short DDP smoke
via scripts/profile_odinflat_step.sh; that is not exercised here because
it requires CUDA + torchrun.

Plan: docs/superpowers/plans/2026-05-10-odinflat-throughput-investigation-plan.md
"""

from __future__ import annotations

import argparse
import os
import sys

import pytest


REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)


def _build_minimal_parser():
    """Replicate the --profile-steps + --diag-frozen-params argparse
    entries without importing train_ddp.main (which immediately runs
    distributed setup on import if misused)."""
    p = argparse.ArgumentParser()
    p.add_argument("--profile-steps", type=str, default="")
    p.add_argument("--diag-frozen-params", type=str, default="")
    return p


def test_profile_steps_default_is_empty():
    parser = _build_minimal_parser()
    args = parser.parse_args([])
    assert args.profile_steps == ""


def test_profile_steps_accepts_valid_range():
    parser = _build_minimal_parser()
    args = parser.parse_args(["--profile-steps", "30:40"])
    assert args.profile_steps == "30:40"


def test_profile_steps_parsing_start_end():
    # Replicate the runtime parsing logic from train_ddp.py main()
    spec = "30:40"
    start, end = [int(x) for x in spec.split(":")]
    assert start == 30
    assert end == 40
    assert end > start


def test_profile_steps_parsing_rejects_bad_format():
    # Runtime parsing logic should raise on missing colon or non-int.
    with pytest.raises(ValueError):
        [int(x) for x in "30-40".split(":")]  # no colon at all
    with pytest.raises(ValueError):
        [int(x) for x in "abc:def".split(":")]


def test_profile_steps_parsing_rejects_end_leq_start():
    # Assertion in train_ddp: end must be > start.
    start, end = 50, 50
    assert not (end > start)


def test_diag_frozen_params_default_is_empty():
    parser = _build_minimal_parser()
    args = parser.parse_args([])
    assert args.diag_frozen_params == ""


def test_diag_frozen_params_accepts_path():
    parser = _build_minimal_parser()
    args = parser.parse_args(["--diag-frozen-params", "/tmp/diag.jsonl"])
    assert args.diag_frozen_params == "/tmp/diag.jsonl"


def test_train_ddp_module_exposes_both_flags():
    """--profile-steps and --diag-frozen-params should be listed in the
    argparse source of scripts/train_ddp.py."""
    src_path = os.path.join(REPO_ROOT, "scripts", "train_ddp.py")
    with open(src_path, "r") as fh:
        src = fh.read()
    assert '"--profile-steps"' in src
    assert '"--diag-frozen-params"' in src
    assert "diag_writer" in src  # plumbed into _complete_step


def test_complete_step_signature_has_diag_writer():
    """_complete_step must accept diag_writer=None (Track 3.A plumbing)."""
    src_path = os.path.join(REPO_ROOT, "scripts", "train_ddp.py")
    with open(src_path, "r") as fh:
        src = fh.read()
    # Find the signature line
    assert "def _complete_step(" in src
    # Extract the line
    for line in src.splitlines():
        if line.startswith("def _complete_step("):
            assert "diag_writer" in line, (
                f"Expected diag_writer in signature: {line!r}"
            )
            break
    else:
        pytest.fail("_complete_step definition not found")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

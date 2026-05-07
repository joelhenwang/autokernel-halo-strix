"""Sprint 1.1 Phase A.5.2: bench_newton_schulz.py smoke test.

Tiny smoke run to confirm the NS benchmark writes the JSON without
crashing. Use very small --iters and --shape to stay under 15s.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
BENCH_SCRIPT = os.path.join(REPO_ROOT, "scripts", "bench_newton_schulz.py")


def test_ns_benchmark_runs():
    """Exit criterion per plan: benchmark completes and writes the JSON."""
    with tempfile.TemporaryDirectory() as tmp:
        out_path = os.path.join(tmp, "ns-bench.json")
        result = subprocess.run(
            [sys.executable, BENCH_SCRIPT,
             "--iters", "10", "--shape", "64", "128", "--out", out_path],
            capture_output=True, text=True, cwd=REPO_ROOT, timeout=60,
        )
        assert result.returncode == 0, (
            f"bench_newton_schulz.py failed:\nSTDOUT:\n{result.stdout}\n"
            f"STDERR:\n{result.stderr}"
        )
        assert os.path.exists(out_path), "Output JSON not written"
        with open(out_path) as f:
            data = json.load(f)
        assert "per_shape" in data
        assert "iters" in data
        assert data["iters"] == 10
        assert len(data["per_shape"]) == 1
        assert data["per_shape"][0]["fp32_ms"] > 0
        assert data["per_shape"][0]["fp16_ms"] > 0
        print(f"  OK: NS bench writes valid JSON "
              f"(fp32={data['per_shape'][0]['fp32_ms']:.3f}ms, "
              f"fp16={data['per_shape'][0]['fp16_ms']:.3f}ms)")


def main():
    print("[TEST] test_ns_benchmark_runs")
    test_ns_benchmark_runs()
    print("\nAll 1 tests passed")


if __name__ == "__main__":
    main()

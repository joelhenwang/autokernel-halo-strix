"""Test scaffold for the OdinHalo Fullgraph Compile Sprint.

Phase B target: verify correctness of compile-safe kernel wrappers and
the static depth_kv_buffer migration. All tests skip today because the
kernels + buffer migrations haven't landed yet; tests will flip on as
each piece is implemented.

Test groups (run per phase):
    Phase B.1 — HIP kernel wrapping:
        test_rope_gate_mul_numeric_parity_fp16
        test_rope_gate_mul_gradcheck_fp32
        test_causal_conv1d_numeric_parity_fp16
        test_causal_conv1d_gradcheck_fp32

    Phase B.2 — Static depth_kv_buffer:
        test_depth_kv_buffer_parity_at_step_10
        test_depth_kv_buffer_parity_at_step_100
        test_depth_kv_buffer_parity_at_step_500

    Phase B.4 — Integration:
        test_fullgraph_compile_no_breaks
        test_fullgraph_loss_parity_100_steps

Usage:
    python scripts/test_compile_safe_kernels.py
    python scripts/test_compile_safe_kernels.py --phase B.1
    python -m pytest scripts/test_compile_safe_kernels.py

Most tests depend on code that lands mid-sprint; they skip with a
clear reason until their prerequisite exists. The skip messages serve
as a sprint-progress checklist.
"""
from __future__ import annotations

import sys
import unittest
from pathlib import Path

import torch

# Allow running as a script from repo root
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def _kernel_wrapper_available(name: str) -> bool:
    """True if the compile-safe wrapper for kernel `name` has been added."""
    if name == "fused_rope_gate_mul":
        try:
            from kernels.hip.fused_rope_gate_mul import fused_rope_gate_mul_op  # noqa: F401
            return True
        except ImportError:
            return False
    if name == "causal_conv1d":
        try:
            from kernels.causal_conv1d.compile_safe import (  # noqa: F401
                causal_conv1d_compile_safe,
            )
            return True
        except ImportError:
            return False
    return False


def _static_depth_kv_buffer_available() -> bool:
    """True if OdinHalo has been migrated to static depth_kv_buffer."""
    try:
        from models.odin_halo import OdinHaloBase
        # The migration adds a _run_shared_block_static method.
        return hasattr(OdinHaloBase, "_run_shared_block_static")
    except ImportError:
        return False


def _fullgraph_path_available() -> bool:
    """True if `--fullgraph-compile` flag + compile_zones(fullgraph=True) are wired."""
    try:
        from models.odin_halo import OdinHaloBase
        import inspect
        sig = inspect.signature(OdinHaloBase.compile_zones)
        return "fullgraph" in sig.parameters
    except (ImportError, AttributeError):
        return False


# -----------------------------------------------------------------------------
# Phase B.1 — HIP kernel wrapping
# -----------------------------------------------------------------------------


class TestFusedRopeGateMulWrapper(unittest.TestCase):
    """Verify the compile-safe wrapper matches the raw HIP kernel output."""

    @unittest.skipUnless(
        _kernel_wrapper_available("fused_rope_gate_mul"),
        "fused_rope_gate_mul_op not defined yet — Phase B.1.1",
    )
    def test_rope_gate_mul_numeric_parity_fp16(self):
        from kernels.hip.fused_rope_gate_mul import fused_rope_gate_mul_op, kernel_fn
        if not torch.cuda.is_available():
            self.skipTest("No CUDA device")
        torch.manual_seed(0)
        B, T, d_conv = 2, 16, 128
        rope_pairs = 16
        b = torch.randn(B * T, d_conv, device="cuda", dtype=torch.float16)
        h = torch.randn(B * T, d_conv, device="cuda", dtype=torch.float16)
        cos = torch.randn(T, rope_pairs, device="cuda", dtype=torch.float32)
        sin = torch.randn(T, rope_pairs, device="cuda", dtype=torch.float32)

        ref = kernel_fn(b, h, cos, sin, T, d_conv, rope_pairs)
        got = fused_rope_gate_mul_op(b, h, cos, sin, T, d_conv, rope_pairs)

        self.assertEqual(ref.shape, got.shape)
        # Output is fp32 per the existing kernel contract
        self.assertTrue(torch.allclose(ref, got, atol=1e-3, rtol=1e-3))

    @unittest.skipUnless(
        _kernel_wrapper_available("fused_rope_gate_mul"),
        "fused_rope_gate_mul_op not defined yet — Phase B.1.1",
    )
    def test_rope_gate_mul_gradcheck_fp32(self):
        """Autograd correctness on a tiny fp32 proxy."""
        from kernels.hip.fused_rope_gate_mul import fused_rope_gate_mul_op
        if not torch.cuda.is_available():
            self.skipTest("No CUDA device")
        torch.manual_seed(0)
        B, T, d_conv = 1, 4, 16
        rope_pairs = 4
        # gradcheck requires double precision for tight tolerance
        b = torch.randn(B * T, d_conv, device="cuda", dtype=torch.float64, requires_grad=True)
        h = torch.randn(B * T, d_conv, device="cuda", dtype=torch.float64, requires_grad=True)
        cos = torch.randn(T, rope_pairs, device="cuda", dtype=torch.float64)
        sin = torch.randn(T, rope_pairs, device="cuda", dtype=torch.float64)

        self.assertTrue(
            torch.autograd.gradcheck(
                lambda b, h: fused_rope_gate_mul_op(b, h, cos, sin, T, d_conv, rope_pairs),
                (b, h), eps=1e-6, atol=1e-4,
            )
        )


class TestCausalConv1dWrapper(unittest.TestCase):
    """Verify the compile-safe wrapper of DaoAILab's causal_conv1d_fn."""

    @unittest.skipUnless(
        _kernel_wrapper_available("causal_conv1d"),
        "causal_conv1d_compile_safe not defined yet — Phase B.1.2",
    )
    def test_causal_conv1d_numeric_parity_fp16(self):
        from kernels.causal_conv1d.compile_safe import causal_conv1d_compile_safe
        from causal_conv1d import causal_conv1d_fn
        if not torch.cuda.is_available():
            self.skipTest("No CUDA device")
        torch.manual_seed(0)
        B, C, T = 2, 128, 64
        K = 3
        x = torch.randn(B, C, T, device="cuda", dtype=torch.float16)
        weight = torch.randn(C, K, device="cuda", dtype=torch.float16)
        bias = torch.randn(C, device="cuda", dtype=torch.float16)

        ref = causal_conv1d_fn(x, weight, bias)
        got = causal_conv1d_compile_safe(x, weight, bias)

        self.assertEqual(ref.shape, got.shape)
        self.assertTrue(torch.allclose(ref, got, atol=1e-3, rtol=1e-3))

    @unittest.skipUnless(
        _kernel_wrapper_available("causal_conv1d"),
        "causal_conv1d_compile_safe not defined yet — Phase B.1.2",
    )
    def test_causal_conv1d_gradcheck_fp32(self):
        """Autograd correctness. Falls back to allow_in_graph if register_autograd fails."""
        from kernels.causal_conv1d.compile_safe import causal_conv1d_compile_safe
        if not torch.cuda.is_available():
            self.skipTest("No CUDA device")
        torch.manual_seed(0)
        B, C, T = 1, 8, 8
        K = 3
        x = torch.randn(B, C, T, device="cuda", dtype=torch.float64, requires_grad=True)
        weight = torch.randn(C, K, device="cuda", dtype=torch.float64, requires_grad=True)
        bias = torch.randn(C, device="cuda", dtype=torch.float64, requires_grad=True)

        self.assertTrue(
            torch.autograd.gradcheck(
                causal_conv1d_compile_safe,
                (x, weight, bias), eps=1e-6, atol=1e-4,
            )
        )


# -----------------------------------------------------------------------------
# Phase B.2 — Static depth_kv_buffer
# -----------------------------------------------------------------------------


class TestStaticDepthKVBuffer(unittest.TestCase):
    """Parity between dynamic list-of-dicts and static tensor buffer."""

    def _run_parity(self, n_steps: int):
        """Run forward for n_steps with both paths, assert close outputs."""
        if not _static_depth_kv_buffer_available():
            self.skipTest("OdinHaloBase._run_shared_block_static not present — Phase B.2")
        if not torch.cuda.is_available():
            self.skipTest("No CUDA device")

        from models.odin_halo import OdinHaloBase
        torch.manual_seed(0)
        # Small proxy config for fast tests
        model = OdinHaloBase(
            vocab_size=1024, d_model=128, embed_rank=64, n_shared_layers=2,
            gqa_positions=(1,), ffn_inner=256, d_conv=64,
            mean_recurrence=2, backprop_depth=2, max_seq_len=64,
        ).to("cuda").half()

        input_ids = torch.randint(0, 1024, (2, 32), device="cuda", dtype=torch.long)

        outputs_dynamic = []
        outputs_static = []
        for step in range(n_steps):
            # Toggle between paths per step to accumulate any drift
            # (each path produces the same output for a given weight state).
            model.use_static_depth_kv = False
            out_dyn = model(input_ids)
            model.use_static_depth_kv = True
            out_stat = model(input_ids)
            outputs_dynamic.append(out_dyn.detach().float().cpu())
            outputs_static.append(out_stat.detach().float().cpu())

            # Small optimizer step to perturb weights between iterations
            loss = out_dyn.float().mean() + out_stat.float().mean()
            loss.backward()
            with torch.no_grad():
                for p in model.parameters():
                    if p.grad is not None:
                        p.data -= 1e-4 * p.grad
                        p.grad = None

        # Compare accumulated outputs
        for i, (d, s) in enumerate(zip(outputs_dynamic, outputs_static)):
            self.assertTrue(
                torch.allclose(d, s, atol=5e-3, rtol=5e-3),
                f"Step {i}: max|Δ| = {(d - s).abs().max().item():.3e}",
            )

    def test_depth_kv_buffer_parity_at_step_10(self):
        self._run_parity(10)

    def test_depth_kv_buffer_parity_at_step_100(self):
        self._run_parity(100)

    def test_depth_kv_buffer_parity_at_step_500(self):
        # Heavy — run only when explicitly requested
        if "--heavy" not in sys.argv:
            self.skipTest("Heavy test; re-run with --heavy to include")
        self._run_parity(500)


# -----------------------------------------------------------------------------
# Phase B.4 — Integration
# -----------------------------------------------------------------------------


class TestFullgraphIntegration(unittest.TestCase):
    """End-to-end: fullgraph compile succeeds + loss parity."""

    @unittest.skipUnless(
        _fullgraph_path_available(),
        "OdinHaloBase.compile_zones(fullgraph=...) not wired yet — Phase B.4",
    )
    def test_fullgraph_compile_no_breaks(self):
        """With use_compile_safe_kernels=True and fullgraph compile,
        no Dynamo Unsupported exceptions should fire."""
        if not torch.cuda.is_available():
            self.skipTest("No CUDA device")
        from models.odin_halo import OdinHaloBase
        import torch._dynamo

        torch._dynamo.reset()
        model = OdinHaloBase(
            vocab_size=1024, d_model=128, embed_rank=64, n_shared_layers=2,
            gqa_positions=(1,), ffn_inner=256, d_conv=64,
            mean_recurrence=2, backprop_depth=2, max_seq_len=64,
            use_compile_safe_kernels=True,
        ).to("cuda").half()

        model.compile_zones(fullgraph=True)
        input_ids = torch.randint(0, 1024, (2, 32), device="cuda", dtype=torch.long)
        with torch.no_grad():
            _ = model(input_ids)
        # If we reached here, fullgraph succeeded. Test passes.

    @unittest.skipUnless(
        _fullgraph_path_available(),
        "Fullgraph path not wired yet — Phase B.4",
    )
    def test_fullgraph_loss_parity_100_steps(self):
        """Loss trajectory with fullgraph compile matches per-layer within fp16 noise."""
        if not torch.cuda.is_available():
            self.skipTest("No CUDA device")
        from models.odin_halo import OdinHaloBase

        def run_once(fullgraph: bool):
            torch.manual_seed(42)
            model = OdinHaloBase(
                vocab_size=1024, d_model=128, embed_rank=64, n_shared_layers=2,
                gqa_positions=(1,), ffn_inner=256, d_conv=64,
                mean_recurrence=2, backprop_depth=2, max_seq_len=64,
                use_compile_safe_kernels=fullgraph,
            ).to("cuda").half()
            model.compile_zones(fullgraph=fullgraph)
            opt = torch.optim.AdamW(model.parameters(), lr=1e-3, fused=True)
            losses = []
            for step in range(100):
                torch.manual_seed(1000 + step)  # deterministic inputs per step
                input_ids = torch.randint(0, 1024, (2, 32), device="cuda", dtype=torch.long)
                targets = torch.randint(0, 1024, (2, 32), device="cuda", dtype=torch.long)
                logits = model(input_ids)
                loss = torch.nn.functional.cross_entropy(
                    logits.reshape(-1, 1024), targets.reshape(-1)
                )
                opt.zero_grad(set_to_none=True)
                loss.backward()
                opt.step()
                losses.append(loss.item())
            return losses

        losses_perlayer = run_once(fullgraph=False)
        losses_fullgraph = run_once(fullgraph=True)

        # Max per-step |Δloss| should stay below the 0.25 fp16 noise band
        deltas = [abs(a - b) for a, b in zip(losses_perlayer, losses_fullgraph)]
        self.assertLess(max(deltas), 0.25,
                        f"Loss parity violated: max|Δ| = {max(deltas):.3f}")


if __name__ == "__main__":
    # Simple argv parsing — allow --phase filter
    phase_filter = None
    for i, arg in enumerate(sys.argv[1:]):
        if arg == "--phase" and i + 2 < len(sys.argv):
            phase_filter = sys.argv[i + 2]
            # Remove for unittest
            sys.argv.pop(i + 2)
            sys.argv.pop(i + 1)
            break

    if phase_filter:
        # Filter tests by phase via test case naming
        # (simple heuristic — users can also just target via -k)
        print(f"[test] Phase filter: {phase_filter} (use -k <name> for finer control)")

    unittest.main(verbosity=2)

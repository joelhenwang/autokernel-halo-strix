"""Micro-benchmark: ActivationMonitor hook overhead, pre- and post-fix.

Measures the cost of ActivationMonitor's forward hook in four scenarios:

  A. monitor detached                    (baseline, no hook)
  B. monitor attached, never set_step    (backward-compat; old slow path —
                                          hook records and .item() syncs on
                                          EVERY forward)
  C. monitor attached, set_step with disarmed step  (new fast path; hook
                                                     returns immediately)
  D. monitor attached, set_step with sample step    (hook records a GPU
                                                     tensor; .item() sync
                                                     deferred to step())

Exit criteria per plan §1.0:
  (C vs A) delta must be <1% on a looped-model-like pattern (layer-count
  dominated by many short forward passes).

Usage:
  python scripts/bench_activation_monitor.py
"""

from __future__ import annotations

import os
import sys
import time

import torch
import torch.nn as nn

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from halo_training.activation_monitor import ActivationMonitor


class DeepModel(nn.Module):
    """Stand-in for a looped model's forward: 18 "layer" passes.

    Matches OdinHalo's 6 shared layers * 3 iters = 18 forwards per step.
    The Linear(512 -> 512) size is small enough that reduction overhead
    is measurable against the layer work itself.
    """

    def __init__(self, n_layers: int = 18, dim: int = 512):
        super().__init__()
        self.layers = nn.ModuleList([nn.Linear(dim, dim) for _ in range(n_layers)])

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x


@torch.no_grad()
def bench_scenario(name: str, model, monitor, n_steps: int, arm_mode: str):
    """Run n_steps forward passes, simulate monitor integration per arm_mode."""
    device = next(model.parameters()).device
    x = torch.randn(16, 256, 512, device=device, dtype=torch.float16)
    # Warmup
    for _ in range(20):
        model(x)
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    if arm_mode == "detached":
        for _ in range(n_steps):
            model(x)
    elif arm_mode == "never_set_step":
        # Default _should_sample=True → hook records every forward (old slow)
        for _ in range(n_steps):
            model(x)
            # No monitor.step() call — we're measuring pure hook overhead
    elif arm_mode == "set_step_disarmed":
        # Arm for NON-sampling step → hook returns immediately
        monitor.set_step(1)  # 1 % 100 != 0 → disarmed
        for step in range(n_steps):
            model(x)
    elif arm_mode == "set_step_armed":
        # Realistic trainer flow: every "opt step" re-arms for the next
        # step. Only opt steps divisible by sample_every (100) are armed.
        # Each opt step represents 1 forward for bench simplicity; real
        # trainer has ``accum_steps`` forwards per opt step all honoring
        # the same _should_sample flag.
        monitor.set_step(1)  # arm for step 1 (will be disarmed: 1%100 != 0)
        for step_idx in range(n_steps):
            global_step = step_idx + 1
            model(x)
            # End of opt step: commit + re-arm for next
            monitor.step(global_step)
            monitor.set_step(global_step + 1)
    else:
        raise ValueError(arm_mode)
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - t0
    tok_s = (n_steps * 16 * 256) / elapsed
    print(f"  {name:<32s}  {elapsed:>6.3f}s    {tok_s:>12,.0f} forward-tokens/s")
    return elapsed


def main():
    assert torch.cuda.is_available(), "need GPU for realistic numbers"
    torch.manual_seed(0)
    model = DeepModel(n_layers=18, dim=512).cuda().half()
    n_steps = 1000

    print(f"Setup: {sum(p.numel() for p in model.parameters())/1e6:.1f}M params, "
          f"{len(model.layers)} layers, batch (16, 256), fp16")
    print(f"Running {n_steps} forward steps per scenario.\n")

    # A. detached baseline
    print("Scenario A: monitor DETACHED (baseline)")
    tA = bench_scenario("  detached", model, None, n_steps, "detached")

    # B. attached, never set_step (old-style behavior: record + sync every forward)
    print("\nScenario B: monitor ATTACHED, never set_step (old-style, records every forward)")
    monB = ActivationMonitor(model, output_path=None, sample_every=100)
    monB.attach()
    # Force old-style by calling .item() synchronously in the hook — emulate pre-fix.
    # To emulate, we can sample directly after each forward via step(n) where n%sample_every==0.
    # But since default _should_sample=True, hook records into _current_stats. To force a sync
    # per forward, commit+clear every forward.
    def bench_old_style(n):
        x = torch.randn(16, 256, 512, device="cuda", dtype=torch.float16)
        for _ in range(20):
            model(x)
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(n):
            model(x)
            # Emulate old code: .item() on every forward per layer
            for stats in monB._current_stats.values():
                if "maxabs_tensor" in stats:
                    _ = float(stats["maxabs_tensor"].item())
        torch.cuda.synchronize()
        return time.perf_counter() - t0
    tB = bench_old_style(n_steps)
    print(f"  {'attached, item() every fwd':<32s}  {tB:>6.3f}s    "
          f"{(n_steps*16*256)/tB:>12,.0f} forward-tokens/s")
    monB.detach()

    # C. attached, set_step disarmed
    print("\nScenario C: monitor ATTACHED, set_step DISARMED (new fast path)")
    monC = ActivationMonitor(model, output_path=None, sample_every=100)
    monC.attach()
    tC = bench_scenario("  attached, disarmed", model, monC, n_steps, "set_step_disarmed")
    monC.detach()

    # D. attached, set_step armed every 100 steps (normal usage)
    print("\nScenario D: monitor ATTACHED, set_step armed per sample_every=100 (real usage)")
    import tempfile
    with tempfile.TemporaryDirectory() as td:
        monD = ActivationMonitor(
            model, output_path=os.path.join(td, "out.jsonl"), sample_every=100,
        )
        monD.attach()
        tD = bench_scenario("  attached, armed@sample", model, monD, n_steps, "set_step_armed")
        monD.detach()

    print("\nSummary:")
    print(f"  A (detached)              : {tA:.3f}s   baseline")
    print(f"  B (item()-every-fwd; old) : {tB:.3f}s   +{100*(tB-tA)/tA:+.1f}%  "
          f"(expected: large overhead)")
    print(f"  C (disarmed; new fast)    : {tC:.3f}s   +{100*(tC-tA)/tA:+.1f}%  "
          f"(expected: <1%)")
    print(f"  D (armed@sample; normal)  : {tD:.3f}s   +{100*(tD-tA)/tA:+.1f}%  "
          f"(expected: <2%)")

    fix_ratio = (tB - tC) / tB * 100
    print(f"\n  Fix recovered {fix_ratio:.1f}% of hook overhead (B -> C).")
    print(f"  Normal-usage residual (D vs A): {100*(tD-tA)/tA:+.2f}% — "
          f"{'PASS' if (tD - tA) / tA < 0.02 else 'FAIL'} (<2% target)")
    print(f"  Disarmed overhead (C vs A):     {100*(tC-tA)/tA:+.2f}% — "
          f"{'PASS' if (tC - tA) / tA < 0.01 else 'FAIL'} (<1% target)")


if __name__ == "__main__":
    main()

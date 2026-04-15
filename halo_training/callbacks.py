"""Training callbacks: phase scheduling, memory monitoring, stability diagnostics."""

from typing import Dict, List, Optional, Set, Tuple

import torch
import torch.nn as nn


class PhaseScheduler:
    """Multi-phase unfreezing scheduler from COOKBOOK.md lines 341-364.

    Usage:
        scheduler = PhaseScheduler([
            (0.4, {"layers.", "embed", "tok_embed"}, "Phase 1: backbone only"),
            (0.3, {"meta_token", "engram"}, "Phase 2: + memory modules"),
            (0.3, {"mtp.", "lm_head"}, "Phase 3: + all"),
        ], total_steps=1000)
    """

    def __init__(self, phases: List[Tuple[float, Set[str], str]], total_steps: int):
        self.phases = phases
        self.boundaries = []
        cumulative = 0
        for frac, _, desc in phases:
            cumulative += int(frac * total_steps)
            self.boundaries.append(cumulative)
        self._last_phase = -1

    def get_phase(self, step: int) -> int:
        for i, boundary in enumerate(self.boundaries):
            if step < boundary:
                return i
        return len(self.phases) - 1

    def __call__(self, model: nn.Module, step: int) -> None:
        phase_idx = self.get_phase(step)

        if phase_idx != self._last_phase:
            desc = self.phases[phase_idx][2]
            print(f"[Phase {phase_idx}] {desc} (step {step})")
            self._last_phase = phase_idx

        active_prefixes = set()
        for i in range(phase_idx + 1):
            active_prefixes.update(self.phases[i][1])

        for name, param in model.named_parameters():
            param.requires_grad_(any(name.startswith(p) for p in active_prefixes))


class MemoryMonitor:
    """Logs GPU memory usage and warns when approaching limits.

    On Strix Halo, GPU-visible memory is ~116 GB of 128 GB total unified.
    OOM on unified memory kills the whole process (no graceful GPU OOM).
    """

    def __init__(self, warn_threshold_gb: float = 104.0):
        self.warn_threshold_gb = warn_threshold_gb
        self._warned = False

    def __call__(self, model: nn.Module, step: int) -> None:
        if not torch.cuda.is_available():
            return

        peak_gb = torch.cuda.max_memory_allocated() / 1e9
        if peak_gb > self.warn_threshold_gb and not self._warned:
            print(
                f"[MemoryMonitor] WARNING: peak memory {peak_gb:.1f} GB "
                f"exceeds {self.warn_threshold_gb:.0f} GB threshold"
            )
            self._warned = True

    def report(self) -> dict:
        if not torch.cuda.is_available():
            return {}
        return {
            "peak_allocated_gb": torch.cuda.max_memory_allocated() / 1e9,
            "peak_reserved_gb": torch.cuda.max_memory_reserved() / 1e9,
            "current_allocated_gb": torch.cuda.memory_allocated() / 1e9,
        }


class StateNormMonitor:
    """Track hidden state growth ratio per recurrent block.

    Adapted from mad_llm_scientist/diagnostics/state_norm_logger.py.
    Detects recurrence instability ~10 steps before gradient explosion.

    Usage:
        monitor = StateNormMonitor(model, warn_ratio=1.05, stop_ratio=1.15)
        # Add as callback: callbacks=[monitor]
        # After training: monitor.report()
    """

    def __init__(
        self,
        model: nn.Module,
        warn_ratio: float = 1.05,
        stop_ratio: float = 1.15,
        log_interval: int = 10,
    ):
        self.warn_ratio = warn_ratio
        self.stop_ratio = stop_ratio
        self.log_interval = log_interval
        self.max_ratio = 0.0
        self._hooks = []
        self._prev_norms: Dict[str, float] = {}
        self._ratios: Dict[str, List[float]] = {}

        # Register forward hooks on recurrent modules
        for name, module in model.named_modules():
            if any(k in name.lower() for k in ("recurrence", "lru", "rec_", "griffin", "ssm")):
                hook = module.register_forward_hook(self._make_hook(name))
                self._hooks.append(hook)
                self._ratios[name] = []

    def _make_hook(self, name: str):
        def hook(module, input, output):
            if isinstance(output, torch.Tensor):
                h = output
            elif isinstance(output, tuple):
                h = output[0]
            else:
                return

            norm = h.float().norm().item()
            if name in self._prev_norms and self._prev_norms[name] > 0:
                ratio = norm / self._prev_norms[name]
                self._ratios[name].append(ratio)
                self.max_ratio = max(self.max_ratio, ratio)
            self._prev_norms[name] = norm

        return hook

    def __call__(self, model: nn.Module, step: int) -> None:
        if not self._ratios:
            return

        if step > 0 and step % self.log_interval == 0:
            if self.max_ratio > self.stop_ratio:
                print(f"[StateNormMonitor] CRITICAL: max_ratio={self.max_ratio:.4f} > {self.stop_ratio} — "
                      f"consider stopping")
            elif self.max_ratio > self.warn_ratio:
                print(f"[StateNormMonitor] WARNING: max_ratio={self.max_ratio:.4f} > {self.warn_ratio}")

    def report(self) -> dict:
        return {
            "max_ratio": self.max_ratio,
            "tracked_modules": list(self._ratios.keys()),
            "stable": self.max_ratio < self.warn_ratio,
        }

    def remove_hooks(self):
        for h in self._hooks:
            h.remove()
        self._hooks.clear()


class PerParamGradMonitor:
    """Log top-K parameters by gradient norm each step.

    Adapted from mad_llm_scientist/diagnostics/per_param_grad_log.py.
    Global grad norm hides which components explode — this identifies culprits.

    Usage:
        monitor = PerParamGradMonitor(top_k=5, log_interval=50)
        # Add as callback: callbacks=[monitor]
    """

    def __init__(self, top_k: int = 5, log_interval: int = 50):
        self.top_k = top_k
        self.log_interval = log_interval

    def __call__(self, model: nn.Module, step: int) -> None:
        if step == 0 or step % self.log_interval != 0:
            return

        grad_norms = []
        for name, param in model.named_parameters():
            if param.grad is not None:
                norm = param.grad.data.float().norm().item()
                grad_norms.append((norm, name))

        grad_norms.sort(reverse=True)
        top = grad_norms[:self.top_k]

        if top:
            parts = [f"{name}={norm:.2f}" for norm, name in top]
            print(f"[GradMonitor step {step}] top-{self.top_k}: {', '.join(parts)}")

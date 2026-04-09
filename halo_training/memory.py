"""Memory budget estimation and mode selection for Halo Training Stack."""

import torch
import torch.nn as nn

from halo_training.model_utils import count_parameters, get_layer_iterator


class MemoryBudget:
    """Estimate and monitor GPU memory for training.

    On Strix Halo (Ryzen AI MAX+ 395):
    - Total unified: 128 GB
    - GPU-visible: ~116 GB (reported by PyTorch)
    - Safe training limit: total_gb - reserve_gb
    """

    def __init__(self, total_gb: float = 116.0, reserve_gb: float = 8.0):
        self.total_gb = total_gb
        self.reserve_gb = reserve_gb
        self.available_gb = total_gb - reserve_gb

    def estimate(
        self,
        model: nn.Module,
        batch_size: int = 16,
        block_size: int = 1024,
        dtype: torch.dtype = torch.float16,
    ) -> dict:
        n_params = count_parameters(model, trainable_only=False)
        bytes_per_param = 2 if dtype in (torch.float16, torch.bfloat16) else 4

        params_gb = n_params * bytes_per_param / 1e9
        optimizer_gb = n_params * 4 * 3 / 1e9  # fp32 m, v, master
        gradients_gb = params_gb

        # Activation estimate per layer
        n_layers = len(get_layer_iterator(model))
        hidden = _guess_hidden(model)
        # Each layer stores: input activation (B * T * D * bytes) + attention intermediates
        act_per_layer_gb = batch_size * block_size * hidden * bytes_per_param / 1e9
        activations_gb = n_layers * act_per_layer_gb * 3  # ~3x for intermediates

        total_gb = params_gb + optimizer_gb + gradients_gb + activations_gb

        return {
            "params_gb": params_gb,
            "optimizer_gb": optimizer_gb,
            "gradients_gb": gradients_gb,
            "activations_gb": activations_gb,
            "total_gb": total_gb,
            "fits": total_gb < self.available_gb,
            "headroom_gb": self.available_gb - total_gb,
        }

    def check_pressure(self) -> str:
        """Check current GPU memory pressure."""
        if not torch.cuda.is_available():
            return "ok"
        used_gb = torch.cuda.memory_allocated() / 1e9
        if used_gb > self.available_gb * 0.95:
            return "critical"
        elif used_gb > self.available_gb * 0.80:
            return "warn"
        return "ok"


def suggest_mode(
    model: nn.Module,
    batch_size: int = 16,
    block_size: int = 1024,
) -> str:
    """Auto-detect Mode A vs Mode B based on memory estimate.

    Mode A: direct training (everything fits comfortably)
    Mode B: layer-streaming with activation checkpointing (memory-tight)

    Crossover heuristic: if estimated total > 60% of available GPU memory,
    use Mode B for safety margin.
    """
    budget = MemoryBudget()
    est = budget.estimate(model, batch_size, block_size)

    threshold = budget.available_gb * 0.60  # 60% of ~108 GB = ~65 GB

    if est["total_gb"] > threshold:
        return "B"
    return "A"


def _guess_hidden(model: nn.Module) -> int:
    for attr in ("dim", "d_model", "hidden_size"):
        if hasattr(model, attr):
            val = getattr(model, attr)
            if isinstance(val, int):
                return val
    # Check config
    if hasattr(model, "config") and hasattr(model.config, "hidden_size"):
        return model.config.hidden_size
    # Fall back to first Linear
    for m in model.modules():
        if isinstance(m, nn.Linear):
            return m.out_features
    return 1024

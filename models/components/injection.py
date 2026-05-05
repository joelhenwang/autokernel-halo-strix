"""SimpleParcaeInjection — stable loop re-injection for uniform dimensions (from griffin_halo)."""

import torch
import torch.nn as nn


class SimpleParcaeInjection(nn.Module):
    """Parcae injection for uniform dimensions (no projection needed)."""

    def __init__(self, d_model: int):
        super().__init__()
        self.log_A = nn.Parameter(torch.full((d_model,), -0.7))
        self.log_B = nn.Parameter(torch.full((d_model,), -0.7))

    def forward(self, h: torch.Tensor, input_embed: torch.Tensor) -> torch.Tensor:
        A = -torch.exp(self.log_A)
        B = torch.exp(self.log_B)
        return A * h + B * input_embed
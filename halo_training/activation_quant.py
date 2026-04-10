"""Int8 activation quantization for backward memory savings.

Stores activations in int8 during forward (per-tensor absmax quantization).
Dequantizes to fp16/fp32 during backward. This reduces activation memory
by ~2x, enabling larger batch sizes which amortize backward overhead.

WARNING: Quantization introduces noise in gradients. Verify convergence.

Usage:
    from halo_training.activation_quant import QuantizedLinear, apply_activation_quant
    apply_activation_quant(model, target_modules=["feed_forward"])
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class QuantizedLinear(torch.autograd.Function):
    """Linear layer that saves activations in int8 for backward."""

    @staticmethod
    def forward(ctx, input, weight, bias):
        output = F.linear(input, weight, bias)

        # Quantize input to int8 for backward
        scale = input.abs().max() / 127.0
        if scale == 0:
            scale = torch.ones(1, device=input.device, dtype=input.dtype)
        input_int8 = (input / scale).round().clamp(-127, 127).to(torch.int8)

        ctx.save_for_backward(input_int8, weight, bias)
        ctx.input_scale = scale
        ctx.input_dtype = input.dtype

        return output

    @staticmethod
    def backward(ctx, grad_output):
        input_int8, weight, bias = ctx.saved_tensors

        # Dequantize input
        input_approx = input_int8.float() * ctx.input_scale

        # grad_input: exact (from weight, not quantized)
        grad_input = grad_output @ weight

        # grad_weight: uses approximate (dequantized) input
        input_2d = input_approx.reshape(-1, input_approx.shape[-1])
        grad_2d = grad_output.reshape(-1, grad_output.shape[-1]).float()
        grad_weight = grad_2d.t() @ input_2d

        grad_bias = None
        if bias is not None:
            grad_bias = grad_output.sum(dim=tuple(range(grad_output.ndim - 1)))

        return grad_input, grad_weight.to(weight.dtype), grad_bias


class QuantizedLinearWrapper(nn.Module):
    """Wraps nn.Linear with int8 activation quantization for backward."""

    def __init__(self, linear: nn.Linear):
        super().__init__()
        self.linear = linear

    def forward(self, x):
        return QuantizedLinear.apply(x, self.linear.weight, self.linear.bias)

    @property
    def weight(self):
        return self.linear.weight

    @property
    def bias(self):
        return self.linear.bias


def apply_activation_quant(
    model: nn.Module,
    target_modules: list = None,
    min_features: int = 512,
) -> nn.Module:
    """Replace targeted Linear layers with QuantizedLinearWrapper.

    Args:
        model: Model to modify in-place
        target_modules: Module name prefixes to target (None = all large Linear)
        min_features: Only quantize layers with in_features >= this

    Returns:
        Modified model
    """
    replacements = {}
    for name, module in model.named_modules():
        if not isinstance(module, nn.Linear):
            continue
        if module.in_features < min_features:
            continue
        if target_modules is not None:
            if not any(t in name for t in target_modules):
                continue
        replacements[name] = module

    for name, module in replacements.items():
        parts = name.split(".")
        parent = model
        for p in parts[:-1]:
            parent = getattr(parent, p)
        setattr(parent, parts[-1], QuantizedLinearWrapper(module))

    if replacements:
        print(f"Applied int8 activation quantization to {len(replacements)} Linear layers")
    return model

"""Model utilities: layer discovery, parameter counting, memory estimation."""

from typing import List, Optional

import torch
import torch.nn as nn


def get_layer_iterator(model: nn.Module) -> List[nn.Module]:
    """Generic layer discovery for any model architecture.

    Checks common layer container attributes in order:
    - model.layers (LlamaModel, most mad_llm_scientist architectures)
    - model.transformer.h (GPT-2)
    - model.model.layers (HuggingFace LlamaForCausalLM)
    - model.encoder.layer (BERT)
    - Falls back to list(model.children())
    """
    for attr_path in [
        "layers",
        "transformer.h",
        "model.layers",
        "encoder.layer",
        "decoder.layers",
    ]:
        obj = model
        try:
            for part in attr_path.split("."):
                obj = getattr(obj, part)
            if isinstance(obj, nn.ModuleList) or (hasattr(obj, "__iter__") and hasattr(obj, "__len__")):
                return list(obj)
        except AttributeError:
            continue

    return list(model.children())


def get_layer_container(model: nn.Module):
    """Return (parent_module, attribute_name) for the layer container.

    Useful for Mode B when you need to iterate and replace layers.
    """
    for attr_path in [
        "layers",
        "transformer.h",
        "model.layers",
        "encoder.layer",
        "decoder.layers",
    ]:
        parts = attr_path.split(".")
        obj = model
        try:
            for part in parts[:-1]:
                obj = getattr(obj, part)
            container = getattr(obj, parts[-1])
            if isinstance(container, nn.ModuleList):
                return obj, parts[-1]
        except AttributeError:
            continue

    return None, None


def count_parameters(model: nn.Module, trainable_only: bool = True) -> int:
    """Count model parameters."""
    if trainable_only:
        return sum(p.numel() for p in model.parameters() if p.requires_grad)
    return sum(p.numel() for p in model.parameters())


def estimate_memory(
    model: nn.Module,
    batch_size: int = 64,
    block_size: int = 1024,
    dtype: torch.dtype = torch.float16,
    include_optimizer: bool = True,
) -> dict:
    """Estimate memory budget for training.

    Returns dict with component-wise GB estimates.
    Based on design spec Section 4 formulas.
    """
    n_params = count_parameters(model, trainable_only=False)
    bytes_per_param = 2 if dtype in (torch.float16, torch.bfloat16) else 4

    # Model parameters
    params_gb = n_params * bytes_per_param / 1e9

    # Optimizer states (fp32 m, v, master weights = 3x params in fp32)
    optimizer_gb = n_params * 4 * 3 / 1e9 if include_optimizer else 0

    # Activations (rough estimate: ~2 bytes per element per layer)
    n_layers = len(get_layer_iterator(model))
    hidden_size = _guess_hidden_size(model)
    acts_per_layer = batch_size * block_size * hidden_size * 2  # bytes
    activations_gb = n_layers * acts_per_layer / 1e9

    # Gradients (same size as params)
    gradients_gb = params_gb

    total_gb = params_gb + optimizer_gb + activations_gb + gradients_gb

    return {
        "params_gb": params_gb,
        "optimizer_gb": optimizer_gb,
        "activations_gb": activations_gb,
        "gradients_gb": gradients_gb,
        "total_gb": total_gb,
        "n_params": n_params,
        "n_layers": n_layers,
    }


def _guess_hidden_size(model: nn.Module) -> int:
    """Try to detect hidden size from model structure."""
    # Check common attributes
    for attr in ("d_model", "hidden_size", "config.hidden_size", "dim"):
        obj = model
        try:
            for part in attr.split("."):
                obj = getattr(obj, part)
            if isinstance(obj, int):
                return obj
        except AttributeError:
            continue

    # Fall back to first Linear layer's output features
    for module in model.modules():
        if isinstance(module, nn.Linear):
            return module.out_features

    return 1024  # conservative default

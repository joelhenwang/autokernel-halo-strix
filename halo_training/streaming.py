"""Mode B: Layer-streaming trainer with activation checkpointing.

For models where optimizer states + activations approach GPU memory limits
(typically >2B params on 116 GB Strix Halo).

Key differences from Mode A:
- Activation checkpointing every N layers (recompute during backward)
- Per-layer torch.compile (not whole model)
- Memory pressure monitoring with escape valves
"""

import time
from typing import Any, Callable, Dict, List, Optional

import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint as torch_checkpoint

from halo_training.model_utils import get_layer_iterator
from halo_training.memory import MemoryBudget


class LayerStreamingTrainer:
    """Mode B training engine with per-layer checkpointing.

    Wraps a model's forward pass to apply activation checkpointing
    every `checkpoint_every` layers, reducing peak activation memory
    at the cost of ~20-30% extra compute (recomputation during backward).

    Usage:
        streamer = LayerStreamingTrainer(model, checkpoint_every=2)
        # In training loop:
        logits = streamer.forward(input_ids)
        loss = loss_fn(logits, targets)
        loss.backward()  # checkpointed layers recompute
    """

    def __init__(
        self,
        model: nn.Module,
        checkpoint_every: int = 2,
        max_memory_gb: float = 108.0,
        compile_layers: bool = False,
    ):
        self.model = model
        self.checkpoint_every = checkpoint_every
        self.max_memory_gb = max_memory_gb
        self.budget = MemoryBudget()

        self.layers = get_layer_iterator(model)
        self.n_layers = len(self.layers)

        # Find pre/post layer components
        self._find_model_components()

        # Optionally compile individual layers
        if compile_layers:
            print(f"Compiling {self.n_layers} layers individually...")
            for i, layer in enumerate(self.layers):
                self.layers[i] = torch.compile(layer, mode="reduce-overhead")

        ckpt_count = sum(1 for i in range(self.n_layers) if i % checkpoint_every == 0)
        print(f"LayerStreamingTrainer: {self.n_layers} layers, "
              f"checkpointing every {checkpoint_every} ({ckpt_count} checkpointed)")

    def _find_model_components(self):
        """Identify embedding, final norm, and output head."""
        model = self.model

        # Embedding
        self.embed_fn = None
        for name in ("tok_embeddings", "embed_tokens", "transformer.wte", "embedding"):
            if hasattr(model, name):
                self.embed_fn = getattr(model, name)
                break
        if self.embed_fn is None:
            for m in model.modules():
                if isinstance(m, nn.Embedding):
                    self.embed_fn = m
                    break

        # Final norm
        self.final_norm = None
        for name in ("norm", "ln_f", "final_layernorm"):
            if hasattr(model, name):
                self.final_norm = getattr(model, name)
                break

        # Output projection
        self.output_proj = None
        for name in ("output", "lm_head", "head"):
            if hasattr(model, name):
                self.output_proj = getattr(model, name)
                break

        # freqs_cis for rotary (LlamaModel-specific)
        self.freqs_cis = getattr(model, "freqs_cis", None)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        """Forward pass with per-layer activation checkpointing."""
        B, T = input_ids.shape

        # Embedding
        h = self.embed_fn(input_ids)

        # Get rotary freqs if available
        freqs = self.freqs_cis[:T] if self.freqs_cis is not None else None

        # Layer-by-layer with checkpointing
        for i, layer in enumerate(self.layers):
            if i % self.checkpoint_every == 0:
                # Checkpointed: activations discarded, recomputed during backward
                if freqs is not None:
                    h = torch_checkpoint(
                        layer, h, freqs,
                        use_reentrant=False,
                    )
                else:
                    h = torch_checkpoint(
                        layer, h,
                        use_reentrant=False,
                    )
            else:
                # Non-checkpointed: activations stored normally
                if freqs is not None:
                    h = layer(h, freqs)
                else:
                    h = layer(h)

            # Memory pressure check every 4 layers
            if i % 4 == 0 and torch.cuda.is_available():
                mem_gb = torch.cuda.memory_allocated() / 1e9
                if mem_gb > self.max_memory_gb:
                    print(f"[LayerStreaming] Memory pressure at layer {i}: "
                          f"{mem_gb:.1f} GB > {self.max_memory_gb} GB")

        # Final norm + output
        if self.final_norm is not None:
            h = self.final_norm(h)
        if self.output_proj is not None:
            h = self.output_proj(h)

        return h

    def train_step(
        self,
        batch: tuple,
        loss_fn: Callable,
        scaler: torch.amp.GradScaler,
        optimizer: torch.optim.Optimizer,
        max_grad_norm: float = 1.0,
        device: torch.device = torch.device("cuda"),
    ) -> Dict[str, Any]:
        """Complete train step: forward + backward + optimizer."""
        input_ids, targets = batch
        input_ids = input_ids.to(device)

        with torch.amp.autocast("cuda", dtype=torch.float16):
            logits = self.forward(input_ids)
            loss = loss_fn(logits, batch)

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)

        grad_norm = nn.utils.clip_grad_norm_(
            self.model.parameters(), max_grad_norm
        )

        if torch.isfinite(grad_norm):
            scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad(set_to_none=True)

        return {
            "loss": loss.item(),
            "grad_norm": grad_norm.item() if torch.isfinite(grad_norm) else float("inf"),
            "tokens": input_ids.numel(),
        }

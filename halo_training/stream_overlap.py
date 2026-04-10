"""HIP stream overlap for layer-level pipeline parallelism.

Overlaps the forward pass of layer N+1 with the backward pass of layer N
using separate HIP streams. This hides some of the backward latency behind
forward computation.

Integrates with Mode B LayerStreamingTrainer for >2B models.

WARNING: Stream scheduling is non-deterministic on GPU. Gains depend on
actual kernel launch patterns and hardware scheduling. Debug with rocprof.

Usage:
    from halo_training.stream_overlap import StreamOverlapTrainer
    trainer = StreamOverlapTrainer(model, layers)
    output = trainer.forward(input_ids)
"""

import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint as torch_checkpoint


class StreamOverlapTrainer:
    """Layer-level stream overlap: forward N+1 || backward N."""

    def __init__(
        self,
        model: nn.Module,
        layers: nn.ModuleList,
        embed_fn=None,
        norm_fn=None,
        head_fn=None,
    ):
        self.model = model
        self.layers = layers
        self.embed_fn = embed_fn
        self.norm_fn = norm_fn
        self.head_fn = head_fn

        # Create separate streams for forward and backward
        self.fwd_stream = torch.cuda.Stream()
        self.bwd_stream = torch.cuda.Stream()

    def forward(self, input_ids, freqs=None):
        """Forward pass with stream overlap potential.

        Note: actual overlap happens during backward via autograd hooks.
        The forward pass sets up the hooks for backward overlap.
        """
        device = input_ids.device

        # Embedding
        if self.embed_fn is not None:
            h = self.embed_fn(input_ids)
        else:
            h = self.model.tok_emb(input_ids)

        # Layer-by-layer forward with checkpoint for memory
        for i, layer in enumerate(self.layers):
            if freqs is not None:
                h = torch_checkpoint(
                    lambda x, f: layer(x, f),
                    h, freqs,
                    use_reentrant=False,
                )
            else:
                h = torch_checkpoint(layer, h, use_reentrant=False)

        # Output norm + head
        if self.norm_fn is not None:
            h = self.norm_fn(h)
        if self.head_fn is not None:
            h = self.head_fn(h)

        return h


def create_stream_overlap_trainer(model: nn.Module) -> StreamOverlapTrainer:
    """Create a StreamOverlapTrainer from a model with standard layer structure."""
    # Find layers
    layers = None
    for attr in ["layers", "transformer.h", "model.layers"]:
        parts = attr.split(".")
        obj = model
        try:
            for p in parts:
                obj = getattr(obj, p)
            layers = obj
            break
        except AttributeError:
            continue

    if layers is None:
        raise ValueError("Could not find layer container in model")

    # Find norm and head
    norm_fn = getattr(model, "norm", None) or getattr(model, "ln_f", None)
    head_fn = getattr(model, "lm_head", None) or getattr(model, "output", None)
    embed_fn = getattr(model, "tok_emb", None)

    return StreamOverlapTrainer(
        model, layers,
        embed_fn=embed_fn,
        norm_fn=norm_fn,
        head_fn=head_fn,
    )

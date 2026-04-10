"""
VIRTUOSO: Composable PLE + MatFormer Wrapper.

Wraps any base architecture (Tempest, Amadeus, Prometheus) with optional
Per-Layer Embeddings and MatFormer nested submodel training. Base model
code is never modified — PLE injects in the outer loop, MatFormer wraps
FFN layers via module surgery.

Usage:
    # Tempest + PLE(a+b) + MatFormer
    python -m halo_training --model models/virtuoso.py --class-name Virtuoso --dataset babylm

    # PLE ablation: mode="a" only
    python -m halo_training --model models/virtuoso.py --class-name VirtuosoPleA --dataset babylm
"""

import torch
import torch.nn as nn

from models.tempest import Tempest, SwiGLU, RMSNorm
from models.ple import PLEModule, PLEConfig
from models.matformer import MatFormerSwiGLU, MatFormerConfig


class Virtuoso(nn.Module):
    """Composable wrapper: base model + optional PLE + optional MatFormer.

    Default: Tempest base + PLE(a+b) + MatFormer. ~244M params.
    """

    def __init__(
        self,
        # Base model args (Tempest defaults)
        vocab_size: int = 50257,
        d_model: int = 1024,
        n_layers: int = 16,
        d_conv: int = 640,
        d_griffin: int = 384,
        ffn_inner: int = 2560,
        conv_kernel: int = 3,
        momentum_beta_init: float = 0.5,
        max_seq_len: int = 1024,
        # PLE args
        use_ple: bool = True,
        ple_mode: str = "a+b",
        ple_dim: int = 64,
        ple_table_rank: int = 32,
        ple_table_dim: int = 64,
        # MatFormer args
        use_matformer: bool = True,
        matformer_granularities: tuple = (0.125, 0.25, 0.5, 1.0),
    ):
        super().__init__()
        self.use_ple = use_ple
        self.use_matformer = use_matformer

        # Build base model
        self.base = Tempest(
            vocab_size=vocab_size, d_model=d_model, n_layers=n_layers,
            d_conv=d_conv, d_griffin=d_griffin, ffn_inner=ffn_inner,
            conv_kernel=conv_kernel, momentum_beta_init=momentum_beta_init,
            max_seq_len=max_seq_len,
        )

        # Attach PLE
        if use_ple:
            self.ple = PLEModule(PLEConfig(
                vocab_size=vocab_size, d_model=d_model, n_layers=n_layers,
                ple_mode=ple_mode, ple_dim=ple_dim,
                ple_table_rank=ple_table_rank, ple_table_dim=ple_table_dim,
            ))

        # Wrap FFN layers with MatFormer
        if use_matformer:
            mf_config = MatFormerConfig(granularities=matformer_granularities)
            self._wrap_ffn_layers(mf_config)

        n_params = sum(p.numel() for p in self.parameters())
        components = []
        if use_ple:
            components.append(f"PLE({ple_mode})")
        if use_matformer:
            g_str = ",".join(f"{g}" for g in matformer_granularities)
            components.append(f"MatFormer({g_str})")
        comp_str = " + ".join(components) if components else "base only"
        print(f"Virtuoso: {n_params / 1e6:.1f}M params ({comp_str})")

    def _wrap_ffn_layers(self, config: MatFormerConfig):
        """Replace every SwiGLU in base model with MatFormerSwiGLU."""
        replacements = []
        for name, module in self.base.named_modules():
            if isinstance(module, SwiGLU):
                replacements.append((name, module))

        for name, module in replacements:
            parts = name.split(".")
            parent = self.base
            for part in parts[:-1]:
                parent = getattr(parent, part)
            wrapped = MatFormerSwiGLU.from_swiglu(module, config)
            setattr(parent, parts[-1], wrapped)

    def forward(self, input_ids: torch.Tensor, targets=None) -> torch.Tensor:
        B, T = input_ids.shape
        h = self.base.tok_embeddings(input_ids)
        velocity = torch.zeros_like(h)

        for i, layer in enumerate(self.base.layers):
            # PLE injection before block
            if self.use_ple:
                h = h + self.ple(h, input_ids, i)

            # Tempest block: returns (h, velocity)
            h, velocity = layer(h, velocity)

        logits = self.base.output(self.base.norm(h))
        return logits


# --- Convenience classes for the 8-run experimental matrix ---

class VirtuosoPleA(Virtuoso):
    """Base + PLE(mode=a) only. Context-aware projection, no token-identity table."""
    def __init__(self, **kwargs):
        kwargs.setdefault("use_ple", True)
        kwargs.setdefault("ple_mode", "a")
        kwargs.setdefault("use_matformer", False)
        super().__init__(**kwargs)


class VirtuosoPleB(Virtuoso):
    """Base + PLE(mode=b) only. Factored token-identity table, no context projection."""
    def __init__(self, **kwargs):
        kwargs.setdefault("use_ple", True)
        kwargs.setdefault("ple_mode", "b")
        kwargs.setdefault("use_matformer", False)
        super().__init__(**kwargs)


class VirtuosoPleAB(Virtuoso):
    """Base + PLE(mode=a+b). Both paths combined."""
    def __init__(self, **kwargs):
        kwargs.setdefault("use_ple", True)
        kwargs.setdefault("ple_mode", "a+b")
        kwargs.setdefault("use_matformer", False)
        super().__init__(**kwargs)


class VirtuosoMatFormer(Virtuoso):
    """Base + MatFormer only. No PLE."""
    def __init__(self, **kwargs):
        kwargs.setdefault("use_ple", False)
        kwargs.setdefault("use_matformer", True)
        super().__init__(**kwargs)


class VirtuosoFull(Virtuoso):
    """Base + PLE(a+b) + MatFormer. Full composition."""
    def __init__(self, **kwargs):
        kwargs.setdefault("use_ple", True)
        kwargs.setdefault("ple_mode", "a+b")
        kwargs.setdefault("use_matformer", True)
        super().__init__(**kwargs)

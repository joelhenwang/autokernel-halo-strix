"""
LAZARUS-lite: Lightweight TTT Fast Weights on AMADEUS backbone.

Stripped-down LAZARUS: no momentum, no chunk loop, only 2 TTT layers.
Targets AMADEUS-equivalent throughput (~13K tok/s) with adaptive FFN.

The TTT mechanism: on layers with ttt_enabled, the FFN's down-projection
is augmented with a fast weight ΔW that adapts per-sequence via damped
outer-product accumulation: ΔW = γ·ΔW + η·V̂ᵀ·Z.

Usage:
    python -m halo_training --model models/lazarus_lite.py --class-name LazarusLite --dataset babylm
    python -m halo_training --model models/lazarus_lite.py --class-name LazarusLite --smoke
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.amadeus import (
    AmadeusConfig, RMSNorm, SwiGLU, GatedConv, Mamba3SISO,
    FiLMConditioner, ParallelHybridBlock,
)

try:
    from causal_conv1d import causal_conv1d_fn
    _HAS_CAUSAL_CONV1D = True
except ImportError:
    _HAS_CAUSAL_CONV1D = False


class LivingSwiGLU(nn.Module):
    """SwiGLU with fast weight on w_down: out = w_down(z) + z @ ΔW^T."""

    def __init__(self, d_model: int, ffn_inner: int):
        super().__init__()
        self.d_model = d_model
        self.ffn_inner = ffn_inner
        self._skip_autokernel = True  # prevent FusedSwiGLU pattern from replacing
        self.w_gate_up = nn.Linear(d_model, 2 * ffn_inner, bias=False)
        self.w_down = nn.Linear(ffn_inner, d_model, bias=False)
        self.w_target = nn.Linear(d_model, ffn_inner, bias=False)
        self.log_gamma = nn.Parameter(torch.tensor(0.0))    # sigmoid(0) = 0.5
        self.log_eta = nn.Parameter(torch.tensor(-4.0))     # softplus(-4) ≈ 0.018

    @property
    def gamma(self):
        return torch.sigmoid(self.log_gamma)

    @property
    def eta(self):
        return F.softplus(self.log_eta)

    def forward(self, x, v_hat=None, delta_W=None):
        gate, up = self.w_gate_up(x).chunk(2, dim=-1)
        z = F.silu(gate) * up
        out = self.w_down(z)

        if delta_W is not None and v_hat is not None:
            out = out + z @ delta_W
            v_proj = self.w_target(v_hat)
            update = torch.einsum('btf,btd->fd', v_proj, x) / x.shape[0]
            delta_W = self.gamma * delta_W + self.eta * update

        return out, delta_W


class NTPTargetEncoder(nn.Module):
    """Causal conv1d on embeddings → NTP-aligned targets. Zero-init start."""

    def __init__(self, d_model: int, kernel_size: int = 4):
        super().__init__()
        self.d_model = d_model
        self.kernel_size = kernel_size
        self.conv_weight = nn.Parameter(torch.zeros(d_model, kernel_size))
        self.conv_bias = nn.Parameter(torch.zeros(d_model))

    def forward(self, embeddings):
        if _HAS_CAUSAL_CONV1D:
            return causal_conv1d_fn(
                embeddings.transpose(1, 2),
                self.conv_weight, self.conv_bias,
            ).transpose(1, 2)
        x = embeddings.transpose(1, 2)
        x = F.pad(x, (self.kernel_size - 1, 0))
        x = F.conv1d(x, self.conv_weight.unsqueeze(1), self.conv_bias,
                     groups=self.d_model)
        return x.transpose(1, 2)


class LazarusBlock(nn.Module):
    """AMADEUS block with optional TTT on FFN. No momentum — standard residual."""

    def __init__(self, cfg: AmadeusConfig, ttt_enabled: bool = False):
        super().__init__()
        self.pre_norm = RMSNorm(cfg.d_model)
        self.conv = GatedConv(cfg.d_model, cfg.d_conv, cfg.conv_kernel)
        self.ssm = Mamba3SISO(cfg.d_model, cfg.d_mamba, cfg.dstate, cfg.n_ssm_heads)
        self.out_proj = nn.Linear(cfg.d_model, cfg.d_model, bias=False)
        self.ffn_norm = RMSNorm(cfg.d_model)
        self.ttt_enabled = ttt_enabled

        if ttt_enabled:
            self.ffn = LivingSwiGLU(cfg.d_model, cfg.ffn_inner)
        else:
            self.ffn = SwiGLU(cfg.d_model, cfg.ffn_inner)

    def forward(self, x, v_hat=None, delta_W=None):
        # Mixer: parallel conv + SSM (same as AMADEUS)
        x_norm = self.pre_norm(x)
        conv_out = self.conv(x_norm)
        ssm_out = self.ssm(x_norm)
        x = x + self.out_proj(torch.cat([conv_out, ssm_out], dim=-1))

        # FFN with optional TTT
        ffn_in = self.ffn_norm(x)
        if self.ttt_enabled:
            ffn_out, delta_W = self.ffn(ffn_in, v_hat, delta_W)
        else:
            ffn_out = self.ffn(ffn_in)
        x = x + ffn_out

        return x, delta_W


class LazarusLite(nn.Module):
    """Lightweight LAZARUS: AMADEUS backbone + 2 TTT layers, no momentum, no chunk loop.

    ~160M parameters. Targets AMADEUS-equivalent throughput.
    """

    def __init__(
        self,
        vocab_size: int = 50257,
        d_model: int = 1024,
        n_layers: int = 10,
        d_conv: int = 640,
        d_mamba: int = 384,
        dstate: int = 64,
        n_ssm_heads: int = 6,
        ffn_inner: int = 2048,
        d_film: int = 64,
        film_start: int = 5,
        max_seq_len: int = 1024,
        conv_kernel: int = 3,
        ttt_layers: tuple = (4, 9),  # 0-indexed: layers 5 and 10 (mid + final)
    ):
        super().__init__()
        assert d_conv + d_mamba == d_model

        self.d_model = d_model
        self.ffn_inner = ffn_inner
        self.n_layers = n_layers
        self.film_start = film_start
        self.ttt_layers = set(ttt_layers)

        cfg = AmadeusConfig(
            vocab_size=vocab_size, d_model=d_model, n_layers=n_layers,
            d_conv=d_conv, d_mamba=d_mamba, dstate=dstate,
            n_ssm_heads=n_ssm_heads, ffn_inner=ffn_inner,
            d_film=d_film, film_start=film_start,
            max_seq_len=max_seq_len, conv_kernel=conv_kernel,
        )

        self.tok_embeddings = nn.Embedding(vocab_size, d_model)
        self.layers = nn.ModuleList([
            LazarusBlock(cfg, ttt_enabled=(i in self.ttt_layers))
            for i in range(n_layers)
        ])
        self.film = FiLMConditioner(d_model, d_film, n_layers - film_start)
        self.target_encoder = NTPTargetEncoder(d_model)
        self.norm = RMSNorm(d_model)
        self.output = nn.Linear(d_model, vocab_size, bias=False)
        self.output.weight = self.tok_embeddings.weight

        self._init_weights()
        n_params = sum(p.numel() for p in self.parameters())
        print(f"LazarusLite: {n_params / 1e6:.1f}M parameters")

    def _init_weights(self):
        for name, p in self.named_parameters():
            if p.dim() >= 2:
                nn.init.xavier_uniform_(p)
        nn.init.zeros_(self.target_encoder.conv_weight)

    def forward(self, input_ids: torch.Tensor, targets=None) -> torch.Tensor:
        B, T = input_ids.shape
        h = self.tok_embeddings(input_ids)

        # NTP targets computed once from embeddings
        v_hat = self.target_encoder(h)

        # Initialize fast weight deltas (one per TTT layer)
        delta_Ws = {
            idx: torch.zeros(self.ffn_inner, self.d_model,
                             device=h.device, dtype=torch.float32)
            for idx in self.ttt_layers
        }

        # Single pass — no chunk loop (seq length ≤ chunk size)
        context = None
        for i, layer in enumerate(self.layers):
            if i == self.film_start:
                context = self.film.compute_context(h)
            if i >= self.film_start and context is not None:
                h = self.film.apply(h, context, i - self.film_start)

            dW = delta_Ws.get(i)
            h, new_dW = layer(h, v_hat, dW)
            if i in self.ttt_layers and new_dW is not None:
                delta_Ws[i] = new_dW

        return self.output(self.norm(h))

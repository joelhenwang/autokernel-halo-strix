"""
LAZARUS-A: AMADEUS backbone + TTT Fast Weights in FFN.

TTT (Test-Time Training) adapts FFN w_down weights per-chunk via damped
outer-product accumulation: ΔW = γ·ΔW + η·V̂ᵀ·Z. The ΔW matrix (5.2MB fp16)
fits in L2 cache on Strix Halo. Zero-init bootstrap: starts as vanilla AMADEUS,
TTT activates organically as conv target weights learn useful signals.

Triple dynamical system:
  - INERTIA: velocity = β·velocity + mixer_out (momentum residual)
  - PLASTICITY: ΔW = γ·ΔW + η·V̂ᵀ·Z (fast weight adaptation)
  - DECAY: γ < 1 (dreams fade, only useful adaptations survive)

Usage:
    python -m halo_training --model models/lazarus.py --class-name LazarusA --dataset babylm
    python -m halo_training --model models/lazarus.py --class-name LazarusA --smoke
"""

import math
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

# Import AMADEUS backbone components
from models.amadeus import (
    AmadeusConfig, RMSNorm, SwiGLU, GatedConv, Mamba3SISO, FiLMConditioner,
)

# Fast causal conv1d backend (10x vs nn.Conv1d)
try:
    from causal_conv1d import causal_conv1d_fn
    _HAS_CAUSAL_CONV1D = True
except ImportError:
    _HAS_CAUSAL_CONV1D = False


# ---------------------------------------------------------------------------
# TTT Components
# ---------------------------------------------------------------------------

class LivingSwiGLU(nn.Module):
    """SwiGLU where w_down is augmented with a fast weight that adapts per chunk.

    Standard SwiGLU:  out = w_down(silu(gate) * up)
    Living SwiGLU:    out = w_down(z) + z @ ΔW^T
                      where ΔW accumulates: ΔW = γ·ΔW + η·mean_B(V̂^T · x)

    The ΔW matrix (ffn_inner × d_model = 2560 × 1024 = 5.2MB fp16) fits in L2 cache.
    """

    def __init__(self, d_model: int, ffn_inner: int):
        super().__init__()
        self.d_model = d_model
        self.ffn_inner = ffn_inner
        self._skip_autokernel = True  # prevent FusedSwiGLU pattern from replacing this
        self.w_gate_up = nn.Linear(d_model, 2 * ffn_inner, bias=False)
        self.w_down = nn.Linear(ffn_inner, d_model, bias=False)  # frozen "slow weight"

        # W_target: projects NTP-aligned target to FFN inner dim
        self.w_target = nn.Linear(d_model, ffn_inner, bias=False)

        # Learned damping and learning rate (constrained to safe ranges)
        self.log_gamma = nn.Parameter(torch.tensor(0.0))   # sigmoid(0) = 0.5
        self.log_eta = nn.Parameter(torch.tensor(-4.0))    # softplus(-4) ≈ 0.018 (conservative start)

    @property
    def gamma(self):
        return torch.sigmoid(self.log_gamma)  # ∈ (0, 1) — damping factor

    @property
    def eta(self):
        return F.softplus(self.log_eta)       # ∈ (0, ∞) — adaptation rate

    def forward(self, x, v_hat_chunk=None, delta_W=None):
        """
        Args:
            x: (B, C, d_model) — chunk of hidden states
            v_hat_chunk: (B, C, d_model) — NTP-aligned target for this chunk (or None)
            delta_W: (ffn_inner, d_model) — current fast weight delta (or None)
        Returns:
            out: (B, C, d_model)
            new_delta_W: updated fast weight delta
        """
        gate, up = self.w_gate_up(x).chunk(2, dim=-1)
        z = F.silu(gate) * up                          # (B, C, ffn_inner)

        # Standard path through frozen w_down
        out = self.w_down(z)                            # (B, C, d_model)

        if delta_W is not None and v_hat_chunk is not None:
            # Fast weight contribution: z @ ΔW^T  (ΔW is ffn_inner × d_model)
            out = out + z @ delta_W                     # (B, C, d_model)

            # Compute NTP-aligned target in FFN space
            v_hat_proj = self.w_target(v_hat_chunk)     # (B, C, ffn_inner)

            # Update ΔW: damped outer product accumulation
            # ΔW_new = γ·ΔW + η·mean_B(V̂_proj^T · x)
            update = torch.einsum('bcf,bcd->fd', v_hat_proj, x) / x.shape[0]
            new_delta_W = self.gamma * delta_W + self.eta * update

            return out, new_delta_W

        return out, delta_W


class NTPTargetEncoder(nn.Module):
    """Compute V̂ from token embeddings via causal conv1d.

    Computed ONCE from embeddings at start of forward pass.
    CausalConv1D blends neighboring token info (kernel=5, causal padding).
    Zero-init: starts as identity (no TTT contribution initially).
    """

    def __init__(self, d_model: int, kernel_size: int = 5):
        super().__init__()
        self.d_model = d_model
        self.kernel_size = kernel_size

        if _HAS_CAUSAL_CONV1D:
            self.conv_weight = nn.Parameter(torch.zeros(d_model, kernel_size))
            self.conv_bias = nn.Parameter(torch.zeros(d_model))
        else:
            # Manual causal conv1d fallback
            self.conv_weight = nn.Parameter(torch.zeros(d_model, kernel_size))
            self.conv_bias = nn.Parameter(torch.zeros(d_model))

    def forward(self, embeddings: torch.Tensor) -> torch.Tensor:
        """embeddings: (B, T, d_model) → v_hat: (B, T, d_model)"""
        if _HAS_CAUSAL_CONV1D:
            v_hat = causal_conv1d_fn(
                embeddings.transpose(1, 2),
                self.conv_weight,
                self.conv_bias,
            ).transpose(1, 2)
        else:
            # Manual causal conv: left-pad by kernel_size-1
            x = embeddings.transpose(1, 2)  # (B, D, T)
            x = F.pad(x, (self.kernel_size - 1, 0))
            # Depthwise conv: each channel independently
            x = F.conv1d(x, self.conv_weight.unsqueeze(1), self.conv_bias,
                         groups=self.d_model)
            v_hat = x.transpose(1, 2)
        return v_hat


class MomentumResidual(nn.Module):
    """Residual connection with depth-wise momentum (from TEMPEST).

    velocity = beta * velocity + layer_output
    h = h + velocity
    """

    def __init__(self, beta_init: float = 0.5):
        super().__init__()
        self.log_beta = nn.Parameter(
            torch.tensor(math.log(beta_init / (1 - beta_init)))
        )

    @property
    def beta(self):
        return torch.sigmoid(self.log_beta)

    def forward(self, h, layer_output, velocity):
        velocity = self.beta * velocity + layer_output
        return h + velocity, velocity


# ---------------------------------------------------------------------------
# LAZARUS Block
# ---------------------------------------------------------------------------

class LazarusBlock(nn.Module):
    """Single LAZARUS layer: parallel conv + SSM → momentum residual → Living/Standard SwiGLU."""

    def __init__(self, cfg: AmadeusConfig, ttt_enabled: bool = False):
        super().__init__()
        self.pre_norm = RMSNorm(cfg.d_model)
        self.conv = GatedConv(cfg.d_model, cfg.d_conv, cfg.conv_kernel)
        self.ssm = Mamba3SISO(cfg.d_model, cfg.d_mamba, cfg.dstate, cfg.n_ssm_heads)
        self.out_proj = nn.Linear(cfg.d_model, cfg.d_model, bias=False)
        self.momentum = MomentumResidual()
        self.ffn_norm = RMSNorm(cfg.d_model)
        self.ttt_enabled = ttt_enabled

        if ttt_enabled:
            self.ffn = LivingSwiGLU(cfg.d_model, cfg.ffn_inner)
        else:
            self.ffn = SwiGLU(cfg.d_model, cfg.ffn_inner)

    def forward(self, x, velocity, v_hat_chunk=None, delta_W=None):
        """
        Args:
            x: (B, C, d_model)
            velocity: (B, C, d_model) — momentum state
            v_hat_chunk: (B, C, d_model) — NTP target (only for TTT layers)
            delta_W: (ffn_inner, d_model) — fast weight delta (only for TTT layers)
        Returns:
            x, velocity, new_delta_W
        """
        # Mixer: parallel conv + SSM
        x_norm = self.pre_norm(x)
        conv_out = self.conv(x_norm)
        ssm_out = self.ssm(x_norm)
        mixed = self.out_proj(torch.cat([conv_out, ssm_out], dim=-1))

        # Momentum residual on mixer
        x, velocity = self.momentum(x, mixed, velocity)

        # FFN with optional TTT
        ffn_in = self.ffn_norm(x)
        if self.ttt_enabled:
            ffn_out, delta_W = self.ffn(ffn_in, v_hat_chunk, delta_W)
        else:
            ffn_out = self.ffn(ffn_in)

        # Standard residual on FFN
        x = x + ffn_out

        return x, velocity, delta_W


# ---------------------------------------------------------------------------
# LAZARUS-A Model
# ---------------------------------------------------------------------------

class LazarusA(nn.Module):
    """LAZARUS-A: AMADEUS backbone + TTT Fast Weights + Momentum Residual.

    16 layers, TTT on layers 4/8/12/16 (0-indexed: 3/7/11/15).
    FiLM conditioning on layers 9-16 (same as AMADEUS).
    Chunk-based processing: ΔW persists across chunks within a sequence.

    ~223M parameters.
    """

    def __init__(
        self,
        vocab_size: int = 50257,
        d_model: int = 1024,
        n_layers: int = 16,
        d_conv: int = 640,
        d_mamba: int = 384,
        dstate: int = 64,
        n_ssm_heads: int = 6,
        ffn_inner: int = 2432,  # reduced from 2560 to fit <250M with TTT overhead
        d_film: int = 64,
        film_start: int = 8,
        max_seq_len: int = 1024,
        conv_kernel: int = 3,
        chunk_size: int = 512,
        ttt_layers: tuple = (3, 7, 11, 15),  # 0-indexed: layers 4, 8, 12, 16
        ntp_conv_kernel: int = 4,  # causal_conv1d supports width 2-4
    ):
        super().__init__()
        assert d_conv + d_mamba == d_model

        self.d_model = d_model
        self.ffn_inner = ffn_inner
        self.n_layers = n_layers
        self.film_start = film_start
        self.chunk_size = chunk_size
        self.ttt_layers = set(ttt_layers)

        cfg = AmadeusConfig(
            vocab_size=vocab_size, d_model=d_model, n_layers=n_layers,
            d_conv=d_conv, d_mamba=d_mamba, dstate=dstate,
            n_ssm_heads=n_ssm_heads, ffn_inner=ffn_inner,
            d_film=d_film, film_start=film_start,
            max_seq_len=max_seq_len, conv_kernel=conv_kernel,
        )

        self.tok_embeddings = nn.Embedding(vocab_size, d_model)

        # Build layers: TTT-enabled on specified layers
        self.layers = nn.ModuleList([
            LazarusBlock(cfg, ttt_enabled=(i in self.ttt_layers))
            for i in range(n_layers)
        ])

        # FiLM conditioning (same as AMADEUS)
        self.film = FiLMConditioner(d_model, d_film, n_layers - film_start)

        # NTP target encoder (computed once from embeddings)
        self.target_encoder = NTPTargetEncoder(d_model, ntp_conv_kernel)

        self.norm = RMSNorm(d_model)
        self.output = nn.Linear(d_model, vocab_size, bias=False)
        self.output.weight = self.tok_embeddings.weight  # weight tying

        self._init_weights()
        n_params = sum(p.numel() for p in self.parameters())
        print(f"LazarusA: {n_params / 1e6:.1f}M parameters")

    def _init_weights(self):
        for name, p in self.named_parameters():
            if p.dim() >= 2:
                nn.init.xavier_uniform_(p)
        # Zero-init NTP target encoder conv weights (TTT starts as identity)
        nn.init.zeros_(self.target_encoder.conv_weight)

    def forward(self, input_ids: torch.Tensor, targets=None) -> torch.Tensor:
        B, T = input_ids.shape
        h = self.tok_embeddings(input_ids)

        # Compute NTP-aligned targets ONCE from raw embeddings
        v_hat = self.target_encoder(h)  # (B, T, d_model)

        # Initialize states
        velocity = torch.zeros_like(h)

        # Initialize fast weight deltas to zero (one per TTT layer)
        delta_Ws = {
            idx: torch.zeros(self.ffn_inner, self.d_model,
                             device=h.device, dtype=torch.float32)
            for idx in self.ttt_layers
        }

        # Process sequence in chunks (ΔW persists across chunks)
        n_chunks = (T + self.chunk_size - 1) // self.chunk_size
        for c in range(n_chunks):
            cs = c * self.chunk_size
            ce = min((c + 1) * self.chunk_size, T)
            h_c = h[:, cs:ce].clone()
            v_c = v_hat[:, cs:ce]
            vel_c = velocity[:, cs:ce].clone()

            # FiLM context (computed per chunk from chunk hidden states)
            context = None

            for i, layer in enumerate(self.layers):
                # FiLM conditioning
                if i == self.film_start:
                    context = self.film.compute_context(h_c)
                if i >= self.film_start and context is not None:
                    h_c = self.film.apply(h_c, context, i - self.film_start)

                # Layer forward with optional TTT
                dW = delta_Ws.get(i)
                h_c, vel_c, new_dW = layer(h_c, vel_c, v_c, dW)

                # Update persistent ΔW
                if i in self.ttt_layers and new_dW is not None:
                    delta_Ws[i] = new_dW

            # Write chunk results back
            h[:, cs:ce] = h_c
            velocity[:, cs:ce] = vel_c

        logits = self.output(self.norm(h))
        return logits


# --- Mini config for smoke testing ---
class LazarusAMini(LazarusA):
    """Tiny LAZARUS-A for smoke testing (d=128, 4 layers, ~2.5M params)."""

    def __init__(self):
        super().__init__(
            vocab_size=1000,
            d_model=128,
            n_layers=4,
            d_conv=80,
            d_mamba=48,
            dstate=16,
            n_ssm_heads=3,
            ffn_inner=256,
            d_film=16,
            film_start=2,
            max_seq_len=128,
            chunk_size=64,
            ttt_layers=(1, 3),  # layers 2 and 4 (0-indexed)
            ntp_conv_kernel=4,
        )

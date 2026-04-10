"""
AMADEUS: Parallel Hybrid (Gated Conv + Mamba-3 SISO) + FiLM Conditioning.

Hypothesis: The simplest competitive architecture combining local pattern
matching (gated conv) with global context tracking (Mamba-3 SSM), plus
FiLM conditioning from a midpoint context fingerprint.

~241M parameters. Targets AMD Strix Halo (gfx1151, RDNA 3.5).

Usage:
    # Full scale (241M)
    python -m halo_training --model models/amadeus.py --class-name Amadeus --dataset babylm

    # Smoke test
    python -m halo_training --model models/amadeus.py --class-name Amadeus --smoke

    # With autokernel + compile (best performance)
    python -m halo_training --model models/amadeus.py --class-name Amadeus \\
        --compile --optimize-kernels --dataset babylm
"""

import math
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

# Fast causal conv1d backend (10x vs nn.Conv1d)
try:
    from causal_conv1d import causal_conv1d_fn
    _HAS_CAUSAL_CONV1D = True
except ImportError:
    _HAS_CAUSAL_CONV1D = False

# Fast selective scan backend (5.6x vs HIP kernel)
try:
    from mamba_ssm.ops.selective_scan_interface import selective_scan_fn as _mamba_ssm_scan_fn
    _HAS_MAMBA_SSM = True
except ImportError:
    _HAS_MAMBA_SSM = False


@dataclass
class AmadeusConfig:
    vocab_size: int = 50257
    d_model: int = 1024
    n_layers: int = 16
    d_conv: int = 640       # gated conv channels (10 heads × 64)
    d_mamba: int = 384      # SSM channels (6 heads × 64)
    dstate: int = 64        # SSM state dimension
    n_ssm_heads: int = 6    # number of SSM heads
    ffn_inner: int = 2560   # SwiGLU expansion (2.5×)
    d_film: int = 64        # FiLM fingerprint dimension
    film_start: int = 8     # first layer with FiLM conditioning
    max_seq_len: int = 1024
    conv_kernel: int = 3


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        norm = torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        return x * norm * self.weight


class SwiGLU(nn.Module):
    """SwiGLU FFN with fused gate+up projection."""

    def __init__(self, d_model: int, ffn_inner: int):
        super().__init__()
        self.w_gate_up = nn.Linear(d_model, 2 * ffn_inner, bias=False)
        self.w_down = nn.Linear(ffn_inner, d_model, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gate, up = self.w_gate_up(x).chunk(2, dim=-1)
        return self.w_down(F.silu(gate) * up)


class GatedConv(nn.Module):
    """Gated short convolution for local pattern matching.

    B, C, h_tilde = proj(x).chunk(3)
    y = B * h_tilde  (element-wise gate)
    z = causal_conv1d(y)
    out = C * z       (output gate)
    """

    def __init__(self, d_model: int, d_conv: int, kernel_size: int = 3):
        super().__init__()
        self.d_conv = d_conv
        self.kernel_size = kernel_size
        self.proj = nn.Linear(d_model, 3 * d_conv, bias=False)
        if _HAS_CAUSAL_CONV1D:
            # causal_conv1d_fn expects weight (D, K) and optional bias (D,)
            self.conv_weight = nn.Parameter(torch.randn(d_conv, kernel_size))
            self.conv_bias = nn.Parameter(torch.zeros(d_conv))
        else:
            # Depthwise causal conv: left-pad by (kernel_size - 1)
            self.conv = nn.Conv1d(
                d_conv, d_conv, kernel_size=kernel_size,
                padding=kernel_size - 1, groups=d_conv, bias=True,
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, _ = x.shape
        b, c, h_tilde = self.proj(x).chunk(3, dim=-1)   # each (B, T, d_conv)
        y = b * h_tilde                                   # gate
        if _HAS_CAUSAL_CONV1D:
            # causal_conv1d_fn: input (B, D, L), weight (D, K) → output (B, D, L)
            z = causal_conv1d_fn(
                y.transpose(1, 2), self.conv_weight, self.conv_bias
            ).transpose(1, 2)
        else:
            # Conv expects (B, C, T) layout
            z = self.conv(y.transpose(1, 2))[:, :, :T]    # causal: truncate future
            z = z.transpose(1, 2)                          # back to (B, T, d_conv)
        return c * z                                       # output gate


_HIP_SCAN_AVAILABLE = None  # lazy check


def _mamba_ssm_scan(x, dt, A_log, B, C, D, n_heads):
    """Wrapper to call mamba-ssm selective_scan_fn with our tensor layout.

    Our layout: x (B, T, D), dt (B, T, D), A_log (D,), B (B, T, N), C (B, T, N), D (D,)
    mamba-ssm: u (B, D, L), delta (B, D, L), A (D, N), B (B, N, L), C (B, N, L), D (D,)
    """
    batch, seqlen, d_inner = x.shape
    dstate = B.shape[-1]

    u = x.transpose(1, 2).contiguous()          # (B, D, L)
    delta = dt.transpose(1, 2).contiguous()      # (B, D, L)
    A = -torch.exp(A_log.float()).view(d_inner // dstate if d_inner > dstate else d_inner, dstate)
    # Reshape A to (D, N): if d_inner = n_heads * dstate, A is (n_heads, dstate) → repeat
    # A_log is (D,) = (n_heads * dstate,). mamba-ssm expects A (D, N).
    # Our A is diagonal per-dim. Expand to (D, N) by repeating.
    A = -torch.exp(A_log.float()).unsqueeze(-1).expand(d_inner, dstate)
    B_t = B.transpose(1, 2).contiguous()         # (B, N, L)
    C_t = C.transpose(1, 2).contiguous()         # (B, N, L)
    D_f = D.float()

    y = _mamba_ssm_scan_fn(u.float(), delta.float(), A, B_t.float(), C_t.float(), D_f)
    return y.transpose(1, 2).to(x.dtype)         # (B, T, D)


def _scan_dispatch(x, dt, A_log, B, C, D, n_heads):
    """Try mamba-ssm > HIP fused scan kernel > chunked Python scan."""
    global _HIP_SCAN_AVAILABLE

    # Priority 1: mamba-ssm (5.6x faster)
    if _HAS_MAMBA_SSM and x.is_cuda:
        try:
            return _mamba_ssm_scan(x, dt, A_log, B, C, D, n_heads)
        except Exception:
            pass

    # Priority 2: HIP kernel
    if _HIP_SCAN_AVAILABLE is None:
        try:
            from kernels.hip.selective_scan import kernel_fn as _hip_scan_fn
            _HIP_SCAN_AVAILABLE = True
        except Exception:
            _HIP_SCAN_AVAILABLE = False

    if _HIP_SCAN_AVAILABLE and x.is_cuda:
        try:
            return _hip_selective_scan(x, dt, A_log, B, C, D, n_heads)
        except Exception:
            pass  # fall through to chunked

    # Priority 3: chunked Python
    return selective_scan_chunked(x, dt, A_log, B, C, D, n_heads)


def _hip_selective_scan(x, dt, A_log, B, C, D, n_heads):
    """Wrapper that prepares tensors for the HIP kernel."""
    from kernels.hip.selective_scan import kernel_fn

    B_exp = B.repeat(1, 1, n_heads)
    C_exp = C.repeat(1, 1, n_heads)

    x_f = x.float()
    dt_f = dt.float()
    A_neg = -torch.exp(A_log.float())

    dA = torch.exp(dt_f * A_neg)
    dBx = dt_f * B_exp.float() * x_f
    C_f = C_exp.float()
    D_f = D.float()

    y = kernel_fn(dA, dBx, C_f, D_f, x_f)
    return y.to(x.dtype)


def selective_scan_chunked(x, dt, A_log, B, C, D, n_heads, chunk_size=64):
    """Chunked diagonal selective scan — vectorized within chunks.

    Instead of T serial steps, does T/chunk_size serial chunk steps.
    Within each chunk, uses cumprod + cumsum for parallel computation.
    For T=512, chunk_size=64: only 8 serial steps instead of 512.

    Args:
        x: (batch, seq, d_inner) — SSM input
        dt: (batch, seq, d_inner) — per-dimension discretization step
        A_log: (d_inner,) — log of diagonal transition matrix
        B: (batch, seq, dstate) — data-dependent input projection
        C: (batch, seq, dstate) — data-dependent output projection
        D: (d_inner,) — skip connection
        n_heads: int — B,C are shared across heads, repeated to d_inner
        chunk_size: int — timesteps per chunk (default 64)
    """
    batch, seqlen, d_inner = x.shape

    # Expand B, C from dstate to d_inner
    B_exp = B.repeat(1, 1, n_heads)
    C_exp = C.repeat(1, 1, n_heads)

    # All in fp32
    x_f = x.float()
    dt_f = dt.float()
    A_neg = -torch.exp(A_log.float())
    D_f = D.float()

    # Per-timestep decay and input
    dA = torch.exp(dt_f * A_neg)              # (batch, seq, d_inner)
    dBx = dt_f * B_exp.float() * x_f         # (batch, seq, d_inner)

    # Pad to multiple of chunk_size
    pad = (chunk_size - seqlen % chunk_size) % chunk_size
    if pad > 0:
        dA = F.pad(dA, (0, 0, 0, pad), value=1.0)    # decay=1 for padding (no change)
        dBx = F.pad(dBx, (0, 0, 0, pad), value=0.0)  # input=0 for padding
        C_exp_f = F.pad(C_exp.float(), (0, 0, 0, pad), value=0.0)
        x_pad = F.pad(x_f, (0, 0, 0, pad), value=0.0)
    else:
        C_exp_f = C_exp.float()
        x_pad = x_f

    total_len = seqlen + pad
    n_chunks = total_len // chunk_size

    # Reshape into chunks: (batch, n_chunks, chunk_size, d_inner)
    dA_c = dA.view(batch, n_chunks, chunk_size, d_inner)
    dBx_c = dBx.view(batch, n_chunks, chunk_size, d_inner)
    C_c = C_exp_f.view(batch, n_chunks, chunk_size, d_inner)
    x_c = x_pad.view(batch, n_chunks, chunk_size, d_inner)

    # Within each chunk, compute cumulative decay from the start of the chunk
    # cum_decay[t] = prod(dA[0:t+1]) = product of decays from chunk start to t
    # Using log domain for numerical stability
    log_dA_c = torch.log(dA_c.clamp(min=1e-10))
    cum_log_dA = torch.cumsum(log_dA_c, dim=2)          # (batch, n_chunks, chunk_size, d_inner)
    cum_dA = torch.exp(cum_log_dA)                       # cumulative decay within chunk

    # Within-chunk state contribution: for each position t in chunk,
    # contribution = sum_{s=0}^{t} prod_{j=s+1}^{t} dA[j] * dBx[s]
    # = sum_{s=0}^{t} (cum_dA[t] / cum_dA[s]) * dBx[s] / dA[s]
    # = cum_dA[t] * sum_{s=0}^{t} (dBx[s] / cum_dA[s])
    # But dBx[s]/cum_dA[s] is "dBx weighted by inverse cumulative decay"
    # Compute: weighted_dBx = dBx / cum_dA (element-wise)
    # Then: intra_state[t] = cum_dA[t] * cumsum(weighted_dBx)[t]
    weighted_dBx = dBx_c / cum_dA.clamp(min=1e-10)
    cum_weighted = torch.cumsum(weighted_dBx, dim=2)
    intra_state = cum_dA * cum_weighted                  # (batch, n_chunks, chunk_size, d_inner)

    # Cross-chunk propagation: state at end of chunk k feeds into chunk k+1
    # chunk_total_decay = prod of all dA within the chunk = cum_dA[:, :, -1, :]
    chunk_total_decay = cum_dA[:, :, -1, :]              # (batch, n_chunks, d_inner)

    # Process chunks serially (only n_chunks = 8 steps for T=512)
    state = torch.zeros(batch, d_inner, dtype=torch.float32, device=x.device)
    all_states = []

    for c in range(n_chunks):
        # State contribution from previous chunks, decayed through this chunk
        # prev_contrib[t] = cum_dA[t] * state_prev (for each t in chunk)
        prev_contrib = cum_dA[:, c] * state.unsqueeze(1)  # (batch, chunk_size, d_inner)
        chunk_states = intra_state[:, c] + prev_contrib    # total state at each position
        all_states.append(chunk_states)

        # Update state for next chunk
        state = chunk_states[:, -1]                         # (batch, d_inner)

    # Concatenate all chunk states and compute output
    states = torch.cat(all_states, dim=1)[:, :seqlen]     # (batch, seq, d_inner)
    y = C_exp.float() * states + D_f * x_f                # (batch, seq, d_inner)
    return y.to(x.dtype)


class Mamba3SISO(nn.Module):
    """Mamba-3 Single-Input Single-Output SSM block.

    Diagonal state space model with data-dependent B, C projections.
    6 heads × 64 dims = 384 channels, shared B/C across heads.
    """

    def __init__(self, d_model: int, d_mamba: int, dstate: int, n_heads: int):
        super().__init__()
        self.d_mamba = d_mamba
        self.dstate = dstate
        self.n_heads = n_heads

        self.x_proj = nn.Linear(d_model, d_mamba, bias=False)
        self.dt_proj = nn.Linear(d_model, d_mamba, bias=True)
        self.B_proj = nn.Linear(d_model, dstate, bias=False)
        self.C_proj = nn.Linear(d_model, dstate, bias=False)

        # A_log init: standard Mamba init — log(1..N) per head
        # Gives decay rates from -1 (fast) to -N (very fast)
        A_init = torch.log(
            torch.arange(1, dstate + 1, dtype=torch.float32)
        ).unsqueeze(0).repeat(n_heads, 1)
        self.A_log = nn.Parameter(A_init)

        # dt_proj bias: initialize to produce small dt via softplus
        # softplus(-4) ≈ 0.018 — starts with gentle state updates
        nn.init.constant_(self.dt_proj.bias, -4.0)
        # Scale dt_proj weights small to keep initial dt near softplus(bias)
        nn.init.normal_(self.dt_proj.weight, std=0.001)

        self.D = nn.Parameter(torch.ones(d_mamba))
        self.out_proj = nn.Linear(d_mamba, d_mamba, bias=False)

    def forward(self, x_norm: torch.Tensor) -> torch.Tensor:
        x_ssm = self.x_proj(x_norm)                        # (B, T, 384)
        dt = F.softplus(self.dt_proj(x_norm))               # (B, T, 384)
        dt = dt.clamp(min=1e-4, max=0.5)                   # conservative clamp
        B = self.B_proj(x_norm)                             # (B, T, 64)
        C = self.C_proj(x_norm)                             # (B, T, 64)
        # Normalize B,C to prevent unbounded state growth
        B = B / (B.norm(dim=-1, keepdim=True).clamp(min=1.0))
        C = C / (C.norm(dim=-1, keepdim=True).clamp(min=1.0))

        y = _scan_dispatch(
            x_ssm, dt, self.A_log.flatten(), B, C, self.D, self.n_heads
        )
        return self.out_proj(y)                             # (B, T, 384)


class ParallelHybridBlock(nn.Module):
    """Single AMADEUS layer: parallel conv + SSM, then SwiGLU FFN."""

    def __init__(self, cfg: AmadeusConfig):
        super().__init__()
        self.pre_norm = RMSNorm(cfg.d_model)
        self.conv = GatedConv(cfg.d_model, cfg.d_conv, cfg.conv_kernel)
        self.ssm = Mamba3SISO(cfg.d_model, cfg.d_mamba, cfg.dstate, cfg.n_ssm_heads)
        # Concat (d_conv + d_mamba = d_model) → output projection
        self.out_proj = nn.Linear(cfg.d_model, cfg.d_model, bias=False)
        self.ffn_norm = RMSNorm(cfg.d_model)
        self.ffn = SwiGLU(cfg.d_model, cfg.ffn_inner)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_norm = self.pre_norm(x)
        conv_out = self.conv(x_norm)                        # (B, T, d_conv=640)
        ssm_out = self.ssm(x_norm)                          # (B, T, d_mamba=384)
        mixed = self.out_proj(torch.cat([conv_out, ssm_out], dim=-1))
        x = x + mixed                                       # residual
        x = x + self.ffn(self.ffn_norm(x))                  # FFN residual
        return x


class FiLMConditioner(nn.Module):
    """Feature-wise Linear Modulation from midpoint context fingerprint.

    After layer film_start, computes a 64-dim fingerprint from the hidden
    state. Layers film_start..n_layers-1 receive per-layer gamma/beta
    modulation. Init to identity (gamma=1, beta=0).
    """

    def __init__(self, d_model: int, d_film: int, n_conditioned_layers: int):
        super().__init__()
        self.context_proj = nn.Linear(d_model, d_film, bias=True)
        self.gamma_projs = nn.ModuleList([
            nn.Linear(d_film, d_model, bias=True)
            for _ in range(n_conditioned_layers)
        ])
        self.beta_projs = nn.ModuleList([
            nn.Linear(d_film, d_model, bias=True)
            for _ in range(n_conditioned_layers)
        ])
        self._init_identity()

    def _init_identity(self):
        """Init so gamma≈1, beta≈0 (FiLM starts as identity transform)."""
        for proj in self.gamma_projs:
            nn.init.zeros_(proj.weight)
            nn.init.zeros_(proj.bias)
        for proj in self.beta_projs:
            nn.init.zeros_(proj.weight)
            nn.init.zeros_(proj.bias)

    def compute_context(self, h: torch.Tensor) -> torch.Tensor:
        """Compute context fingerprint from hidden state."""
        return self.context_proj(h.mean(dim=1))             # (B, d_film)

    def apply(self, h: torch.Tensor, context: torch.Tensor, layer_idx: int) -> torch.Tensor:
        """Apply FiLM modulation: gamma * h + beta."""
        gamma = self.gamma_projs[layer_idx](context) + 1.0  # (B, d_model)
        beta = self.beta_projs[layer_idx](context)           # (B, d_model)
        return gamma.unsqueeze(1) * h + beta.unsqueeze(1)


class Amadeus(nn.Module):
    """AMADEUS: Parallel Hybrid + FiLM Conditioning.

    Architecture:
        16 Parallel Hybrid Blocks (gated conv || Mamba-3 SISO → concat → proj → SwiGLU)
        FiLM conditioning on layers 9-16 from layer 8 fingerprint
        Weight-tied embedding/LM head

    Config: d=1024, 16 layers, conv=640, mamba=384, ffn=2560, FiLM d=64
    ~241M parameters.
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
        ffn_inner: int = 2560,
        d_film: int = 64,
        film_start: int = 8,
        max_seq_len: int = 1024,
        conv_kernel: int = 3,
        adaptive_head: bool = False,
    ):
        super().__init__()
        assert d_conv + d_mamba == d_model, (
            f"d_conv({d_conv}) + d_mamba({d_mamba}) must equal d_model({d_model})"
        )

        self.film_start = film_start
        self.n_layers = n_layers
        self.adaptive_head = adaptive_head

        cfg = AmadeusConfig(
            vocab_size=vocab_size, d_model=d_model, n_layers=n_layers,
            d_conv=d_conv, d_mamba=d_mamba, dstate=dstate,
            n_ssm_heads=n_ssm_heads, ffn_inner=ffn_inner,
            d_film=d_film, film_start=film_start,
            max_seq_len=max_seq_len, conv_kernel=conv_kernel,
        )

        self.tok_embeddings = nn.Embedding(vocab_size, d_model)
        self.layers = nn.ModuleList([
            ParallelHybridBlock(cfg) for _ in range(n_layers)
        ])
        self.film = FiLMConditioner(d_model, d_film, n_layers - film_start)
        self.norm = RMSNorm(d_model)

        if adaptive_head:
            from models.adaptive_head import AdaptiveLMHead
            # 3-tier: 8192 full-rank + 16384 low-rank(256) + 25681 low-rank(128)
            remainder = vocab_size - 8192 - 16384
            self.output = AdaptiveLMHead(
                d_model, vocab_size,
                tier_sizes=(8192, 16384, remainder),
                tier_ranks=(None, 256, 128),
            )
        else:
            self.output = nn.Linear(d_model, vocab_size, bias=False)
            # Weight tying (only for standard head)
            self.output.weight = self.tok_embeddings.weight

        self._init_weights()
        n_params = sum(p.numel() for p in self.parameters())
        print(f"Amadeus: {n_params / 1e6:.1f}M parameters")

    def _init_weights(self):
        """Xavier/normal init following COOKBOOK.md."""
        for name, p in self.named_parameters():
            if p.dim() >= 2:
                nn.init.xavier_uniform_(p)
            # A_log and D are already initialized in Mamba3SISO.__init__

    def forward(self, input_ids: torch.Tensor, targets: torch.Tensor = None) -> torch.Tensor:
        B, T = input_ids.shape
        h = self.tok_embeddings(input_ids)

        context = None
        for i, layer in enumerate(self.layers):
            if i == self.film_start:
                context = self.film.compute_context(h)
            if i >= self.film_start and context is not None:
                h = self.film.apply(h, context, i - self.film_start)
            h = layer(h)

        h = self.norm(h)

        if self.adaptive_head and targets is not None:
            # Adaptive head: return chunked CE loss (scalar)
            return self.output(h, targets=targets)
        else:
            # Standard: return logits
            return self.output(h)


# --- Mini config for COOKBOOK mandatory validation ---
class AmadeusMini(Amadeus):
    """Tiny AMADEUS for smoke testing (d=128, 4 layers, ~2M params)."""

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
        )

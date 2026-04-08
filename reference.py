"""
Reference implementations -- PyTorch-only ground truth for correctness verification.
DO NOT MODIFY. These are the oracles that the benchmark harness checks against.
"""

import torch
import torch.nn.functional as F

# Matrix Multiplication
def matmul_ref(A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
    """Standard matrix multiplication. A @ B."""
    return torch.matmul(A, B)

# Softmax
def softmax_ref(x: torch.Tensor, dim: int = -1) -> torch.Tensor:
    """Standard softmax along dim."""
    return F.softmax(x, dim=dim)

# Layer Normalization
def layernorm_ref(x: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor, eps: float = 1e-5) -> torch.Tensor:
    """Layer normalization over last dimension."""
    normalized_shape = x.shape[-1:]
    return F.layer_norm(x, normalized_shape, weight, bias, eps)

# RMS Normalization
def rmsnorm_ref(x: torch.Tensor, weight: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """RMS normalization."""
    rms = torch.sqrt(torch.mean(x ** 2, dim=-1, keepdim=True) + eps)
    return (x / rms) * weight

# Flash Attention
def flash_attention_ref(Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor, causal: bool = True, sm_scale: float = None) -> torch.Tensor:
    """Standard scaled dot-product attention."""
    if sm_scale is None:
        sm_scale = Q.shape[-1] ** -0.5
    attn = torch.matmul(Q, K.transpose(-2, -1)) * sm_scale
    if causal:
        seq_len_q, seq_len_k = Q.shape[-2], K.shape[-2]
        mask = torch.triu(torch.ones(seq_len_q, seq_len_k, device=Q.device, dtype=torch.bool), diagonal=1)
        attn = attn.masked_fill(mask, float('-inf'))
    attn = F.softmax(attn, dim=-1)
    return torch.matmul(attn, V)

# Fused MLP (SwiGLU-style)
def fused_mlp_ref(x: torch.Tensor, w_gate: torch.Tensor, w_up: torch.Tensor, w_down: torch.Tensor, activation: str = "silu") -> torch.Tensor:
    """SwiGLU-style fused MLP: down(activation(gate(x)) * up(x))."""
    gate = x @ w_gate.T
    up = x @ w_up.T
    if activation == "silu":
        gate = F.silu(gate)
    elif activation == "gelu":
        gate = F.gelu(gate)
    elif activation == "relu2":
        gate = F.relu(gate) ** 2
    return (gate * up) @ w_down.T

# Cross Entropy Loss
def cross_entropy_ref(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    """Standard cross entropy loss."""
    return F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))

# Rotary Position Embedding
def rotary_embedding_ref(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """Apply rotary position embeddings."""
    x1 = x[..., ::2]
    x2 = x[..., 1::2]
    rx1 = x1 * cos - x2 * sin
    rx2 = x1 * sin + x2 * cos
    return torch.stack([rx1, rx2], dim=-1).flatten(-2)

# Fused Residual Add + RMSNorm
def fused_residual_add_rmsnorm_ref(x: torch.Tensor, residual: torch.Tensor, weight: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """RMSNorm(x + residual, weight). Fuses residual add with RMS normalization."""
    hidden = x + residual
    # Compute RMS in fp32 to avoid fp16 overflow when squaring large values
    hidden_f = hidden.float()
    rms = torch.sqrt(torch.mean(hidden_f ** 2, dim=-1, keepdim=True) + eps)
    return ((hidden_f / rms) * weight.float()).to(x.dtype)

# Parallel Reductions
def reduce_sum_ref(x: torch.Tensor, dim: int = -1) -> torch.Tensor:
    """Sum reduction."""
    return x.sum(dim=dim)

def reduce_max_ref(x: torch.Tensor, dim: int = -1) -> torch.Tensor:
    """Max reduction."""
    return x.max(dim=dim).values

# SiLU Activation
def silu_ref(x: torch.Tensor) -> torch.Tensor:
    """SiLU (Swish) activation: x * sigmoid(x)."""
    return F.silu(x)

# GELU Activation
def gelu_ref(x: torch.Tensor) -> torch.Tensor:
    """GELU activation: x * 0.5 * (1 + erf(x / sqrt(2)))."""
    return F.gelu(x)

# Fused Residual Add + LayerNorm
# Fused Bias Add + SiLU Activation
def fused_bias_silu_ref(x: torch.Tensor, bias: torch.Tensor) -> torch.Tensor:
    """Fused bias add + SiLU: SiLU(x + bias)."""
    return F.silu(x + bias)

# Fused Bias Add + GELU Activation
def fused_bias_gelu_ref(x: torch.Tensor, bias: torch.Tensor) -> torch.Tensor:
    """Fused bias add + GELU: GELU(x + bias)."""
    return F.gelu(x + bias)

# Fused SiLU-Gate-Multiply (SwiGLU activation)
def silu_gate_mul_ref(gate: torch.Tensor, up: torch.Tensor) -> torch.Tensor:
    """SwiGLU: SiLU(gate) * up. Fuses activation with elementwise multiply."""
    return F.silu(gate) * up

# MoE Top-K Gating
def moe_gating_ref(router_logits: torch.Tensor, k: int = 2) -> torch.Tensor:
    """MoE gating: softmax over expert scores, return top-k routing weights (sparse).
    Output shape matches input: non-top-k positions are zeroed."""
    probs = F.softmax(router_logits, dim=-1)
    topk_vals, topk_idx = torch.topk(probs, k, dim=-1)
    # Normalize top-k weights to sum to 1
    topk_vals = topk_vals / topk_vals.sum(dim=-1, keepdim=True)
    out = torch.zeros_like(probs)
    out.scatter_(-1, topk_idx, topk_vals)
    return out

# Parallel Prefix Sum (Cumulative Sum)
def prefix_scan_ref(x: torch.Tensor) -> torch.Tensor:
    """Inclusive prefix sum along last dimension."""
    return torch.cumsum(x, dim=-1)

# Top-K Sampling
def top_k_sampling_ref(logits: torch.Tensor, k: int = 50, temperature: float = 1.0) -> torch.Tensor:
    """Top-k sampling: scale by temperature, zero out non-top-k, return filtered logits (not sampled indices)."""
    scaled = logits / temperature
    topk_vals, topk_idx = torch.topk(scaled, k, dim=-1)
    # Zero out everything except top-k
    out = torch.full_like(scaled, float('-inf'))
    out.scatter_(-1, topk_idx, topk_vals)
    # Apply softmax over the filtered logits
    return F.softmax(out, dim=-1)

# Int4 Dequantization
def dequantize_int4_ref(x_packed: torch.Tensor, scale: torch.Tensor, zero_point: torch.Tensor) -> torch.Tensor:
    """Per-channel int4 dequantization. x_packed is uint8 with 2 int4 values per byte.
    Low nibble first, high nibble second. Values are unsigned 0-15, zero_point shifts them."""
    lo = (x_packed & 0x0F).to(torch.float32) - zero_point.float()
    hi = ((x_packed >> 4) & 0x0F).to(torch.float32) - zero_point.float()
    # Interleave: even cols from lo, odd cols from hi
    M = x_packed.size(0)
    N_packed = x_packed.size(1)
    N_out = N_packed * 2
    out = torch.empty(M, N_out, device=x_packed.device, dtype=torch.float32)
    out[:, 0::2] = lo * scale.float()
    out[:, 1::2] = hi * scale.float()
    return out.half()

# Int8 Dequantization
def dequantize_int8_ref(x_int8: torch.Tensor, scale: torch.Tensor, zero_point: torch.Tensor) -> torch.Tensor:
    """Per-channel int8 dequantization: ((int8 - zero_point) * scale) -> fp16."""
    return ((x_int8.float() - zero_point.float()) * scale.float()).half()

def fused_residual_add_layernorm_ref(x: torch.Tensor, residual: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor, eps: float = 1e-5) -> torch.Tensor:
    """LayerNorm(x + residual, weight, bias). Fuses residual add with layer normalization."""
    hidden = x + residual
    hidden_f = hidden.float()
    mean = hidden_f.mean(dim=-1, keepdim=True)
    var = ((hidden_f - mean) ** 2).mean(dim=-1, keepdim=True)
    return (((hidden_f - mean) / torch.sqrt(var + eps)) * weight.float() + bias.float()).to(x.dtype)

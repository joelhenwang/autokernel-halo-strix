"""Isolated RoPE+gate mul parity test to find mismatch."""
import sys, torch
sys.path.insert(0, '.')

from kernels.hip.fused_rope_gate_mul import kernel_fn as hip_fused_rope_mul
from models._components import precompute_freqs_cis

device = 'cuda'
torch.manual_seed(42)

# Use OdinHalo's actual dimensions
B, T, H, D_head = 4, 256, 8, 64
d_conv = H * D_head  # 512
pairs = D_head // 2  # 32

print(f"B={B} T={T} H={H} D_head={D_head} d_conv={d_conv} pairs={pairs}")

b = torch.randn(B * T, d_conv, device=device, dtype=torch.float16)
h = torch.randn(B * T, d_conv, device=device, dtype=torch.float16)

freqs_cis = precompute_freqs_cis(D_head, T).to(device)

# HIP path expects freqs_cos, freqs_sin as fp32
freqs_cos = freqs_cis.real[:T, :pairs].contiguous().float()
freqs_sin = freqs_cis.imag[:T, :pairs].contiguous().float()
out_hip = hip_fused_rope_mul(b, h, freqs_cos, freqs_sin, T, d_conv, pairs)

# Native path (same math as kernel)
def native_rope_gate_mul(b, h, freqs_cis, T, n_rope_heads, rope_head_dim):
    pairs = rope_head_dim // 2
    B_T, d_conv = b.shape
    B = B_T // T
    cos = freqs_cis.real[:T, :pairs].contiguous().float()
    sin = freqs_cis.imag[:T, :pairs].contiguous().float()
    b_r = b.view(B, T, n_rope_heads, rope_head_dim).float()
    h_r = h.view(B, T, n_rope_heads, rope_head_dim).float()
    b_even = b_r[..., 0::2]
    b_odd  = b_r[..., 1::2]
    h_even = h_r[..., 0::2]
    h_odd  = h_r[..., 1::2]
    cos = cos[None, :, None, :]
    sin = sin[None, :, None, :]
    b_rot_even = b_even * cos - b_odd * sin
    b_rot_odd  = b_even * sin + b_odd * cos
    y_even = b_rot_even * h_even
    y_odd  = b_rot_odd  * h_odd
    y = torch.stack([y_even, y_odd], dim=-1).flatten(-2)
    return y.view(B_T, d_conv).to(b.dtype)

out_native = native_rope_gate_mul(b, h, freqs_cis, T, H, D_head)

diff = (out_hip.float() - out_native.float()).abs()
print(f"HIP vs native: max_err={diff.max().item():.4e} mean_err={diff.mean().item():.4e}")

# Where is the biggest diff?
idx = diff.flatten().argmax().item()
row = idx // d_conv
col = idx % d_conv
print(f"  biggest diff at (row={row}, col={col}): hip={out_hip.flatten()[idx].item()} native={out_native.flatten()[idx].item()}")
print(f"  t_idx={row % T}  head={col // D_head}  pair_idx_in_head={(col // 2) % pairs}  even/odd={col % 2}")

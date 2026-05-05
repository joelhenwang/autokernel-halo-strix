"""Trace native RoPE inside the actual block to find the mismatch."""
import sys, torch
sys.path.insert(0, '.')

import models.components.conv_blocks as cb
from models.odin_halo import OdinHalo
from models._components import precompute_freqs_cis

device = 'cuda'
torch.manual_seed(42)

model = OdinHalo(use_chunked_ce=False).to(device)
model.eval()
freqs_cis = precompute_freqs_cis(model.head_dim, 256).to(device)
input_ids = torch.randint(0, model.vocab_size, (4, 256), device=device)

layer = None
for l in model.shared_layers:
    if isinstance(l, cb.HyPEShortConvBlock):
        layer = l
        break

print(f"rope_head_dim={layer.rope_head_dim} n_rope_heads={layer.n_rope_heads} d_conv={layer.d_conv}")

# Build b, c, h_tilde exactly as the block does
with torch.no_grad(), torch.amp.autocast('cuda', dtype=torch.float16):
    h = model.tok_embeddings(input_ids).half()
    normed = layer.pre_norm(h)
    b, c, h_tilde = layer.proj(normed).chunk(3, dim=-1)

    B, T, _ = h.shape

    b_flat = b.reshape(B * T, layer.d_conv).contiguous()
    h_flat = h_tilde.reshape(B * T, layer.d_conv).contiguous()

    # HIP path WITH .contiguous() on fcos/fsin (as per my fix)
    from kernels.hip.fused_rope_gate_mul import kernel_fn as hip_fn
    pairs = layer.rope_head_dim // 2
    fcos_fixed = freqs_cis.real[:T, :pairs].contiguous().float()
    fsin_fixed = freqs_cis.imag[:T, :pairs].contiguous().float()
    print(f"fcos_fixed contiguous={fcos_fixed.is_contiguous()}")
    y_hip_fixed = hip_fn(b_flat.half(), h_flat.half(), fcos_fixed, fsin_fixed,
                         T, layer.d_conv, pairs)

    # Native path
    y_native = layer._rope_gate_mul_native(b_flat, h_flat, freqs_cis, T)

    diff = (y_hip_fixed.float() - y_native.float()).abs()
    print(f"Fixed HIP vs native: max_err={diff.max().item():.4e} mean_err={diff.mean().item():.4e}")

    # Also test HIP WITHOUT contiguous (the old buggy way)
    fcos_bad = freqs_cis.real[:T, :pairs].float()
    fsin_bad = freqs_cis.imag[:T, :pairs].float()
    print(f"fcos_bad contiguous={fcos_bad.is_contiguous()}")
    y_hip_bad = hip_fn(b_flat.half(), h_flat.half(), fcos_bad, fsin_bad,
                       T, layer.d_conv, pairs)
    diff_b = (y_hip_fixed.float() - y_hip_bad.float()).abs()
    print(f"Fixed HIP vs buggy HIP: max_err={diff_b.max().item():.4e}")
    diff_bn = (y_hip_bad.float() - y_native.float()).abs()
    print(f"Buggy HIP vs native:   max_err={diff_bn.max().item():.4e}")

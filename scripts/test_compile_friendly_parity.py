"""Verify compile-friendly HyPEShortConvBlock produces same output as HIP kernel path.

Run 3 comparisons:
  (A) Full native (native RoPE + manual conv)  vs  HIP (HIP RoPE + causal_conv1d)
  (B) HIP RoPE + manual conv                   vs  HIP (baseline)  — tests conv
  (C) Native RoPE + causal_conv1d              vs  HIP (baseline)  — tests RoPE
"""
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
h = model.tok_embeddings(input_ids).half()

layer = None
for l in model.shared_layers:
    if isinstance(l, cb.HyPEShortConvBlock):
        layer = l
        break

def run(rope_native, conv_manual, label):
    # Monkey-patch flags
    layer._compile_friendly = rope_native
    # Also test manual conv in isolation by patching method
    if not rope_native and conv_manual:
        # Replace forward to use HIP RoPE + manual conv
        orig = cb.HyPEShortConvBlock.forward
        def patched(self, x, fc):
            B, T, _ = x.shape
            normed = self.pre_norm(x)
            b, c, h_tilde = self.proj(normed).chunk(3, dim=-1)
            from kernels.hip.fused_rope_gate_mul import kernel_fn as fused_rope_mul
            freqs_cos = fc.real[:T, :self.rope_head_dim // 2].float()
            freqs_sin = fc.imag[:T, :self.rope_head_dim // 2].float()
            y = fused_rope_mul(b.reshape(B*T, self.d_conv).half(),
                               h_tilde.reshape(B*T, self.d_conv).half(),
                               freqs_cos, freqs_sin,
                               T, self.d_conv, self.rope_head_dim // 2).float().view(B, T, self.d_conv)
            z = self._manual_causal_conv1d(y)
            conv_out = self.out_proj(c * z)
            ffn_out = self.ffn(self.ffn_norm(x + conv_out))
            return x + conv_out + ffn_out
        cb.HyPEShortConvBlock.forward = patched
    elif rope_native and not conv_manual:
        # Native RoPE + causal_conv1d
        orig = cb.HyPEShortConvBlock.forward
        def patched(self, x, fc):
            B, T, _ = x.shape
            normed = self.pre_norm(x)
            b, c, h_tilde = self.proj(normed).chunk(3, dim=-1)
            y = self._rope_gate_mul_native(
                b.reshape(B * T, self.d_conv),
                h_tilde.reshape(B * T, self.d_conv),
                fc, T,
            ).view(B, T, self.d_conv)
            if cb._HAS_CAUSAL_CONV1D:
                z = cb.causal_conv1d_fn(
                    y.transpose(1, 2), self.conv_weight, self.conv_bias
                ).transpose(1, 2)
            else:
                z = self.conv(y.transpose(1, 2))[:, :, :T].transpose(1, 2)
            conv_out = self.out_proj(c * z)
            ffn_out = self.ffn(self.ffn_norm(x + conv_out))
            return x + conv_out + ffn_out
        cb.HyPEShortConvBlock.forward = patched
    elif rope_native and conv_manual:
        orig = cb.HyPEShortConvBlock.forward
        # Use default forward with _compile_friendly=True (native + manual)
        layer._compile_friendly = True
    else:
        orig = cb.HyPEShortConvBlock.forward
        layer._compile_friendly = False

    with torch.no_grad(), torch.amp.autocast('cuda', dtype=torch.float16):
        out = layer(h, freqs_cis)

    # Restore
    if not (rope_native and conv_manual) and not (not rope_native and not conv_manual):
        cb.HyPEShortConvBlock.forward = orig

    return out.clone()


out_ref   = run(False, False, "HIP RoPE + causal_conv1d (baseline)")
out_full  = run(True,  True,  "native RoPE + manual conv")
out_roped = run(False, True,  "HIP RoPE + manual conv")
out_conv  = run(True,  False, "native RoPE + causal_conv1d")

def cmp(a, b, label):
    d = (a.float() - b.float()).abs()
    print(f"  {label:40s} max_err={d.max().item():.4e} mean_err={d.mean().item():.4e}")

print("vs HIP baseline:")
cmp(out_full,  out_ref, "full native (A)")
cmp(out_roped, out_ref, "HIP RoPE + manual conv (B)")
cmp(out_conv,  out_ref, "native RoPE + causal_conv1d (C)")


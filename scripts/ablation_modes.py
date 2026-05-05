"""Ablation on production OdinHalo with mode comparison (fused vs tiny)."""
import sys, time, torch
sys.path.insert(0, '.')

STEPS = 30
device = 'cuda'

from halo_training.data import BabyLMDataset, build_dataloader
ds = BabyLMDataset(root='datasets/babylm-odin32k.bin', block_size=256,
                   tokenizer_path='tokenizers/odin-32k/tokenizer.json')


def bench(label, build_model_fn, loss_fn=None):
    model = build_model_fn()
    model.train()
    dl = build_dataloader(ds, batch_size=4, num_workers=0, shuffle=True)
    data_iter = iter(dl)
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    t0 = time.time()
    tokens = 0
    for step in range(STEPS):
        batch = next(data_iter)
        input_ids, targets = batch
        input_ids, targets = input_ids.to(device), targets.to(device)
        with torch.amp.autocast('cuda', dtype=torch.float16):
            logits = model(input_ids)
            if isinstance(logits, dict):
                logits = logits['logits']
            if loss_fn:
                loss = loss_fn(logits.view(-1, logits.size(-1)), targets.view(-1))
            else:
                loss = torch.nn.functional.cross_entropy(
                    logits.view(-1, logits.size(-1)), targets.view(-1))
        loss.backward()
        model.zero_grad(set_to_none=True)
        tokens += input_ids.numel()
    torch.cuda.synchronize()
    elapsed = time.time() - t0
    tok_s = tokens / elapsed
    mem = torch.cuda.max_memory_allocated() / 1e9
    print(f'{label}: {tok_s:,.0f} tok/s ({mem:.1f} GB peak)')
    return tok_s


import models.components.conv_blocks as cb
_orig_forward = cb.HyPEShortConvBlock.forward

def _patched_forward(self, x, freqs_cis):
    B, T, _ = x.shape
    normed = self.pre_norm(x)
    b, c, h_tilde = self.proj(normed).chunk(3, dim=-1)
    b = self._rope_on_gate(b, freqs_cis)
    y = b * h_tilde
    if cb._HAS_CAUSAL_CONV1D:
        z = cb.causal_conv1d_fn(
            y.transpose(1, 2), self.conv_weight, self.conv_bias
        ).transpose(1, 2)
    else:
        z = self.conv(y.transpose(1, 2))[:, :, :T].transpose(1, 2)
    conv_out = self.out_proj(c * z)
    ffn_out = self.ffn(self.ffn_norm(x + conv_out))
    return x + conv_out + ffn_out

cb.HyPEShortConvBlock.forward = _patched_forward

from models.odin_halo import OdinHalo
import kernel as opt_ce

def ce_fused(logits, targets):
    return opt_ce.ce_full(logits, targets, mode="fused")

def ce_tiny(logits, targets):
    return opt_ce.ce_full(logits, targets, mode="tiny")


print("=== NO RoPE fusion (isolating CE) ===")
b_base = bench('Baseline (PyTorch CE)       ', lambda: OdinHalo().to(device))
b_fused = bench('+ HIP CE fused              ', lambda: OdinHalo().to(device), ce_fused)
b_tiny = bench('+ HIP CE tiny               ', lambda: OdinHalo().to(device), ce_tiny)

print()
print("=== WITH RoPE fusion ===")
cb.HyPEShortConvBlock.forward = _orig_forward
b_rope_pt = bench('PyTorch CE + RoPE fusion    ', lambda: OdinHalo().to(device))
b_rope_fused = bench('HIP CE fused + RoPE fusion  ', lambda: OdinHalo().to(device), ce_fused)
b_rope_tiny = bench('HIP CE tiny + RoPE fusion   ', lambda: OdinHalo().to(device), ce_tiny)

print()
print('=== SUMMARY (OdinHalo prod, V=32768) ===')
print(f'baseline (PyTorch CE)                {b_base:>7,.0f} tok/s')
print(f'+HIP CE fused                        {b_fused:>7,.0f} tok/s ({b_fused/b_base:.3f}x)')
print(f'+HIP CE tiny                         {b_tiny:>7,.0f} tok/s ({b_tiny/b_base:.3f}x)')
print(f'+RoPE fusion (PyTorch CE)            {b_rope_pt:>7,.0f} tok/s ({b_rope_pt/b_base:.3f}x)')
print(f'+RoPE fusion + HIP CE fused          {b_rope_fused:>7,.0f} tok/s ({b_rope_fused/b_base:.3f}x)')
print(f'+RoPE fusion + HIP CE tiny           {b_rope_tiny:>7,.0f} tok/s ({b_rope_tiny/b_base:.3f}x)')

"""Full production ablation: Phase 1 + Phase 2 (chunked CE) on OdinHalo.

Measures every combination so we can quantify each optimization's contribution.
"""
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
            output = model(input_ids)
            loss = loss_fn(model, output, targets)
        loss.backward()
        model.zero_grad(set_to_none=True)
        tokens += input_ids.numel()
    torch.cuda.synchronize()
    elapsed = time.time() - t0
    tok_s = tokens / elapsed
    mem = torch.cuda.max_memory_allocated() / 1e9
    print(f'{label}: {tok_s:,.0f} tok/s ({mem:.2f} GB peak)')
    return tok_s, mem


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


from models.odin_halo import OdinHalo
import kernel as opt_ce
from kernels.hip.chunked_linear_cross_entropy import ChunkedLinearCrossEntropyLoss


def pytorch_ce(model, logits, targets):
    return torch.nn.functional.cross_entropy(
        logits.view(-1, logits.size(-1)), targets.view(-1))


def hip_ce(model, logits, targets):
    return opt_ce.ce_full(
        logits.view(-1, logits.size(-1)), targets.view(-1),
        mode="tiny",  # winner at prod scale per earlier benchmark
    )


chunked_ce_fn = ChunkedLinearCrossEntropyLoss(chunk_size=256, softcap=30.0)

def chunked_ce(model, h_low, targets):
    return chunked_ce_fn(
        h_low.view(-1, h_low.size(-1)),
        model.lm_head.embed_table.weight,
        targets.view(-1),
    )


print("=== CE path comparison (RoPE fusion OFF) ===")
cb.HyPEShortConvBlock.forward = _patched_forward
b_base, m_base = bench('Baseline (PyTorch CE, logits)     ',
                        lambda: OdinHalo().to(device), pytorch_ce)
b_hip,  m_hip  = bench('+HIP CE (Phase 1, tiny mode)      ',
                        lambda: OdinHalo().to(device), hip_ce)
b_chunk, m_chunk = bench('+Chunked CE (Phase 2)           ',
                        lambda: OdinHalo(use_chunked_ce=True).to(device), chunked_ce)

print()
print("=== Full stack (RoPE fusion ON) ===")
cb.HyPEShortConvBlock.forward = _orig_forward
b_rope_pt,  m_rope_pt  = bench('RoPE fusion + PyTorch CE           ',
                               lambda: OdinHalo().to(device), pytorch_ce)
b_rope_hip, m_rope_hip = bench('RoPE fusion + HIP CE               ',
                               lambda: OdinHalo().to(device), hip_ce)
b_rope_chunk, m_rope_chunk = bench('RoPE fusion + Chunked CE       ',
                               lambda: OdinHalo(use_chunked_ce=True).to(device), chunked_ce)

print()
print("=" * 70)
print("=== FULL STACK ABLATION SUMMARY (OdinHalo, V=32768) ===")
print("=" * 70)
print(f"Baseline:                                 {b_base:>7,.0f} tok/s {m_base:>5.2f} GB")
print(f"+ Phase 1 (online-softmax CE):            {b_hip:>7,.0f} tok/s {m_hip:>5.2f} GB  "
      f"({b_hip/b_base:.3f}x, Δmem {(m_hip-m_base)*1000:+.0f} MB)")
print(f"+ Phase 2 (chunked linear+CE):            {b_chunk:>7,.0f} tok/s {m_chunk:>5.2f} GB  "
      f"({b_chunk/b_base:.3f}x, Δmem {(m_chunk-m_base)*1000:+.0f} MB)")
print(f"+ RoPE fusion (PyTorch CE):               {b_rope_pt:>7,.0f} tok/s {m_rope_pt:>5.2f} GB  "
      f"({b_rope_pt/b_base:.3f}x)")
print(f"+ RoPE fusion + HIP CE:                   {b_rope_hip:>7,.0f} tok/s {m_rope_hip:>5.2f} GB  "
      f"({b_rope_hip/b_base:.3f}x)")
print(f"+ RoPE fusion + Chunked CE (FULL STACK):  {b_rope_chunk:>7,.0f} tok/s {m_rope_chunk:>5.2f} GB  "
      f"({b_rope_chunk/b_base:.3f}x, Δmem {(m_rope_chunk-m_base)*1000:+.0f} MB)")

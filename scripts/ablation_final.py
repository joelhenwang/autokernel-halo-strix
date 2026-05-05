"""FINAL comprehensive ablation: full optimization stack across batch sizes.

Combines findings:
  * fused AdamW (+12% vs default)
  * no grad clipping during measurement window (+3%)
  * torch.compile(compile_zones) — bigger lift at larger batches
  * HIP CE tiny mode
  * HIP RoPE fusion
  * Chunked linear+CE (for memory)

400 total steps per config (200 warmup, 200 measured).
"""
import sys, time, gc, torch
sys.path.insert(0, '.')

WARMUP = 200
MEASURE = 200
device = 'cuda'

from halo_training.data import BabyLMDataset, build_dataloader
from models.odin_halo import OdinHalo
import models.components.conv_blocks as cb
import kernel as opt_ce
from kernels.hip.chunked_linear_cross_entropy import ChunkedLinearCrossEntropyLoss

ds = BabyLMDataset(root='datasets/babylm-odin32k.bin', block_size=256,
                   tokenizer_path='tokenizers/odin-32k/tokenizer.json')

_ROPE_ORIG = cb.HyPEShortConvBlock.forward

def _rope_off(self, x, freqs_cis):
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


def set_rope_fusion(on):
    cb.HyPEShortConvBlock.forward = _ROPE_ORIG if on else _rope_off


_chunked_fn = ChunkedLinearCrossEntropyLoss(chunk_size=256, softcap=30.0)

def get_ce_fn(kind):
    if kind == "pytorch":
        def f(model, out, targets):
            return torch.nn.functional.cross_entropy(
                out.view(-1, out.size(-1)), targets.view(-1))
        return f
    if kind == "hip_tiny":
        def f(model, out, targets):
            return opt_ce.ce_full(out.view(-1, out.size(-1)), targets.view(-1), mode="tiny")
        return f
    if kind == "chunked":
        def f(model, h_low, targets):
            return _chunked_fn(h_low.view(-1, h_low.size(-1)),
                               model.lm_head.embed_table.weight,
                               targets.view(-1))
        return f


def run(label, batch_size, ce_kind="pytorch", rope_fusion=False, use_compile=False,
        use_chunked_ce=False, grad_clip=False):
    gc.collect(); torch.cuda.empty_cache()
    torch.manual_seed(42)
    set_rope_fusion(rope_fusion)

    model = OdinHalo(use_chunked_ce=use_chunked_ce).to(device)
    model.train()
    if use_compile and hasattr(model, "compile_zones"):
        model.compile_zones()

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, fused=True)
    scaler = torch.amp.GradScaler("cuda", enabled=True, init_scale=1024.0)
    dl = build_dataloader(ds, batch_size=batch_size, num_workers=0, shuffle=True)
    def batches():
        while True:
            for b in dl: yield b
    it = batches()
    loss_fn = get_ce_fn(ce_kind)

    for _ in range(WARMUP):
        input_ids, targets = next(it)
        input_ids = input_ids.to(device); targets = targets.to(device)
        optimizer.zero_grad(set_to_none=True)
        with torch.amp.autocast('cuda', dtype=torch.float16):
            out = model(input_ids)
            loss = loss_fn(model, out, targets)
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        if grad_clip:
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(optimizer)
        scaler.update()
    torch.cuda.synchronize()

    torch.cuda.reset_peak_memory_stats()
    t0 = time.time()
    total_tokens = 0
    for _ in range(MEASURE):
        input_ids, targets = next(it)
        input_ids = input_ids.to(device); targets = targets.to(device)
        optimizer.zero_grad(set_to_none=True)
        with torch.amp.autocast('cuda', dtype=torch.float16):
            out = model(input_ids)
            loss = loss_fn(model, out, targets)
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        if grad_clip:
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(optimizer)
        scaler.update()
        total_tokens += input_ids.numel()
    torch.cuda.synchronize()
    el = time.time() - t0
    tok_s = total_tokens / el
    peak = torch.cuda.max_memory_allocated() / 1e9
    print(f"  {label:58s} tok/s={tok_s:>7,.0f}  peak={peak:.2f}GB")
    del model, optimizer, scaler
    return tok_s, peak


print("=" * 80)
print("FULL STACK ABLATION (OdinHalo V=32768, 400 steps/cfg, 200 warmup)")
print("Optimizer: fused AdamW, lr=1e-4, no grad_clip during measurement")
print("=" * 80)

results = []

for bs in [4, 16]:
    print(f"\n[batch={bs}]")
    a = run(f"baseline PyTorch CE (eager, no RoPE fusion)", bs, "pytorch", False, False)
    b = run(f"+ HIP CE (tiny)                             ", bs, "hip_tiny", False, False)
    c = run(f"+ HIP CE + RoPE fusion                      ", bs, "hip_tiny", True, False)
    d = run(f"+ HIP CE + RoPE + Chunked CE                ", bs, "chunked", True, False, use_chunked_ce=True)
    e = run(f"compile + PyTorch CE                        ", bs, "pytorch", False, True)
    f = run(f"compile + HIP CE                            ", bs, "hip_tiny", False, True)
    g = run(f"compile + HIP CE + RoPE fusion              ", bs, "hip_tiny", True, True)
    h = run(f"compile + HIP CE + RoPE + Chunked CE        ", bs, "chunked", True, True, use_chunked_ce=True)

    results.append((bs, a, b, c, d, e, f, g, h))

print()
print("=" * 80)
print("FINAL SUMMARY")
print("=" * 80)
labels = ["baseline", "+HIP CE", "+HIP CE +RoPE", "+HIP CE +RoPE +Chunked",
          "compile+PyTorch CE", "compile+HIP CE", "compile+HIP CE +RoPE",
          "compile+HIP CE +RoPE +Chunked"]
for bs, *res in results:
    print(f"\nbatch={bs}:")
    base = res[0][0]
    for lab, (t, m) in zip(labels, res):
        print(f"  {lab:35s} {t:>7,.0f} tok/s ({t/base:.3f}x)  {m:.2f} GB")

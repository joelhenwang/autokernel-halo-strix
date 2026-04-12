"""Fair comparison: LlamaModel vs Tempest vs AMADEUS at ~124M params."""

import torch
import torch.nn as nn
import time
import sys
import gc
import importlib.util

sys.path.insert(0, '.')


def load(path, cls):
    spec = importlib.util.spec_from_file_location('m', path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return getattr(mod, cls)()


def bench(model, label, batch=8, seq=256, steps=30, warmup=5):
    device = torch.device('cuda')
    model = model.to(device).train()
    vocab = None
    for m in model.modules():
        if isinstance(m, nn.Embedding):
            vocab = m.weight.shape[0]
            break
    if vocab is None:
        vocab = 32000
    n_params = sum(p.numel() for p in model.parameters())
    scaler = torch.amp.GradScaler('cuda')
    opt = torch.optim.AdamW(model.parameters(), lr=8e-4, fused=True)
    tps = batch * seq
    for _ in range(warmup):
        x = torch.randint(0, vocab, (batch, seq), device=device)
        t = torch.randint(0, vocab, (batch, seq), device=device)
        opt.zero_grad(set_to_none=True)
        with torch.amp.autocast('cuda', dtype=torch.float16):
            out = model(x)
            logits = out if isinstance(out, torch.Tensor) else out.logits
            loss = nn.functional.cross_entropy(logits.view(-1, logits.size(-1)), t.view(-1))
        scaler.scale(loss).backward()
        scaler.step(opt)
        scaler.update()
    torch.cuda.synchronize()

    times = []
    for _ in range(steps):
        x = torch.randint(0, vocab, (batch, seq), device=device)
        t = torch.randint(0, vocab, (batch, seq), device=device)
        opt.zero_grad(set_to_none=True)
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        with torch.amp.autocast('cuda', dtype=torch.float16):
            out = model(x)
            logits = out if isinstance(out, torch.Tensor) else out.logits
            loss = nn.functional.cross_entropy(logits.view(-1, logits.size(-1)), t.view(-1))
        scaler.scale(loss).backward()
        scaler.unscale_(opt)
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(opt)
        scaler.update()
        torch.cuda.synchronize()
        times.append((time.perf_counter() - t0) * 1000)

    skip = 3
    t_sorted = sorted(times[skip:])
    med = t_sorted[len(t_sorted) // 2]
    tok = tps / (med / 1000)
    mfu = tok * 6 * n_params / 59.4e12 * 100
    mem = torch.cuda.max_memory_allocated() / 1e9
    print(f"  {label:40s} {n_params/1e6:7.1f}M  {med:8.2f}ms  {tok:8.0f} tok/s  {mfu:5.1f}% MFU  {mem:.2f}GB")
    return model


def main():
    print(f"GPU: {torch.cuda.get_device_name()}")
    print(f"Batch=8, Seq=256, 30 steps")
    print()

    configs = [
        ("models/llama_7b.py", "LlamaModel"),
        ("models/tempest_124m.py", "Tempest124M"),
        ("models/amadeus_124m.py", "Amadeus124M"),
    ]

    # Eager
    print("=== EAGER (no optimizations) ===")
    for path, cls in configs:
        torch.cuda.reset_peak_memory_stats()
        m = load(path, cls)
        bench(m, f"{cls} eager")
        del m
        torch.cuda.empty_cache()
        gc.collect()

    print()

    # With autokernel
    print("=== AUTOKERNEL.OPTIMIZE ===")
    import autokernel
    for path, cls in configs:
        torch.cuda.reset_peak_memory_stats()
        m = load(path, cls)
        m = m.to("cuda").train()
        m = autokernel.optimize(m, training=True)
        bench(m, f"{cls} autokernel")
        del m
        torch.cuda.empty_cache()
        gc.collect()

    print()

    # With autokernel + compile
    print("=== AUTOKERNEL + TORCH.COMPILE ===")
    for path, cls in configs:
        torch.cuda.reset_peak_memory_stats()
        m = load(path, cls)
        m = m.to("cuda").train()
        m = autokernel.optimize(m, training=True)
        try:
            m = torch.compile(m, mode="default")
            bench(m, f"{cls} autokernel+compile", warmup=10)
        except Exception as e:
            print(f"  {cls:40s} COMPILE FAILED: {e}")
        del m
        torch.cuda.empty_cache()
        gc.collect()


if __name__ == "__main__":
    main()

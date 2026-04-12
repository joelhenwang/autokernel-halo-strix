"""Benchmark all hypothesis architectures at ~170M params.

Each model is instantiated with scaled-down dimensions targeting ~170M params.
Tests eager, autokernel, and autokernel+compile configurations.
"""

import torch
import torch.nn as nn
import time
import sys
import gc
import importlib.util

sys.path.insert(0, '.')


# ~170M configurations for each model
# Target: 160-180M params, vocab=50257, d must be multiple of 128
CONFIGS_170M = {
    "Tempest": {
        "path": "models/tempest.py",
        "class": "Tempest",
        "kwargs": dict(
            vocab_size=50257, d_model=896, n_layers=14,
            d_conv=512, d_griffin=384, ffn_inner=2304,
        ),
    },
    "SpectralHydra": {
        "path": "models/spectral_hydra.py",
        "class": "SpectralHydra",
        "kwargs": dict(
            vocab_size=50257, d_model=896, n_layers=14,
            n_heads=14, d_head=64, ffn_inner=2304,
        ),
    },
    "ResonantLoop": {
        "path": "models/resonant_loop.py",
        "class": "ResonantLoop",
        "kwargs": dict(
            vocab_size=50257, d_model=896, ffn_inner=1792,
            max_iterations=16,
        ),
    },
    "MaestroPrima": {
        "path": "models/maestro_prima.py",
        "class": "MaestroPrima",
        "kwargs": dict(
            vocab_size=50257, d_model=896, n_layers=12,
            d_conv=512, d_mamba=384, dstate=64, n_ssm_heads=6,
            ffn_inner=2304, d_film=64, film_start=6,
        ),
    },
    "Prometheus": {
        "path": "models/prometheus.py",
        "class": "Prometheus",
        "kwargs": dict(
            vocab_size=50257, d_model=896, n_layers=14,
            d_conv=512, d_griffin=384, ffn_inner=2304,
            n_attn_heads=8, n_kv_heads=2, head_dim=112,
            attn_layers=(3, 10),
        ),
    },
    "DualCortex": {
        "path": "models/dual_cortex.py",
        "class": "DualCortex",
        "kwargs": dict(
            vocab_size=50257, d_embed=896, d_fast=256,
            d_slow=896, n_fast_layers=6, n_slow_layers=8,
            ffn_fast=512, ffn_slow=2304,
        ),
    },
    "Obsidian": {
        "path": "models/obsidian.py",
        "class": "Obsidian",
        "kwargs": dict(
            vocab_size=50257, d_model=896, d_reflex=256,
            n_reflex_layers=6, d_reflex_ffn=512,
            n_genius_layers=8, d_genius_rec=896, ffn_genius=2304,
        ),
    },
    "Amadeus": {
        "path": "models/amadeus.py",
        "class": "Amadeus",
        "kwargs": dict(
            vocab_size=50257, d_model=896, n_layers=12,
            d_conv=512, d_mamba=384, dstate=64, n_ssm_heads=6,
            ffn_inner=2304, d_film=64, film_start=6,
        ),
    },
    "Virtuoso": {
        "path": "models/virtuoso.py",
        "class": "Virtuoso",
        "kwargs": dict(
            vocab_size=50257, d_model=896, n_layers=14,
            d_conv=512, d_griffin=384, ffn_inner=2304,
        ),
    },
}


def load_model(path, cls_name, kwargs):
    spec = importlib.util.spec_from_file_location("model_module", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    cls = getattr(mod, cls_name)
    return cls(**kwargs)


def bench(model, label, batch=8, seq=256, steps=30, warmup=5):
    device = torch.device('cuda')
    model = model.to(device).train()
    vocab = None
    for m in model.modules():
        if isinstance(m, nn.Embedding):
            vocab = m.weight.shape[0]
            break
    if vocab is None:
        vocab = 50257
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
    print(f"  {label:40s} {n_params/1e6:7.1f}M  {med:8.2f}ms  {tok:8.0f} tok/s  {mfu:5.1f}% MFU  {mem:.1f}GB")
    return n_params, tok, mfu, mem


def main():
    print(f"GPU: {torch.cuda.get_device_name()}")
    print(f"All models at ~170M params | Batch=8, Seq=256, 30 steps")
    print()

    results = {}

    # === EAGER ===
    print("=== EAGER ===")
    for name, cfg in CONFIGS_170M.items():
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.empty_cache()
        gc.collect()
        try:
            m = load_model(cfg["path"], cfg["class"], cfg["kwargs"])
            n_params, tok, mfu, mem = bench(m, f"{name} eager")
            results[f"{name}_eager"] = (n_params, tok, mfu, mem)
            del m
        except Exception as e:
            print(f"  {name:40s} FAILED: {e}")

    print()

    # === AUTOKERNEL ===
    print("=== AUTOKERNEL ===")
    import autokernel
    for name, cfg in CONFIGS_170M.items():
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.empty_cache()
        gc.collect()
        try:
            m = load_model(cfg["path"], cfg["class"], cfg["kwargs"])
            m = m.to("cuda").train()
            m = autokernel.optimize(m, training=True)
            n_params, tok, mfu, mem = bench(m, f"{name} autokernel")
            results[f"{name}_autokernel"] = (n_params, tok, mfu, mem)
            del m
        except Exception as e:
            print(f"  {name:40s} FAILED: {e}")

    print()

    # === AUTOKERNEL + COMPILE ===
    print("=== AUTOKERNEL + COMPILE ===")
    for name, cfg in CONFIGS_170M.items():
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.empty_cache()
        gc.collect()
        try:
            m = load_model(cfg["path"], cfg["class"], cfg["kwargs"])
            m = m.to("cuda").train()
            m = autokernel.optimize(m, training=True)
            m = torch.compile(m, mode="default")
            n_params, tok, mfu, mem = bench(m, f"{name} autokernel+compile", warmup=10)
            results[f"{name}_compiled"] = (n_params, tok, mfu, mem)
            del m
        except Exception as e:
            print(f"  {name:40s} COMPILE FAILED: {str(e)[:80]}")

    # Summary
    print()
    print("=" * 95)
    print("SUMMARY (best config per model)")
    print("=" * 95)
    print(f"{'Model':20s} {'Params':>8s} {'Eager':>10s} {'Autokernel':>12s} {'Compiled':>12s} {'Best':>12s}")
    print("-" * 95)
    for name in CONFIGS_170M:
        eager = results.get(f"{name}_eager", (0, 0, 0, 0))
        ak = results.get(f"{name}_autokernel", (0, 0, 0, 0))
        comp = results.get(f"{name}_compiled", (0, 0, 0, 0))
        best = max(eager[1], ak[1], comp[1])
        print(f"{name:20s} {eager[0]/1e6:7.1f}M "
              f"{eager[1]:>9.0f} "
              f"{ak[1]:>11.0f} "
              f"{comp[1]:>11.0f} "
              f"{best:>11.0f} tok/s")


if __name__ == "__main__":
    main()

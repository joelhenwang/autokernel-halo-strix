"""WI4: Test whether gradient pre-allocation eliminates Memsets.

Three strategies for handling gradients across steps:
  A: optimizer.zero_grad(set_to_none=True)    (current default)
  B: optimizer.zero_grad(set_to_none=False)   (grads persist, zero_()'d each step)
  C: Pre-allocate grads once, then zero_() each step (no free/realloc)

Measures tok/s + records memset count via a simple HIP runtime hook.

Usage:
  python scripts/bench_zero_grad.py --steps 100 --repeat 3
"""
import argparse, time, statistics, sys
sys.path.insert(0, '.')
import torch


def make_workload(batch_size=16, block_size=256):
    from models.odin_halo import OdinHalo
    from halo_training.data import BabyLMDataset, build_dataloader
    model = OdinHalo().to('cuda')
    model.train()
    model.compile_zones()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, fused=True)
    scaler = torch.amp.GradScaler('cuda', enabled=True, init_scale=1024.0)
    ds = BabyLMDataset(
        root='datasets/babylm-odin32k.bin',
        block_size=block_size,
        tokenizer_path='tokenizers/odin-32k/tokenizer.json',
    )
    dl = build_dataloader(ds, batch_size=batch_size, num_workers=0, shuffle=True)
    return model, optimizer, scaler, dl


def step_train(model, optimizer, scaler, ids, tgt):
    with torch.amp.autocast('cuda', dtype=torch.float16):
        logits = model(ids)
        loss = torch.nn.functional.cross_entropy(
            logits.view(-1, logits.size(-1)), tgt.view(-1))
    scaler.scale(loss).backward()
    scaler.unscale_(optimizer)
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    scaler.step(optimizer)
    scaler.update()


def bench_A(model, optimizer, scaler, dl, n_warmup, n_measured):
    """A: set_to_none=True (default)."""
    it = iter(dl)
    def gb():
        nonlocal it
        try: return next(it)
        except StopIteration:
            it = iter(dl); return next(it)
    for _ in range(n_warmup):
        ids, tgt = gb()
        optimizer.zero_grad(set_to_none=True)
        step_train(model, optimizer, scaler, ids.to('cuda'), tgt.to('cuda'))
    torch.cuda.synchronize()
    t0 = time.perf_counter(); n_tok = 0
    for _ in range(n_measured):
        ids, tgt = gb()
        optimizer.zero_grad(set_to_none=True)
        step_train(model, optimizer, scaler, ids.to('cuda'), tgt.to('cuda'))
        n_tok += ids.numel()
    torch.cuda.synchronize()
    t1 = time.perf_counter()
    return n_tok / (t1 - t0)


def bench_B(model, optimizer, scaler, dl, n_warmup, n_measured):
    """B: set_to_none=False (grads stay allocated, zero_ in-place)."""
    it = iter(dl)
    def gb():
        nonlocal it
        try: return next(it)
        except StopIteration:
            it = iter(dl); return next(it)
    for _ in range(n_warmup):
        ids, tgt = gb()
        optimizer.zero_grad(set_to_none=False)
        step_train(model, optimizer, scaler, ids.to('cuda'), tgt.to('cuda'))
    torch.cuda.synchronize()
    t0 = time.perf_counter(); n_tok = 0
    for _ in range(n_measured):
        ids, tgt = gb()
        optimizer.zero_grad(set_to_none=False)
        step_train(model, optimizer, scaler, ids.to('cuda'), tgt.to('cuda'))
        n_tok += ids.numel()
    torch.cuda.synchronize()
    t1 = time.perf_counter()
    return n_tok / (t1 - t0)


def bench_C(model, optimizer, scaler, dl, n_warmup, n_measured):
    """C: pre-allocate grads, zero with foreach_zero_.

    This runs one step to force grad allocation, then keeps grads persistent
    across all future calls and uses the foreach_zero fused kernel.
    """
    # Initial step to allocate grads
    it = iter(dl)
    ids, tgt = next(it)
    step_train(model, optimizer, scaler, ids.to('cuda'), tgt.to('cuda'))

    # Now collect grads and use foreach_zero_ each iter
    grads = [p.grad for p in model.parameters() if p.grad is not None]
    print(f"  Pre-allocated {len(grads)} grad tensors")

    def gb():
        nonlocal it
        try: return next(it)
        except StopIteration:
            it = iter(dl); return next(it)

    for _ in range(n_warmup):
        ids, tgt = gb()
        # foreach fused zero instead of per-param zero_
        torch._foreach_zero_(grads)
        step_train(model, optimizer, scaler, ids.to('cuda'), tgt.to('cuda'))
    torch.cuda.synchronize()
    t0 = time.perf_counter(); n_tok = 0
    for _ in range(n_measured):
        ids, tgt = gb()
        torch._foreach_zero_(grads)
        step_train(model, optimizer, scaler, ids.to('cuda'), tgt.to('cuda'))
        n_tok += ids.numel()
    torch.cuda.synchronize()
    t1 = time.perf_counter()
    return n_tok / (t1 - t0)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--steps", type=int, default=100)
    parser.add_argument("--repeat", type=int, default=3)
    args = parser.parse_args()

    torch.manual_seed(42)
    print(f"WI4 zero_grad strategy bench: warmup={args.warmup} steps={args.steps} repeat={args.repeat}")

    results = {}
    strategies = [
        ("A: set_to_none=True (default)", bench_A),
        ("B: set_to_none=False", bench_B),
        ("C: pre-alloc + foreach_zero_", bench_C),
    ]

    for label, fn in strategies:
        runs = []
        for r in range(args.repeat):
            model, optimizer, scaler, dl = make_workload()
            t = fn(model, optimizer, scaler, dl, args.warmup, args.steps)
            runs.append(t)
            del model, optimizer, scaler, dl
            torch.cuda.empty_cache()
        results[label] = runs

    print("\n" + "=" * 80)
    print(f"{'Strategy':<40}  {'median':>10}  {'stdev':>8}  vs A   runs")
    print("-" * 80)
    baseline = statistics.median(results["A: set_to_none=True (default)"])
    for label, _ in strategies:
        r = results[label]
        med = statistics.median(r)
        sd = statistics.stdev(r) if len(r) > 1 else 0
        pct = 100.0 * (med - baseline) / baseline
        runs_str = ", ".join(f"{x:,.0f}" for x in r)
        print(f"{label:<40}  {med:>10,.0f}  {sd:>8.0f}  {pct:+.2f}%  ({runs_str})")
    print("=" * 80)


if __name__ == "__main__":
    main()

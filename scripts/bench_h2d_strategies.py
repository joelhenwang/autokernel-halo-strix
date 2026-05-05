"""WI5: Benchmark H2D copy strategies for the input-batch upload.

Compares 4 strategies:
  A: pin_memory=False, .to(device)               (baseline)
  B: pin_memory=False, .to(device, non_blocking=True)
  C: pin_memory=True,  .to(device, non_blocking=True)
  D: pin_memory=True,  CUDA-stream double-buffered prefetch

Each strategy runs a simulated training loop (forward + dummy backward + optimizer step)
for a fixed number of iterations. Reports tok/s for each.

This lets us decide if Strix Halo's unified memory makes pin_memory meaningful, and
whether the full prefetcher is worth the code complexity vs just flipping a flag.

Usage:
  python scripts/bench_h2d_strategies.py --steps 100
"""
import argparse, time, statistics, sys
sys.path.insert(0, '.')
import torch
from torch.utils.data import DataLoader


def make_workload():
    """Build OdinHalo + fp16 autocast + fused AdamW matching production config."""
    from models.odin_halo import OdinHalo
    model = OdinHalo().to('cuda')
    model.train()
    model.compile_zones()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, fused=True)
    scaler = torch.amp.GradScaler('cuda', enabled=True, init_scale=1024.0)
    return model, optimizer, scaler


def make_dataloader(pin_memory, batch_size=16, block_size=256, num_workers=0):
    from halo_training.data import BabyLMDataset
    ds = BabyLMDataset(
        root='datasets/babylm-odin32k.bin',
        block_size=block_size,
        tokenizer_path='tokenizers/odin-32k/tokenizer.json',
    )
    return DataLoader(
        ds, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=pin_memory, drop_last=True,
    )


def one_step(model, optimizer, scaler, input_ids, targets):
    optimizer.zero_grad(set_to_none=True)
    with torch.amp.autocast('cuda', dtype=torch.float16):
        logits = model(input_ids)
        loss = torch.nn.functional.cross_entropy(
            logits.view(-1, logits.size(-1)), targets.view(-1))
    scaler.scale(loss).backward()
    scaler.unscale_(optimizer)
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    scaler.step(optimizer)
    scaler.update()
    return loss


def bench_strategy_A(model, optimizer, scaler, dl, n_warmup, n_measured):
    """A: pin_memory=False, plain .to(device)."""
    it = iter(dl)

    def get_batch():
        nonlocal it
        try:
            return next(it)
        except StopIteration:
            it = iter(dl)
            return next(it)

    for _ in range(n_warmup):
        ids, tgt = get_batch()
        ids = ids.to('cuda')
        tgt = tgt.to('cuda')
        one_step(model, optimizer, scaler, ids, tgt)
    torch.cuda.synchronize()

    t0 = time.perf_counter()
    n_tok = 0
    for _ in range(n_measured):
        ids, tgt = get_batch()
        ids = ids.to('cuda')
        tgt = tgt.to('cuda')
        one_step(model, optimizer, scaler, ids, tgt)
        n_tok += ids.numel()
    torch.cuda.synchronize()
    t1 = time.perf_counter()
    return n_tok / (t1 - t0), t1 - t0


def bench_strategy_B(model, optimizer, scaler, dl, n_warmup, n_measured):
    """B: pin_memory=False, non_blocking=True (no-op on unpinned)."""
    it = iter(dl)

    def get_batch():
        nonlocal it
        try:
            return next(it)
        except StopIteration:
            it = iter(dl)
            return next(it)

    for _ in range(n_warmup):
        ids, tgt = get_batch()
        ids = ids.to('cuda', non_blocking=True)
        tgt = tgt.to('cuda', non_blocking=True)
        one_step(model, optimizer, scaler, ids, tgt)
    torch.cuda.synchronize()

    t0 = time.perf_counter()
    n_tok = 0
    for _ in range(n_measured):
        ids, tgt = get_batch()
        ids = ids.to('cuda', non_blocking=True)
        tgt = tgt.to('cuda', non_blocking=True)
        one_step(model, optimizer, scaler, ids, tgt)
        n_tok += ids.numel()
    torch.cuda.synchronize()
    t1 = time.perf_counter()
    return n_tok / (t1 - t0), t1 - t0


def bench_strategy_C(model, optimizer, scaler, dl, n_warmup, n_measured):
    """C: pin_memory=True, non_blocking=True (should actually async)."""
    # dl already created with pin_memory=True
    it = iter(dl)

    def get_batch():
        nonlocal it
        try:
            return next(it)
        except StopIteration:
            it = iter(dl)
            return next(it)

    for _ in range(n_warmup):
        ids, tgt = get_batch()
        ids = ids.to('cuda', non_blocking=True)
        tgt = tgt.to('cuda', non_blocking=True)
        one_step(model, optimizer, scaler, ids, tgt)
    torch.cuda.synchronize()

    t0 = time.perf_counter()
    n_tok = 0
    for _ in range(n_measured):
        ids, tgt = get_batch()
        ids = ids.to('cuda', non_blocking=True)
        tgt = tgt.to('cuda', non_blocking=True)
        one_step(model, optimizer, scaler, ids, tgt)
        n_tok += ids.numel()
    torch.cuda.synchronize()
    t1 = time.perf_counter()
    return n_tok / (t1 - t0), t1 - t0


class CUDAPrefetcher:
    """Double-buffered H2D prefetcher using a dedicated CUDA stream.

    Issues .to(device, non_blocking=True) onto copy_stream, then records an event.
    The compute stream waits on that event before consuming the batch.
    """
    def __init__(self, loader, device='cuda'):
        self.loader = loader
        self.device = torch.device(device)
        self.copy_stream = torch.cuda.Stream(device=self.device)
        self.it = iter(loader)
        self.next_batch = None
        self.next_event = None
        self._preload()

    def _preload(self):
        try:
            batch = next(self.it)
        except StopIteration:
            self.it = iter(self.loader)
            batch = next(self.it)
        with torch.cuda.stream(self.copy_stream):
            batch = tuple(t.to(self.device, non_blocking=True) for t in batch)
            ev = torch.cuda.Event()
            ev.record(self.copy_stream)
        self.next_batch = batch
        self.next_event = ev

    def next(self):
        # Make compute stream wait on copy_stream
        if self.next_event is not None:
            self.next_event.wait(torch.cuda.current_stream(self.device))
        batch = self.next_batch
        self._preload()
        return batch


def bench_strategy_D(model, optimizer, scaler, dl, n_warmup, n_measured):
    """D: pin_memory=True, CUDA-stream double-buffered prefetch."""
    pf = CUDAPrefetcher(dl, device='cuda')

    for _ in range(n_warmup):
        ids, tgt = pf.next()
        one_step(model, optimizer, scaler, ids, tgt)
    torch.cuda.synchronize()

    t0 = time.perf_counter()
    n_tok = 0
    for _ in range(n_measured):
        ids, tgt = pf.next()
        one_step(model, optimizer, scaler, ids, tgt)
        n_tok += ids.numel()
    torch.cuda.synchronize()
    t1 = time.perf_counter()
    return n_tok / (t1 - t0), t1 - t0


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--warmup", type=int, default=25)
    parser.add_argument("--steps", type=int, default=100)
    parser.add_argument("--num-workers", type=int, default=0,
                        help="DataLoader num_workers (0=main thread)")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--block-size", type=int, default=256)
    parser.add_argument("--repeat", type=int, default=3,
                        help="Repeat each strategy N times, report median")
    args = parser.parse_args()

    torch.manual_seed(42)
    print(f"WI5 H2D strategy benchmark: batch={args.batch_size} block={args.block_size} "
          f"warmup={args.warmup} steps={args.steps} workers={args.num_workers} repeat={args.repeat}")

    # Build workload once
    model, optimizer, scaler = make_workload()

    results = {}
    strategies = [
        ("A: pin=F, .to()", False, bench_strategy_A),
        ("B: pin=F, .to(non_blocking)", False, bench_strategy_B),
        ("C: pin=T, .to(non_blocking)", True, bench_strategy_C),
        ("D: pin=T, prefetcher", True, bench_strategy_D),
    ]

    for label, pin, fn in strategies:
        toks_all = []
        elapsed_all = []
        for r in range(args.repeat):
            dl = make_dataloader(pin_memory=pin, batch_size=args.batch_size,
                                 block_size=args.block_size, num_workers=args.num_workers)
            tok_s, elapsed = fn(model, optimizer, scaler, dl,
                                args.warmup if r == 0 else 5,  # full warm on first rep
                                args.steps)
            toks_all.append(tok_s)
            elapsed_all.append(elapsed)
            # Flush between runs
            del dl
            torch.cuda.empty_cache()
        results[label] = {
            "median_tok_s": statistics.median(toks_all),
            "mean_tok_s": statistics.mean(toks_all),
            "stdev_tok_s": statistics.stdev(toks_all) if len(toks_all) > 1 else 0.0,
            "all_runs": toks_all,
            "elapsed_sec_median": statistics.median(elapsed_all),
        }

    # Report
    print("\n" + "=" * 80)
    print(f"{'Strategy':<32}  {'median tok/s':>12}  {'stdev':>8}  {'runs':}")
    print("-" * 80)
    baseline = results["A: pin=F, .to()"]["median_tok_s"]
    for label, _, _ in strategies:
        r = results[label]
        pct = 100.0 * (r["median_tok_s"] - baseline) / baseline
        runs_str = ", ".join(f"{x:,.0f}" for x in r["all_runs"])
        print(f"{label:<32}  {r['median_tok_s']:>12,.0f}  {r['stdev_tok_s']:>8.0f}  "
              f"{pct:+.2f}%  ({runs_str})")
    print("=" * 80)


if __name__ == "__main__":
    main()

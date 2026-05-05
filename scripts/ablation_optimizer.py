"""Test optimizations beyond compile: fused AdamW, no grad clip, larger batch."""
import sys, time, gc, torch
sys.path.insert(0, '.')

WARMUP = 200
MEASURE = 200
device = 'cuda'

from halo_training.data import BabyLMDataset, build_dataloader
from models.odin_halo import OdinHalo

ds = BabyLMDataset(root='datasets/babylm-odin32k.bin', block_size=256,
                   tokenizer_path='tokenizers/odin-32k/tokenizer.json')


def run(label, build_model, build_opt, use_compile=False,
        grad_clip=True, batch_size=4):
    gc.collect()
    torch.cuda.empty_cache()
    torch.manual_seed(42)
    model = build_model().to(device)
    model.train()

    if use_compile:
        if hasattr(model, "compile_zones"):
            model.compile_zones()

    optimizer = build_opt(model)
    scaler = torch.amp.GradScaler("cuda", enabled=True, init_scale=1024.0)

    dl = build_dataloader(ds, batch_size=batch_size, num_workers=0, shuffle=True)

    def batches():
        while True:
            for b in dl:
                yield b

    it = batches()
    # Warmup
    for _ in range(WARMUP):
        input_ids, targets = next(it)
        input_ids = input_ids.to(device); targets = targets.to(device)
        optimizer.zero_grad(set_to_none=True)
        with torch.amp.autocast('cuda', dtype=torch.float16):
            logits = model(input_ids)
            loss = torch.nn.functional.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
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
            logits = model(input_ids)
            loss = torch.nn.functional.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
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
    print(f"  {label:50s} tok/s={tok_s:>7,.0f}  peak={peak:.2f}GB")
    del model, optimizer, scaler

print("=== Optimizer / batch-size sensitivity (eager baseline, batch=4) ===")
run("default AdamW + grad_clip",
    lambda: OdinHalo(),
    lambda m: torch.optim.AdamW(m.parameters(), lr=1e-4))
run("fused AdamW + grad_clip",
    lambda: OdinHalo(),
    lambda m: torch.optim.AdamW(m.parameters(), lr=1e-4, fused=True))
run("fused AdamW, NO grad_clip",
    lambda: OdinHalo(),
    lambda m: torch.optim.AdamW(m.parameters(), lr=1e-4, fused=True),
    grad_clip=False)
run("fused AdamW + compile",
    lambda: OdinHalo(),
    lambda m: torch.optim.AdamW(m.parameters(), lr=1e-4, fused=True),
    use_compile=True)
run("fused AdamW + compile, NO grad_clip",
    lambda: OdinHalo(),
    lambda m: torch.optim.AdamW(m.parameters(), lr=1e-4, fused=True),
    use_compile=True, grad_clip=False)

print()
print("=== Batch size sensitivity ===")
for bs in [4, 8, 16]:
    try:
        run(f"batch={bs} eager",
            lambda: OdinHalo(),
            lambda m: torch.optim.AdamW(m.parameters(), lr=1e-4, fused=True),
            batch_size=bs)
        run(f"batch={bs} compile",
            lambda: OdinHalo(),
            lambda m: torch.optim.AdamW(m.parameters(), lr=1e-4, fused=True),
            use_compile=True, batch_size=bs)
    except torch.cuda.OutOfMemoryError:
        print(f"  batch={bs}: OOM")
        torch.cuda.empty_cache()

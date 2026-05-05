"""W4 ablation: torch.compile(optimizer.step) effect on throughput.

Two runs:
  1. Baseline: compile_zones + HIP CE + fused AdamW
  2. Same + TORCH_COMPILE_OPTIMIZER=1 (wrap optimizer.step in torch.compile)

400 steps each (200 warmup, 200 measured).
"""
import sys, os, time, gc, torch
sys.path.insert(0, '.')

WARMUP = 200
MEASURE = 200
BATCH_SIZE = 16
BLOCK_SIZE = 256
device = 'cuda'

from halo_training.data import BabyLMDataset, build_dataloader
from models.odin_halo import OdinHalo


def run(label, compile_optimizer=False):
    gc.collect(); torch.cuda.empty_cache()
    torch.manual_seed(42)

    model = OdinHalo().to(device)
    model.train()
    model.compile_zones()

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, fused=True)

    if compile_optimizer:
        try:
            pt_ver = tuple(int(x) for x in torch.__version__.split(".")[:2])
            if pt_ver >= (2, 5):
                optimizer.step = torch.compile(optimizer.step, fullgraph=False)
                print("  optimizer.step compiled")
            else:
                print(f"  WARN: torch < 2.5 ({torch.__version__}), skipping")
                return None
        except Exception as e:
            print(f"  compile(optimizer.step) failed: {e}")
            return None

    scaler = torch.amp.GradScaler("cuda", enabled=True, init_scale=1024.0)
    ds = BabyLMDataset(root='datasets/babylm-odin32k.bin', block_size=BLOCK_SIZE,
                       tokenizer_path='tokenizers/odin-32k/tokenizer.json')
    dl = build_dataloader(ds, batch_size=BATCH_SIZE, num_workers=0, shuffle=True)

    def batches():
        while True:
            for b in dl: yield b
    it = batches()

    for _ in range(WARMUP):
        input_ids, targets = next(it)
        input_ids = input_ids.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
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
    torch.cuda.synchronize()

    torch.cuda.reset_peak_memory_stats()
    t0 = time.time()
    total_tokens = 0
    for _ in range(MEASURE):
        input_ids, targets = next(it)
        input_ids = input_ids.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
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
        total_tokens += input_ids.numel()
    torch.cuda.synchronize()
    elapsed = time.time() - t0
    tok_s = total_tokens / elapsed
    peak = torch.cuda.max_memory_allocated() / 1e9

    print(f"  {label:50s} tok/s={tok_s:>7,.0f}  peak={peak:.2f}GB")
    del model, optimizer, scaler
    return tok_s, peak


print("=== W4: compile(optimizer.step) ablation (OdinHalo, bs=16, 400 steps) ===\n")
print(f"PyTorch: {torch.__version__}\n")

base = run("baseline (compile_zones + fused AdamW)", compile_optimizer=False)
copt = run("+ TORCH_COMPILE_OPTIMIZER=1", compile_optimizer=True)

print()
if base and copt:
    lift = copt[0] / base[0]
    print(f"compile(optimizer.step) lift: {lift:.3f}x  "
          f"({base[0]:,.0f} -> {copt[0]:,.0f} tok/s)")
    print(f"peak memory delta: {copt[1] - base[1]:+.2f} GB")

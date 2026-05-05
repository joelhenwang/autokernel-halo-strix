"""Test chunked CE with different chunk sizes to optimize."""
import sys, time, gc, torch
sys.path.insert(0, '.')

WARMUP = 200
MEASURE = 200
device = 'cuda'

from halo_training.data import BabyLMDataset, build_dataloader
from models.odin_halo import OdinHalo
from kernels.hip.chunked_linear_cross_entropy import ChunkedLinearCrossEntropyLoss

ds = BabyLMDataset(root='datasets/babylm-odin32k.bin', block_size=256,
                   tokenizer_path='tokenizers/odin-32k/tokenizer.json')


def run(label, chunk_size, use_compile=False):
    gc.collect(); torch.cuda.empty_cache()
    torch.manual_seed(42)
    model = OdinHalo(use_chunked_ce=True).to(device)
    model.train()
    if use_compile:
        model.compile_zones()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, fused=True)
    scaler = torch.amp.GradScaler("cuda", enabled=True, init_scale=1024.0)
    chunked_fn = ChunkedLinearCrossEntropyLoss(chunk_size=chunk_size, softcap=30.0)
    dl = build_dataloader(ds, batch_size=16, num_workers=0, shuffle=True)

    def batches():
        while True:
            for b in dl: yield b
    it = batches()

    def loss_fn(h_low, targets):
        return chunked_fn(
            h_low.view(-1, h_low.size(-1)),
            model.lm_head.embed_table.weight,
            targets.view(-1))

    for _ in range(WARMUP):
        input_ids, targets = next(it)
        input_ids = input_ids.to(device); targets = targets.to(device)
        optimizer.zero_grad(set_to_none=True)
        with torch.amp.autocast('cuda', dtype=torch.float16):
            h_low = model(input_ids)
            loss = loss_fn(h_low, targets)
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        scaler.step(optimizer)
        scaler.update()
    torch.cuda.synchronize()

    torch.cuda.reset_peak_memory_stats()
    t0 = time.time()
    total = 0
    for _ in range(MEASURE):
        input_ids, targets = next(it)
        input_ids = input_ids.to(device); targets = targets.to(device)
        optimizer.zero_grad(set_to_none=True)
        with torch.amp.autocast('cuda', dtype=torch.float16):
            h_low = model(input_ids)
            loss = loss_fn(h_low, targets)
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        scaler.step(optimizer)
        scaler.update()
        total += input_ids.numel()
    torch.cuda.synchronize()
    el = time.time() - t0
    tok_s = total / el
    peak = torch.cuda.max_memory_allocated() / 1e9
    print(f"  {label:50s} tok/s={tok_s:>7,.0f}  peak={peak:.2f}GB")
    del model, optimizer, scaler

print("=== Chunk size sensitivity (OdinHalo batch=16) ===")
print("[eager]")
for cs in [128, 256, 512, 1024, 2048, 4096]:
    run(f"chunk_size={cs}", cs, use_compile=False)
print("[compiled]")
for cs in [128, 256, 512, 1024, 2048, 4096]:
    run(f"chunk_size={cs}", cs, use_compile=True)

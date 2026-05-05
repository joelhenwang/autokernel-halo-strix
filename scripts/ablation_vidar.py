"""Quick VidarHalo ablation at batch=16 to compare with OdinHalo."""
import sys, time, gc, torch
sys.path.insert(0, '.')

WARMUP = 200
MEASURE = 200
device = 'cuda'

from halo_training.data import BabyLMDataset, build_dataloader
ds = BabyLMDataset(root='datasets/babylm-odin32k.bin', block_size=256,
                   tokenizer_path='tokenizers/odin-32k/tokenizer.json')

from models.vidar_halo import VidarHalo
from kernels.hip.chunked_linear_cross_entropy import ChunkedLinearCrossEntropyLoss
_chunked = ChunkedLinearCrossEntropyLoss(chunk_size=512)


def run(label, use_compile=False, use_chunked_ce=False):
    gc.collect(); torch.cuda.empty_cache()
    torch.manual_seed(42)
    model = VidarHalo(use_chunked_ce=use_chunked_ce, vocab_size=32768).to(device)
    model.train()
    if use_compile:
        model.compile_zones()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, fused=True)
    scaler = torch.amp.GradScaler("cuda", enabled=True)
    dl = build_dataloader(ds, batch_size=16, num_workers=0, shuffle=True)
    def batches():
        while True:
            for b in dl: yield b
    it = batches()

    def loss_fn(out, targets):
        if use_chunked_ce:
            return _chunked(out.view(-1, out.size(-1)),
                            model.lm_head.embed_table.weight,
                            targets.view(-1))
        return torch.nn.functional.cross_entropy(
            out.view(-1, out.size(-1)), targets.view(-1))

    for _ in range(WARMUP):
        input_ids, targets = next(it)
        input_ids = input_ids.to(device); targets = targets.to(device)
        optimizer.zero_grad(set_to_none=True)
        with torch.amp.autocast('cuda', dtype=torch.float16):
            out = model(input_ids)
            loss = loss_fn(out, targets)
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
            out = model(input_ids)
            loss = loss_fn(out, targets)
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        scaler.step(optimizer)
        scaler.update()
        total += input_ids.numel()
    torch.cuda.synchronize()
    el = time.time() - t0
    tok_s = total / el
    peak = torch.cuda.max_memory_allocated() / 1e9
    print(f"  {label:40s} tok/s={tok_s:>7,.0f}  peak={peak:.2f}GB")
    del model, optimizer, scaler

print("=== VidarHalo @ batch=16 (400 steps, 200 warmup) ===")
run("baseline eager (PyTorch CE)")
run("eager + Chunked CE", use_chunked_ce=True)
run("compile + PyTorch CE", use_compile=True)
run("compile + Chunked CE", use_compile=True, use_chunked_ce=True)

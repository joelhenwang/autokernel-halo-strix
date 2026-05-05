"""Test at batch=32 + reduce-overhead mode to push limits."""
import sys, time, gc, torch
sys.path.insert(0, '.')

WARMUP = 100  # shorter for larger batch to avoid OOM during measurement
MEASURE = 100
device = 'cuda'

from halo_training.data import BabyLMDataset, build_dataloader
from models.odin_halo import OdinHalo
import kernel as opt_ce
from kernels.hip.chunked_linear_cross_entropy import ChunkedLinearCrossEntropyLoss

ds = BabyLMDataset(root='datasets/babylm-odin32k.bin', block_size=256,
                   tokenizer_path='tokenizers/odin-32k/tokenizer.json')

def hip_ce(model, out, targets):
    return opt_ce.ce_full(out.view(-1, out.size(-1)), targets.view(-1), mode="tiny")

_chunked_fn = ChunkedLinearCrossEntropyLoss(chunk_size=256, softcap=30.0)
def chunked_ce(model, h_low, targets):
    return _chunked_fn(h_low.view(-1, h_low.size(-1)),
                       model.lm_head.embed_table.weight,
                       targets.view(-1))


def run(label, batch_size, use_compile=False, compile_mode="default",
        use_chunked_ce=False, ce="hip_tiny"):
    gc.collect(); torch.cuda.empty_cache()
    torch.manual_seed(42)
    model = OdinHalo(use_chunked_ce=use_chunked_ce).to(device)
    model.train()
    if use_compile:
        for i in range(len(model.shared_layers)):
            model.shared_layers[i] = torch.compile(
                model.shared_layers[i], mode=compile_mode)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, fused=True)
    scaler = torch.amp.GradScaler("cuda", enabled=True, init_scale=1024.0)
    dl = build_dataloader(ds, batch_size=batch_size, num_workers=0, shuffle=True)
    def batches():
        while True:
            for b in dl: yield b
    it = batches()
    loss_fn = chunked_ce if use_chunked_ce else hip_ce

    try:
        for _ in range(WARMUP):
            input_ids, targets = next(it)
            input_ids = input_ids.to(device); targets = targets.to(device)
            optimizer.zero_grad(set_to_none=True)
            try: torch.compiler.cudagraph_mark_step_begin()
            except: pass
            with torch.amp.autocast('cuda', dtype=torch.float16):
                out = model(input_ids)
                loss = loss_fn(model, out, targets)
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
            try: torch.compiler.cudagraph_mark_step_begin()
            except: pass
            with torch.amp.autocast('cuda', dtype=torch.float16):
                out = model(input_ids)
                loss = loss_fn(model, out, targets)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            scaler.step(optimizer)
            scaler.update()
            total += input_ids.numel()
        torch.cuda.synchronize()
        el = time.time() - t0
        tok_s = total / el
        peak = torch.cuda.max_memory_allocated() / 1e9
        print(f"  {label:55s} tok/s={tok_s:>7,.0f}  peak={peak:.2f}GB")
    except torch.cuda.OutOfMemoryError:
        print(f"  {label:55s} OOM")
        torch.cuda.empty_cache()
    except Exception as e:
        print(f"  {label:55s} FAIL: {type(e).__name__}: {str(e)[:120]}")
    del model, optimizer, scaler


print("=== batch=32 + large-batch scaling ===")
for bs in [16, 32]:
    print(f"\n[batch={bs}]")
    run(f"eager + HIP CE", bs, False)
    run(f"eager + Chunked CE", bs, False, use_chunked_ce=True)
    run(f"compile default + HIP CE", bs, True, "default")
    run(f"compile default + Chunked CE", bs, True, "default", use_chunked_ce=True)
    run(f"compile reduce-overhead + HIP CE", bs, True, "reduce-overhead")
    run(f"compile reduce-overhead + Chunked CE", bs, True, "reduce-overhead", use_chunked_ce=True)

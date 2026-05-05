"""Test torch.compile modes: default vs reduce-overhead vs max-autotune."""
import sys, time, gc, torch
sys.path.insert(0, '.')

WARMUP = 200
MEASURE = 200
device = 'cuda'

from halo_training.data import BabyLMDataset, build_dataloader

ds = BabyLMDataset(root='datasets/babylm-odin32k.bin', block_size=256,
                   tokenizer_path='tokenizers/odin-32k/tokenizer.json')


def load_batches():
    dl = build_dataloader(ds, batch_size=4, num_workers=0, shuffle=True)
    while True:
        for batch in dl:
            yield batch


def run(label, compile_mode=None, friendly=False):
    gc.collect()
    torch.cuda.empty_cache()
    torch.manual_seed(42)

    from models.odin_halo import OdinHalo
    model = OdinHalo().to(device)
    model.train()

    if compile_mode:
        if friendly:
            # Custom compile with friendly path
            from models.components.conv_blocks import HyPEShortConvBlock
            for layer in model.shared_layers:
                if isinstance(layer, HyPEShortConvBlock):
                    layer._compile_friendly = True
            for i in range(len(model.shared_layers)):
                model.shared_layers[i] = torch.compile(model.shared_layers[i], mode=compile_mode)
        else:
            for i in range(len(model.shared_layers)):
                model.shared_layers[i] = torch.compile(model.shared_layers[i], mode=compile_mode)

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    scaler = torch.amp.GradScaler("cuda", enabled=True, init_scale=1024.0)
    batches = load_batches()

    # Warmup
    w0 = time.time()
    for _ in range(WARMUP):
        input_ids, targets = next(batches)
        input_ids = input_ids.to(device); targets = targets.to(device)
        optimizer.zero_grad(set_to_none=True)
        # cudagraph step boundary (for reduce-overhead mode)
        try:
            torch.compiler.cudagraph_mark_step_begin()
        except Exception:
            pass
        with torch.amp.autocast('cuda', dtype=torch.float16):
            logits = model(input_ids)
            loss = torch.nn.functional.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(optimizer)
        scaler.update()
    torch.cuda.synchronize()
    warm = time.time() - w0

    torch.cuda.reset_peak_memory_stats()
    t0 = time.time()
    total_tokens = 0
    for _ in range(MEASURE):
        input_ids, targets = next(batches)
        input_ids = input_ids.to(device); targets = targets.to(device)
        optimizer.zero_grad(set_to_none=True)
        try:
            torch.compiler.cudagraph_mark_step_begin()
        except Exception:
            pass
        with torch.amp.autocast('cuda', dtype=torch.float16):
            logits = model(input_ids)
            loss = torch.nn.functional.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(optimizer)
        scaler.update()
        total_tokens += input_ids.numel()
    torch.cuda.synchronize()
    el = time.time() - t0
    tok_s = total_tokens / el
    peak = torch.cuda.max_memory_allocated() / 1e9
    print(f"  {label:45s} tok/s={tok_s:>6,.0f}  peak={peak:.2f}GB  warmup={warm:.1f}s")

    del model, optimizer, scaler
    gc.collect()
    torch.cuda.empty_cache()


print("=== compile mode comparison (OdinHalo V=32768, baseline CE) ===")
run("eager (no compile)", None)
run("compile mode=default", "default")
run("compile mode=default + friendly", "default", friendly=True)
run("compile mode=reduce-overhead", "reduce-overhead")
run("compile mode=reduce-overhead + friendly", "reduce-overhead", friendly=True)
# max-autotune takes a long time to warm up; try optionally
try:
    run("compile mode=max-autotune + friendly", "max-autotune", friendly=True)
except Exception as e:
    print(f"  max-autotune failed: {str(e)[:200]}")

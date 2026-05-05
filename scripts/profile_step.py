"""Profile one training step to find where time is spent."""
import sys, torch
sys.path.insert(0, '.')

from models.odin_halo import OdinHalo
from halo_training.data import BabyLMDataset, build_dataloader

device = 'cuda'
torch.manual_seed(42)

model = OdinHalo().to(device)
model.train()

ds = BabyLMDataset(root='datasets/babylm-odin32k.bin', block_size=256,
                   tokenizer_path='tokenizers/odin-32k/tokenizer.json')
dl = build_dataloader(ds, batch_size=4, num_workers=0, shuffle=True)
batches = iter(dl)

optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
scaler = torch.amp.GradScaler("cuda", enabled=True, init_scale=1024.0)

# Warmup
for _ in range(20):
    input_ids, targets = next(batches)
    input_ids = input_ids.to(device); targets = targets.to(device)
    optimizer.zero_grad(set_to_none=True)
    with torch.amp.autocast('cuda', dtype=torch.float16):
        logits = model(input_ids)
        loss = torch.nn.functional.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
    scaler.scale(loss).backward()
    scaler.unscale_(optimizer)
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    scaler.step(optimizer)
    scaler.update()
torch.cuda.synchronize()

# Profile 5 steps
from torch.profiler import profile, ProfilerActivity, schedule
with profile(
    activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
    schedule=schedule(wait=1, warmup=1, active=3, repeat=1),
    record_shapes=False,
) as prof:
    for _ in range(5):
        input_ids, targets = next(batches)
        input_ids = input_ids.to(device); targets = targets.to(device)
        optimizer.zero_grad(set_to_none=True)
        with torch.amp.autocast('cuda', dtype=torch.float16):
            logits = model(input_ids)
            loss = torch.nn.functional.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(optimizer)
        scaler.update()
        prof.step()

print("=== TOP OPS BY GPU TIME (3 steps averaged) ===")
print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=25))

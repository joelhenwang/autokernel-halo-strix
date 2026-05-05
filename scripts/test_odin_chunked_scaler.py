"""Repro: test OdinHalo chunked CE with GradScaler to match trainer's fp16 path."""
import sys, torch, traceback
sys.path.insert(0, '.')

from models.odin_halo import OdinHalo
from kernels.hip.chunked_linear_cross_entropy import ChunkedLinearCrossEntropyLoss

device = 'cuda'
torch.manual_seed(42)

model = OdinHalo(use_chunked_ce=True).to(device)
model.train()
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, fused=True)
scaler = torch.amp.GradScaler("cuda", enabled=True, init_scale=1024.0)

chunked_loss_fn = ChunkedLinearCrossEntropyLoss(
    chunk_size=256, softcap=30.0, ignore_index=-100,
)

for step in range(3):
    input_ids = torch.randint(0, model.vocab_size, (4, 256), device=device)
    targets = torch.randint(0, model.vocab_size, (4, 256), device=device)
    optimizer.zero_grad(set_to_none=True)
    with torch.amp.autocast("cuda", dtype=torch.float16):
        h_low = model(input_ids)
        print(f"step {step} h_low: shape={tuple(h_low.shape)} dtype={h_low.dtype} requires_grad={h_low.requires_grad}")
        weight = model.lm_head.embed_table.weight
        print(f"  weight: shape={tuple(weight.shape)} dtype={weight.dtype} requires_grad={weight.requires_grad}")
        loss = chunked_loss_fn(
            h_low.view(-1, h_low.size(-1)),
            weight,
            targets.view(-1),
        )
        print(f"  loss: {loss.item():.4f} dtype={loss.dtype}")
    try:
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        scaler.step(optimizer)
        scaler.update()
        torch.cuda.synchronize()
        print(f"  step {step} OK scale={scaler.get_scale()}")
    except Exception as e:
        print(f"  step {step} FAILED: {e}")
        traceback.print_exc()
        break

print("done")

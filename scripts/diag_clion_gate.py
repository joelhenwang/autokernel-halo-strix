"""Diagnose CLion gate firing rate on OdinHalo (per_coord and per_tensor)."""
import sys, torch
sys.path.insert(0, '.')

from models.odin_halo import OdinHalo
from halo_training.clion import CLion
from halo_training.data import BabyLMDataset, build_dataloader

device = 'cuda'
torch.manual_seed(42)

# Build OdinHalo as the shootout does
model = OdinHalo().to(device)
model.train()
opt = CLion(model.parameters(), lr=3e-5, betas=(0.9, 0.99),
            weight_decay=0.1, nu=1e-3, gate_mode="per_coord")

ds = BabyLMDataset(root='datasets/babylm-odin32k.bin', block_size=256,
                   tokenizer_path='tokenizers/odin-32k/tokenizer.json')
dl = build_dataloader(ds, batch_size=16, num_workers=0, shuffle=True)
scaler = torch.amp.GradScaler("cuda", enabled=True)
batches = iter(dl)

for step in range(5):
    input_ids, targets = next(batches)
    input_ids = input_ids.to(device); targets = targets.to(device)
    opt.zero_grad(set_to_none=True)
    with torch.amp.autocast('cuda', dtype=torch.float16):
        logits = model(input_ids)
        loss = torch.nn.functional.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
    scaler.scale(loss).backward()
    scaler.unscale_(opt)

    # Probe: for the lm_head weight, how many coords cross the gate?
    total_elems = 0
    total_sign_coords = 0
    sample_stats = []
    for group in opt.param_groups:
        beta1, _ = group["betas"]
        nu = group["nu"]
        for p in group["params"]:
            if p.grad is None:
                continue
            state = opt.state[p]
            exp_avg = state.get("exp_avg", torch.zeros_like(p))
            c = exp_avg * beta1 + p.grad * (1 - beta1)
            n = c.numel()
            sign_mask = c.abs() >= nu
            n_sign = sign_mask.sum().item()
            total_elems += n
            total_sign_coords += n_sign
            # Median and mean abs
            c_flat = c.abs().flatten()
            sample_stats.append((tuple(p.shape), n, n_sign,
                                 c_flat.median().item(),
                                 c_flat.mean().item()))

    pct = 100.0 * total_sign_coords / total_elems
    print(f"Step {step}: loss={loss.item():.4f}, sign_coords={total_sign_coords:,}/{total_elems:,} ({pct:.1f}%)")
    # Show 3 tensors with smallest % coords passing gate
    sample_stats.sort(key=lambda x: x[2]/x[1] if x[1] > 0 else 0)
    for shape, n, n_sign, med, mean in sample_stats[:3]:
        print(f"  {shape}: n={n:,} sign={n_sign:,} ({100*n_sign/n:.1f}%)  median|c|={med:.2e}  mean|c|={mean:.2e}")

    scaler.step(opt)
    scaler.update()

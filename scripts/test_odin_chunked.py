"""End-to-end integration test: OdinHalo with use_chunked_ce=True.

Verifies that:
  1. OdinHalo builds with use_chunked_ce=True
  2. Training step runs without error
  3. Loss is close to baseline OdinHalo (same init)
  4. Peak memory is lower
"""
import sys, time, torch
sys.path.insert(0, '.')

from models.odin_halo import OdinHalo

device = 'cuda'
torch.manual_seed(42)


def run_steps(model_fn, loss_fn, n_steps=5, seed=42):
    torch.manual_seed(seed)
    model = model_fn().to(device)
    model.train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()

    losses = []
    t0 = time.time()
    for step in range(n_steps):
        input_ids = torch.randint(0, model.vocab_size, (4, 256), device=device)
        targets = torch.randint(0, model.vocab_size, (4, 256), device=device)
        with torch.amp.autocast('cuda', dtype=torch.float16):
            output = model(input_ids)
            if isinstance(output, torch.Tensor) and output.dim() == 0:
                loss = output
            else:
                loss = loss_fn(model, output, targets)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        losses.append(loss.item())
    torch.cuda.synchronize()
    elapsed = time.time() - t0
    peak_mb = torch.cuda.max_memory_allocated() / 1e6
    return losses, peak_mb, elapsed


# Baseline: standard OdinHalo, F.cross_entropy
def baseline_loss(model, logits, targets):
    return torch.nn.functional.cross_entropy(
        logits.view(-1, logits.size(-1)), targets.view(-1))

from kernels.hip.chunked_linear_cross_entropy import ChunkedLinearCrossEntropyLoss
chunked_loss_fn = ChunkedLinearCrossEntropyLoss(
    chunk_size=256, softcap=30.0, ignore_index=-100,
)

def chunked_loss(model, h_low, targets):
    return chunked_loss_fn(
        h_low.view(-1, h_low.size(-1)),
        model.lm_head.embed_table.weight,
        targets.view(-1),
    )

print("=" * 70)
print("OdinHalo end-to-end chunked CE integration test")
print("=" * 70)

print("\n[Baseline OdinHalo (F.cross_entropy)]")
losses1, peak1, t1 = run_steps(
    lambda: OdinHalo(use_chunked_ce=False),
    baseline_loss, n_steps=5, seed=42,
)
print(f"  losses: {[f'{l:.4f}' for l in losses1]}")
print(f"  peak_mb: {peak1:.1f}")
print(f"  time: {t1:.2f}s")

print("\n[OdinHalo with use_chunked_ce=True]")
losses2, peak2, t2 = run_steps(
    lambda: OdinHalo(use_chunked_ce=True),
    chunked_loss, n_steps=5, seed=42,
)
print(f"  losses: {[f'{l:.4f}' for l in losses2]}")
print(f"  peak_mb: {peak2:.1f}")
print(f"  time: {t2:.2f}s")

# Compare first-step loss — should be within ~1% (same init, different path)
first_diff = abs(losses1[0] - losses2[0]) / max(abs(losses1[0]), 1e-6)
print(f"\nFirst-step loss rel diff: {first_diff:.4f}  "
      f"({'PASS' if first_diff < 0.02 else 'FAIL'})")
print(f"Peak memory delta: {peak1-peak2:+.1f} MB  "
      f"({'PASS' if peak2 < peak1 else 'NO IMPROVEMENT'})")

print("\nDone.")

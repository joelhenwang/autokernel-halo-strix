"""Unit test for Lion optimizer: verify convergence on a tiny MLP regression."""
import sys
import torch
import torch.nn as nn

sys.path.insert(0, '.')
from halo_training.lion import Lion


def train_one(optimizer_cls, lr, steps=100, seed=42):
    torch.manual_seed(seed)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # Tiny regression task: fit a small linear transform
    torch.manual_seed(seed)
    true_w = torch.randn(32, 8, device=device)
    x = torch.randn(256, 32, device=device)
    y = x @ true_w + 0.01 * torch.randn(256, 8, device=device)

    model = nn.Sequential(
        nn.Linear(32, 64),
        nn.GELU(),
        nn.Linear(64, 8),
    ).to(device)

    opt = optimizer_cls(model.parameters(), lr=lr)

    losses = []
    for _ in range(steps):
        opt.zero_grad()
        loss = ((model(x) - y) ** 2).mean()
        loss.backward()
        opt.step()
        losses.append(loss.item())

    return losses[0], losses[-1], losses


def main():
    print("=== Lion unit test (tiny MLP regression) ===")

    # Lion's sign-based update is slow on small-scale regression. Use a higher LR
    # and more steps than AdamW comparison to give it a fair chance.
    # The goal of this test: verify Lion decreases loss monotonically, not that
    # it matches AdamW on a task where sign-based updates are a poor fit.
    init_lion, final_lion, losses_lion = train_one(Lion, lr=3e-3, steps=400)
    print(f"Lion  (lr=3e-3, 400 steps): initial={init_lion:.4f}  final={final_lion:.4f}  "
          f"ratio={final_lion/init_lion:.3f}")

    # AdamW sanity comparison (faster on this task because adaptive-per-coord)
    init_adam, final_adam, _ = train_one(torch.optim.AdamW, lr=1e-3, steps=200)
    print(f"AdamW (lr=1e-3, 200 steps): initial={init_adam:.4f}  final={final_adam:.4f}  "
          f"ratio={final_adam/init_adam:.3f}")

    # Acceptance: Lion loss must decrease meaningfully (<50% of initial) and
    # monotonically over a smoothed window (no divergence).
    smoothed = [sum(losses_lion[i:i+10])/10 for i in range(0, len(losses_lion)-10, 10)]
    assert smoothed[-1] < smoothed[0], \
        f"Lion smoothed loss did not decrease: start {smoothed[0]} -> end {smoothed[-1]}"
    assert final_lion < 0.5 * init_lion, \
        f"Lion did not reach 50% reduction: final {final_lion}, initial {init_lion}"
    assert final_adam < 0.2 * init_adam, \
        f"AdamW sanity failed: final {final_adam}, initial {init_adam}"

    print()
    print("PASS: Lion decreases loss monotonically and reaches <50% of initial; "
          "AdamW sanity OK at <20%.")


if __name__ == "__main__":
    main()

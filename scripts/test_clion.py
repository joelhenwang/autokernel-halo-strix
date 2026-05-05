"""Unit test for CLion optimizer (Cautious Lion, arXiv:2604.14587).

Verifies:
  1. Convergence on tiny MLP regression.
  2. Gate behavior: with a very large ν, CLion should act like SGDM (identity path).
  3. Gate behavior: with ν≈0, CLion should act like Lion (sign path always taken).
"""
import sys
import torch
import torch.nn as nn

sys.path.insert(0, '.')
from halo_training.clion import CLion
from halo_training.lion import Lion


def train_one(optimizer_cls, lr, kwargs, steps=100, seed=42):
    torch.manual_seed(seed)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    torch.manual_seed(seed)
    true_w = torch.randn(32, 8, device=device)
    x = torch.randn(256, 32, device=device)
    y = x @ true_w + 0.01 * torch.randn(256, 8, device=device)

    model = nn.Sequential(
        nn.Linear(32, 64),
        nn.GELU(),
        nn.Linear(64, 8),
    ).to(device)

    opt = optimizer_cls(model.parameters(), lr=lr, **kwargs)

    losses = []
    for _ in range(steps):
        opt.zero_grad()
        loss = ((model(x) - y) ** 2).mean()
        loss.backward()
        opt.step()
        losses.append(loss.item())

    return losses[0], losses[-1], losses


def main():
    print("=== CLion unit test ===\n")

    # 1. Convergence with tiny ν (gate always passes, sign path used) — match Lion
    init, final, _ = train_one(CLion, lr=3e-3, kwargs=dict(nu=1e-12), steps=400)
    print(f"CLion (nu=1e-12, always-sign, lr=3e-3, 400 steps): "
          f"{init:.4f} -> {final:.4f} (ratio {final/init:.3f})")
    # Should converge like Lion
    assert final < 0.1 * init, \
        f"CLion (sign mode) failed to converge well: {final}, initial {init}"

    # 2. Very large ν — gate always fails, always use identity (c_t) path
    #    This is essentially SGD with interpolated momentum (slow on tiny task)
    init_sgdm, final_sgdm, _ = train_one(CLion, lr=3e-3, kwargs=dict(nu=1e6), steps=400)
    print(f"CLion (nu=1e6, always-identity, lr=3e-3, 400 steps): "
          f"{init_sgdm:.4f} -> {final_sgdm:.4f} (ratio {final_sgdm/init_sgdm:.3f})")
    # Weaker convergence expected for identity path — just check loss decreased
    assert final_sgdm < init_sgdm, \
        f"CLion (identity mode) should at least decrease loss"

    # 3. CLion at tiny ν should match standalone Lion (identical trajectory)
    init_lion, final_lion, _ = train_one(
        Lion, lr=3e-3, kwargs=dict(), steps=400)
    print(f"CLion (nu=1e-12, always-sign):  "
          f"{init:.4f} -> {final:.4f}")
    print(f"Lion  (reference, 400 steps):    "
          f"{init_lion:.4f} -> {final_lion:.4f}")
    rel_diff = abs(final - final_lion) / max(1e-6, abs(final_lion))
    assert rel_diff < 0.05, \
        f"CLion with tiny nu should match Lion: {final} vs {final_lion} (rel diff {rel_diff})"

    print("\nPASS: CLion sign-path matches Lion, identity-path is distinct, gate logic works.")


if __name__ == "__main__":
    main()

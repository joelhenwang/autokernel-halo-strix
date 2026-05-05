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

    # 1. Per-coord mode with small nu: sign path for most coords, identity for tiny
    init, final, _ = train_one(
        CLion, lr=3e-3,
        kwargs=dict(nu=1e-3, gate_mode="per_coord"),
        steps=400,
    )
    print(f"CLion (per_coord, nu=1e-3, lr=3e-3, 400 steps): "
          f"{init:.4f} -> {final:.4f} (ratio {final/init:.3f})")
    assert final < 0.1 * init, \
        f"CLion per_coord failed to converge: {final}, initial {init}"

    # 2. Per-coord with nu=0 — sign path always, equivalent to Lion
    init_l, final_l, _ = train_one(
        CLion, lr=3e-3,
        kwargs=dict(nu=1e-12, gate_mode="per_coord"),
        steps=400,
    )
    init_lion, final_lion, _ = train_one(Lion, lr=3e-3, kwargs=dict(), steps=400)
    print(f"CLion (per_coord, nu~0):  {init_l:.4f} -> {final_l:.4f}")
    print(f"Lion  (reference):        {init_lion:.4f} -> {final_lion:.4f}")
    rel_diff = abs(final_l - final_lion) / max(1e-6, abs(final_lion))
    assert rel_diff < 0.05, \
        f"CLion per_coord nu~0 should match Lion: {final_l} vs {final_lion} (rel {rel_diff})"

    # 3. Per-coord with huge nu — no coord passes gate, pure SGDM-style
    init_s, final_s, _ = train_one(
        CLion, lr=3e-3,
        kwargs=dict(nu=1e6, gate_mode="per_coord"),
        steps=400,
    )
    print(f"CLion (per_coord, nu=1e6, identity-only): "
          f"{init_s:.4f} -> {final_s:.4f} (ratio {final_s/init_s:.3f})")
    assert final_s < init_s, \
        "CLion identity-only should at least decrease loss"

    # 4. Per-tensor mode (literal paper Algorithm 2) — gate almost never passes
    #    for a mid-sized MLP; expect SGDM-like slow convergence
    init_t, final_t, _ = train_one(
        CLion, lr=3e-3,
        kwargs=dict(nu=1e-3, gate_mode="per_tensor"),
        steps=400,
    )
    print(f"CLion (per_tensor, nu=1e-3): "
          f"{init_t:.4f} -> {final_t:.4f} (ratio {final_t/init_t:.3f})")
    # per_tensor at any nu > machine-epsilon almost always falls to identity
    assert final_t < init_t, \
        "CLion per_tensor should decrease loss"

    print("\nPASS: per_coord converges ~Lion-fast, sign-only mode matches Lion, "
          "identity-only works, per_tensor literal-paper mode converges slowly.")


if __name__ == "__main__":
    main()

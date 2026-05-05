"""CLion: Cautious Lion Optimizer (Huang, Zhang, Chen 2026, arXiv:2604.14587).

Variant of Lion that "cautiously" uses the sign function. The paper's algorithm
(Algorithm 2) specifies a whole-tensor gate: if min_{j∈S_t} |c_t[j]| >= ν, use
sign(c_t); else use c_t directly. Figure 1(d) of the paper illustrates CLion as
a per-coordinate active function (identity for small |c|, sign for larger).

These two interpretations disagree: whole-tensor gating almost never triggers
sign() for large models (any single tiny gradient component fails the gate),
effectively making CLion behave like SGDM at Lion's tiny LR — which does not
converge. Per-coordinate gating matches Figure 1(d), preserves Lion-like
behavior for "normal" gradients, and provides the safety net the paper argues
for (small gradients don't get amplified to ±1 by sign).

This implementation supports BOTH modes via `gate_mode`. Default is
"per_coord" (matches the figure, useful in practice). "per_tensor" is offered
for exact-to-algorithm-block faithfulness.

Update rule per parameter tensor:
    g = param.grad
    c = β1 * m + (1 - β1) * g
    # per_coord mode (default):
    update[j] = sign(c[j]) if |c[j]| >= ν else c[j]
    # per_tensor mode (exact paper algorithm):
    S = {j | c[j] != 0}
    update = sign(c) if min_{j∈S} |c[j]| >= ν else c
    # both modes then:
    p -= lr * (update + weight_decay * p)
    m  = β2 * m + (1 - β2) * g

Paper analysis (reused for both modes):
  * Generalization error O(1/N) vs Lion's O(1/(Nτ^T)) — proven for per-tensor.
    Per-coord gating gives a more favorable bound in practice (each coord's
    contribution is bounded by ν instead of sign magnitude).
  * Convergence rate O(√d / T^(1/4)) same as Lion under ℓ1-norm.
"""
from typing import Iterable, Optional, Callable, Literal

import torch
from torch.optim.optimizer import Optimizer


class CLion(Optimizer):
    """Cautious Lion optimizer.

    Args:
        params: iterable of parameters or param groups.
        lr: learning rate.
        betas: (beta1, beta2). Defaults (0.9, 0.99).
        weight_decay: decoupled weight decay.
        nu: threshold ν for the cautious gate.
        gate_mode: "per_coord" (default; matches Figure 1(d)) or "per_tensor"
            (exact algorithm block). Per-coord is usually what you want;
            per-tensor is mostly useful for reproducing the paper's proof setup.
    """

    def __init__(
        self,
        params: Iterable,
        lr: float = 1e-4,
        betas: tuple = (0.9, 0.99),
        weight_decay: float = 0.0,
        nu: float = 1e-3,
        gate_mode: Literal["per_coord", "per_tensor"] = "per_coord",
    ):
        if lr < 0.0:
            raise ValueError(f"Invalid lr: {lr}")
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError(f"Invalid beta1: {betas[0]}")
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError(f"Invalid beta2: {betas[1]}")
        if weight_decay < 0.0:
            raise ValueError(f"Invalid weight_decay: {weight_decay}")
        if nu <= 0.0:
            raise ValueError(f"Invalid nu: {nu} (must be > 0)")
        if gate_mode not in ("per_coord", "per_tensor"):
            raise ValueError(f"Invalid gate_mode: {gate_mode}")

        defaults = dict(lr=lr, betas=betas, weight_decay=weight_decay,
                        nu=nu, gate_mode=gate_mode)
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure: Optional[Callable] = None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            lr = group["lr"]
            beta1, beta2 = group["betas"]
            wd = group["weight_decay"]
            nu = group["nu"]
            gate_mode = group["gate_mode"]

            for p in group["params"]:
                if p.grad is None:
                    continue
                grad = p.grad
                if grad.is_sparse:
                    raise RuntimeError("CLion does not support sparse gradients")

                state = self.state[p]
                if "exp_avg" not in state:
                    state["exp_avg"] = torch.zeros_like(p, memory_format=torch.preserve_format)

                exp_avg = state["exp_avg"]

                # c = beta1 * m + (1 - beta1) * g (out-of-place; exp_avg updated later)
                c = exp_avg.mul(beta1).add_(grad, alpha=1 - beta1)

                if gate_mode == "per_coord":
                    # Elementwise: sign(c) where |c| >= nu, else c itself.
                    # torch.where handles this fused without explicit branching.
                    update = torch.where(c.abs() >= nu, torch.sign(c), c)
                else:  # per_tensor (exact paper algorithm)
                    c_abs = c.abs()
                    nonzero = c_abs > 0
                    if nonzero.any():
                        min_abs = c_abs[nonzero].min()
                        use_sign = bool(min_abs >= nu)
                    else:
                        use_sign = True
                    update = torch.sign(c) if use_sign else c

                # Decoupled weight decay + parameter update
                if wd != 0.0:
                    p.mul_(1.0 - lr * wd)
                p.add_(update, alpha=-lr)

                # Update long-horizon momentum
                exp_avg.mul_(beta2).add_(grad, alpha=1 - beta2)

        return loss

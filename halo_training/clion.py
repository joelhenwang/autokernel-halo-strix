"""CLion: Cautious Lion Optimizer (Huang, Zhang, Chen 2026, arXiv:2604.14587).

Variant of Lion that "cautiously" uses the sign function: for each parameter
tensor, check whether the smallest absolute value among non-zero elements of the
momentum-gradient blend c_t exceeds a threshold ν. If yes, use sign(c_t)
(standard Lion). If no, use c_t directly (identity update, no sign binarization).

Advantages per paper:
  * Generalization error O(1/N) vs Lion's O(1/(Nτ^T)), where τ can be very small.
  * Same convergence rate as Lion: O(√d / T^(1/4)) under ℓ1-norm of gradient.
  * Avoids gradient explosion when c_t has tiny components (sign amplifies them
    to ±1, which the identity path leaves small).

Update rule per param tensor:
    g = param.grad
    c = β1 * m + (1 - β1) * g
    S = {j | c[j] != 0}
    if min_{j∈S} |c[j]| >= ν:
        update = sign(c)
    else:
        update = c
    p -= lr * (update + weight_decay * p)
    m  = β2 * m + (1 - β2) * g

Notes:
  * Default ν = 1.0 per Theorem 2 (generalization). Paper's convergence analysis
    uses ν >= O(1/√d), so smaller ν is also theoretically OK.
  * The gating is whole-tensor (all-or-nothing), not per-coordinate.
  * Edge case: if all c[j] == 0 (vacuous S_t), we use sign (which is zero
    everywhere) — consistent with Lion's behavior on zero gradients.
"""
from typing import Iterable, Optional, Callable

import torch
from torch.optim.optimizer import Optimizer


class CLion(Optimizer):
    """Cautious Lion optimizer.

    Args:
        params: iterable of parameters or param groups.
        lr: learning rate (same scale as Lion; recommend ~3x smaller than AdamW).
        betas: (beta1, beta2). Defaults (0.9, 0.99).
        weight_decay: decoupled weight decay.
        nu: threshold ν for the cautious gate. Default 1.0 (paper generalization
            theorem value). For large models, try ν = 1/√d or smaller.
    """

    def __init__(
        self,
        params: Iterable,
        lr: float = 1e-4,
        betas: tuple = (0.9, 0.99),
        weight_decay: float = 0.0,
        nu: float = 1.0,
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

        defaults = dict(lr=lr, betas=betas, weight_decay=weight_decay, nu=nu)
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

                # c = beta1 * m + (1 - beta1) * g  (without mutating exp_avg yet)
                c = exp_avg.mul(beta1).add_(grad, alpha=1 - beta1)

                # Cautious gate: check min absolute value over non-zero entries.
                # Use .abs() once; gate is a scalar branch per tensor.
                c_abs = c.abs()
                # Non-zero mask — in fp16/fp32, denormals are vanishingly rare
                # for momentum-blended values. We test strictly != 0.
                nonzero = c_abs > 0
                if nonzero.any():
                    min_abs = c_abs[nonzero].min()
                    use_sign = bool(min_abs >= nu)
                else:
                    # All zeros: sign(0) = 0, c itself is 0 — either path is a no-op.
                    use_sign = True

                if use_sign:
                    update = torch.sign(c)
                else:
                    update = c

                # Decoupled weight decay + update
                if wd != 0.0:
                    p.mul_(1.0 - lr * wd)
                p.add_(update, alpha=-lr)

                # Update long-horizon momentum: m = beta2 * m + (1 - beta2) * g
                exp_avg.mul_(beta2).add_(grad, alpha=1 - beta2)

        return loss

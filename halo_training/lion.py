"""Lion optimizer (Chen et al. 2023, https://arxiv.org/abs/2302.06675).

Lion (EvoLved sIgn mOmentum): one EMA buffer per param (vs AdamW's two),
update is sign of interpolated momentum. Typical LR ~3-10x smaller than AdamW.

Update rule per step:
    g = param.grad
    update = sign(β1 * m + (1 - β1) * g)
    param -= lr * (update + weight_decay * param)
    m     = β2 * m + (1 - β2) * g

Reference: arXiv 2302.06675.
Implementation notes:
  * foreach dispatch is NOT used here; one explicit torch.sign call per param.
    If this proves a bottleneck for large param counts, a foreach/fused version
    can be added later.
  * decoupled weight decay (applied directly to param before sign update)
    matches AdamW's "W" (weight-decay) variant.
"""
from typing import Iterable, Optional, Callable

import torch
from torch.optim.optimizer import Optimizer


class Lion(Optimizer):
    """Lion: sign-based first-moment optimizer.

    Args:
        params: iterable of parameters or param groups.
        lr: learning rate (recommend ~3x smaller than AdamW base).
        betas: (beta1, beta2) — beta1 for update-direction EMA, beta2 for momentum store.
            Defaults (0.9, 0.99) from the paper.
        weight_decay: decoupled weight decay.
    """

    def __init__(
        self,
        params: Iterable,
        lr: float = 1e-4,
        betas: tuple = (0.9, 0.99),
        weight_decay: float = 0.0,
    ):
        if lr < 0.0:
            raise ValueError(f"Invalid lr: {lr}")
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError(f"Invalid beta1: {betas[0]}")
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError(f"Invalid beta2: {betas[1]}")
        if weight_decay < 0.0:
            raise ValueError(f"Invalid weight_decay: {weight_decay}")

        defaults = dict(lr=lr, betas=betas, weight_decay=weight_decay)
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

            for p in group["params"]:
                if p.grad is None:
                    continue
                grad = p.grad
                if grad.is_sparse:
                    raise RuntimeError("Lion does not support sparse gradients")

                state = self.state[p]
                if "exp_avg" not in state:
                    state["exp_avg"] = torch.zeros_like(p, memory_format=torch.preserve_format)

                exp_avg = state["exp_avg"]

                # update_direction = sign(beta1 * m + (1 - beta1) * g)
                # Compute without allocating a full intermediate: use exp_avg as scratch.
                # update = beta1 * exp_avg + (1 - beta1) * grad
                update = exp_avg.mul(beta1).add_(grad, alpha=1 - beta1)
                update = torch.sign(update)

                # Decoupled weight decay + parameter update in one fused mul_+add_.
                # p = p * (1 - lr*wd) - lr * update
                if wd != 0.0:
                    p.mul_(1.0 - lr * wd)
                p.add_(update, alpha=-lr)

                # Now update the long-horizon momentum: m = beta2 * m + (1 - beta2) * g
                exp_avg.mul_(beta2).add_(grad, alpha=1 - beta2)

        return loss

"""Muon optimizer: MomentUm Orthogonalized by Newton-schulz.

Replaces AdamW's second moment with gradient orthogonalization via 5
Newton-Schulz iterations. ~2x token-efficiency, ~50% less optimizer memory.

Based on: KellerJordan/Muon + MoonshotAI/Moonlight (2025 scaling paper).

Usage:
    from halo_training.muon import Muon
    optimizer = Muon(muon_params, adamw_params, lr=0.02)
"""

import math
import torch
import torch.nn as nn


def zeropower_via_newtonschulz5(G, steps=5):
    """Compute the zeroth power (UV^T from SVD) via 5 Newton-Schulz iterations.

    Orthogonalizes the gradient matrix, equalizing all singular values.
    Runs in bfloat16 for numerical stability with the quintic polynomial.
    """
    assert G.ndim == 2
    a, b, c = (3.4445, -4.7750, 2.0315)
    X = G.bfloat16()
    # Transpose if tall — work with the wider matrix for efficiency
    if X.shape[0] > X.shape[1]:
        X = X.T
        transposed = True
    else:
        transposed = False
    X = X / (X.norm() + 1e-7)
    for _ in range(steps):
        A = X @ X.T
        B = b * A + c * A @ A
        X = a * X + B @ X
    if transposed:
        X = X.T
    return X.to(G.dtype)


# NOTE: Do NOT torch.compile the NS function — each unique weight shape
# triggers a separate compilation graph, causing massive memory blowup
# (29 GB for 193 params). The eager NS is only ~5 matmuls per param,
# overhead is <1% of total training time.


class Muon(torch.optim.Optimizer):
    """Muon optimizer with internal AdamW fallback for non-2D params.

    2D weight matrices get Muon updates (momentum + Newton-Schulz orthogonalization).
    Embeddings, norms, biases, and other 1D params get standard AdamW updates.

    Args:
        muon_params: Iterable of param groups for Muon (2D weights).
        lr: Learning rate for Muon params (default 0.02, different scale from AdamW).
        momentum: Momentum coefficient (default 0.95).
        nesterov: Use Nesterov momentum (default True).
        ns_steps: Newton-Schulz iterations (default 5).
        adamw_params: Iterable of param groups for AdamW fallback (1D params).
        adamw_lr: Learning rate for AdamW params (default 8e-4).
        adamw_betas: AdamW beta coefficients (default (0.9, 0.95)).
        adamw_wd: AdamW weight decay (default 0.0, applied per-group).
    """

    def __init__(
        self,
        muon_params,
        lr=0.02,
        momentum=0.95,
        nesterov=True,
        ns_steps=5,
        weight_decay=0.01,
        adamw_params=None,
        adamw_lr=8e-4,
        adamw_betas=(0.9, 0.95),
        adamw_wd=0.0,
    ):
        # Build Muon param groups
        if isinstance(muon_params, dict):
            muon_params = [muon_params]
        muon_groups = []
        for group in muon_params:
            if isinstance(group, dict):
                g = dict(group)
                g.setdefault("lr", lr)
                g.setdefault("momentum", momentum)
                g.setdefault("nesterov", nesterov)
                g.setdefault("ns_steps", ns_steps)
                g.setdefault("weight_decay", weight_decay)
                g["_optimizer_type"] = "muon"
                muon_groups.append(g)
            else:
                # Bare parameter or iterable of parameters
                muon_groups.append({
                    "params": list(group) if hasattr(group, "__iter__") and not isinstance(group, torch.Tensor) else [group],
                    "lr": lr, "momentum": momentum, "nesterov": nesterov,
                    "ns_steps": ns_steps, "weight_decay": weight_decay,
                    "_optimizer_type": "muon",
                })

        # Build AdamW param groups
        adamw_groups = []
        if adamw_params is not None:
            if isinstance(adamw_params, dict):
                adamw_params = [adamw_params]
            for group in adamw_params:
                if isinstance(group, dict):
                    g = dict(group)
                    g.setdefault("lr", adamw_lr)
                    g.setdefault("betas", adamw_betas)
                    g.setdefault("weight_decay", g.pop("weight_decay", adamw_wd))
                    g["_optimizer_type"] = "adamw"
                    adamw_groups.append(g)
                else:
                    adamw_groups.append({
                        "params": list(group) if hasattr(group, "__iter__") and not isinstance(group, torch.Tensor) else [group],
                        "lr": adamw_lr, "betas": adamw_betas,
                        "weight_decay": adamw_wd,
                        "_optimizer_type": "adamw",
                    })

        all_groups = muon_groups + adamw_groups
        defaults = dict(lr=lr, momentum=momentum, nesterov=nesterov,
                        ns_steps=ns_steps, weight_decay=weight_decay)
        super().__init__(all_groups, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            opt_type = group.get("_optimizer_type", "muon")
            if opt_type == "muon":
                self._muon_step(group)
            else:
                self._adamw_step(group)

        return loss

    def _muon_step(self, group):
        lr = group["lr"]
        mu = group["momentum"]
        nesterov = group["nesterov"]
        ns_steps = group["ns_steps"]
        wd = group["weight_decay"]

        ns_fn = zeropower_via_newtonschulz5

        for p in group["params"]:
            if p.grad is None:
                continue

            grad = p.grad

            # Momentum buffer
            state = self.state[p]
            if "momentum_buffer" not in state:
                state["momentum_buffer"] = torch.zeros_like(grad)

            buf = state["momentum_buffer"]
            buf.mul_(mu).add_(grad)

            # Nesterov lookahead
            if nesterov:
                g = grad.add(buf, alpha=mu)
            else:
                g = buf.clone()

            # Newton-Schulz orthogonalization
            g = ns_fn(g, steps=ns_steps)

            # Per-parameter scaling (built-in muP)
            scale = max(g.shape[0], g.shape[1]) ** 0.5 * 0.2
            g.mul_(scale)

            # Decoupled weight decay
            if wd > 0:
                p.data.mul_(1 - lr * wd)

            # Update
            p.data.add_(g, alpha=-lr)

    def _adamw_step(self, group):
        lr = group["lr"]
        beta1, beta2 = group.get("betas", (0.9, 0.95))
        wd = group.get("weight_decay", 0.0)
        eps = group.get("eps", 1e-8)

        for p in group["params"]:
            if p.grad is None:
                continue

            grad = p.grad

            state = self.state[p]
            if "step" not in state:
                state["step"] = 0
                state["exp_avg"] = torch.zeros_like(p)
                state["exp_avg_sq"] = torch.zeros_like(p)

            state["step"] += 1
            exp_avg, exp_avg_sq = state["exp_avg"], state["exp_avg_sq"]

            # Bias correction
            bias_correction1 = 1 - beta1 ** state["step"]
            bias_correction2 = 1 - beta2 ** state["step"]

            # Decoupled weight decay
            if wd > 0:
                p.data.mul_(1 - lr * wd)

            # Update moments
            exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)
            exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)

            # Adam update
            denom = (exp_avg_sq.sqrt() / math.sqrt(bias_correction2)).add_(eps)
            step_size = lr / bias_correction1
            p.data.addcdiv_(exp_avg, denom, value=-step_size)


# Params with these name fragments go to AdamW even if 2D.
# SSM/conv/special params have different gradient dynamics than MLP weights.
_ADAMW_FORCE_PATTERNS = (
    "ssm", "mamba", "conv", "scan", "A_log", "dt_", "D_param",
    "target", "film", "embedding", "embed", "output.weight",
    "log_gamma", "log_eta", "log_beta", "omega", "gamma_param",
    "decay", "conductor", "engram", "meta_token",
)


def split_params_for_muon(model: nn.Module):
    """Split model parameters into Muon-eligible (2D weights) and AdamW (rest).

    Only standard MLP/attention weight matrices go to Muon.
    SSM, conv, embedding, and special params stay on AdamW.

    Returns:
        muon_params: list of 2D weight tensors for Muon
        adamw_params: list of (name, param) tuples for AdamW
    """
    embedding_params = set()
    for module in model.modules():
        if isinstance(module, nn.Embedding):
            for p in module.parameters():
                embedding_params.add(id(p))

    muon_params = []
    adamw_params = []

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue

        # Force AdamW for embeddings
        if id(param) in embedding_params:
            adamw_params.append((name, param))
            continue

        # Force AdamW for SSM/conv/special params
        if any(pat in name for pat in _ADAMW_FORCE_PATTERNS):
            adamw_params.append((name, param))
            continue

        # Exactly 2D weight matrices → Muon (MLP, attention, projection weights)
        # Newton-Schulz requires ndim == 2; 3D+ params (e.g. QK-Norm scales) go to AdamW
        if param.ndim == 2:
            muon_params.append(param)
        else:
            adamw_params.append((name, param))

    return muon_params, adamw_params

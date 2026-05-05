"""
EGGROLL ES Pretraining for TYR-HALO.

Three perturbation strategies (--strategy):
  hooks:     Forward hooks on nn.Linear (simple, moderate overhead)
  perturbed: PerturbedLinear module replacement (no hooks, compile-friendly)
  vmap:      torch.func.vmap over functional_call (maximum parallelism)

Usage:
    python scripts/eggroll_pretrain.py \
        --model models/tyr_halo.py --class-name TyrHaloFast \
        --dataset datasets/stem-crawl-solo.bin \
        --population-size 10000 --chunk-size 2000 --strategy perturbed \
        --sigma 0.01 --lr 0.001 --seq-len 100 --max-steps 10000
"""

import argparse
import copy
import json
import math
import os
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from halo_training.cli import load_model_from_file


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_token_pool(dataset_path: str) -> torch.Tensor:
    root = Path(dataset_path)
    if root.is_file() and root.suffix == ".bin":
        import numpy as np
        tokens = torch.tensor(np.fromfile(str(root), dtype=np.uint16), dtype=torch.long)
        print(f"Token pool: {tokens.shape[0]:,} tokens from {root}")
        return tokens

    import tiktoken
    enc = tiktoken.get_encoding("gpt2")
    eos = 50256
    zst_files = sorted(root.rglob("*.jsonl.zst"))
    if zst_files:
        import zstandard, json as jmod, io
        all_tokens = []
        for f in zst_files:
            dctx = zstandard.ZstdDecompressor()
            with open(f, "rb") as fh:
                reader = dctx.stream_reader(fh)
                text_reader = io.TextIOWrapper(reader, encoding="utf-8")
                for line in text_reader:
                    try:
                        row = jmod.loads(line)
                    except jmod.JSONDecodeError:
                        continue
                    text = row.get("text", "")
                    if text.strip():
                        all_tokens.extend(enc.encode(text, allowed_special={"<|endoftext|>"}))
                        all_tokens.append(eos)
            print(f"  {f.name}: {len(all_tokens):,} tokens")
        return torch.tensor(all_tokens, dtype=torch.long)

    bin_files = sorted(root.glob("*.bin")) if root.is_dir() else []
    if bin_files:
        import numpy as np
        arrays = [np.fromfile(str(f), dtype=np.uint16) for f in bin_files]
        tokens = torch.tensor(np.concatenate(arrays), dtype=torch.long)
        print(f"Token pool: {tokens.shape[0]:,} tokens")
        return tokens

    from halo_training.data import BabyLMDataset
    ds = BabyLMDataset(root=dataset_path, block_size=128)
    tokens = ds.tokens if isinstance(ds.tokens, torch.Tensor) else torch.tensor(ds.tokens, dtype=torch.long)
    print(f"Token pool: {tokens.shape[0]:,} tokens")
    return tokens


# ---------------------------------------------------------------------------
# Strategy A: PerturbedLinear (Option A — compile-friendly, no hooks)
# ---------------------------------------------------------------------------

class PerturbedLinear(nn.Module):
    """Drop-in nn.Linear with inline rank-1 perturbation support."""

    def __init__(self, linear: nn.Linear):
        super().__init__()
        self.weight = linear.weight
        self.bias = linear.bias
        self._A: Optional[torch.Tensor] = None
        self._B: Optional[torch.Tensor] = None
        self._sigma: float = 0.0

    def set_perturbation(self, A: torch.Tensor, B: torch.Tensor, sigma: float):
        self._A = A
        self._B = B
        self._sigma = sigma

    def clear_perturbation(self):
        self._A = self._B = None
        self._sigma = 0.0

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        base = F.linear(x, self.weight, self.bias)
        if self._A is not None:
            if x.dim() == 3:
                dots = (x * self._B.unsqueeze(1)).sum(-1, keepdim=True)
                base = base + self._sigma * dots * self._A.unsqueeze(1)
            elif x.dim() == 2:
                dots = (x * self._B).sum(-1, keepdim=True)
                base = base + self._sigma * dots * self._A
        return base


def replace_linears_with_perturbed(model: nn.Module) -> List[Tuple[str, PerturbedLinear]]:
    """Replace all nn.Linear modules with PerturbedLinear. Returns list of (name, module)."""
    perturbed = []
    for name, mod in list(model.named_modules()):
        if isinstance(mod, nn.Linear) and mod.weight.dim() == 2:
            pl = PerturbedLinear(mod)
            parts = name.rsplit('.', 1)
            if len(parts) == 2:
                parent = dict(model.named_modules())[parts[0]]
                setattr(parent, parts[1], pl)
            else:
                setattr(model, name, pl)
            perturbed.append((name, pl))
    return perturbed


# ---------------------------------------------------------------------------
# Strategy B: vmap over functional_call (Option B — maximum parallelism)
# ---------------------------------------------------------------------------

def setup_vmap(model: nn.Module):
    """Prepare model for vmap-based evaluation. Returns (params, buffers, fn)."""
    from torch.func import functional_call, vmap, stack_module_state

    params = {k: v for k, v in model.named_parameters() if v.dim() == 2}
    buffers = dict(model.named_buffers())
    all_params = dict(model.named_parameters())

    def single_forward(perturbed_params, input_ids):
        merged = {**all_params, **perturbed_params}
        return functional_call(model, (merged, buffers), (input_ids.unsqueeze(0),))

    return params, single_forward


# ---------------------------------------------------------------------------
# Main trainer
# ---------------------------------------------------------------------------

class EggrollTrainer:
    STRATEGIES = ("hooks", "perturbed", "vmap")

    def __init__(self, model: nn.Module, token_pool: torch.Tensor,
                 population_size: int = 10000, chunk_size: int = 2000,
                 sigma: float = 0.01, lr: float = 0.001, seq_len: int = 100,
                 strategy: str = "perturbed", use_compile: bool = True,
                 device: torch.device = None):
        self.token_pool = token_pool
        self.population_size = population_size
        self.chunk_size = chunk_size
        self.sigma = sigma
        self.lr = lr
        self.seq_len = seq_len
        self.strategy = strategy
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.model = model.to(self.device).eval()
        self.perturbed_layers: List[Tuple[str, nn.Module]] = []

        if strategy == "perturbed":
            self.perturbed_layers = replace_linears_with_perturbed(self.model)
            if use_compile:
                try:
                    self.model = torch.compile(self.model, mode="default")
                    print("Model compiled (PerturbedLinear + torch.compile)")
                except Exception as e:
                    print(f"Compile failed ({e}), running eager")
        elif strategy == "hooks":
            for name, mod in model.named_modules():
                if isinstance(mod, nn.Linear) and mod.weight.dim() == 2:
                    self.perturbed_layers.append((name, mod))
        elif strategy == "vmap":
            self._vmap_params, self._vmap_fn = setup_vmap(self.model)
            self.perturbed_layers = [(k, None) for k in self._vmap_params.keys()]

        total_params = sum(
            (m.weight.numel() if hasattr(m, 'weight') else 0) if m else 0
            for _, m in self.perturbed_layers
        )
        if strategy == "vmap":
            total_params = sum(p.numel() for p in self._vmap_params.values())
        print(f"EGGROLL [{strategy}]: {len(self.perturbed_layers)} layers "
              f"({total_params/1e6:.1f}M params), pop={population_size}, "
              f"chunk={chunk_size}, sigma={sigma}, lr={lr}")

    def _sample_sequence(self) -> torch.Tensor:
        max_start = self.token_pool.shape[0] - self.seq_len - 1
        start = torch.randint(0, max_start, (1,)).item()
        return self.token_pool[start:start + self.seq_len + 1].to(self.device)

    def _sample_perturbations(self, N: int) -> List[Tuple[torch.Tensor, torch.Tensor]]:
        perturbations = []
        for name, mod in self.perturbed_layers:
            if self.strategy == "vmap":
                p = self._vmap_params[name]
                d_out, d_in = p.shape
            else:
                d_out, d_in = mod.weight.shape
            A = torch.randn(N, d_out, device=self.device)
            B = torch.randn(N, d_in, device=self.device)
            perturbations.append((A, B))
        return perturbations

    # --- Strategy: hooks ---
    def _evaluate_hooks(self, input_ids, targets, perturbations, sign):
        sigma_scaled = sign * self.sigma
        handles = []
        for (_, mod), (A, B) in zip(self.perturbed_layers, perturbations):
            def make_hook(A_mat, B_mat, scale):
                def hook_fn(module, inp, output):
                    x = inp[0]
                    if x.dim() == 3:
                        dots = (x * B_mat.unsqueeze(1)).sum(-1, keepdim=True)
                        return output + scale * dots * A_mat.unsqueeze(1)
                    elif x.dim() == 2:
                        dots = (x * B_mat).sum(-1, keepdim=True)
                        return output + scale * dots * A_mat
                    return output
                return hook_fn
            h = mod.register_forward_hook(make_hook(A, B, sigma_scaled))
            handles.append(h)

        with torch.no_grad():
            output = self.model(input_ids)
            logits = output["logits"] if isinstance(output, dict) else output

        for h in handles:
            h.remove()
        return self._compute_fitness(logits, targets)

    # --- Strategy: perturbed ---
    def _evaluate_perturbed(self, input_ids, targets, perturbations, sign):
        sigma_scaled = sign * self.sigma
        for (_, mod), (A, B) in zip(self.perturbed_layers, perturbations):
            mod.set_perturbation(A, B, sigma_scaled)

        with torch.no_grad():
            output = self.model(input_ids)
            logits = output["logits"] if isinstance(output, dict) else output

        for _, mod in self.perturbed_layers:
            mod.clear_perturbation()
        return self._compute_fitness(logits, targets)

    # --- Strategy: vmap ---
    def _evaluate_vmap(self, input_ids, targets, perturbations, sign):
        from torch.func import vmap
        sigma_scaled = sign * self.sigma
        N = perturbations[0][0].shape[0]

        # Build per-member perturbed params: {name: (N, d_out, d_in)}
        stacked_params = {}
        for (name, _), (A, B) in zip(self.perturbed_layers, perturbations):
            base = self._vmap_params[name]
            perturbed = base.unsqueeze(0).expand(N, -1, -1) + sigma_scaled * torch.bmm(
                A.unsqueeze(-1), B.unsqueeze(-2))
            stacked_params[name] = perturbed

        # vmap single_forward over the N dimension
        input_single = input_ids[0]  # all same, (T,)
        target_single = targets[0]

        try:
            batched_fn = vmap(
                lambda pp: self._vmap_fn(pp, input_single),
                in_dims=(0,),
            )
            with torch.no_grad():
                outputs = batched_fn(stacked_params)
                logits = outputs["logits"] if isinstance(outputs, dict) else outputs
                logits = logits.squeeze(1)  # remove batch dim from functional_call
        except Exception as e:
            print(f"vmap failed: {e}, falling back to perturbed strategy")
            self.strategy = "perturbed"
            self.perturbed_layers = replace_linears_with_perturbed(self.model)
            return self._evaluate_perturbed(input_ids, targets, perturbations, sign)

        return self._compute_fitness(logits, targets)

    def _compute_fitness(self, logits, targets):
        N, T_out, V = logits.shape
        losses = F.cross_entropy(
            logits.reshape(N * T_out, V),
            targets[:, :T_out].reshape(N * T_out),
            reduction='none'
        ).view(N, T_out).mean(dim=1)
        return -losses

    def _evaluate_chunk(self, input_ids, targets, perturbations, sign):
        if self.strategy == "hooks":
            return self._evaluate_hooks(input_ids, targets, perturbations, sign)
        elif self.strategy == "perturbed":
            return self._evaluate_perturbed(input_ids, targets, perturbations, sign)
        elif self.strategy == "vmap":
            return self._evaluate_vmap(input_ids, targets, perturbations, sign)

    def step(self, step_idx: int) -> dict:
        seq = self._sample_sequence()
        input_template = seq[:-1]
        target_template = seq[1:]

        half_pop = self.population_size // 2
        n_chunks = max(1, (half_pop + self.chunk_size - 1) // self.chunk_size)

        grad_accum = [(torch.zeros_like(
            self._vmap_params[name] if self.strategy == "vmap" else mod.weight
        ), 0) for name, mod in self.perturbed_layers]

        all_fitness_pos = []
        all_fitness_neg = []
        all_perturbations = []

        for chunk_idx in range(n_chunks):
            start = chunk_idx * self.chunk_size
            end = min(start + self.chunk_size, half_pop)
            N = end - start
            if N <= 0:
                break

            input_ids = input_template.unsqueeze(0).expand(N, -1).contiguous()
            targets = target_template.unsqueeze(0).expand(N, -1).contiguous()

            perturbations = self._sample_perturbations(N)
            fitness_pos = self._evaluate_chunk(input_ids, targets, perturbations, +1.0)
            fitness_neg = self._evaluate_chunk(input_ids, targets, perturbations, -1.0)

            all_fitness_pos.append(fitness_pos)
            all_fitness_neg.append(fitness_neg)
            all_perturbations.append((perturbations, N))

        fitness_pos = torch.cat(all_fitness_pos)
        fitness_neg = torch.cat(all_fitness_neg)
        shaped = torch.sign(fitness_pos - fitness_neg)

        offset = 0
        for perturbations, N in all_perturbations:
            chunk_shaped = shaped[offset:offset + N]
            for layer_idx, (A, B) in enumerate(perturbations):
                weighted_A = A * chunk_shaped.unsqueeze(-1)
                grad_est = weighted_A.T @ B
                old_grad, old_count = grad_accum[layer_idx]
                grad_accum[layer_idx] = (old_grad + grad_est, old_count + N)
            offset += N

        lr_t = self.lr / (0.015 * step_idx + 1)
        with torch.no_grad():
            if self.strategy == "vmap":
                for layer_idx, (name, _) in enumerate(self.perturbed_layers):
                    grad_est, count = grad_accum[layer_idx]
                    if count > 0:
                        self._vmap_params[name].add_(lr_t / count * grad_est)
            else:
                for layer_idx, (_, mod) in enumerate(self.perturbed_layers):
                    grad_est, count = grad_accum[layer_idx]
                    if count > 0:
                        mod.weight.add_(lr_t / count * grad_est)

        # Base loss for logging
        with torch.no_grad():
            output = self.model(input_template.unsqueeze(0))
            logits = output["logits"] if isinstance(output, dict) else output
            base_loss = F.cross_entropy(
                logits.view(-1, logits.shape[-1]),
                target_template[:logits.shape[1]].unsqueeze(0).view(-1)
            ).item()

        return {
            "loss": base_loss,
            "lr": lr_t,
            "update_fraction": (shaped != 0).float().mean().item(),
            "mean_fitness_gap": (fitness_pos - fitness_neg).abs().mean().item(),
            "tokens_this_step": self.population_size * self.seq_len,
        }

    def train(self, max_steps: int = 10000, log_interval: int = 10,
              checkpoint_dir: str = "checkpoints/eggroll",
              checkpoint_interval: int = 500):
        os.makedirs(checkpoint_dir, exist_ok=True)
        log_path = os.path.join(checkpoint_dir, "train_log.jsonl")
        total_tokens = 0
        t0 = time.time()

        print(f"EGGROLL [{self.strategy}]: {max_steps} steps, "
              f"pop={self.population_size}, chunk={self.chunk_size}")

        for step in range(max_steps):
            step_t0 = time.time()
            metrics = self.step(step)
            step_time = time.time() - step_t0

            total_tokens += metrics["tokens_this_step"]
            tok_per_sec = metrics["tokens_this_step"] / max(step_time, 1e-6)

            entry = {
                "step": step,
                "loss": round(metrics["loss"], 6),
                "lr": round(metrics["lr"], 8),
                "tok_per_sec": round(tok_per_sec),
                "total_tokens": total_tokens,
                "update_fraction": round(metrics["update_fraction"], 3),
                "mean_fitness_gap": round(metrics["mean_fitness_gap"], 6),
                "step_time": round(step_time, 2),
                "wall_time": round(time.time() - t0, 1),
            }

            with open(log_path, "a") as f:
                f.write(json.dumps(entry) + "\n")

            if step % log_interval == 0:
                print(f"step={step:>6} loss={metrics['loss']:.4f} "
                      f"tok/s={tok_per_sec:,.0f} upd={metrics['update_fraction']:.2f} "
                      f"gap={metrics['mean_fitness_gap']:.4f} "
                      f"lr={metrics['lr']:.6f} t={step_time:.1f}s")

            if checkpoint_interval and step > 0 and step % checkpoint_interval == 0:
                ckpt_path = os.path.join(checkpoint_dir, f"step_{step}.pt")
                torch.save({
                    "step": step,
                    "model_state_dict": self.model.state_dict(),
                    "total_tokens": total_tokens,
                }, ckpt_path)
                print(f"Checkpoint: {ckpt_path}")

        final_path = os.path.join(checkpoint_dir, "final.pt")
        torch.save({
            "step": max_steps,
            "model_state_dict": self.model.state_dict(),
            "total_tokens": total_tokens,
        }, final_path)
        print(f"Final: {final_path}, tokens={total_tokens:,}, time={time.time()-t0:.1f}s")


def main():
    parser = argparse.ArgumentParser(description="EGGROLL ES Pretraining")
    parser.add_argument("--model", required=True)
    parser.add_argument("--class-name", required=True)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--population-size", type=int, default=10000)
    parser.add_argument("--chunk-size", type=int, default=2000)
    parser.add_argument("--sigma", type=float, default=0.01)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--seq-len", type=int, default=100)
    parser.add_argument("--max-steps", type=int, default=10000)
    parser.add_argument("--log-interval", type=int, default=10)
    parser.add_argument("--checkpoint-dir", default="checkpoints/eggroll")
    parser.add_argument("--checkpoint-interval", type=int, default=500)
    parser.add_argument("--strategy", default="perturbed",
                        choices=EggrollTrainer.STRATEGIES,
                        help="Perturbation strategy: hooks, perturbed, vmap")
    parser.add_argument("--no-compile", action="store_true",
                        help="Disable torch.compile (for perturbed strategy)")
    args = parser.parse_args()

    model = load_model_from_file(args.model, args.class_name)
    token_pool = load_token_pool(args.dataset)

    trainer = EggrollTrainer(
        model=model,
        token_pool=token_pool,
        population_size=args.population_size,
        chunk_size=args.chunk_size,
        sigma=args.sigma,
        lr=args.lr,
        seq_len=args.seq_len,
        strategy=args.strategy,
        use_compile=not args.no_compile,
    )
    trainer.train(
        max_steps=args.max_steps,
        log_interval=args.log_interval,
        checkpoint_dir=args.checkpoint_dir,
        checkpoint_interval=args.checkpoint_interval,
    )


if __name__ == "__main__":
    main()

"""Unified alignment training loop for ORPO, SimPO, KTO.

Supports all Tier 1 losses (ORPO, SimPO, KTO) and Tier 2 drop-in modifiers
(D2PO temporal decay, ConfPO token selection, AlphaPO reward shaping).

Usage:
    python -m halo_training.alignment_trainer \
        --model models/fenrir_halo.py --class-name FenrirHalo \
        --checkpoint checkpoints/fenrir_halo_dolma/step_XXXXX.pt \
        --method orpo \
        --dataset datasets/alignment/magpie_50k.jsonl \
        --d2po-gamma 0.98 --confpo-frac 0.5
"""

import argparse
import copy
import importlib.util
import json
import math
import os
import sys
import time
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from halo_training.alignment import orpo_loss, simpo_loss, kto_loss
from halo_training.dpo import DPODataset
from halo_training.chat_template import ChatMLTokenizer, CHATML_TOKENS


# ---------------------------------------------------------------------------
# KTO Dataset (unpaired: each example is prompt+response+label)
# ---------------------------------------------------------------------------

class KTODataset(torch.utils.data.Dataset):
    """Dataset for KTO: each item is (input_ids, mask, label).

    Accepts both paired data (splits into 2 examples) and unpaired data.
    """

    def __init__(self, data_path: str, tokenizer: ChatMLTokenizer,
                 block_size: int = 1024, max_examples: Optional[int] = None):
        self.tokenizer = tokenizer
        self.block_size = block_size
        self.examples = []

        with open(data_path, "r", encoding="utf-8") as f:
            raw = [json.loads(line.strip()) for line in f if line.strip()]

        if max_examples:
            raw = raw[:max_examples]

        for ex in raw:
            if "chosen" in ex and "rejected" in ex:
                self._add(ex["chosen"], True)
                self._add(ex["rejected"], False)
            else:
                messages = ex.get("messages", ex.get("conversation", []))
                label = ex.get("label", ex.get("desirable", True))
                if isinstance(label, str):
                    label = label.lower() in ("true", "desirable", "chosen", "good")
                self._add(messages, label)

        print(f"  KTODataset: {len(self.examples)} examples "
              f"({sum(1 for _, _, l in self.examples if l)} desirable, "
              f"{sum(1 for _, _, l in self.examples if not l)} undesirable)")

    def _add(self, messages, label):
        if isinstance(messages, str):
            return
        ids, mask = self._tokenize(messages)
        if len(ids) < 4 or sum(mask) < 2:
            return
        if len(ids) > self.block_size:
            ids, mask = ids[:self.block_size], mask[:self.block_size]
        self.examples.append((ids, mask, label))

    def _tokenize(self, messages):
        all_ids, all_mask = [], []
        for msg in messages:
            role = msg["role"]
            content = msg["content"]
            header = [self.tokenizer.im_start_id]
            header.extend(self.tokenizer._base.encode_ordinary(role + "\n"))
            content_ids = self.tokenizer.encode(content) if role == "assistant" else self.tokenizer._base.encode_ordinary(content)
            footer = [self.tokenizer.im_end_id]
            footer.extend(self.tokenizer._base.encode_ordinary("\n"))
            is_asst = role == "assistant"
            all_ids.extend(header)
            all_mask.extend([0] * len(header))
            all_ids.extend(content_ids)
            all_mask.extend([1 if is_asst else 0] * len(content_ids))
            all_ids.extend(footer)
            all_mask.extend([1 if is_asst else 0] * len(footer))
        return all_ids, all_mask

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        ids, mask, label = self.examples[idx]
        pad_len = self.block_size - len(ids)
        ids = ids + [self.tokenizer.pad_id] * pad_len
        mask = mask + [0] * pad_len
        return (
            torch.tensor(ids, dtype=torch.long),
            torch.tensor(mask, dtype=torch.long),
            torch.tensor(1 if label else 0, dtype=torch.long),
        )


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------

def load_model_from_file(model_path: str, class_name: str):
    spec = importlib.util.spec_from_file_location("user_model", model_path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["user_model"] = mod
    spec.loader.exec_module(mod)
    return getattr(mod, class_name)()


def resize_embeddings(model, new_vocab_size: int):
    """Resize embedding + LM head for ChatML special tokens."""
    for module in model.modules():
        if isinstance(module, nn.Embedding):
            old_weight = module.weight
            old_vocab, dim = old_weight.shape
            if old_vocab >= new_vocab_size:
                return
            new_weight = torch.zeros(new_vocab_size, dim, dtype=old_weight.dtype,
                                     device=old_weight.device)
            new_weight[:old_vocab] = old_weight
            nn.init.normal_(new_weight[old_vocab:], std=0.02)
            module.weight = nn.Parameter(new_weight)
            module.num_embeddings = new_vocab_size
            print(f"  Resized embedding: {old_vocab} -> {new_vocab_size}")
            return


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------

def train_alignment(
    model: nn.Module,
    dataset,
    method: str = "orpo",
    epochs: int = 1,
    batch_size: int = 4,
    lr: float = 5e-6,
    checkpoint_dir: str = "checkpoints/alignment",
    log_interval: int = 10,
    max_steps: Optional[int] = None,
    # Method-specific
    orpo_lambda: float = 0.1,
    simpo_beta: float = 2.0,
    simpo_gamma_ratio: float = 0.5,
    simpo_alpha: float = 1.0,
    kto_beta: float = 0.1,
    kto_des_weight: float = 1.0,
    kto_und_weight: float = 1.0,
    # Drop-in modifiers
    d2po_gamma: float = 0.0,
    confpo_frac: float = 0.0,
    # Reference model (KTO only)
    ref_model: Optional[nn.Module] = None,
):
    device = next(model.parameters()).device
    ckpt_dir = Path(checkpoint_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)

    if method == "kto":
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True,
                                num_workers=0, drop_last=True)
    else:
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True,
                                num_workers=0, drop_last=True)

    total_steps = len(dataloader) * epochs
    warmup_steps = max(1, total_steps // 10)

    def lr_lambda(step):
        if step < warmup_steps:
            return step / warmup_steps
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return 0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * progress))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    log_path = ckpt_dir / "train_log.jsonl"
    log_file = open(log_path, "w")

    modifiers = f"d2po={d2po_gamma}" if d2po_gamma > 0 else ""
    modifiers += f" confpo={confpo_frac}" if confpo_frac > 0 else ""
    if method == "simpo" and simpo_alpha != 1.0:
        modifiers += f" alpha={simpo_alpha}"
    print(f"Alignment: {method.upper()}, {len(dataset)} examples, "
          f"batch={batch_size}, lr={lr} {modifiers}")

    global_step = 0
    start_time = time.time()
    best_accuracy = 0

    for epoch in range(epochs):
        model.train()
        for batch in dataloader:
            if max_steps and global_step >= max_steps:
                break

            if method in ("orpo", "simpo"):
                c_ids, c_mask, r_ids, r_mask = [x.to(device) for x in batch]

                if method == "orpo":
                    loss, metrics = orpo_loss(
                        model, c_ids, c_mask, r_ids, r_mask,
                        lambda_weight=orpo_lambda,
                        d2po_gamma=d2po_gamma, confpo_frac=confpo_frac,
                    )
                else:
                    loss, metrics = simpo_loss(
                        model, c_ids, c_mask, r_ids, r_mask,
                        beta=simpo_beta, gamma_beta_ratio=simpo_gamma_ratio,
                        alpha=simpo_alpha,
                        d2po_gamma=d2po_gamma, confpo_frac=confpo_frac,
                    )

            elif method == "kto":
                ids, mask, labels = [x.to(device) for x in batch]
                loss, metrics = kto_loss(
                    model, ref_model, ids, mask, labels,
                    beta=kto_beta,
                    desirable_weight=kto_des_weight,
                    undesirable_weight=kto_und_weight,
                    d2po_gamma=d2po_gamma, confpo_frac=confpo_frac,
                )

            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()

            global_step += 1

            if global_step % log_interval == 0:
                elapsed = time.time() - start_time
                metrics["step"] = global_step
                metrics["lr"] = scheduler.get_last_lr()[0]
                metrics["elapsed_s"] = elapsed
                log_file.write(json.dumps(metrics) + "\n")
                log_file.flush()
                acc_str = f"acc={metrics.get('accuracy', 0):.3f}"
                print(f"  [step {global_step:>5}] loss={metrics['loss']:.4f} "
                      f"{acc_str} lr={metrics['lr']:.2e}")
                best_accuracy = max(best_accuracy, metrics.get("accuracy", 0))

    # Save final
    raw = model._orig_mod if hasattr(model, "_orig_mod") else model
    ckpt_path = ckpt_dir / f"step_{global_step}.pt"
    torch.save({
        "model_state_dict": raw.state_dict(),
        "step": global_step,
        "method": method,
        "accuracy": best_accuracy,
    }, ckpt_path)
    print(f"Saved: {ckpt_path}")

    log_file.close()
    elapsed = time.time() - start_time
    print(f"\nDone: {global_step} steps in {elapsed:.0f}s, best acc={best_accuracy:.3f}")
    return {"steps": global_step, "best_accuracy": best_accuracy}


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Alignment training (ORPO/SimPO/KTO)")
    parser.add_argument("--model", required=True)
    parser.add_argument("--class-name", required=True)
    parser.add_argument("--checkpoint", required=True, help="Base model checkpoint")
    parser.add_argument("--dataset", required=True, help="JSONL alignment data")
    parser.add_argument("--method", choices=["orpo", "simpo", "kto"], default="orpo")
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=5e-6)
    parser.add_argument("--block-size", type=int, default=1024)
    parser.add_argument("--checkpoint-dir", default="checkpoints/alignment")
    parser.add_argument("--log-interval", type=int, default=10)
    parser.add_argument("--max-steps", type=int, default=None)
    # ORPO
    parser.add_argument("--orpo-lambda", type=float, default=0.1)
    # SimPO
    parser.add_argument("--simpo-beta", type=float, default=2.0)
    parser.add_argument("--simpo-gamma-ratio", type=float, default=0.5)
    # AlphaPO (applies to SimPO)
    parser.add_argument("--alpha", type=float, default=1.0)
    # KTO
    parser.add_argument("--kto-beta", type=float, default=0.1)
    # Drop-in modifiers
    parser.add_argument("--d2po-gamma", type=float, default=0.0)
    parser.add_argument("--confpo-frac", type=float, default=0.0)

    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load model
    sys.path.insert(0, ".")
    model = load_model_from_file(args.model, args.class_name)

    # Autokernel before checkpoint load
    try:
        import autokernel
        model = autokernel.optimize(model, training=True)
    except Exception:
        pass

    # Resize for ChatML tokens
    max_token = max(CHATML_TOKENS.values()) + 1
    resize_embeddings(model, max_token)

    model = model.to(device)

    # Load checkpoint
    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    sd = ckpt.get("model_state_dict", ckpt)
    clean = {k.replace("_orig_mod.", ""): v for k, v in sd.items()}
    model.load_state_dict(clean, strict=False)
    print(f"Loaded checkpoint: {args.checkpoint}")

    # Tokenizer
    tokenizer = ChatMLTokenizer()

    # Dataset
    if args.method == "kto":
        dataset = KTODataset(args.dataset, tokenizer, block_size=args.block_size)
    else:
        dataset = DPODataset(args.dataset, tokenizer, block_size=args.block_size)

    # Reference model (KTO only)
    ref_model = None
    if args.method == "kto":
        ref_model = copy.deepcopy(model)
        ref_model.eval()
        for p in ref_model.parameters():
            p.requires_grad = False

    train_alignment(
        model=model,
        dataset=dataset,
        method=args.method,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        checkpoint_dir=args.checkpoint_dir,
        log_interval=args.log_interval,
        max_steps=args.max_steps,
        orpo_lambda=args.orpo_lambda,
        simpo_beta=args.simpo_beta,
        simpo_gamma_ratio=args.simpo_gamma_ratio,
        simpo_alpha=args.alpha,
        kto_beta=args.kto_beta,
        d2po_gamma=args.d2po_gamma,
        confpo_frac=args.confpo_frac,
        ref_model=ref_model,
    )


if __name__ == "__main__":
    main()

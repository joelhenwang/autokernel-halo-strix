"""Direct Preference Optimization (DPO) for context grounding.

Implements offline DPO: trains a policy model to prefer chosen completions
over rejected ones, using a frozen reference model for the KL constraint.

DPO loss: -log(sigmoid(beta * (log_pi(chosen) - log_ref(chosen) - log_pi(rejected) + log_ref(rejected))))

Reference: Rafailov et al., "Direct Preference Optimization" (NeurIPS 2023)
"""

import copy
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset

from halo_training.chat_template import ChatMLTokenizer, IGNORE_INDEX


class DPODataset(Dataset):
    """Dataset of (chosen, rejected) preference pairs for DPO training.

    Each item returns:
        chosen_ids: (seq_len,) token IDs for chosen completion
        chosen_mask: (seq_len,) 1 for assistant tokens, 0 for prompt
        rejected_ids: (seq_len,) token IDs for rejected completion
        rejected_mask: (seq_len,) 1 for assistant tokens, 0 for prompt
    """

    def __init__(
        self,
        data_path: str,
        tokenizer: ChatMLTokenizer,
        block_size: int = 1536,
        max_examples: Optional[int] = None,
    ):
        self.tokenizer = tokenizer
        self.block_size = block_size

        examples = self._load_jsonl(data_path)
        if max_examples:
            examples = examples[:max_examples]

        self.pairs = []
        skipped = 0
        for ex in examples:
            chosen = ex.get("chosen", [])
            rejected = ex.get("rejected", [])
            if not chosen or not rejected:
                skipped += 1
                continue

            c_ids, c_mask = self._tokenize_messages(chosen)
            r_ids, r_mask = self._tokenize_messages(rejected)

            if len(c_ids) > block_size or len(r_ids) > block_size:
                c_ids, c_mask = c_ids[:block_size], c_mask[:block_size]
                r_ids, r_mask = r_ids[:block_size], r_mask[:block_size]

            if sum(c_mask) < 2 or sum(r_mask) < 2:
                skipped += 1
                continue

            self.pairs.append((c_ids, c_mask, r_ids, r_mask))

        print(f"  DPODataset: {len(examples)} examples -> {len(self.pairs)} pairs "
              f"(skipped {skipped}, block_size={block_size})")

    def _load_jsonl(self, path: str) -> list:
        examples = []
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    examples.append(json.loads(line))
        return examples

    def _tokenize_messages(self, messages: List[Dict[str, str]]) -> Tuple[List[int], List[int]]:
        """Tokenize messages and return (token_ids, assistant_mask).

        assistant_mask[i] = 1 if token i is part of an assistant turn's content,
        0 otherwise (system, user, tool, headers, footers).
        """
        all_ids = []
        all_mask = []

        for msg in messages:
            role = msg["role"]
            content = msg["content"]

            header_ids = [self.tokenizer.im_start_id]
            header_ids.extend(self.tokenizer._base.encode_ordinary(role + "\n"))

            if role == "assistant":
                content_ids = self.tokenizer.encode(content)
            else:
                content_ids = self.tokenizer._base.encode_ordinary(content)

            footer_ids = [self.tokenizer.im_end_id]
            footer_ids.extend(self.tokenizer._base.encode_ordinary("\n"))

            is_asst = role == "assistant"
            all_ids.extend(header_ids)
            all_mask.extend([0] * len(header_ids))
            all_ids.extend(content_ids)
            all_mask.extend([1 if is_asst else 0] * len(content_ids))
            all_ids.extend(footer_ids)
            all_mask.extend([1 if is_asst else 0] * len(footer_ids))

        return all_ids, all_mask

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        c_ids, c_mask, r_ids, r_mask = self.pairs[idx]

        # Pad to block_size
        def pad(ids, mask):
            pad_len = self.block_size - len(ids)
            ids = ids + [self.tokenizer.pad_id] * pad_len
            mask = mask + [0] * pad_len
            return torch.tensor(ids, dtype=torch.long), torch.tensor(mask, dtype=torch.long)

        c_ids_t, c_mask_t = pad(c_ids, c_mask)
        r_ids_t, r_mask_t = pad(r_ids, r_mask)

        return c_ids_t, c_mask_t, r_ids_t, r_mask_t


def compute_log_probs(model, input_ids, mask):
    """Compute per-token log probabilities for masked positions.

    Args:
        model: Language model that returns logits
        input_ids: (B, T) token IDs
        mask: (B, T) 1 for positions to score, 0 to ignore

    Returns:
        (B,) sum of log-probs over masked positions
    """
    with torch.amp.autocast("cuda", dtype=torch.float16):
        logits = model(input_ids)  # (B, T, V)

    # Shift: predict next token
    shift_logits = logits[:, :-1, :]  # (B, T-1, V)
    shift_labels = input_ids[:, 1:]   # (B, T-1)
    shift_mask = mask[:, 1:]          # (B, T-1)

    # Per-token log probs
    log_probs = F.log_softmax(shift_logits.float(), dim=-1)
    token_log_probs = log_probs.gather(2, shift_labels.unsqueeze(-1)).squeeze(-1)  # (B, T-1)

    # Mask and sum
    masked_log_probs = token_log_probs * shift_mask.float()
    return masked_log_probs.sum(dim=-1)  # (B,)


def dpo_loss(
    policy_model,
    ref_model,
    chosen_ids,
    chosen_mask,
    rejected_ids,
    rejected_mask,
    beta: float = 0.1,
):
    """Compute DPO loss.

    Returns:
        loss: scalar DPO loss
        metrics: dict with chosen_reward, rejected_reward, accuracy
    """
    # Policy log probs
    pi_chosen = compute_log_probs(policy_model, chosen_ids, chosen_mask)
    pi_rejected = compute_log_probs(policy_model, rejected_ids, rejected_mask)

    # Reference log probs (no grad)
    with torch.no_grad():
        ref_chosen = compute_log_probs(ref_model, chosen_ids, chosen_mask)
        ref_rejected = compute_log_probs(ref_model, rejected_ids, rejected_mask)

    # DPO: reward difference
    chosen_reward = beta * (pi_chosen - ref_chosen)
    rejected_reward = beta * (pi_rejected - ref_rejected)

    loss = -F.logsigmoid(chosen_reward - rejected_reward).mean()

    # Metrics
    with torch.no_grad():
        accuracy = (chosen_reward > rejected_reward).float().mean()

    return loss, {
        "chosen_reward": chosen_reward.mean().item(),
        "rejected_reward": rejected_reward.mean().item(),
        "accuracy": accuracy.item(),
        "loss": loss.item(),
    }


def train_dpo(
    model,
    dataset: DPODataset,
    epochs: int = 2,
    batch_size: int = 4,
    lr: float = 5e-6,
    beta: float = 0.1,
    checkpoint_dir: str = "checkpoints/dpo",
    log_interval: int = 10,
    max_steps: Optional[int] = None,
    time_budget_minutes: float = 120,
    compile: bool = False,
):
    """Run DPO training loop."""
    import time
    from torch.utils.data import DataLoader

    device = next(model.parameters()).device
    ckpt_dir = Path(checkpoint_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    # Create reference model (frozen copy)
    print("Creating reference model (frozen copy)...")
    ref_model = copy.deepcopy(model)
    ref_model.eval()
    for p in ref_model.parameters():
        p.requires_grad = False

    if compile:
        print("Compiling policy model...")
        model = torch.compile(model)
        print("Compiling reference model...")
        ref_model = torch.compile(ref_model)

    # Optimizer (AdamW, not Muon — DPO needs stable, small updates)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)

    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True,
                            num_workers=0, pin_memory=True)

    # Training log
    log_path = ckpt_dir / "train_log.jsonl"
    log_file = open(log_path, "w")

    start_time = time.time()
    budget_secs = time_budget_minutes * 60
    global_step = 0
    best_accuracy = 0

    print(f"DPO training: {len(dataset)} pairs, batch_size={batch_size}, "
          f"beta={beta}, lr={lr}")
    print(f"Time budget: {time_budget_minutes} min")

    for epoch in range(epochs):
        model.train()
        epoch_loss = 0
        epoch_acc = 0
        n_batches = 0

        for batch in dataloader:
            if time.time() - start_time > budget_secs:
                print(f"Time budget reached at step {global_step}")
                break
            if max_steps and global_step >= max_steps:
                break

            c_ids, c_mask, r_ids, r_mask = [x.to(device) for x in batch]

            loss, metrics = dpo_loss(
                model, ref_model, c_ids, c_mask, r_ids, r_mask, beta=beta
            )

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            global_step += 1
            epoch_loss += metrics["loss"]
            epoch_acc += metrics["accuracy"]
            n_batches += 1

            if global_step % log_interval == 0:
                avg_loss = epoch_loss / n_batches
                avg_acc = epoch_acc / n_batches
                elapsed = time.time() - start_time
                log_entry = {
                    "step": global_step,
                    "loss": avg_loss,
                    "accuracy": avg_acc,
                    "chosen_reward": metrics["chosen_reward"],
                    "rejected_reward": metrics["rejected_reward"],
                    "elapsed_s": elapsed,
                }
                log_file.write(json.dumps(log_entry) + "\n")
                log_file.flush()
                print(f"[step {global_step:>5}] loss={avg_loss:.4f} "
                      f"acc={avg_acc:.3f} "
                      f"c_rew={metrics['chosen_reward']:.3f} "
                      f"r_rew={metrics['rejected_reward']:.3f}")

                if avg_acc > best_accuracy:
                    best_accuracy = avg_acc

        if max_steps and global_step >= max_steps:
            break
        if time.time() - start_time > budget_secs:
            break

    # Save final checkpoint
    ckpt_path = ckpt_dir / f"step_{global_step}.pt"
    torch.save({
        "model_state_dict": model.state_dict() if not compile else model._orig_mod.state_dict(),
        "step": global_step,
        "dpo_accuracy": best_accuracy,
    }, ckpt_path)
    print(f"Saved checkpoint: {ckpt_path}")

    log_file.close()
    elapsed = time.time() - start_time
    print(f"\nDPO training done: {global_step} steps in {elapsed:.0f}s, "
          f"best accuracy={best_accuracy:.3f}")

    return {"steps": global_step, "best_accuracy": best_accuracy, "elapsed_s": elapsed}

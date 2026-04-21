# Part 12: SFT & Alignment -- From Base Model to Chat Model

## Goal
Transform a pretrained base model into one that follows instructions, holds conversations, uses tools, and aligns with human preferences. You will implement supervised fine-tuning (SFT) with ChatML format, loss masking, and DPO alignment -- all from scratch.

## Why This Matters
A base model is a next-token predictor. It can complete text, but it cannot follow instructions, answer questions coherently, or know when to stop talking. SFT and alignment are what turn a text generator into a useful assistant.

## Prerequisites
- Part 02: You understand the training loop.
- Part 09: You have evaluation tools.
- Part 11: You have a trained base model (or at minimum, GPT-2 124M from Part 02).

---

## 12.1 The Post-Training Pipeline

Every capable chat model goes through three stages:

```
Stage 1: Pretraining (Parts 02-11)
    Input:  Raw text (books, web, code)
    Output: Base model -- predicts next token, no concept of "conversation"
    Data:   Billions of tokens, broad coverage
    LR:     8e-4, cosine schedule

Stage 2: Supervised Fine-Tuning (SFT)
    Input:  (instruction, response) pairs
    Output: Instruction-following model -- responds to questions
    Data:   10K-1M curated examples
    LR:     1e-5 to 5e-5, constant or cosine with warmup

Stage 3: Alignment (DPO/RLHF)
    Input:  (prompt, chosen_response, rejected_response) triples
    Output: Aligned model -- prefers helpful, harmless, honest responses
    Data:   1K-100K preference pairs
    LR:     5e-7 to 5e-6, constant
```

### Why Each Stage Matters

**Without SFT,** the base model continues whatever text you give it. Ask "What is Python?" and it might generate "What is Java? What is C++?" -- continuing the pattern of questions rather than answering yours.

**Without alignment,** the SFT model follows instructions but may generate harmful, biased, or factually wrong content with equal enthusiasm. DPO teaches it to prefer better responses.

### The Data Funnel

```
Pretraining:  100B+ tokens    (broad, noisy)
SFT:          1M tokens       (curated, structured)
DPO:          100K tokens     (paired, judged)
```

Each stage uses less but higher-quality data.

---

## 12.2 ChatML Format

ChatML is the standard format for structured conversations. Every turn has a role (system, user, assistant) and is wrapped in special tokens.

### Message Structure

```
<|im_start|>system
You are a helpful assistant.<|im_end|>
<|im_start|>user
What is Python?<|im_end|>
<|im_start|>assistant
Python is a high-level programming language known for its readability and versatility.<|im_end|>
```

Each turn follows the pattern:
```
<|im_start|>{role}\n{content}<|im_end|>\n
```

### Special Tokens

The base GPT-2 tokenizer has 50,257 tokens (IDs 0-50,256). We add three new tokens:

| Token | ID | Purpose |
|-------|----|---------|
| `<\|im_start\|>` | 50,257 | Marks the beginning of a turn |
| `<\|im_end\|>` | 50,258 | Marks the end of a turn |
| `<\|pad\|>` | 50,259 | Padding token for batching |

For tool-use support (Stage B), we add two more:

| Token | ID | Purpose |
|-------|----|---------|
| `<tool_call>` | 50,260 | Model wants to call a tool |
| `</tool_call>` | 50,261 | End of tool call |

### Tokenizer Extension

Since tiktoken does not natively support custom tokens, we wrap it:

```python
import re
import tiktoken

# Constants
IM_START = "<|im_start|>"
IM_END = "<|im_end|>"
PAD = "<|pad|>"
IGNORE_INDEX = -100  # standard PyTorch ignore label

CHATML_TOKENS = {
    IM_START: 50257,
    IM_END: 50258,
    PAD: 50259,
}


class ChatMLTokenizer:
    """Wraps tiktoken with support for ChatML special tokens."""

    def __init__(self, base="gpt2", extra_tokens=None):
        self._base = tiktoken.get_encoding(base)
        self._extra = dict(extra_tokens) if extra_tokens else dict(CHATML_TOKENS)
        self._id_to_token = {v: k for k, v in self._extra.items()}

        # Regex to split text around special tokens
        escaped = [re.escape(tok) for tok in sorted(self._extra, key=len, reverse=True)]
        self._split_pattern = re.compile("(" + "|".join(escaped) + ")")

    @property
    def vocab_size(self):
        return self._base.n_vocab + len(self._extra)

    @property
    def eos_token_id(self):
        return self._base.n_vocab - 1  # 50256 for GPT-2

    @property
    def pad_id(self):
        return self._extra[PAD]

    def encode(self, text):
        """Encode text, handling special tokens by splitting around them."""
        parts = self._split_pattern.split(text)
        ids = []
        for part in parts:
            if not part:
                continue
            if part in self._extra:
                ids.append(self._extra[part])
            else:
                ids.extend(self._base.encode_ordinary(part))
        return ids

    def decode(self, ids):
        """Decode token IDs back to text."""
        result = []
        buffer = []
        for tid in ids:
            if tid in self._id_to_token:
                if buffer:
                    result.append(self._base.decode(buffer))
                    buffer = []
                result.append(self._id_to_token[tid])
            else:
                buffer.append(tid)
        if buffer:
            result.append(self._base.decode(buffer))
        return "".join(result)
```

### Resizing Model Embeddings

The base model has `vocab_size=50257`. After adding special tokens, we need `vocab_size=50260` (or 50262 with tool tokens). The embedding table and LM head must be resized:

```python
def resize_embeddings(model, new_vocab_size):
    """Safely resize token embeddings and LM head, preserving weight tying."""
    old_embed = model.tok_embeddings
    old_vocab = old_embed.num_embeddings
    d_model = old_embed.embedding_dim

    if new_vocab_size == old_vocab:
        return model

    # Create new embedding with mean-initialization for new tokens
    # (Zero-init causes RMSNorm corruption and training divergence)
    new_embed = torch.nn.Embedding(
        new_vocab_size, d_model,
        device=old_embed.weight.device, dtype=old_embed.weight.dtype
    )
    with torch.no_grad():
        new_embed.weight[:old_vocab] = old_embed.weight
        embed_mean = old_embed.weight.mean(dim=0)
        embed_std = old_embed.weight.std()
        for i in range(old_vocab, new_vocab_size):
            new_embed.weight[i] = embed_mean + torch.randn_like(embed_mean) * (embed_std * 0.01)

    model.tok_embeddings = new_embed

    # Recreate output linear and retie weights
    model.output = torch.nn.Linear(
        d_model, new_vocab_size, bias=False,
        device=new_embed.weight.device, dtype=new_embed.weight.dtype
    )
    model.output.weight = model.tok_embeddings.weight

    return model
```

**Critical ordering:** Call `resize_embeddings` AFTER loading the pretrained checkpoint and BEFORE `autokernel.optimize()`. The fused QKV pattern matching needs to see the final model shape.

---

## 12.3 SFT Dataset Preparation

### Dataset Formats

The two most common SFT dataset formats are Alpaca and ShareGPT.

**Alpaca format** (Stanford Alpaca): Each example has `instruction`, optional `input`, and `output` fields.

```json
{
    "instruction": "Explain what a list comprehension is in Python.",
    "input": "",
    "output": "A list comprehension is a concise way to create lists..."
}
```

**ShareGPT format** (OpenHermes, WizardLM): Multi-turn conversations with `from` and `value` fields.

```json
{
    "conversations": [
        {"from": "system", "value": "You are a helpful assistant."},
        {"from": "human", "value": "What is Python?"},
        {"from": "gpt", "value": "Python is a programming language..."},
        {"from": "human", "value": "How do I install it?"},
        {"from": "gpt", "value": "You can install Python from..."}
    ]
}
```

### Format Adapters

Convert any format to ChatML messages:

```python
def convert_alpaca(example, system_prompt):
    """Convert Alpaca format to ChatML messages."""
    messages = [{"role": "system", "content": system_prompt}]
    user_content = example["instruction"]
    if example.get("input"):
        user_content += "\n" + example["input"]
    messages.append({"role": "user", "content": user_content})
    messages.append({"role": "assistant", "content": example["output"]})
    return messages


def convert_sharegpt(example, system_prompt):
    """Convert ShareGPT format to ChatML messages."""
    role_map = {"human": "user", "gpt": "assistant", "system": "system"}
    conversations = example.get("conversations", [])
    messages = []
    has_system = any(c.get("from") == "system" for c in conversations)
    if not has_system:
        messages.append({"role": "system", "content": system_prompt})
    for turn in conversations:
        role = role_map.get(turn["from"], turn["from"])
        messages.append({"role": role, "content": turn["value"]})
    return messages
```

### Loss Masking: Only Train on Assistant Tokens

This is the most important concept in SFT. You ONLY compute loss on the tokens the model is supposed to generate (assistant turns). The system prompt and user messages are context, not targets.

```
Tokens:  <|im_start|> system \n You are helpful <|im_end|> \n <|im_start|> user \n What is Python? <|im_end|> \n <|im_start|> assistant \n Python is... <|im_end|> \n
Mask:    IGNORE IGNORE    IGNORE IGNORE          IGNORE    IGNORE IGNORE      IGNORE IGNORE          IGNORE    IGNORE IGNORE            TRAIN TRAIN     TRAIN      TRAIN
```

The mask uses `IGNORE_INDEX = -100`, which is the standard value that `nn.CrossEntropyLoss` skips.

```python
def build_example(messages, tokenizer):
    """Convert ChatML messages to (tokens, labels) with loss masking.

    labels[i] = tokens[i+1] if tokens[i+1] is assistant-generated, else -100.
    """
    all_tokens = []
    all_is_assistant = []

    for msg in messages:
        role = msg["role"]
        content = msg["content"]

        # Header: <|im_start|>role\n -- never learnable
        header_ids = [tokenizer.im_start_id]
        header_ids.extend(tokenizer._base.encode_ordinary(role + "\n"))

        # Content tokens
        content_ids = tokenizer.encode(content)

        # Footer: <|im_end|>\n -- learnable for assistant (must learn to stop)
        footer_ids = [tokenizer.im_end_id]
        footer_ids.extend(tokenizer._base.encode_ordinary("\n"))

        turn_ids = header_ids + content_ids + footer_ids
        is_assistant = role == "assistant"

        # Mark which tokens are learnable
        turn_mask = (
            [False] * len(header_ids) +          # header: never
            [is_assistant] * len(content_ids) +   # content: only for assistant
            [is_assistant] * len(footer_ids)       # footer: only for assistant
        )

        all_tokens.extend(turn_ids)
        all_is_assistant.extend(turn_mask)

    # Build labels with next-token shift
    labels = []
    for i in range(len(all_tokens)):
        if i + 1 < len(all_tokens) and all_is_assistant[i + 1]:
            labels.append(all_tokens[i + 1])
        else:
            labels.append(IGNORE_INDEX)

    return all_tokens, labels
```

**Why the footer is learnable for assistant turns:** The model must learn to emit `<|im_end|>` to signal it is done. Without training on this token, the model generates forever.

### Packing: Fill Every Sequence

Short conversations waste compute if padded to `block_size`. Packing concatenates multiple conversations into a single sequence:

```
Before packing (block_size=2048, 3 conversations of ~500 tokens each):
  Sequence 1: [conv1 tokens ... PAD PAD PAD PAD ... PAD]  (500 real + 1548 pad)
  Sequence 2: [conv2 tokens ... PAD PAD PAD PAD ... PAD]  (600 real + 1448 pad)
  Sequence 3: [conv3 tokens ... PAD PAD PAD PAD ... PAD]  (450 real + 1598 pad)
  Total: 3 sequences, 75% padding = wasted compute

After packing:
  Sequence 1: [conv1 tokens | conv2 tokens | conv3 tokens | PAD ... PAD]
  Total: 1 sequence, 24% padding = much more efficient
```

The loss mask handles boundaries automatically -- padded positions get `IGNORE_INDEX`.

```python
def pack_conversations(examples, block_size, pad_id):
    """Pack short conversations into block_size+1 sequences."""
    target_len = block_size + 1
    packed = []
    cur_tokens, cur_labels = [], []

    for tokens, labels in examples:
        # If adding this example would overflow, finalize current sequence
        if cur_tokens and len(cur_tokens) + len(tokens) > target_len:
            remaining = target_len - len(cur_tokens)
            cur_tokens.extend([pad_id] * remaining)
            cur_labels.extend([IGNORE_INDEX] * remaining)
            packed.append((cur_tokens, cur_labels))
            cur_tokens, cur_labels = [], []

        # Truncate if single example is too long
        if len(tokens) > target_len:
            tokens = tokens[:target_len]
            labels = labels[:target_len]

        cur_tokens.extend(tokens)
        cur_labels.extend(labels)

    # Finalize last partial sequence
    if cur_tokens:
        remaining = target_len - len(cur_tokens)
        cur_tokens.extend([pad_id] * remaining)
        cur_labels.extend([IGNORE_INDEX] * remaining)
        packed.append((cur_tokens, cur_labels))

    return packed
```

### The SFT Dataset Class

Putting it all together into a PyTorch Dataset:

```python
class SFTDataset(torch.utils.data.Dataset):
    """Instruction-tuning dataset with ChatML formatting and loss masking.

    Returns (input_ids, labels) where:
        - input_ids: shape (block_size,)
        - labels: shape (block_size,), with -100 for non-assistant tokens
    """

    def __init__(self, data_path, tokenizer, format="alpaca",
                 block_size=2048, system_prompt="You are a helpful assistant.",
                 pack=True, max_examples=None):
        self.tokenizer = tokenizer
        self.block_size = block_size
        adapter = {"alpaca": convert_alpaca, "sharegpt": convert_sharegpt}[format]

        # Load and convert examples
        raw_examples = load_jsonl(data_path)
        if max_examples:
            raw_examples = raw_examples[:max_examples]

        tokenized = []
        for ex in raw_examples:
            messages = adapter(ex, system_prompt)
            tokens, labels = build_example(messages, tokenizer)
            if len(tokens) > block_size + 1:
                tokens = tokens[:block_size + 1]
                labels = labels[:block_size + 1]
            if len(tokens) >= 4:
                tokenized.append((tokens, labels))

        if pack:
            self.sequences = pack_conversations(tokenized, block_size, tokenizer.pad_id)
        else:
            self.sequences = pad_conversations(tokenized, block_size, tokenizer.pad_id)

        print(f"SFTDataset: {len(raw_examples)} examples -> {len(self.sequences)} sequences")

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        tokens, labels = self.sequences[idx]
        tokens_t = torch.tensor(tokens, dtype=torch.long)
        labels_t = torch.tensor(labels, dtype=torch.long)
        return tokens_t[:-1], labels_t[:-1]  # standard next-token shift
```

---

## 12.4 SFT Training

### Key Differences from Pretraining

| Parameter | Pretraining | SFT |
|-----------|------------|-----|
| Learning rate | 8e-4 | 1e-5 to 5e-5 |
| Epochs | 1 (over huge data) | 1-3 (over small data) |
| Batch size | 64-128 | 8-32 |
| Data size | 10B+ tokens | 10K-1M tokens |
| Loss | All tokens | Assistant tokens only |
| Optimizer | AdamW or Muon | AdamW only |

### Why Lower Learning Rate

The pretrained model already has good representations. SFT is about teaching new behavior (following instructions) without destroying existing knowledge. A high learning rate causes **catastrophic forgetting** -- the model learns to chat but forgets how to write coherent text.

### Weighted Cross-Entropy for EOS

One common failure mode: the model never stops generating. It produces good responses but cannot decide when to end. We fix this by upweighting the EOS and `<|im_end|>` tokens:

```python
class WeightedCrossEntropyLoss(nn.Module):
    """Cross-entropy with per-token weight multipliers."""

    def __init__(self, token_weights=None, ignore_index=-100):
        super().__init__()
        self.token_weights = token_weights or {}
        self.ignore_index = ignore_index

    def forward(self, logits, targets):
        logits_flat = logits.view(-1, logits.size(-1))
        targets_flat = targets.view(-1)

        # Per-token CE (no reduction)
        losses = F.cross_entropy(
            logits_flat, targets_flat,
            reduction="none", ignore_index=self.ignore_index,
        )

        # Build weight mask
        mask = targets_flat != self.ignore_index
        weights = torch.ones_like(losses)
        weights[~mask] = 0.0

        # Upweight specific tokens
        for token_id, weight in self.token_weights.items():
            token_mask = targets_flat == token_id
            weights[token_mask] = weight

        total_weight = weights.sum()
        if total_weight == 0:
            return losses.sum() * 0  # no valid targets
        return (losses * weights).sum() / total_weight
```

Usage:
```python
# 5x weight on <|im_end|> (50258) and EOS (50256)
loss_fn = WeightedCrossEntropyLoss(token_weights={50256: 5.0, 50258: 5.0})
```

### Preventing Catastrophic Forgetting

Strategy 1: **Mix general data with SFT data.** Reserve 10-20% of each batch for standard language modeling examples (from your pretraining data). This keeps the model's general capabilities alive.

Strategy 2: **Low learning rate + few epochs.** 1-3 epochs at 2e-5 is usually enough. More epochs means more overfitting to the small SFT dataset.

Strategy 3: **Regularization.** Mild weight decay (0.01) and gradient clipping (max_norm=1.0) prevent large parameter updates.

### Complete SFT Training Script

```python
"""train_sft.py -- Supervised fine-tuning with ChatML and loss masking."""
import torch
from torch.utils.data import DataLoader

from halo_training.chat_template import ChatMLTokenizer, resize_embeddings, CHATML_TOKENS
from halo_training.sft_data import SFTDataset
from halo_training.sft_loss import build_sft_loss_fn


def train_sft(
    model,
    data_path,
    format="alpaca",
    epochs=2,
    batch_size=8,
    block_size=1024,
    lr=2e-5,
    eos_weight=5.0,
    checkpoint_dir="checkpoints/sft",
    max_steps=None,
):
    device = torch.device("cuda")

    # 1. Build tokenizer with ChatML tokens
    tokenizer = ChatMLTokenizer(base="gpt2", extra_tokens=CHATML_TOKENS)

    # 2. Resize model embeddings to fit new tokens
    model = resize_embeddings(model, tokenizer.vocab_size)
    model = model.to(device)

    # 3. Build dataset
    dataset = SFTDataset(
        data_path=data_path,
        tokenizer=tokenizer,
        format=format,
        block_size=block_size,
        pack=True,
    )
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=2)

    # 4. Loss function with EOS upweighting
    loss_fn = build_sft_loss_fn(eos_weight=eos_weight)

    # 5. Optimizer (AdamW, NOT Muon -- DPO/SFT needs stable updates)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)

    # 6. Mixed precision
    scaler = torch.amp.GradScaler("cuda")

    # 7. Training loop
    model.train()
    global_step = 0

    for epoch in range(epochs):
        for batch in loader:
            input_ids, labels = [x.to(device) for x in batch]

            with torch.amp.autocast("cuda", dtype=torch.float16):
                logits = model(input_ids)
                loss = loss_fn(logits, batch)

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()

            global_step += 1
            if global_step % 10 == 0:
                print(f"Step {global_step}: loss={loss.item():.4f}")

            if max_steps and global_step >= max_steps:
                break
        if max_steps and global_step >= max_steps:
            break

    # Save checkpoint
    torch.save(model.state_dict(), f"{checkpoint_dir}/sft_final.pt")
    return model
```

---

## 12.5 DPO (Direct Preference Optimization)

### The Idea

After SFT, the model follows instructions but does not distinguish between good and bad responses. DPO teaches this distinction using pairs of (chosen, rejected) responses to the same prompt.

Example:
```
Prompt: "Explain recursion in Python."
Chosen:  "Recursion is when a function calls itself..."  (clear, accurate)
Rejected: "Recursion is a thing in Python where you do recursion..." (vague, circular)
```

### The DPO Formula

DPO optimizes the policy (our model) to prefer chosen over rejected:

```
loss = -log(sigmoid(beta * (log_pi(chosen) - log_ref(chosen)
                          - log_pi(rejected) + log_ref(rejected))))
```

Where:
- `pi` is our policy model (being trained)
- `ref` is a frozen copy of the SFT model (the reference)
- `beta` controls how much the policy can deviate from the reference (typically 0.1)
- `log_pi(x)` is the sum of log-probabilities of assistant tokens in response x

### Why a Reference Model

Without the reference model, DPO would simply maximize `log_pi(chosen)` and minimize `log_pi(rejected)`. This causes the model to collapse -- it would assign all probability mass to a few "safe" tokens and become useless for generation.

The reference model acts as a KL-divergence constraint: the policy cannot stray too far from the original SFT model. This is what `beta` controls -- lower beta means less constraint.

### Computing Log Probabilities

```python
def compute_log_probs(model, input_ids, mask):
    """Compute per-token log probabilities for masked (assistant) positions.

    Args:
        model: Language model that returns logits
        input_ids: (B, T) token IDs
        mask: (B, T) 1 for assistant positions, 0 to ignore

    Returns:
        (B,) sum of log-probs over masked positions
    """
    with torch.amp.autocast("cuda", dtype=torch.float16):
        logits = model(input_ids)  # (B, T, V)

    # Next-token shift
    shift_logits = logits[:, :-1, :]   # (B, T-1, V)
    shift_labels = input_ids[:, 1:]     # (B, T-1)
    shift_mask = mask[:, 1:]            # (B, T-1)

    # Per-token log probs
    log_probs = F.log_softmax(shift_logits.float(), dim=-1)
    token_log_probs = log_probs.gather(2, shift_labels.unsqueeze(-1)).squeeze(-1)

    # Mask and sum
    return (token_log_probs * shift_mask.float()).sum(dim=-1)  # (B,)
```

### DPO Loss Implementation

```python
def dpo_loss(policy_model, ref_model,
             chosen_ids, chosen_mask, rejected_ids, rejected_mask,
             beta=0.1):
    """Compute DPO loss.

    Returns:
        loss: scalar DPO loss
        metrics: dict with rewards and accuracy
    """
    # Policy log probs (gradient flows through these)
    pi_chosen = compute_log_probs(policy_model, chosen_ids, chosen_mask)
    pi_rejected = compute_log_probs(policy_model, rejected_ids, rejected_mask)

    # Reference log probs (frozen, no gradient)
    with torch.no_grad():
        ref_chosen = compute_log_probs(ref_model, chosen_ids, chosen_mask)
        ref_rejected = compute_log_probs(ref_model, rejected_ids, rejected_mask)

    # Implicit rewards
    chosen_reward = beta * (pi_chosen - ref_chosen)
    rejected_reward = beta * (pi_rejected - ref_rejected)

    # DPO loss: maximize chosen-rejected reward gap
    loss = -F.logsigmoid(chosen_reward - rejected_reward).mean()

    with torch.no_grad():
        accuracy = (chosen_reward > rejected_reward).float().mean()

    return loss, {
        "chosen_reward": chosen_reward.mean().item(),
        "rejected_reward": rejected_reward.mean().item(),
        "accuracy": accuracy.item(),
        "loss": loss.item(),
    }
```

### DPO Dataset

Each example is a preference pair: same prompt, two different assistant completions.

```python
class DPODataset(torch.utils.data.Dataset):
    """Dataset of (chosen, rejected) preference pairs.

    JSONL format:
    {"chosen": [messages...], "rejected": [messages...]}

    Each item returns:
        chosen_ids, chosen_mask, rejected_ids, rejected_mask
    """

    def __init__(self, data_path, tokenizer, block_size=1536):
        self.tokenizer = tokenizer
        self.block_size = block_size
        self.pairs = []

        for ex in load_jsonl(data_path):
            chosen = ex.get("chosen", [])
            rejected = ex.get("rejected", [])
            if not chosen or not rejected:
                continue

            c_ids, c_mask = tokenize_messages(chosen, tokenizer)
            r_ids, r_mask = tokenize_messages(rejected, tokenizer)

            # Truncate if needed
            if len(c_ids) > block_size:
                c_ids, c_mask = c_ids[:block_size], c_mask[:block_size]
            if len(r_ids) > block_size:
                r_ids, r_mask = r_ids[:block_size], r_mask[:block_size]

            if sum(c_mask) >= 2 and sum(r_mask) >= 2:
                self.pairs.append((c_ids, c_mask, r_ids, r_mask))

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        c_ids, c_mask, r_ids, r_mask = self.pairs[idx]

        def pad(ids, mask):
            pad_len = self.block_size - len(ids)
            ids = ids + [self.tokenizer.pad_id] * pad_len
            mask = mask + [0] * pad_len
            return torch.tensor(ids, dtype=torch.long), torch.tensor(mask, dtype=torch.long)

        c_ids_t, c_mask_t = pad(c_ids, c_mask)
        r_ids_t, r_mask_t = pad(r_ids, r_mask)
        return c_ids_t, c_mask_t, r_ids_t, r_mask_t
```

### Generating Preference Data

You need (chosen, rejected) pairs. The easiest approach: use a strong model as a judge.

```python
"""generate_preferences.py -- Create DPO training pairs."""
import json
import random


def generate_preference_pair(prompt, model, tokenizer, n_samples=4):
    """Generate multiple responses and pair best with worst.

    Strategy:
      1. Generate n_samples responses with different temperatures
      2. Score each by a quality heuristic (or strong model judge)
      3. Pair highest-scored with lowest-scored
    """
    responses = []
    for temp in [0.3, 0.6, 0.9, 1.2]:
        response = generate(model, tokenizer, prompt, temperature=temp, max_tokens=256)
        score = score_response(prompt, response)  # could be rule-based or LLM-as-judge
        responses.append((response, score))

    responses.sort(key=lambda x: x[1])
    chosen = responses[-1][0]   # highest score
    rejected = responses[0][0]  # lowest score

    return {
        "chosen": [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": prompt},
            {"role": "assistant", "content": chosen},
        ],
        "rejected": [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": prompt},
            {"role": "assistant", "content": rejected},
        ],
    }


def score_response(prompt, response):
    """Simple rule-based scoring (replace with LLM judge for production)."""
    score = 0
    score += min(len(response.split()) / 50, 1.0)  # prefer moderate length
    score -= response.count(response[:20]) * 0.5     # penalize repetition
    if response.strip().endswith("."):
        score += 0.3  # prefer complete sentences
    if "I don't" in response or "I cannot" in response:
        score -= 0.2  # penalize refusals on benign prompts
    return score
```

### DPO Training Loop

```python
def train_dpo(model, dataset, epochs=2, batch_size=4, lr=5e-6, beta=0.1):
    """Run DPO training."""
    import copy
    device = next(model.parameters()).device

    # Create frozen reference model
    ref_model = copy.deepcopy(model)
    ref_model.eval()
    for p in ref_model.parameters():
        p.requires_grad = False

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    model.train()
    for epoch in range(epochs):
        for batch in loader:
            c_ids, c_mask, r_ids, r_mask = [x.to(device) for x in batch]

            loss, metrics = dpo_loss(
                model, ref_model, c_ids, c_mask, r_ids, r_mask, beta=beta
            )

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            print(f"loss={metrics['loss']:.4f} acc={metrics['accuracy']:.3f} "
                  f"c_rew={metrics['chosen_reward']:.3f} r_rew={metrics['rejected_reward']:.3f}")
```

### DPO Monitoring

Watch these metrics during training:

| Metric | Healthy Range | Problem |
|--------|--------------|---------|
| Accuracy | 0.55-0.80 | < 0.5 means not learning; > 0.95 means overfitting |
| Chosen reward | Positive, increasing | Negative means chosen is worse than reference |
| Rejected reward | Near zero or negative | Strongly positive means rejected is preferred |
| Loss | 0.4-0.7 | < 0.3 means overfitting; > 0.9 means not learning |

---

## 12.6 Tool-Use and Function Calling

### The Concept

A tool-using model generates structured tool calls instead of (or alongside) natural language. The agent loop detects these, executes them, and feeds results back.

### Special Tokens

```python
TOOL_CALL_START = "<tool_call>"    # ID: 50260
TOOL_CALL_END = "</tool_call>"     # ID: 50261
```

### Training Data Format

The system prompt defines available tools:

```
<|im_start|>system
You are a helpful assistant with access to the following tools:

[{"name": "search", "description": "Search the web", "parameters": {"query": "string"}}]
[{"name": "calculate", "description": "Evaluate math", "parameters": {"expression": "string"}}]

When you need to use a tool, output a tool call in this format:
<tool_call>{"name": "tool_name", "arguments": {"key": "value"}}</tool_call>
<|im_end|>
<|im_start|>user
What is 15% of 2340?
<|im_end|>
<|im_start|>assistant
I'll calculate that for you.
<tool_call>{"name": "calculate", "arguments": {"expression": "2340 * 0.15"}}</tool_call>
<|im_end|>
<|im_start|>tool
{"result": 351.0}
<|im_end|>
<|im_start|>assistant
15% of 2340 is 351.
<|im_end|>
```

### The Agent Loop

```python
"""agent.py -- Simple tool-use agent loop."""
import json
import re


TOOL_CALL_PATTERN = re.compile(r"<tool_call>(.*?)</tool_call>", re.DOTALL)


def agent_loop(model, tokenizer, messages, tools, max_turns=5):
    """Run the agent loop: generate -> detect tool call -> execute -> repeat."""

    for turn in range(max_turns):
        # Generate assistant response
        response = generate_chat(model, tokenizer, messages)

        # Check for tool calls
        match = TOOL_CALL_PATTERN.search(response)
        if not match:
            # No tool call -- return the final response
            messages.append({"role": "assistant", "content": response})
            return messages

        # Parse tool call
        tool_call_json = match.group(1)
        try:
            tool_call = json.loads(tool_call_json)
        except json.JSONDecodeError:
            messages.append({"role": "assistant", "content": response})
            return messages

        # Add assistant message (includes tool call)
        messages.append({"role": "assistant", "content": response})

        # Execute tool
        tool_name = tool_call.get("name")
        tool_args = tool_call.get("arguments", {})
        result = execute_tool(tool_name, tool_args, tools)

        # Add tool result
        messages.append({"role": "tool", "content": json.dumps(result)})

    return messages


def execute_tool(name, args, tools):
    """Execute a tool by name with given arguments."""
    for tool in tools:
        if tool["name"] == name:
            return tool["function"](**args)
    return {"error": f"Unknown tool: {name}"}
```

### Training for Tool Use

The SFT dataset includes examples where the assistant generates tool calls. The same loss masking applies -- we train on assistant tokens (including the `<tool_call>` tokens). The tool results (`role: "tool"`) are masked like user messages.

Mix tool-use examples with regular conversation examples (70% regular, 30% tools) to avoid the model trying to use tools for every question.

---

## 12.7 Evaluation

### MT-Bench: Multi-Turn Conversation Quality

MT-Bench presents 80 multi-turn questions across 8 categories (writing, roleplay, extraction, reasoning, math, coding, knowledge, STEM). A strong model (GPT-4) judges responses on a 1-10 scale.

For small models (~150M), expect scores of 2-4. This is still useful for relative comparison between your SFT model and baselines.

```python
"""eval_mtbench.py -- Simplified MT-Bench evaluation."""


MT_BENCH_SAMPLES = [
    {
        "category": "writing",
        "turns": [
            "Write a short poem about autumn leaves.",
            "Now rewrite it from the perspective of a single leaf."
        ],
    },
    {
        "category": "reasoning",
        "turns": [
            "If all roses are flowers, and some flowers fade quickly, can we conclude that some roses fade quickly?",
            "Explain why or why not, using formal logic."
        ],
    },
    # ... 78 more ...
]


def evaluate_mtbench(model, tokenizer, samples=None, max_tokens=512):
    """Generate responses for MT-Bench prompts."""
    samples = samples or MT_BENCH_SAMPLES
    results = []

    for sample in samples:
        messages = [{"role": "system", "content": "You are a helpful assistant."}]
        responses = []

        for turn in sample["turns"]:
            messages.append({"role": "user", "content": turn})
            response = generate_chat(model, tokenizer, messages, max_tokens=max_tokens)
            messages.append({"role": "assistant", "content": response})
            responses.append(response)

        results.append({
            "category": sample["category"],
            "responses": responses,
        })

    return results
```

### IFEval: Instruction Following

IFEval tests whether the model follows specific formatting instructions: "Write exactly 3 paragraphs", "Use no more than 50 words", "Include the word 'however'".

```python
def eval_ifeval_simple(model, tokenizer):
    """Simple instruction-following tests."""
    tests = [
        {
            "prompt": "Write exactly 3 sentences about dogs.",
            "check": lambda r: len([s for s in r.split(".") if s.strip()]) == 3,
        },
        {
            "prompt": "List 5 colors, one per line.",
            "check": lambda r: len([l for l in r.strip().split("\n") if l.strip()]) == 5,
        },
        {
            "prompt": "Respond with only the word 'yes' or 'no': Is the sky blue?",
            "check": lambda r: r.strip().lower() in ("yes", "no", "yes.", "no."),
        },
    ]

    passed = 0
    for test in tests:
        response = generate_chat(model, tokenizer, [
            {"role": "system", "content": "Follow instructions exactly."},
            {"role": "user", "content": test["prompt"]},
        ])
        if test["check"](response):
            passed += 1
            print(f"  PASS: {test['prompt'][:50]}...")
        else:
            print(f"  FAIL: {test['prompt'][:50]}... -> {response[:100]}")

    print(f"\nIFEval: {passed}/{len(tests)} passed")
```

### Manual Inspection

The most important evaluation is reading the model's outputs yourself. Generate 10-20 responses and check for:

1. **Coherence:** Does the response make sense?
2. **Relevance:** Does it address the question?
3. **Completion:** Does it end naturally (not mid-sentence)?
4. **Repetition:** Does it loop or repeat phrases?
5. **Hallucination:** Does it state false facts confidently?

```python
def manual_eval(model, tokenizer, prompts=None):
    """Generate responses for manual inspection."""
    if prompts is None:
        prompts = [
            "What is a neural network?",
            "Write a haiku about programming.",
            "Explain the difference between a list and a tuple in Python.",
            "What causes rain?",
            "Tell me a short story about a robot learning to cook.",
        ]

    tokenizer_chat = ChatMLTokenizer(base="gpt2", extra_tokens=CHATML_TOKENS)

    for prompt in prompts:
        messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": prompt},
        ]

        # Tokenize the conversation so far
        input_ids = tokenizer_chat.encode_chatml(messages)
        # Add assistant turn start
        input_ids.extend(tokenizer_chat.start_assistant_turn())

        # Generate
        input_tensor = torch.tensor([input_ids], device="cuda")
        response_ids = generate_with_kv_cache(
            model, input_tensor,
            max_new_tokens=256,
            temperature=0.7,
            top_p=0.9,
            eos_token_id=tokenizer_chat.im_end_id,
        )

        response_text = tokenizer_chat.decode(response_ids)
        print(f"\n{'='*60}")
        print(f"USER: {prompt}")
        print(f"ASST: {response_text}")
        print(f"{'='*60}")
```

### Decoding Parameters for Small Models

Small models (< 350M) are more sensitive to decoding parameters. These settings help:

```python
# Good defaults for 150M-350M chat models
generation_config = {
    "temperature": 0.7,
    "top_p": 0.9,
    "top_k": 50,
    "repetition_penalty": 1.3,    # prevents loops
    "frequency_penalty": 0.5,     # discourages repeated tokens
    "max_new_tokens": 256,
}
```

Without `repetition_penalty`, small models tend to enter repetition loops ("The the the the..."). A value of 1.3 is a good starting point.

---

## Exercises

### Exercise 1: Fine-Tune on Alpaca with ChatML

Download the Alpaca cleaned dataset and fine-tune your base model:

```bash
# Download dataset
pip install datasets
python -c "
from datasets import load_dataset
import json
ds = load_dataset('yahma/alpaca-cleaned', split='train')
with open('data/alpaca_cleaned.jsonl', 'w') as f:
    for ex in ds:
        f.write(json.dumps(ex) + '\n')
print(f'Saved {len(ds)} examples')
"

# Train SFT
python train_sft.py \
    --model-path checkpoints/base_model/best.pt \
    --data-path data/alpaca_cleaned.jsonl \
    --format alpaca \
    --epochs 2 \
    --lr 2e-5 \
    --eos-weight 5.0
```

Verify:
- Loss decreases from approximately 3.0 to approximately 1.5
- Model responds to "What is Python?" with a relevant answer (not a continuation)
- Model stops generating (emits `<|im_end|>`) within 256 tokens

### Exercise 2: Generate Text in Conversation Mode

Write a simple interactive loop:

```python
"""chat.py -- Interactive conversation with your SFT model."""
import torch

def chat(model, tokenizer):
    messages = [{"role": "system", "content": "You are a helpful assistant."}]
    print("Chat with your model (type 'quit' to exit)")

    while True:
        user_input = input("\nYou: ").strip()
        if user_input.lower() == "quit":
            break

        messages.append({"role": "user", "content": user_input})

        # Generate response
        response = generate_chat(model, tokenizer, messages, max_tokens=256)
        messages.append({"role": "assistant", "content": response})

        print(f"Bot: {response}")
```

Test with at least 5 different questions. Note which ones produce good responses and which produce garbage -- this tells you where DPO could help.

### Exercise 3: Create 100 Preference Pairs and Run DPO

1. Generate 100 prompts covering different topics
2. For each prompt, generate 4 responses at different temperatures
3. Pair the best and worst into preference pairs
4. Save as JSONL in DPO format
5. Run DPO training for 100 steps

```bash
# Generate preference data
python generate_preferences.py --model checkpoints/sft/sft_final.pt --n-prompts 100

# Train DPO
python train_dpo.py --model checkpoints/sft/sft_final.pt \
    --data checkpoints/sft/preferences.jsonl \
    --epochs 2 --lr 5e-6 --beta 0.1
```

Verify:
- DPO accuracy starts above 0.5 and increases to 0.6-0.7
- Chosen reward is positive and increasing
- Rejected reward is near zero or slightly negative
- Qualitatively: DPO model gives better responses than SFT model on a held-out set of 10 prompts

---

## Checkpoint

Before moving to Part 13, verify:

- [ ] You understand the full post-training pipeline: Base -> SFT -> DPO
- [ ] ChatML tokenizer works: encode/decode round-trips correctly
- [ ] Embeddings resize correctly: model trains without shape errors
- [ ] Loss masking: only assistant tokens contribute to loss
- [ ] SFT model follows instructions (answers questions, does not just continue text)
- [ ] SFT model stops generating (emits im_end within max_tokens)
- [ ] DPO training shows increasing accuracy
- [ ] You can generate text interactively and judge quality manually

**Expected time:** 6 hours. The most time-consuming part is generating preference data and debugging tokenization edge cases. If your model produces gibberish after SFT, the first thing to check is your loss mask -- make sure assistant tokens are NOT being masked.

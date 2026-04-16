# SFT Pipeline Design: EOS Warm-Up + Staged Instruction Tuning

**Date:** 2026-04-16
**Status:** Approved
**Base checkpoint:** `argus_prime_cc_ddp_v2/step_72076.pt` (168M params, loss 3.81, Dolma 10B CC 2 epochs)

## Problem

ARGUS-PRIME generates coherent text but has two issues:
1. **Poor EOS behavior** — rambles past natural stopping points, repetition degeneration on creative prompts
2. **No instruction following** — pre-trained on raw Common Crawl, has no concept of prompt/response format

## Solution: 4-Phase Staged Pipeline

```
Phase 0: EOS Warm-Up + Context Extension (sharpen boundaries, 1024 -> 2048 seq_len)
    ↓
Stage C: Basic SFT (teach prompt/response + clean stops)
    ↓
Stage A: Broad SFT (general chat assistant)
    ↓
Stage B: Domain SFT (code, STEM, tool calls)
```

Each phase resumes from the previous checkpoint. Each phase is independently validated before proceeding.

---

## 1. Chat Template & Special Tokens

### Format: ChatML

```
<|im_start|>system
You are a helpful assistant.<|im_end|>
<|im_start|>user
What is photosynthesis?<|im_end|>
<|im_start|>assistant
Photosynthesis is the process by which plants convert sunlight into energy...<|im_end|>
```

### Token Registration

| Token | ID | Purpose |
|-------|-----|---------|
| `<\|im_start\|>` | 50257 | Turn start marker |
| `<\|im_end\|>` | 50258 | Turn end marker (model must learn to emit this) |
| `<\|pad\|>` | 50259 | Padding for batched variable-length conversations |

Vocab size grows from 50257 to 50260. Embedding and LM head layers are resized with zero-initialized new rows. The existing `<|endoftext|>` (50256) remains as document-level separator; `<|im_end|>` becomes the conversation turn terminator.

### Stage B Additional Tokens (registered at stage B only)

| Token | ID | Purpose |
|-------|-----|---------|
| `<tool_call>` | 50260 | Tool invocation start |
| `</tool_call>` | 50261 | Tool invocation end |

---

## 2. Loss Masking

During SFT, the model is only trained to predict **assistant response tokens**, not prompt/system tokens.

```
Tokens:  <|im_start|> user \n What is X? <|im_end|> \n <|im_start|> assistant \n The answer is... <|im_end|>
Labels:     -100      -100 -100  -100    -100     -100    -100        -100    -100  answer is...  <|im_end|>
```

- All tokens before and including `assistant\n` are set to `-100` (PyTorch cross_entropy ignore index)
- The model learns to predict response tokens AND the `<|im_end|>` terminator
- For multi-turn conversations, each assistant turn is unmasked; all user/system turns stay masked

### Trainer Change

Currently `trainer.py` computes loss internally via shifted `input_ids`. For SFT, the dataset returns `(input_ids, labels)` pairs where labels are pre-masked, and the trainer uses those labels directly.

---

## 3. EOS Warm-Up (Phase 0)

A short continued pre-training phase that serves two purposes: sharpen EOS behavior and extend context length from 1024 to 2048.

**What it does:**
- Takes Dolma CC data re-chunked with **clean document boundaries** — sequences are split only at `<|endoftext|>` tokens (real document separators), never mid-document. Each chunk contains 1-N complete documents that fit within `block_size`, with EOS tokens at each boundary. Short trailing documents that don't fill a chunk are kept as-is (not padded, not merged with the next document).
- Applies **weighted loss** on EOS tokens: loss for predicting `<|endoftext|>` (50256) multiplied by configurable weight (default 5x)
- **Extends context to 2048 tokens** — the model's RoPE `freqs_cis` buffer is already pre-computed for `max_seq_len * 2 = 2048` positions, so no RoPE modification is needed. The model simply trains at `block_size=2048` to adapt to longer sequences. Batch size is halved (16 -> 8) to compensate for the doubled sequence length, keeping total tokens per step constant.
- Short phase: ~3000-5000 steps at 1/10th pre-training LR (5e-5) to avoid destabilizing weights

**Why combine EOS + context extension:** Both are lightweight adaptation tasks that don't change the model's core knowledge — EOS refines boundary prediction, context extension adapts positional representations. Doing them together saves a training phase and the low LR suits both goals. The model sees longer documents with clean EOS boundaries, reinforcing both behaviors simultaneously.

**Why 2048 is enough:** At 168M params, 2048 tokens comfortably fits 5-8 chat turns. If 4096 is later needed for Stage B (long code, multi-step tool calls), NTK-aware RoPE scaling (adjusting base frequency from 10000 to ~40000) can be applied as a separate mini-phase at that point.

**Implementation:**
- `WeightedCrossEntropyLoss` wrapper accepting `token_weights` dict (token ID -> multiplier)
- `--eos-weight` CLI flag (default 5.0)
- `--phase eos-warmup` mode that sets low LR, `block_size=2048`, and short step budget automatically

---

## 4. Training Stages & Hyperparameters

| | Phase 0: EOS Warm-Up + Context Extension | Stage C: Basic SFT | Stage A: Broad SFT | Stage B: Domain SFT |
|--|-----|-----|-----|-----|
| **Base checkpoint** | step_72076.pt | Phase 0 output | Stage C output | Stage A output |
| **Dataset** | Dolma CC (clean-chunked) | Alpaca-cleaned 52K | OpenHermes 2.5 1M | Curated (TBD) |
| **Epochs** | 1 | 3 | 2 | 3-5 |
| **LR** | 5e-5 | 2e-5 | 2e-5 | 1e-5 |
| **Warmup** | 100 steps | 200 steps | 500 steps | 100 steps |
| **Schedule** | Cosine | Cosine | Cosine | Cosine |
| **Batch size** | 8 x 8 accum | 8 x 4 accum | 8 x 4 accum | 8 x 4 accum |
| **Seq length** | 2048 | 2048 | 2048 | 2048 |
| **Optimizer** | AdamW | AdamW | AdamW | AdamW |
| **Special** | EOS weight 5x, context extension | Loss masking | Loss masking, multi-turn | Loss masking, tool tokens |

### Why AdamW (not Muon) for SFT

Muon's 2x token efficiency shines for pre-training from scratch. For fine-tuning with small LR on small data, AdamW is safer — Muon's Newton-Schulz orthogonalization can be too aggressive when making small adjustments to already-trained weights.

### Why Lower LR at Each Stage

Classic fine-tuning principle: each stage is further from pre-training, so we use smaller updates to avoid catastrophic forgetting. 2e-5 is the standard SFT LR for models at this scale (Alpaca, Vicuna, etc).

### Why More Epochs on Smaller Data

Alpaca (52K) and domain data (TBD) are small enough that the model needs multiple passes. OpenHermes at 1M is large enough that 2 epochs suffices. Overfitting risk managed by monitoring val loss.

---

## 5. Dataset Loaders & Format Pipeline

### Pipeline

```
Raw dataset (HuggingFace / local files)
    ↓
Format adapter (per-dataset: alpaca, sharegpt, chatml)
    ↓
ChatML formatter (standardize all conversations to ChatML)
    ↓
Tokenize + build labels (apply loss masking)
    ↓
Pack/pad to seq_length
    ↓
(input_ids, labels) tensors
```

### Format Adapters

| Format | Datasets | Fields | Conversion |
|--------|----------|--------|------------|
| **alpaca** | Alpaca-cleaned | `instruction`, `input`, `output` | Single-turn: user=instruction+input, assistant=output |
| **sharegpt** | OpenHermes 2.5 | `conversations: [{from, value}]` | Multi-turn: map `human`->user, `gpt`->assistant |
| **chatml** | Any pre-formatted | `messages: [{role, content}]` | Pass through directly |

### Tool Call Format (Stage B)

```
<|im_start|>assistant
<tool_call>{"name": "calculate", "args": {"expr": "2+2"}}</tool_call><|im_end|>
<|im_start|>tool
{"result": 4}<|im_end|>
<|im_start|>assistant
The answer is 4.<|im_end|>
```

The `tool` role is just another ChatML turn — no special handling beyond the `<tool_call>` / `</tool_call>` token pair.

### Conversation Packing

Short conversations are packed into a single sequence separated by `<|pad|>` tokens. The standard causal attention mask is used (no custom block-diagonal mask) — this means later conversations in a packed sequence can attend to earlier ones. This is the pragmatic choice: most open-source SFT implementations (Axolotl, TRL) do the same, the cross-contamination is minor in practice, and it avoids modifying the attention code. Loss masking already ensures the model only learns from assistant tokens, so the attention leakage only affects context, not gradients. If this proves problematic (diagnosed by comparing packed vs unpacked val loss), we can switch to right-padding without packing as a fallback.

---

## 6. Implementation Plan

### New Files

| File | Purpose |
|------|---------|
| `halo_training/sft_data.py` | ChatML formatter, format adapters (alpaca/sharegpt/chatml), loss mask builder, conversation packing |
| `halo_training/sft_loss.py` | `WeightedCrossEntropyLoss` with per-token weights, EOS weight support, compatibility with existing chunked CE |
| `halo_training/chat_template.py` | ChatML constants, special token registration, `resize_embeddings()` helper, tokenizer factory |

### Modified Files

| File | Change |
|------|--------|
| `halo_training/trainer.py` | Accept `labels` from dataset (bypass internal target shift). Add `--phase` flag dispatch. |
| `halo_training/data.py` | Add `SFTDataset` alongside existing `BabyLMDataset`. Factory function picks based on `--phase`. |
| `halo_training/cli.py` | New flags: `--phase` (eos-warmup/sft), `--sft-dataset` (alpaca/openhermes/custom path), `--sft-format` (alpaca/sharegpt/chatml), `--eos-weight`, `--chat-template` |
| `scripts/generate_text.py` | Add `--chat` mode that wraps prompt in ChatML and stops on `<|im_end|>` instead of `<|endoftext|>` |

### Untouched

`memory.py`, `streaming.py`, `metrics.py`, `callbacks.py`, `smoke.py`, all kernel code, all model code. The SFT pipeline slots alongside the existing pre-training pipeline.

### CLI Usage

```bash
# Phase 0: EOS warm-up + context extension (1024 -> 2048)
python -m halo_training --model models/argus_prime.py --class-name ArgusPrime \
    --resume-from checkpoints/argus_prime_cc_ddp_v2/step_72076.pt \
    --phase eos-warmup --eos-weight 5.0 --lr 5e-5 --block-size 2048 \
    --batch-size 8 --accum-steps 8 --dataset datasets/dolma_cc_clean

# Stage C: Basic SFT on Alpaca
python -m halo_training --model models/argus_prime.py --class-name ArgusPrime \
    --resume-from checkpoints/argus_prime_eos/best.pt \
    --phase sft --sft-dataset alpaca --lr 2e-5 --epochs 3 --block-size 2048

# Stage A: Broad SFT on OpenHermes
python -m halo_training --model models/argus_prime.py --class-name ArgusPrime \
    --resume-from checkpoints/argus_prime_sft_alpaca/best.pt \
    --phase sft --sft-dataset openhermes --lr 2e-5 --epochs 2 --block-size 2048

# Stage B: Domain SFT (future)
python -m halo_training --model models/argus_prime.py --class-name ArgusPrime \
    --resume-from checkpoints/argus_prime_sft_openhermes/best.pt \
    --phase sft --sft-dataset /path/to/domain_data --sft-format chatml --lr 1e-5 --epochs 3

# Chat inference
python scripts/generate_text.py --model models/argus_prime.py --class-name ArgusPrime \
    --checkpoint checkpoints/argus_prime_sft_openhermes/best.pt \
    --chat --prompt "What is photosynthesis?"
```

---

## 7. Evaluation & Success Criteria

### Per-Stage Validation

| Stage | Success Metric | Target | How to Measure |
|-------|---------------|--------|----------------|
| Phase 0: EOS | Model predicts EOS within 5 tokens of true document boundary | >80% of val samples | Run val set, check if EOS appears near ground truth boundary |
| Stage C | Follows prompt/response format, stops at `<|im_end|>` | >95% of generations end cleanly | Generate 100 samples, count those with exactly one `<|im_end|>` |
| Stage A | Coherent multi-turn, stays on topic, no repetition | Val loss < 2.5 on held-out OpenHermes split | Standard val loss + manual spot-check of 20 diverse prompts |
| Stage B | Correct code/STEM/tool-call format | Domain-specific accuracy TBD | Exact match on tool call JSON, code execution pass rate |

### Regression Checks

After each stage, re-run the previous stage's eval to check for catastrophic forgetting. If stage A degrades stage C's clean-stop rate below 85%, reduce LR or mix in Alpaca data during stage A training (replay buffer).

### Fixed Test Suite (10 prompts, evaluated after every stage)

1. "What is the capital of France?" (factual, should be short)
2. "Explain quantum entanglement in simple terms" (explanation, moderate length)
3. "Write a haiku about rain" (creative, should be exactly 3 lines)
4. "List 5 prime numbers" (structured output)
5. "Tell me a joke" (should stop after punchline)
6. Multi-turn: "What is Python?" -> "Show me a hello world example"
7. "Summarize this: [paste 200 words]" (follows instruction)
8. Empty-ish prompt: "Hi" (should respond briefly, not ramble)
9. "Calculate 15 * 23" (stage B: tool call)
10. "What is the derivative of x^3?" (stage B: STEM)

Saved and compared across stages for qualitative progression tracking.

---

## 8. Decoding Parameters

Best config from inference testing (pre-SFT baseline):

| Param | Value |
|-------|-------|
| temperature | 0.8-0.9 |
| top_k | 50 |
| top_p | 0.92 |
| repetition_penalty | 1.3 |
| frequency_penalty | 0.5 |

Post-SFT, repetition penalty can likely be reduced (1.1-1.2) since the model will have learned cleaner stopping behavior. Re-evaluate after each stage.

---

## 9. Hardware Constraints (Strix Halo)

- **fp16 + GradScaler**, NOT bf16 (24% slower, compile crashes)
- **torch.compile model only**, never the optimizer
- **autokernel before checkpoint load** (fused QKV keys must exist before `load_state_dict()`)
- **Unified memory** — `pin_memory=False`, gloo backend for DDP
- **Inference: 101 tok/s eager** (no KV cache). torch.compile hurts inference (recompiles per length).
- **Training throughput:** ~16.8K tok/s single GPU, ~35K tok/s DDP. SFT throughput will be similar since model architecture is unchanged.

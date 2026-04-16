---
title: "SFT Pipeline: EOS Warm-Up + Staged Instruction Tuning"
domain: training
type: reference
status: active
tags: [sft, instruction-tuning, chatml, eos, argus-prime, decoding]
related:
  - argus_prime_results.md
  - ../architectures/looped_model_design_lessons.md
  - ddp_setup_guide.md
---

# SFT Pipeline

Staged instruction fine-tuning pipeline for ARGUS-PRIME (168M params, pre-trained on Dolma 10B CC, loss 3.81).

## Architecture

```
Phase 0: EOS Warm-Up + Context Extension (1024 → 2048 seq_len)
    ↓
Stage C: Basic SFT (Alpaca 52K — teach prompt/response + clean stops)
    ↓
Stage A: Broad SFT (OpenHermes 2.5 1M — general chat)
    ↓
Stage B: Domain SFT (TBD — code, STEM, tool calls)
```

Each phase resumes from the previous checkpoint.

## Chat Template: ChatML

```
<|im_start|>system
You are a helpful assistant.<|im_end|>
<|im_start|>user
What is photosynthesis?<|im_end|>
<|im_start|>assistant
Photosynthesis is the process...<|im_end|>
```

### Special Tokens

| Token | ID | Purpose |
|-------|-----|---------|
| `<\|im_start\|>` | 50257 | Turn start |
| `<\|im_end\|>` | 50258 | Turn end (model must learn to emit) |
| `<\|pad\|>` | 50259 | Padding |
| `<tool_call>` | 50260 | Stage B only |
| `</tool_call>` | 50261 | Stage B only |

Vocab resized from 50257 → 50260 (or 50262 for Stage B). New rows zero-initialized. Weight tying preserved.

## Loss Masking

Only assistant response tokens contribute to loss. System/user tokens set to -100 (PyTorch ignore_index). The `<|im_end|>` token is NOT masked — the model must learn to predict it.

## EOS Warm-Up (Phase 0)

- **Purpose:** Sharpen EOS prediction + adapt to 2048 context length
- **Method:** `WeightedCrossEntropyLoss` with 5x weight on EOS token (50256)
- **LR:** 5e-5 (1/10th pre-training), cosine schedule
- **Dataset:** Dolma CC (same pre-training data, clean document boundaries)
- **Batch:** 8 × 8 accum = 64 effective, seq_len=2048

## Hyperparameters

| | Phase 0 | Stage C | Stage A | Stage B |
|--|---------|---------|---------|---------|
| **Dataset** | Dolma CC | Alpaca 52K | OpenHermes 1M | TBD |
| **Epochs** | 1 | 3 | 2 | 3-5 |
| **LR** | 5e-5 | 2e-5 | 2e-5 | 1e-5 |
| **Warmup** | 100 | 200 | 500 | 100 |
| **Batch** | 8×8 | 8×4 | 8×4 | 8×4 |
| **Seq len** | 2048 | 2048 | 2048 | 2048 |
| **Optimizer** | AdamW | AdamW | AdamW | AdamW |

AdamW (not Muon) for SFT — Muon's Newton-Schulz is too aggressive for small LR fine-tuning.

## Best Decoding Parameters

| Param | Value |
|-------|-------|
| temperature | 0.8-0.9 |
| top_k | 50 |
| top_p | 0.92 |
| repetition_penalty | 1.3 |
| frequency_penalty | 0.5 |

Post-SFT: repetition penalty can likely be reduced to 1.1-1.2.

## Implementation

### New Files

| File | Purpose |
|------|---------|
| `halo_training/chat_template.py` | ChatMLTokenizer (tiktoken wrapper), resize_embeddings(), constants |
| `halo_training/sft_data.py` | SFTDataset with alpaca/sharegpt/chatml adapters, loss masking, packing |
| `halo_training/sft_loss.py` | WeightedCrossEntropyLoss, build_sft_loss_fn() |

### CLI

```bash
# Phase 0: EOS warm-up + context extension
python -m halo_training --model models/argus_prime.py --class-name ArgusPrime \
    --resume-from checkpoints/argus_prime_cc_ddp_v2/step_72076.pt \
    --phase eos-warmup --eos-weight 5.0 --lr 5e-5 --block-size 2048 \
    --batch-size 8 --accum-steps 8 --dataset datasets/dolma_cc_clean

# Stage C: Alpaca SFT
python -m halo_training --model models/argus_prime.py --class-name ArgusPrime \
    --resume-from checkpoints/argus_prime_eos/best.pt \
    --phase sft --sft-dataset alpaca --lr 2e-5 --epochs 3 --block-size 2048

# Chat inference
python scripts/generate_text.py --model models/argus_prime.py --class-name ArgusPrime \
    --checkpoint checkpoints/argus_prime_sft_alpaca/best.pt \
    --chat --prompt "What is photosynthesis?"
```

### Critical Ordering

Embedding resize must happen AFTER checkpoint load but BEFORE autokernel:

1. `model.to(device)`
2. `resume_from` checkpoint load (shapes match at vocab=50257)
3. `resize_embeddings(model, 50260)` — copies old weights, zero-inits new rows, reties output
4. `autokernel.optimize()`

This is handled by the `resize_vocab` parameter in `train()`.

## Inference Performance (Pre-SFT Baseline)

| Mode | tok/s | Latency/token |
|------|-------|---------------|
| Eager (no compile) | 101 | 9.9 ms |
| Compiled | 3.0 | 336 ms |

Compiled inference is slow due to recompilation per sequence length. KV-cache needed for compiled inference.

## Phase 0 Training Results (Machine B)

| Metric | Value |
|--------|-------|
| Throughput | 15.9K tok/s |
| MFU | 27.0% |
| Memory | 18.4 GB |
| Loss (start) | 43.7 (with 5x EOS weight) |
| Loss (step 320) | 32.7 |
| Grad norm | 6.5 → 0.8 (stable) |

## Evaluation Criteria

| Stage | Metric | Target |
|-------|--------|--------|
| Phase 0 | EOS predicted within 5 tokens of boundary | >80% |
| Stage C | Generations end with `<\|im_end\|>` | >95% |
| Stage A | Val loss on held-out OpenHermes | < 2.5 |
| Stage B | Tool call JSON exact match | TBD |

## Design Spec

Full spec: `docs/superpowers/specs/2026-04-16-sft-pipeline-design.md`

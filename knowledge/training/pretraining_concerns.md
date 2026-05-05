---
title: "Pretraining Concerns"
domain: architectures
type: reference
status: active
related:
  - mad_llm_scientist/EVALUATION_GUIDE.md
tags: [%pretraining, %token-budget, %stability, %convergence]
---

The Cold, Hard Math First
Let's just run the numbers before I give you my opinion.
The Arithmetic
One full epoch:
$$\frac{16{,}000{,}000 \text{ tokens}}{10{,}000 \text{ tokens/sec}} = 1{,}600 \text{ sec} = 26 \text{ min } 40 \text{ sec}$$
Tokens you can actually see in 15 minutes:
$$15 \times 60 \times 10{,}000 = 9{,}000{,}000 \text{ tokens} = 9\text{M tokens}$$
That's 56.25% of your dataset. You can't even finish one epoch.
$$\frac{9\text{M}}{16\text{M}} = 0.5625 \approx 56%$$

My Opinion: Three Separate Problems
Problem 1: You're not even seeing your full dataset once
This is the most immediate issue. You're leaving 7M tokens on the floor. The model will never learn from 44% of your data. Whatever distribution your dataset has — the model gets a skewed, incomplete view of it.
If your data loader isn't shuffling well, you might be seeing all of topic A and none of topic B. Even with good shuffling, statistical coverage is poor.
Problem 2: 16M tokens is already absurdly small for pretraining
Let's put 16M in perspective. GPT-2 (124M parameters) was trained on ~10B tokens. The Chinchilla-optimal ratio is roughly 20 tokens per parameter:
$$N_{\text{optimal}} \approx 20 \times 124\text{M} = 2.48\text{B tokens}$$
Your dataset is:
$$\frac{16\text{M}}{2{,}480\text{M}} = 0.0065 = 0.65%$$
You have less than 1% of what Chinchilla scaling laws say you need for compute-optimal training. Even for a single epoch scenario, this model is going to be massively undertrained.
And you're not even completing that single epoch.
Problem 3: 10K tokens/sec feels low for this hardware
A GPT-2 124M param model on a Strix Halo should be doing better than this. Let me sanity-check:
Model memory footprint (training, FP16 mixed precision):

Parameters: ~248 MB (FP16)
Optimizer states (AdamW): ~992 MB (FP32 params + momentum + variance)
Gradients: ~248 MB (FP16)
Activations: depends on sequence length and batch size, but manageable
Total: ~1.5-3 GB — fits trivially in your 128 GB (~116 GB GPU-visible) unified memory

Arithmetic intensity consideration: The 8060S has no MFMA/matrix cores, so all matmuls go through VALU (vector ALU). That hurts. But 40 CUs × wave32 still gives you meaningful throughput for a 124M model. 10K tok/s suggests you might have bottlenecks in:

Batch size too small → not saturating the CUs
No kernel fusion → excessive memory round-trips
Framework overhead → Python-level or launch latency dominating
Not using FP16 properly → doing FP32 when you shouldn't be


What I'd Actually Do
Option A: Fix the time budget (easiest)
Just... train longer. You need at minimum 1 full epoch:
$$t_{\text{1 epoch}} \approx 27 \text{ minutes}$$
And honestly, with only 16M tokens, you want multiple epochs. The dataset is so small that multi-epoch training is not just acceptable — it's necessary. For a dataset this small, 4-8 epochs is reasonable before overfitting becomes catastrophic:
$$t_{\text{4 epochs}} \approx 107 \text{ minutes} \approx 1.8 \text{ hours}$$
That's still very fast. This isn't a big ask on Strix Halo.
Option B: Increase throughput (better long-term)
Target: 30-50K tokens/sec, which would let you do 3+ epochs in 15 minutes.



Optimization
Expected Gain
Effort



Increase batch size (maximize CU occupancy)
1.5-2x
Low


torch.compile() with mode="reduce-overhead"
1.3-1.8x
Low


Full FP16 / BF16 (if not already)
1.5-2x
Low


Gradient accumulation + larger effective batch
1.2x
Low


Flash-style fused attention (even without matrix cores, fusion helps)
1.3x
Medium


Custom ROCm/HIP kernels for your architecture
1.5-2x
High


Quick wins to try immediately:
import torch

# 1. Make sure you're using mixed precision
scaler = torch.amp.GradScaler("cuda")
with torch.amp.autocast("cuda", dtype=torch.float16):
    outputs = model(input_ids, labels=labels)
    loss = outputs.loss

scaler.scale(loss).backward()
scaler.step(optimizer)
scaler.update()

# 2. Compile the model (PyTorch 2.x)
model = torch.compile(model, mode="reduce-overhead")

# 3. Increase batch size until you OOM, then back off one step
# With 128GB unified memory (~116GB GPU-visible) and a 124M model, you can go BIG
# Try batch_size=64 or even 128 with seq_len=1024

Option C: Reconsider the dataset size (the real question)
16M tokens for pretraining from scratch is... a toy experiment. I don't say that dismissively — toy experiments are how you validate architectures and pipelines. But you need to be honest with yourself about what you'll get:



Dataset Size
What You Get



16M tokens, <1 epoch
A model that generates semi-coherent babble


16M tokens, 4-8 epochs
A model that memorizes your dataset and maybe generalizes locally


200M+ tokens
A model that starts showing real language understanding


1B+ tokens
A GPT-2 that's actually somewhat useful


If this is architecture validation — fine, 16M tokens at 15 minutes is a reasonable smoke test. Just know what you're measuring.
If you want a useful model — you need at least 100-500x more data, and your 15-minute constraint needs to become a multi-hour or multi-day constraint.

The Bottom Line



Aspect
Verdict



15 min budget for 16M tokens
❌ Can't even finish 1 epoch


16M tokens for GPT-2 pretraining
⚠️ Fine for debugging, useless for a real model


10K tok/s on Strix Halo
⚠️ Likely leaving 2-3x performance on the table


My honest take: If this is a pipeline/architecture test — extend to 30 minutes, do 1 full epoch, verify loss decreases, sanity-check generations, and move on. If you're trying to build something useful — find more data first, optimize throughput second, and throw the 15-minute constraint away. On your hardware, a 124M model on 1B tokens at 30K tok/s would take about 9 hours. That's a weekend run. That's where the real learning starts.

---

## Update (2026-04-09): Verified Training Baselines

The `halo_training/` package is now operational. Actual measured throughput on Strix Halo:

| Architecture | Params | Config | tok/s | MFU | Memory |
|-------------|--------|--------|-------|-----|--------|
| LlamaModel (transformer) | 124.7M | eager, no autokernel | 14,500 | 17% | ~17 GB |
| LlamaModel (transformer) | 124.7M | compile + autokernel | **43,000** | **54%** | ~17 GB |
| AMADEUS (SSM hybrid) | 243.8M | eager, chunked scan | 6,400 | 16% | 12.7 GB |

**Token budget recalculation with actual throughput:**

| Throughput | 15 min | 45 min | 120 min |
|-----------|--------|--------|---------|
| 6.4K tok/s (SSM eager) | 5.76M | 17.3M (1.1 epochs) | 46.1M (2.9 epochs) |
| 14.5K tok/s (transformer eager) | 13.1M | 39.2M (2.5 epochs) | 104.4M (6.5 epochs) |
| 43K tok/s (transformer + autokernel) | 38.7M | 116.1M (7.3 epochs) | 309.6M (19.4 epochs) |

**Key finding for SSM architectures:** Sequential scan and `torch.associative_scan` are equally slow on gfx1151 (~1.3K tok/s). **Chunked linear recurrence** (chunk_size=64) gives 5x speedup to 6.4K tok/s. Use `models/amadeus.py:selective_scan_chunked` as the reference implementation. A fused HIP kernel is expected to push this to ~9.5K tok/s.

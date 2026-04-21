# Part 14: Putting It All Together

## Goal
Connect every piece you have built into a single end-to-end pipeline: data preparation, training with optimized kernels, evaluation, SFT, DPO, and inference. Run the complete workflow and verify it works.

## Why This Matters
Each part of this series taught one skill in isolation. Real projects require all of them to work together. This part is the integration test -- the proof that you can build an LLM from scratch on consumer hardware.

---

## 14.1 The Full Pipeline

Here is every step, in order, from raw data to deployed model:

```
1. DATA         Prepare training corpus (CLIMB mixture, quality filtered)
2. TRAIN        Pretrain base model (autokernel + compile, 3x speedup)
3. EVAL         Benchmark on HellaSwag, ARC, MMLU (lm-evaluation-harness)
4. SFT          Fine-tune on instruction data (ChatML, loss masking)
5. DPO          Align with preference pairs (chosen vs rejected)
6. INFERENCE    Serve with KV-cache, INT8, streaming API
```

### How Each Part Connects

```
Part 08 (Data)
    |
    v
Part 02-07 (Training + Kernels)  ----  Part 11 (Architecture)
    |
    v
Part 09 (Evaluation)
    |
    v
Part 12 (SFT + DPO)
    |
    v
Part 13 (Inference + Deployment)
```

Each arrow is a checkpoint file. The output of training is a `.pt` file that SFT loads. SFT produces another `.pt` file that DPO loads. DPO produces the final `.pt` file that the inference server loads.

### The 10-Minute Smoke Test

Before any long training run, verify the entire pipeline in 10 minutes:

```bash
# Step 1: Prepare a tiny dataset (30 seconds)
python -c "
import tiktoken
enc = tiktoken.get_encoding('gpt2')
with open('data/smoke_test.txt', 'w') as f:
    for i in range(1000):
        f.write(f'The quick brown fox jumps over the lazy dog. Sentence {i}.\n')
print('Smoke dataset ready.')
"

# Step 2: Train for 200 steps (2 minutes)
python -m halo_training \
    --model models/chimera.py --class-name ChimeraMini \
    --dataset data/smoke_test.txt \
    --max-steps 200 --batch-size 8 --block-size 256

# Step 3: Check loss decreased
python -c "
import json
with open('checkpoints/chimera_halo_mini/train_log.jsonl') as f:
    lines = [json.loads(l) for l in f if l.strip()]
first, last = lines[0]['loss'], lines[-1]['loss']
print(f'Loss: {first:.3f} -> {last:.3f} (decreased={last < first})')
assert last < first, 'Loss did not decrease!'
"

# Step 4: Generate text (30 seconds)
python -c "
import torch
from models.chimera import ChimeraMini
model = ChimeraMini()
model.load_state_dict(torch.load('checkpoints/chimera_halo_mini/final.pt')['model_state_dict'])
model.eval().cuda()
ids = torch.tensor([[464, 2068, 7586]], device='cuda')  # 'The quick brown'
with torch.no_grad():
    logits = model(ids)
    pred = logits[0, -1].argmax().item()
print(f'Predicted next token ID: {pred}')
print('Smoke test PASSED.')
"
```

If this fails, do not proceed to full training. Debug here where iteration is fast.

---

## 14.2 Scaling Up

### From Smoke Test to BabyLM (10M tokens)

```bash
# Download BabyLM strict-small (10M tokens of child-directed speech)
python scripts/download_babylm.py

# Train your architecture for real (45 minutes)
python -m halo_training \
    --model models/chimera.py --class-name Chimera \
    --dataset babylm \
    --compile --optimize-kernels --muon \
    --time-budget-minutes 45 \
    --batch-size 64 --block-size 1024 --accum-steps 4

# Monitor (do NOT watch SSH stdout -- it will timeout)
# Instead, tail the log:
tail -f checkpoints/chimera/train_log.jsonl | python -c "
import sys, json
for line in sys.stdin:
    d = json.loads(line)
    print(f\"step={d['step']:>5} loss={d['loss']:.3f} tok/s={d.get('tok_s', 0):.0f}\")
"
```

### From BabyLM to Dolma (10B tokens)

```bash
# Download Dolma v1.7 sample (or full dataset)
python scripts/download_dolma.py --subset 10B

# Train for 24 hours (use resume for restarts)
python -m halo_training \
    --model models/chimera.py --class-name Chimera \
    --dataset datasets/dolma-10b-train \
    --compile --optimize-kernels --muon \
    --time-budget-minutes 1440 \
    --checkpoint-interval 1000 \
    --batch-size 64 --block-size 1024 --accum-steps 8

# If the process dies (OOM, SSH timeout), resume:
python -m halo_training \
    --model models/chimera.py --class-name Chimera \
    --dataset datasets/dolma-10b-train \
    --resume-from checkpoints/chimera/step_12000.pt \
    --compile --optimize-kernels --muon \
    --time-budget-minutes 1440
```

### Monitoring Long Runs

Never rely on SSH stdout for long training runs. SSH sessions timeout and you lose all output. Instead:

```bash
# Check if training is still running
ps aux | grep python3 | grep halo_training

# Tail the training log
tail -5 checkpoints/chimera/train_log.jsonl

# Plot loss curve
python -c "
import json, matplotlib.pyplot as plt
with open('checkpoints/chimera/train_log.jsonl') as f:
    data = [json.loads(l) for l in f if l.strip()]
steps = [d['step'] for d in data]
losses = [d['loss'] for d in data]
plt.plot(steps, losses)
plt.xlabel('Step'); plt.ylabel('Loss')
plt.title('Training Loss')
plt.savefig('loss_curve.png')
print(f'Saved loss_curve.png ({len(data)} points)')
"
```

### The Data Scaling Ladder

| Stage | Dataset | Tokens | Time | Expected Loss |
|-------|---------|--------|------|---------------|
| Smoke | Random sentences | 100K | 2 min | 8.0 -> 4.0 |
| BabyLM | Child-directed speech | 10M | 45 min | 4.5 -> 3.5 |
| GPT-training-small | Web text | 100M | 4 hours | 3.8 -> 3.0 |
| Dolma sample | Web + books + code | 10B | 24 hours | 3.5 -> 2.8 |
| Full Dolma | Everything | 100B | 10 days | 3.0 -> 2.4 |

Each stage uses roughly 10x more data and 5-10x more time. The first jump (smoke to BabyLM) has the most dramatic loss improvement.

---

## 14.3 Experiment Checklist

### Pre-Flight

Before starting any multi-hour run:

```bash
# 1. Commit your config
git add models/chimera.py
git commit -m "chimera: freeze config for dolma-10b run"

# 2. Smoke test passes
python -m halo_training --model models/chimera.py --class-name ChimeraMini --smoke

# 3. Check disk space (checkpoints are 200-700 MB each)
df -h /home/  # need at least 20 GB free

# 4. Check GPU memory (no other processes)
nvidia-smi  # should show 0 MB used

# 5. Check dataset exists and is readable
ls -la datasets/dolma-10b-train/
wc -l datasets/dolma-10b-train/*.txt  # or *.jsonl
```

### During Training

Monitor every 1-2 hours:

```bash
# Loss should be decreasing (or at least not increasing)
tail -1 checkpoints/chimera/train_log.jsonl

# GPU utilization should be > 80%
nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader

# Memory should be stable (not creeping up)
nvidia-smi --query-gpu=memory.used --format=csv,noheader

# Process should still be alive
ps aux | grep python3 | grep -v grep
```

Red flags that require action:
- Loss is NaN -> model has diverged, restart from last checkpoint with lower LR
- Loss suddenly spikes -> bad data batch or learning rate issue
- GPU utilization < 50% -> data loading bottleneck, increase num_workers
- Memory creeping up -> memory leak, possibly from logging tensors

### Post-Training

```bash
# 1. Run evaluation
python -c "
from halo_training.evaluate import evaluate_bpb
from models.chimera import Chimera
import torch

model = Chimera()
ckpt = torch.load('checkpoints/chimera/final.pt')
model.load_state_dict(ckpt['model_state_dict'])
model.eval().cuda()

results = evaluate_bpb(model)
print(f\"Val BPB: {results['bpb']:.3f}\")
"

# 2. Run lm-evaluation-harness benchmarks
lm_eval --model hf --model_args pretrained=checkpoints/chimera/ \
    --tasks hellaswag,arc_easy,mmlu \
    --batch_size 8

# 3. Save results
echo "chimera, $(date), loss=$(tail -1 checkpoints/chimera/train_log.jsonl | jq .loss)" \
    >> results/experiment_log.csv

# 4. Generate sample text for qualitative check
python generate_samples.py --checkpoint checkpoints/chimera/final.pt \
    --prompts "Once upon a time" "The fundamental theorem of" "def fibonacci("
```

---

## 14.4 Baselines to Beat

### Published Small Model Benchmarks

| Model | Params | Tokens | HellaSwag | ARC-Easy | MMLU |
|-------|--------|--------|-----------|----------|------|
| SmolLM2-135M | 135M | 2T | 42.1 | 43.9 | 25.8 |
| Nandi-Mini-150M | 150M | 500B | 25.63 avg | - | - |
| GPT-2 (124M) | 124M | ~40B | 31.6 | 43.7 | 25.9 |
| OLMo-1B | 1B | 3T | 62.5 | 55.3 | 26.2 |

### What These Numbers Mean

**HellaSwag:** Multiple-choice sentence completion. Tests language understanding and common sense. Random baseline is 25%. GPT-2 124M gets 31.6% with only 40B tokens of training.

**ARC-Easy:** Science questions from elementary school exams. Tests factual knowledge. Random baseline is 25%.

**MMLU:** Massive Multitask Language Understanding. 57 subjects from STEM to humanities. For models under 1B parameters, scores cluster around 25-26% (near random) because the knowledge simply does not fit.

### Your Target

With a 150M model on 10B tokens (Dolma), you should aim for:

| Metric | Minimum | Good | Great |
|--------|---------|------|-------|
| HellaSwag | 30 | 35 | 40+ |
| ARC-Easy | 35 | 40 | 45+ |
| Val BPB | < 1.2 | < 1.0 | < 0.9 |
| Tok/s (training) | 10K | 25K | 40K+ |

The key advantage of a novel architecture (Parcae looping, factorized embeddings) is parameter efficiency -- you get more effective parameters for the same unique parameter budget, which should close the gap with SmolLM2's 2T-token advantage.

---

## 14.5 Common Failure Modes and Fixes

Over the course of building this system, you will encounter recurring problems. Here is a reference table of the most common failures and how to fix them.

### Training Failures

| Symptom | Cause | Fix |
|---------|-------|-----|
| Loss is NaN at step 1 | Learning rate too high | Reduce LR by 10x, check for inf in embeddings |
| Loss is NaN after 100 steps | Multi-step TTT in Parcae loop | Switch to single-step TTT or no TTT |
| Loss spikes then recovers | Bad data batch | Usually benign; if persistent, check data for encoding errors |
| Loss plateaus early (> 4.0) | Model too small or data too repetitive | Increase d_model or switch to more diverse data |
| Loss oscillates wildly | LR too high or batch too small | Reduce LR, increase effective batch via accum_steps |
| OOM crash | Model + optimizer + activations > 16GB | Reduce batch_size, enable gradient checkpointing |
| Very slow training | Data loading bottleneck | Increase num_workers, preprocess data to binary format |
| torch.compile crash | Python loop in model forward | Use compile_zones() instead of compiling full model |
| bf16 crash with RoPE | torch.compile + bf16 + complex numbers | Always use fp16 + GradScaler, never bf16 |

### SFT Failures

| Symptom | Cause | Fix |
|---------|-------|-----|
| Model outputs gibberish after SFT | Loss mask is wrong (training on all tokens) | Verify IGNORE_INDEX is -100, check mask counts |
| Model never stops generating | Did not train on im_end token | Ensure footer tokens are in assistant mask |
| Model repeats user's question | Too few epochs or LR too low | Train for 2-3 epochs at 2e-5 |
| Model forgets general knowledge | Catastrophic forgetting | Lower LR to 1e-5, mix in general data |
| Embedding resize causes NaN | Zero-initialized new embeddings | Use mean-init with small noise, not zeros |

### DPO Failures

| Symptom | Cause | Fix |
|---------|-------|-----|
| Accuracy stays at 0.5 | LR too low or beta too high | Increase LR to 5e-6, reduce beta to 0.05 |
| Accuracy > 0.95 | Overfitting to preference pairs | Add more diverse pairs, reduce epochs |
| Model becomes repetitive | KL constraint too loose (beta too low) | Increase beta to 0.2 |
| All rewards negative | Reference model mismatch | Ensure ref model is the exact SFT checkpoint |

### Inference Failures

| Symptom | Cause | Fix |
|---------|-------|-----|
| KV-cache gives different output than naive | RoPE position offset wrong | Verify position starts at cache.position, not 0 |
| Generation is very slow | No KV-cache | Implement KV-cache from Section 13.2 |
| INT8 model outputs garbage | Quantization on critical layers | Exclude embedding and head from quantization |
| Streaming server hangs | No async yield | Add await asyncio.sleep(0) in generation loop |

---

## 14.6 Experiment Tracking Best Practices

### The Experiment Log

Keep a CSV or JSONL file that records every training run:

```python
"""log_experiment.py -- Record experiment results."""
import json
import datetime
from pathlib import Path


def log_experiment(
    name: str,
    model_class: str,
    dataset: str,
    config: dict,
    results: dict,
    notes: str = "",
    log_path: str = "results/experiments.jsonl",
):
    """Append experiment results to the log."""
    entry = {
        "timestamp": datetime.datetime.now().isoformat(),
        "name": name,
        "model_class": model_class,
        "dataset": dataset,
        "config": config,
        "results": results,
        "notes": notes,
    }

    Path(log_path).parent.mkdir(parents=True, exist_ok=True)
    with open(log_path, "a") as f:
        f.write(json.dumps(entry) + "\n")

    print(f"Logged: {name} -> {results}")


# Example usage after training
log_experiment(
    name="chimera_babylm_v3",
    model_class="Chimera",
    dataset="babylm",
    config={
        "d_model": 768, "n_shared_layers": 8,
        "mean_recurrence": 3, "lr": 8e-4,
        "batch_size": 64, "accum_steps": 4,
    },
    results={
        "final_loss": 3.55, "hellaswag": 33.1,
        "arc_easy": 35.2, "tok_s": 37000,
        "total_steps": 5000, "wall_time_min": 45,
    },
    notes="First run with depth cache enabled. XSA +0.03 loss improvement.",
)
```

### Comparing Runs

```python
"""compare_experiments.py -- Compare results across runs."""
import json


def load_experiments(log_path="results/experiments.jsonl"):
    experiments = []
    with open(log_path) as f:
        for line in f:
            if line.strip():
                experiments.append(json.loads(line))
    return experiments


def compare(experiments, metric="final_loss"):
    """Print experiments sorted by a metric."""
    valid = [e for e in experiments if metric in e.get("results", {})]
    valid.sort(key=lambda e: e["results"][metric])

    print(f"{'Name':<30} {'Model':<20} {metric:<15} {'Notes'}")
    print("-" * 85)
    for e in valid:
        name = e["name"][:30]
        model = e["model_class"][:20]
        value = e["results"][metric]
        notes = e.get("notes", "")[:40]
        print(f"{name:<30} {model:<20} {value:<15.4f} {notes}")


# Usage
experiments = load_experiments()
compare(experiments, metric="final_loss")
compare(experiments, metric="hellaswag")
```

### When to Stop an Experiment

Clear stopping criteria prevent wasting compute on dead ends:

1. **Loss not decreasing after 500 steps:** Something is fundamentally wrong. Stop and debug.
2. **Loss curve matches baseline exactly:** Your novel component is not helping. Stop, remove it, try something else.
3. **Loss diverges (NaN or > 10):** Model is broken. Stop immediately.
4. **Quality metrics worse than Bare variant:** Your additions are hurting. Remove them.
5. **Time budget exceeded:** Save checkpoint and evaluate what you have.

---

## 14.7 What's Next

### Multi-GPU Training

Your 150M model fits on a single RTX 4060 Ti. Larger models require distributing across GPUs.

**DDP (Distributed Data Parallel):** Replicate the model on each GPU, split the data. Each GPU computes gradients independently, then they all-reduce. Best for models that fit in one GPU's memory.

```python
# DDP setup (2+ GPUs)
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

dist.init_process_group(backend="nccl")
model = model.to(local_rank)
model = DDP(model, device_ids=[local_rank])
```

**FSDP (Fully Sharded Data Parallel):** Shard model parameters across GPUs. Each GPU holds only 1/N of the model. Enables training models that do not fit on a single GPU.

```python
# FSDP setup (for models that don't fit on 1 GPU)
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP

model = FSDP(model, auto_wrap_policy=transformer_auto_wrap_policy)
```

### Longer Context

Your current model uses 1024-token context. Extending to 4K or 8K requires:

1. **RoPE scaling:** Apply NTK-aware scaling to extend RoPE beyond training length.
2. **Sliding window attention:** Only attend to the most recent W tokens, reducing memory from O(T^2) to O(T * W).
3. **More data:** Longer context models need more training data to learn long-range dependencies.

```python
# RoPE scaling for 4x context extension
def precompute_freqs_cis_scaled(dim, max_seq_len, base=10000.0, scale=4.0):
    # NTK-aware interpolation
    base = base * (scale ** (dim / (dim - 2)))
    freqs = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
    t = torch.arange(max_seq_len, dtype=torch.float32)
    freqs = torch.outer(t, freqs)
    return torch.polar(torch.ones_like(freqs), freqs)
```

### Multimodal (Vision + Language)

Add a vision encoder (ViT) that projects image patches into the same embedding space as text tokens:

```
Image -> ViT encoder -> project to d_model -> prepend to text tokens -> LLM
```

The LLM sees image tokens as just another input modality. Fine-tune the projection layer and optionally the LLM on image-text pairs.

### Mixture of Experts (MoE)

Replace dense FFN layers with sparse MoE layers. Each token is routed to K experts (out of N total), so the model has N * params_per_expert total parameters but only uses K * params_per_expert per token.

```python
class MoELayer(nn.Module):
    def __init__(self, d_model, ffn_inner, n_experts=8, top_k=2):
        super().__init__()
        self.router = nn.Linear(d_model, n_experts)
        self.experts = nn.ModuleList([
            SwiGLU(d_model, ffn_inner) for _ in range(n_experts)
        ])
        self.top_k = top_k

    def forward(self, x):
        # Route each token to top-k experts
        router_logits = self.router(x)
        weights, indices = torch.topk(router_logits, self.top_k, dim=-1)
        weights = F.softmax(weights, dim=-1)

        # Compute expert outputs (simplified; production uses scatter/gather)
        output = torch.zeros_like(x)
        for k in range(self.top_k):
            expert_idx = indices[:, :, k]  # (B, T)
            expert_weight = weights[:, :, k:k+1]  # (B, T, 1)
            for e in range(len(self.experts)):
                mask = expert_idx == e
                if mask.any():
                    output[mask] += (expert_weight[mask] * self.experts[e](x[mask]))
        return output
```

MoE lets you scale to billions of parameters while keeping per-token compute constant. This is how Mixtral 8x7B achieves 47B total params but only activates 13B per token.

### Contributing to Open Source

You now have the skills to contribute to:
- **PyTorch:** Custom CUDA kernels, quantization backends
- **vLLM:** Inference optimizations for new architectures
- **lm-evaluation-harness:** New evaluation tasks
- **HuggingFace Transformers:** Model implementations
- **CUDA/cuDNN:** Performance analysis and bug reports

---

## 14.8 Resources

### Video Courses

- **Karpathy's "Zero to Hero"** (YouTube): The best introduction to neural networks and language models. Watch the "Let's build GPT" video for a clean implementation of the transformer.
- **3Blue1Brown "Essence of Linear Algebra"** (YouTube): Visual intuition for the math behind everything. Essential for understanding attention as matrix operations.
- **MIT 6.S898 "Deep Learning"**: More rigorous mathematical foundations.

### Papers We Referenced

| Paper | Key Contribution | Section |
|-------|-----------------|---------|
| Vaswani et al., 2017 | Transformer architecture | Part 10 |
| Touvron et al., 2023 | LLaMA: efficient pretraining | Part 02 |
| Gu & Dao, 2023 | Mamba: selective state spaces | Part 10, 11 |
| Su et al., 2021 | RoPE: rotary position embeddings | Part 10, 11 |
| Shazeer, 2020 | SwiGLU activation | Part 11 |
| LFM2 (Liquid AI) | 75:25 conv:attention hybrid | Part 11 |
| Parcae (ETH Zurich) | Stable looping, scaling laws | Part 11 |
| XSA (Zhai, 2026) | Exclusive self-attention | Part 11 |
| Nandi (Databricks) | Factorized embeddings | Part 11 |
| CLIMB (Wettig, 2024) | Data mixture optimization | Part 08 |
| Rafailov et al., 2023 | DPO: preference alignment | Part 12 |
| Self-Improving Pretraining | Curate-then-train loop | Part 08 |

### Communities

- **r/LocalLLaMA** (Reddit): The best community for running and training LLMs on consumer hardware.
- **EleutherAI Discord**: Research-focused community. Hosts lm-evaluation-harness.
- **HuggingFace Forums**: Help with Transformers, datasets, and model deployment.
- **NVIDIA Developer Forums**: CUDA programming questions and optimization tips.

### Tools

| Tool | Purpose | Part |
|------|---------|------|
| PyTorch | Training framework | All |
| tiktoken | Tokenization | Part 02, 12 |
| torch.compile | Graph optimization | Part 07 |
| lm-evaluation-harness | Standardized benchmarks | Part 09 |
| FastAPI | Inference serving | Part 13 |
| auto-gptq | INT4 quantization | Part 13 |
| nsight systems | GPU profiling | Part 03 |
| matplotlib | Loss curves, analysis | Part 09 |

---

## 14.9 Closing

### What You Built

Over these 14 parts, you built every component of an LLM system from scratch:

1. **A validated hardware profile** -- you know your GPU's bandwidth, compute, and memory limits.
2. **A complete training stack** -- data loading, mixed precision, gradient accumulation, learning rate scheduling.
3. **A profiling toolkit** -- you can find bottlenecks in any model.
4. **Custom CUDA kernels** -- fused operations that run 2-5x faster than PyTorch defaults.
5. **An autokernel library** -- pattern-matching kernel replacement that works on any model.
6. **torch.compile integration** -- custom ops registered with the compiler for maximum fusion.
7. **A data pipeline** -- CLIMB-style mixture optimization and quality filtering.
8. **An evaluation framework** -- standardized benchmarks for comparing models.
9. **Mathematical foundations** -- you can read papers about SSMs, attention, and control theory.
10. **A novel architecture** -- designed from research papers, with ablation discipline.
11. **Post-training** -- SFT with ChatML, DPO alignment, tool-use.
12. **Inference and deployment** -- KV-cache, quantization, streaming API.

### What It Means

You do not need a cluster of A100s or H100s to do meaningful LLM research. An RTX 4060 Ti with 16GB of VRAM can train a competitive 150M model in 24 hours. That model can beat published baselines if you make the right architecture and data choices.

The skills transfer directly to larger scale. The same CUDA kernel techniques, the same training loop structure, the same evaluation discipline -- everything works the same at 7B or 70B, just with more GPUs and more data.

The field moves fast. Papers that were state-of-the-art when this tutorial was written will be superseded. But the engineering skills -- profiling, kernel optimization, ablation discipline, end-to-end integration -- are permanent. They are how you turn a paper into a working system.

### The Full Command Sequence

For reference, here is the complete pipeline as shell commands:

```bash
# 0. Setup
python3 -m venv .venv && source .venv/bin/activate
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124
pip install tiktoken numpy matplotlib datasets fastapi uvicorn

# 1. Validate hardware
python benchmark_bandwidth.py    # expect > 250 GB/s
python benchmark_tensorcore.py   # expect > 100 TFLOPS

# 2. Smoke test
python -m halo_training --model models/chimera.py --class-name ChimeraMini --smoke

# 3. Pretrain on BabyLM (45 min)
python -m halo_training \
    --model models/chimera.py --class-name Chimera \
    --dataset babylm \
    --compile --optimize-kernels --muon \
    --time-budget-minutes 45

# 4. Evaluate
python eval_benchmarks.py --checkpoint checkpoints/chimera/final.pt

# 5. SFT (30 min)
python train_sft.py \
    --model-path checkpoints/chimera/final.pt \
    --data-path data/alpaca_cleaned.jsonl \
    --format alpaca --epochs 2 --lr 2e-5

# 6. DPO (30 min)
python generate_preferences.py --model checkpoints/sft/sft_final.pt --n-prompts 200
python train_dpo.py \
    --model checkpoints/sft/sft_final.pt \
    --data checkpoints/sft/preferences.jsonl \
    --epochs 2 --lr 5e-6 --beta 0.1

# 7. Quantize + Serve
python quantize_int8.py --checkpoint checkpoints/dpo/final.pt
uvicorn server:app --host 0.0.0.0 --port 8000

# 8. Test
curl -X POST http://localhost:8000/v1/chat/completions \
    -H "Content-Type: application/json" \
    -d '{"messages": [{"role": "user", "content": "Hello!"}]}'
```

That is the entire pipeline. From raw hardware to a deployed chat model, on a single consumer GPU.

---

## Final Checkpoint

- [ ] Complete pipeline runs end-to-end: data -> train -> eval -> SFT -> DPO -> serve
- [ ] Smoke test passes in under 60 seconds
- [ ] Base model achieves HellaSwag > 30 on BabyLM
- [ ] SFT model follows instructions and stops generating
- [ ] DPO model shows improved accuracy over SFT
- [ ] Inference server streams tokens at > 100 tok/s
- [ ] You can explain every step and why it matters

Congratulations. You built an LLM system from scratch.

---

**Previous: [Part 13 -- Inference & Deployment](13_inference_and_deployment.md)**
**Back to: [Tutorial Index](README.md)**

---
title: "Evaluation Guide for Continued Pre-Training"
domain: architectures
type: guide
status: active
related:
  - mad_llm_scientist/BPB_MFU_ANALYSIS.md
tags: [%evaluation, %benchmarks, %methodology]
---

# Evaluation Guide for Continued Pre-Training

## Purpose

Use this guide to determine whether continued pre-training is producing a model that is:

1. Learning the target domain
2. Retaining general capability
3. Remaining stable during training
4. Improving practical usefulness
5. Meeting Strix Halo inference constraints

A run is only acceptable if the logs support a clear tradeoff between adaptation and retention.

---

## 1. Evaluation Principles

Continued pre-training must be evaluated on four axes:

- **Adaptation**: does the model improve on the target domain?
- **Retention**: does the model preserve general capability?
- **Stability**: are loss and gradients healthy?
- **Utility**: is the model usable in practice on the target hardware?

Use both automated metrics and qualitative inspection. Do not rely on training loss alone.

---

## 2. During Training

### 2.1 Monitor Core Training Signals

Track these every step:

| Signal | Meaning | Action |
|---|---|---|
| Training loss decreases, validation loss decreases | Healthy learning | Continue |
| Training loss decreases, validation loss increases | Overfitting | Increase replay or reduce LR |
| Training loss decreases slowly, validation loss is flat | Weak adaptation | Tune LR or data mix |
| Training loss spikes and recovers | Temporary instability | Observe; intervene if repeated |
| Both losses plateau | Training signal exhausted | Stop or change setup |

### 2.2 Use Two Validation Sets

Always maintain both:

| Validation set | Purpose |
|---|---|
| Domain validation set | Measures adaptation to the new domain |
| General validation set | Measures forgetting of original capability |

Interpretation:

- Domain loss down, general loss stable: good
- Domain loss down, general loss up: forgetting
- Both flat: weak training signal
- Both down: ideal outcome

### 2.3 Track Perplexity

Measure perplexity on multiple datasets:

- New domain corpus
- General corpus
- Instruction-following samples

Expected behavior:

- Domain perplexity should decrease
- General perplexity should remain stable
- Instruction-following perplexity should not degrade

Perplexity is:

$$
\text{PPL} = e^{\mathcal{L}}
$$

where $\mathcal{L}$ is cross-entropy loss.

#### Reference implementation

```python
import math
import torch

def compute_perplexity(model, dataloader, device="cuda"):
    model.eval()
    total_loss = 0.0
    total_tokens = 0

    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = input_ids.clone()

            outputs = model(input_ids, attention_mask=attention_mask, labels=labels)

            active_tokens = attention_mask.sum().item()
            total_loss += outputs.loss.item() * active_tokens
            total_tokens += active_tokens

    avg_loss = total_loss / total_tokens
    return math.exp(avg_loss)
```

### 2.4 Monitor Gradient Norms

Track gradient norms every step or every few steps.

Look for:

- exploding gradients
- vanishing gradients
- divergence across layer types
- sudden spikes after batch changes

#### Reference implementation

```python
def log_gradient_norms(model):
    norms = {}
    for name, param in model.named_parameters():
        if param.grad is not None:
            norms[name] = param.grad.data.norm(2).item()
    return norms
```

If gradient behavior is unstable, treat it as a training issue, not a logging issue.

---

## 3. Online Probing During Training

Run a lightweight probe suite every `500–2000` steps.

### 3.1 Recommended Probe Tasks

| Probe task | Purpose | Failure condition |
|---|---|---|
| HellaSwag (0-shot) | Common sense and coherence | Drops more than 2% from base |
| ARC-Easy (0-shot) | Basic reasoning | Drops more than 3% from base |
| MMLU (5-shot, a few subjects) | Knowledge breadth | Any subject drops more than 5% |
| GSM8K (5-shot, small sample) | Math reasoning | Any drop is concerning |
| Domain-specific QA | Domain absorption | No improvement after early training |

Use `lm-evaluation-harness` with a small sample for speed.

### 3.2 Quick Probe Command

```bash
lm_eval --model hf \
    --model_args pretrained=./checkpoint-step-5000,dtype=float16 \
    --tasks hellaswag,arc_easy \
    --num_fewshot 0 \
    --batch_size 8 \
    --device cuda:0 \
    --limit 200
```

### 3.3 Strix Halo Guidance

On Strix Halo:

- unified memory removes PCIe transfer overhead
- the model can remain resident during evaluation
- batch size should be increased until memory pressure becomes uncomfortable

For a `1B–3B` FP16 model, batch sizes around `16–32` are usually practical.

### 3.4 Sanity Generation

At each checkpoint, generate outputs for a fixed prompt set.

Use this to detect:

- repetition loops
- coherence loss
- domain regression
- safety drift
- instruction-following drift

#### Example prompt set

```python
SANITY_PROMPTS = [
    "Explain photosynthesis in one paragraph.",
    "List 3 pros and 3 cons of nuclear energy.",
    "What is the mechanism of action of metformin?",
    "If a train leaves at 3pm going 60mph and another at 4pm going 90mph, when do they meet?",
    "Write a haiku about machine learning.",
]
```

#### Reference implementation

```python
import torch

def sanity_check(model, tokenizer, prompts, max_new_tokens=200):
    model.eval()
    results = {}

    for prompt in prompts:
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        with torch.no_grad():
            output = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=0.7,
                do_sample=True,
            )
        results[prompt] = tokenizer.decode(output[0], skip_special_tokens=True)

    return results
```

### 3.5 What to Watch For

| Observation | Meaning | Response |
|---|---|---|
| Coherence degrades | Possible forgetting | Inspect data and probes |
| Repetition loops appear | Instability or collapse | Reduce LR or stop |
| Domain answers improve | Adaptation is working | Continue |
| General answers degrade | Catastrophic forgetting | Increase replay |

---

## 4. Forgetting Metric

Compute forgetting for every probe task.

$$
\text{FG}_{\text{task}} = \frac{\text{Score}_{\text{base}} - \text{Score}_{\text{checkpoint}}}{\text{Score}_{\text{base}}} \times 100\%
$$

### 4.1 Thresholds

| FG value | Interpretation | Action |
|---|---|---|
| `FG <= 5%` | Usually acceptable | Continue |
| `FG > 5%` on a critical capability | Warning | Intervene |
| `FG > 10%` | Severe regression | Stop and correct course |

### 4.2 Remediation

If forgetting is too high:

- increase replay ratio of original data
- reduce learning rate
- improve data quality
- rebalance the data mixture
- consider regularization such as EWC

---

## 5. Learning Rate and Stability

Use a schedule with re-warmup followed by cosine decay.

### 5.1 Rule

If loss does not improve after warmup, the peak learning rate is likely:

- too high, if the model is diverging
- too low, if adaptation is weak

A typical CPT peak LR is about `1/10` to `1/5` of the original pre-training peak LR.

### 5.2 Schedule Sketch

```text
LR Schedule for CPT:
   ┌─ re-warm ─┐┌──────── cosine decay ────────┐
   │           ╱╲                               │
   │         ╱    ╲                              │
   │       ╱       ╲─────────────────╲           │
   │     ╱                             ╲         │
   │   ╱                                 ╲        │
   └──╱───────────────────────────────────╲──────┘
    step 0    warm_end              total_steps
```

---

## 6. After Continued Pre-Training

Run the full evaluation suite after training completes.

### 6.1 Full Benchmark Battery

```bash
lm_eval --model hf \
    --model_args pretrained=./final-model,dtype=float16 \
    --tasks mmlu,hellaswag,arc_challenge,arc_easy,winogrande,piqa,boolq,gsm8k,ifeval \
    --num_fewshot 0,0,0,0,0,0,0,5,0 \
    --batch_size 16 \
    --device cuda:0
```

### 6.2 Scorecard

| Category | Benchmarks | Minimum threshold for 1–3B | Notes |
|---|---|---|---|
| Knowledge | MMLU (5-shot), GPQA (0-shot) | MMLU > 45% | Lower values indicate weak general knowledge |
| Reasoning | ARC-C, HellaSwag, WinoGrande, PIQA | ARC-C > 35%, HellaSwag > 55% | Core intelligence signals |
| Math | GSM8K (0/5-shot), MATH-500 | GSM8K > 20% | Useful reasoning indicator |
| Instruction | IFEval, IFBench | IFEval > 30% | Usability gate |
| Domain | Custom eval set | Better than base | Main objective of CPT |

---

## 7. Forgetting Scorecard

Compare the final model against the base model on all general benchmarks.

$$
\Delta_{\text{task}} = \text{Score}_{\text{final}} - \text{Score}_{\text{base}}
$$

### 7.1 Acceptance Budget

| Condition | Status |
|---|---|
| `Δ > -2%` on any individual task | Acceptable |
| `Δ > -5%` aggregate | Acceptable |
| `Δ < -5%` on any task | Unacceptable |

If the model exceeds the forgetting budget, retrain with more replay data or a better data mix.

---

## 8. Practical Usefulness Tests

Benchmarks are necessary but not sufficient.

### 8.1 LLM-as-Judge

Use a stronger model to compare the candidate model against the base model on open-ended prompts.

Judge for:

- helpfulness
- accuracy
- relevance
- coherence

#### Judge prompt

```python
JUDGE_PROMPT = """You are evaluating two AI assistant responses.
Rate which is better on: helpfulness, accuracy, relevance, coherence.

User Question: {question}

Response A: {response_a}
Response B: {response_b}

Which is better? Answer "A", "B", or "Tie" with a brief explanation."""
```

Target:

- win rate above `55%` against the base model

### 8.2 Constrained Output Tests

Verify that the model can follow structure precisely.

| Test | Pass criteria |
|---|---|
| Respond in JSON with keys `name`, `age`, `city` | Valid JSON with correct keys |
| Answer in exactly 3 bullet points | Exactly 3 bullets |
| Translate to French: `Hello world` | Correct translation |
| Summarize in ≤50 words | Coherent output within limit |

### 8.3 Inference Performance on Strix Halo

A model that is accurate but slow is not good enough for local deployment.

Measure on the target hardware.

#### Reference implementation

```python
import time
import torch

def benchmark_inference(model, tokenizer, prompt, num_runs=10):
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    input_len = inputs["input_ids"].shape[1]

    for _ in range(3):
        model.generate(**inputs, max_new_tokens=128)

    times = []
    for _ in range(num_runs):
        torch.cuda.synchronize()
        start = time.perf_counter()
        output = model.generate(**inputs, max_new_tokens=128, do_sample=False)
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - start
        gen_tokens = output.shape[1] - input_len
        times.append(gen_tokens / elapsed)

    avg_tps = sum(times) / len(times)
    print(f"Average: {avg_tps:.1f} tokens/sec")
    print(f"Prefill: {input_len} tokens")
    return avg_tps
```

### 8.4 Strix Halo Targets

| Metric | Target | Unacceptable |
|---|---|---|
| Decode throughput | `> 30 tok/s` | `< 15 tok/s` |
| Prefill (512 tokens) | `< 200 ms` | `> 500 ms` |
| Peak VRAM | `< 6 GB` for 1B, `< 12 GB` for 3B | More than 2x target |
| Time to first token | `< 300 ms` | `> 1 s` |

---

## 9. Continuous Evaluation Dashboard

Maintain a single checkpoint dashboard.

### 9.1 Dashboard Template

```text
╔══════════════════════════════════════════════════════════╗
║           CONTINUED PRE-TRAINING MONITOR v2             ║
╠══════════════════════════════════════════════════════════╣
║ Step: 15,000 / 50,000          LR: 2.1e-5              ║
║ Train Loss: 2.34 (↓)          Val Loss (domain): 2.51  ║
║ Val Loss (general): 2.89 (stable)                       ║
║ Grad Norm (avg): 0.42         Grad Norm (max): 1.8     ║
╠══════════════════════════════════════════════════════════╣
║ PROBES (every 2000 steps)                               ║
║ HellaSwag: 58.2% (base: 59.1%)    FG: 1.5% ✅          ║
║ ARC-Easy:  71.4% (base: 72.0%)    FG: 0.8% ✅          ║
║ MMLU-mini: 46.1% (base: 45.3%)    FG: -1.8% ✅         ║
║ Domain QA: 34.2% (base: 18.5%)    Δ: +15.7% 🚀        ║
║ GSM8K-50:  22.0% (base: 23.0%)    FG: 4.3% ⚠️          ║
╠══════════════════════════════════════════════════════════╣
║ SANITY CHECK                                             ║
║ Coherence: ✅  Repetition: ✅  Domain: ✅  Safety: ✅     ║
╠══════════════════════════════════════════════════════════╣
║ INFERENCE (Strix Halo)                                   ║
║ Decode: 42.3 tok/s    Prefill (512): 145ms    VRAM: 4.8G║
╚══════════════════════════════════════════════════════════╝
```

---

## 10. Decision Tree

Use the final evaluation to decide whether to ship, continue, or stop.

```text
All general benchmarks within
5% of base model?
        /         \
      Yes          No
      /              \
Domain metrics      FG > 10% on
improved?           any task?
   /      \          /       \
 Yes       No      Yes        No
 /           \      /           \
Instruction  Check data   ❌ STOP   Increase
following OK? quality     Restart   replay,
 /      \                 with more  reduce LR
Yes      No               replay
/          \
✅ SHIP!   Fine-tune
           with instruct
           data (SFT)
```

---

## 11. TL;DR Checklist

| When | What | Tool | Frequency |
|---|---|---|---|
| During | Train/val loss and gap | Training loop | Every step |
| During | Perplexity across domains | Custom script | Every 500 steps |
| During | Gradient norms | Training loop | Every step |
| During | Quick probes | lm-eval-harness subsample | Every 2000 steps |
| During | Sanity generations | Custom script | Every 2000 steps |
| During | Forgetting metric | Computed from probes | Every probe |
| After | Full benchmark battery | lm-eval-harness | Once |
| After | LLM-as-Judge | Custom + API | Once |
| After | Constrained output tests | Custom | Once |
| After | Inference benchmarks | Custom on Strix Halo | Once |
| After | Forgetting scorecard | Computed from benchmarks | Once |

---

## 12. Agent Procedure

Follow this order:

1. Monitor training health
   - loss curves
   - validation gap
   - gradient norms

2. Run lightweight probes
   - HellaSwag
   - ARC-Easy
   - MMLU mini
   - GSM8K mini
   - domain QA

3. Compute forgetting
   - compare against base model
   - log FG for every probe task

4. Inspect generations
   - coherence
   - repetition
   - domain quality
   - safety

5. Validate final usefulness
   - full benchmark battery
   - judge comparisons
   - constrained formatting
   - Strix Halo throughput

6. Decide
   - ship
   - continue training
   - adjust replay or LR
   - stop and restart

### Decision rules

| Condition | Action |
|---|---|
| Domain improves, general capability stable | Continue training |
| Domain improves, general capability drops slightly | Increase replay or lower LR |
| General capability drops too much | Stop and retrain |
| Output quality improves and inference stays fast | Candidate for shipping |
| Output quality improves but inference is too slow | Optimize architecture or quantization |

---

## 13. Final Principle

A valid continued pre-training run must produce a clear narrative:

- what improved
- what regressed
- whether the tradeoff is acceptable

If the logs do not support that narrative, the evaluation is incomplete.
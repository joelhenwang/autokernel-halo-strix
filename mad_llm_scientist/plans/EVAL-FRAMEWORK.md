# EVAL FRAMEWORK

**Custom Evaluation Suite for 15-Minute Architecture Comparison at 250M Scale**

## Why Standard Benchmarks Fail

The E2LM NeurIPS 2025 competition (2506.07731) proved that standard benchmarks (MMLU, GSM8K, HellaSwag) give **noisy, non-discriminative signal** for small models during early training. At 250M params with 25M tokens of training, these benchmarks produce near-random scores.

We need metrics that give **meaningful signal in 15 minutes** and can **differentiate between architectures** at equal training budget.

---

## Three Pillars

### Pillar 1: Horse Race (Architecture Comparison)

Metrics that rank architectures against each other:

| Metric | What It Measures | How to Compute |
|--------|-----------------|----------------|
| **Loss-per-token curve** | Learning speed | Plot loss vs tokens_seen on same data |
| **Loss-per-FLOP** | Compute efficiency | loss / estimated_FLOPs_per_token |
| **Content word loss** | Hard pattern learning | CE loss masked to content tokens only |
| **Glue word loss** | Basic competence | CE loss masked to glue tokens only |
| **Bigram completion** | Factual pattern learning | Top-5 accuracy on 100 factual completions |
| **Function word fill** | Grammar acquisition | Exact match on 200 cloze tests |
| **Simple copying** | Retrieval capability | Exact match on 50 copy tasks |
| **Counting** | State tracking | Exact match on 50 counting tasks |

### Pillar 2: Component Probes (Does Each Ingredient Work?)

Architecture-specific diagnostics (uses `hasattr` — works for any model):

| Probe | Applies To | What It Measures | Good Signal |
|-------|-----------|-----------------|-------------|
| **mHC branch similarity** | Models with mHC | Cosine sim between H_pre weight vectors | < 0.5 = specialized |
| **Engram gate selectivity** | Models with Engram | Mean gate alpha for entities vs random | Entities > 0.7, random < 0.3 |
| **Decay spectrum preservation** | Models with decay_bias | Std of learned bias values | > 1.0 = spectrum maintained |
| **MTP head accuracy** | Models with MTP | % correct predictions per head | Head2 > 30%, Head3 > 15% |
| **Meta token gradient flow** | Models with meta_tokens | Mean |grad| through meta positions | > 0 = contributing |

### Pillar 3: Training Health (Catch Failures Early)

| Metric | What It Catches | Alert Threshold |
|--------|----------------|-----------------|
| **Gradient norm per layer** | Exploding/vanishing gradients | > 100 or < 1e-7 |
| **Activation mean/std** | Dead/exploding neurons | std < 1e-4 or mean > 100 |
| **Loss variance** | Noisy/unstable training | variance > 2× rolling average |
| **Dead neuron %** | Wasted capacity | > 5% of neurons always zero |
| **Weight norm growth** | Unbounded weights | > 10× initial norm |

---

## Implementation Design

### Part A: TrainingMonitor (in training loop, ~0.1ms/step)

```python
class TrainingMonitor:
    def __init__(self, glue_token_ids, log_path):
        self.glue_set = set(glue_token_ids)
        self.log_file = open(log_path, 'w')
        self.loss_buffer = deque(maxlen=100)
    
    def log_step(self, step, model, loss, logits, targets, log_grads=True):
        # 1. Total loss
        self.log_file.write(f"{step}\tloss\t{loss.item():.4f}\n")
        
        # 2. Content/glue split
        glue_mask = torch.tensor([t in self.glue_set for t in targets.view(-1)])
        if glue_mask.any():
            glue_loss = F.cross_entropy(logits.view(-1, V)[glue_mask], targets.view(-1)[glue_mask])
            self.log_file.write(f"{step}\tloss_glue\t{glue_loss.item():.4f}\n")
        content_mask = ~glue_mask
        if content_mask.any():
            content_loss = F.cross_entropy(logits.view(-1, V)[content_mask], targets.view(-1)[content_mask])
            self.log_file.write(f"{step}\tloss_content\t{content_loss.item():.4f}\n")
        
        # 3. Gradient norms (every step)
        if log_grads:
            total_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), float('inf'))
            self.log_file.write(f"{step}\tgrad_norm\t{total_norm.item():.4f}\n")
        
        # 4. Loss variance
        self.loss_buffer.append(loss.item())
        if len(self.loss_buffer) >= 10:
            var = torch.tensor(list(self.loss_buffer)).var().item()
            self.log_file.write(f"{step}\tloss_var\t{var:.6f}\n")
        
        # 5. Activation stats (every 50 steps to limit overhead)
        if step % 50 == 0:
            self._log_activation_stats(step, model)
```

### Part B: ArchitectureEvaluator (on checkpoints, ~30s)

```python
class ArchitectureEvaluator:
    def __init__(self, eval_data, synthetic_tasks, glue_token_ids):
        self.eval_data = eval_data
        self.tasks = synthetic_tasks
        self.glue_set = set(glue_token_ids)
    
    def evaluate(self, model, tokenizer):
        results = {}
        
        # --- Pillar 1: Horse Race ---
        results["ppl/total"] = self.compute_perplexity(model)
        results["ppl/content"] = self.compute_perplexity(model, mask="content")
        results["ppl/glue"] = self.compute_perplexity(model, mask="glue")
        results["flops_per_token"] = self.estimate_flops(model)
        
        for task_name, task_data in self.tasks.items():
            results[f"task/{task_name}"] = self.run_task(model, tokenizer, task_data)
        
        # --- Pillar 2: Component Probes ---
        results.update(self.run_component_probes(model))
        
        # --- Pillar 3: Health Check ---
        results.update(self.run_health_check(model))
        
        return results
    
    def run_component_probes(self, model):
        probes = {}
        if hasattr(model, 'mhc_layers'):
            probes["probe/mhc_branch_sim"] = self.probe_mhc(model)
        if hasattr(model, 'engram'):
            probes["probe/engram_selectivity"] = self.probe_engram(model)
        if hasattr(model, 'decay_bias'):
            probes["probe/decay_spectrum"] = self.probe_decay(model)
        if hasattr(model, 'mtp_heads'):
            probes.update(self.probe_mtp(model))
        if hasattr(model, 'meta_tokens'):
            probes["probe/meta_gradient"] = self.probe_meta(model)
        return probes
```

### Part C: Synthetic Tasks (JSON fixtures)

Pre-generated, deterministic test sets:

**bigram_completion.json** (100 items):
```json
[
  {"input": "The capital of France is", "targets": ["Paris"], "top_k": 5},
  {"input": "Water freezes at zero degrees", "targets": ["Celsius", "Fahrenheit"], "top_k": 5},
  ...
]
```

**function_word_fill.json** (200 items):
```json
[
  {"input": "She went", "target": "to", "context": "She went ___ the store"},
  {"input": "He is looking", "target": "for", "context": "He is looking ___ his keys"},
  ...
]
```

**simple_copying.json** (50 items):
```json
[
  {"input": "Repeat the word: elephant. The word is:", "target": "elephant"},
  {"input": "Copy: hello world. Copy:", "target": "hello"},
  ...
]
```

**counting.json** (50 items):
```json
[
  {"input": "How many times does 'a' appear in banana? Answer:", "target": "3"},
  {"input": "Count 'the' in: the cat and the dog. Count:", "target": "2"},
  ...
]
```

### Part D: Comparison Dashboard

```python
def generate_comparison(results_dict):
    """results_dict: {arch_name: {metric: value}}"""
    # Markdown table
    # Ranking by loss-per-FLOP
    # Highlight best-in-class per metric
    # Flag any health alerts
```

Output: `mad_llm_scientist/eval_results/comparison_TIMESTAMP.md`

---

## Glue Token Dictionary

Shared with architecture plans. ~500 tiktoken GPT-2 token IDs covering articles, prepositions, conjunctions, auxiliaries, pronouns, determiners, punctuation. Stored as a Python set in a shared module.

```python
# mad_llm_scientist/glue_tokens.py
import tiktoken
enc = tiktoken.get_encoding("gpt2")

GLUE_WORDS = [
    "the", "a", "an", "of", "in", "to", "for", "on", "at", "with", "by",
    "from", "and", "or", "but", "is", "are", "was", "were", "be", "been",
    "being", "has", "have", "had", "it", "he", "she", "they", "we", "I",
    "you", "this", "that", "these", "those", "some", "any", "not", "no",
    "if", "so", "as", "than", "then", "when", "where", "which", "who",
    "what", "how", "do", "does", "did", "will", "would", "could", "should",
    "may", "might", "can", "shall", "must", "its", "his", "her", "their",
    "our", "my", "your", ",", ".", ";", ":", "!", "?", '"', "'", "(", ")",
    # ... plus all cased/whitespace variants
]

GLUE_TOKEN_IDS = set()
for word in GLUE_WORDS:
    for variant in [word, " " + word, word.capitalize(), " " + word.capitalize()]:
        tokens = enc.encode(variant)
        GLUE_TOKEN_IDS.update(tokens)
```

---

## Usage

### During Training
```python
monitor = TrainingMonitor(GLUE_TOKEN_IDS, log_path="logs/archon_train.tsv")

for step, batch in enumerate(dataloader):
    loss, logits = model(batch)
    loss.backward()
    optimizer.step()
    monitor.log_step(step, model, loss, logits, batch.targets)
```

### Checkpoint Evaluation (every 2 min)
```python
evaluator = ArchitectureEvaluator(eval_data, tasks, GLUE_TOKEN_IDS)

for ckpt_path in checkpoints:
    model = load_checkpoint(ckpt_path)
    results = evaluator.evaluate(model, tokenizer)
    save_json(results, f"eval_results/{ckpt_path.stem}.json")

# Final comparison
all_results = load_all_results("eval_results/")
generate_comparison(all_results)
```

## Implementation Roadmap

1. Create `glue_tokens.py` with the shared glue dictionary
2. Create `eval_tasks/` with 4 synthetic task JSON fixtures
3. Implement `TrainingMonitor` class (lightweight, in-loop)
4. Implement `ArchitectureEvaluator` class (heavy, on checkpoints)
5. Implement component probes (mHC, Engram, decay, MTP, meta)
6. Implement `generate_comparison` dashboard
7. Test on GPT-2 small (124M) to validate metrics give signal
8. Integrate into training scripts

---

## Hardware Optimization Notes (Strix Halo gfx1151)

> Added 2026-04-09 based on AMADEUS implementation findings.

### SSM-Specific Evaluation Thresholds
Standard smoke test criteria (10K tok/s, 6 GB memory) don't apply to SSM architectures:
- **SSM throughput baseline:** 6.4K tok/s (AMADEUS 243M, eager) — NOT 10K
- **SSM memory baseline:** 12.7 GB (AMADEUS 243M) — NOT 6 GB
- **Transformer baseline:** 14.5K tok/s, ~17 GB (LlamaModel 124M, eager)
- Adjust evaluation thresholds per architecture type

### State Norm Monitoring for SSMs
The `StateNormMonitor` (designed for Griffin-style fixed recurrence) gives **false positives** for data-dependent SSMs (Mamba). Max ratios of 3-5 between batches are normal. Only flag sustained growth over consecutive steps, not per-batch variance.

### halo_training Integration
Evaluation can use `halo_training.evaluate.evaluate_bpb()` and `benchmark_inference()` directly.

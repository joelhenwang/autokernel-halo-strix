# Sprint 2 — Evaluation Scorecard Infrastructure

**Date:** 2026-05-06
**Track:** C (of roadmap A → C → B → D)
**Status:** Design approved, awaiting implementation plan
**Related:**
- `docs/superpowers/specs/2026-05-06-sprint1-foundation-wins-design.md` (Sprint 1, which this unlocks measurement for)
- `docs/research/broad-research-synthesis-2026-05-06.md` (identifies eval as the second-biggest gap)
- `knowledge/architectures/reliable_small_lm_insights.md` GAP 3 (the single-metric evaluation problem)

---

## 1. Goal

Ship a per-checkpoint evaluation scorecard that unlocks multi-dimensional capability measurement for all subsequent sprints. Before Sprint 2, the only training-time signal is loss/BPB. After Sprint 2, every checkpoint emits a machine-readable JSON report covering generalization (per-domain BPB), deployment readiness (int4 BPB, inference throughput, memory profile), sampling quality (distinct-N, self-PPL), regression risk (sample-pack diff), and architectural health (activation statistics).

Sprint 2 delivers the infrastructure; Sprints 3 and 4 consume it.

## 2. Roadmap context

Sprint 2 is the second sprint in the A → C → B → D roadmap:

```
Sprint 1 (A) → Sprint 2 (C) → Sprint 3 (B) → Sprint 4 (D)
              ^^^^^^^^^^^^
              you are here
```

**Gate A → C:** Sprint 1's success criteria must be met to enter Sprint 2 (so we have a stable recipe to evaluate).

**Gate C → B:** Sprint 2's success criteria (§6) gate entry to Sprint 3's T²-optimal dolma training. Sprint 3 uses the scorecard to verify its massive compute investment is paying off.

**Sprint 1.5 interaction:** Sprint 1.5 (optional) may execute between Sprint 2 and Sprint 3; it consumes the scorecard as its own measurement harness (ablation results flow through the same JSON schema).

## 3. Scope

### In scope

Comprehensive scorecard minus lm-evaluation-harness benchmarks (deferred to Sprint 4):

| Category | Metrics | Notes |
|----------|---------|-------|
| Per-domain BPB | wikitext-val, gpt-small-val, stem-crawl-val, dolma-val | Custom BPB on our `.bin` validation splits |
| Quantized performance | int4 BPB on same 4 splits | On-the-fly int4 quantization; deployment-readiness delta |
| Sampling quality | distinct-2, distinct-3, self-PPL at winning sampling config | Wraps `scripts/ablate_odin_flat_sampling.py` |
| Inference throughput | tok/s at seq={256, 512, 1024} batch=1 autoregressive | Custom HIP-aware inference harness |
| Memory profile | peak VRAM at seq={256, 512, 1024} batch=1 | `torch.cuda.max_memory_allocated` |
| Sample-pack regression | 20 hand-curated prompts; output diff vs prior checkpoint | Deterministic generation with fixed seed |
| Activation diagnostics | per-layer kurtosis, RMS norm, attention head entropy | Forward hooks, averaged over 10 batches |

### Out of scope

- **lm-evaluation-harness integration** (HellaSwag, ARC-easy, MMLU-CF, PIQA, BLiMP) — deferred to Sprint 4 (post-training). At 122M pre-SFT these benchmarks are near-chance and provide noisy signal.
- **Instruction-following benchmarks** (MT-Bench, AlpacaEval, HumanEval) — require SFT model, out of scope until Sprint 4
- **Automated dashboards / visualization** — JSONL index is sufficient; `jq` is our dashboard
- **Distributed eval** — single-node inference is fast enough for 122M; not worth DDP coordination
- **Auto-triggered watcher daemon** — we chose manual + optional trainer hook (Option D) over background daemon

### Model scope

Sprint 2 supports any Odin variant. Primary targets:
- OdinFlat (122M, flat)
- OdinHalo (57.6M unique, looped) — validates the scorecard handles both architectures
- OdinFlat30M (if Sprint 1.5 executes) — validates small-scale evaluation

## 4. Invocation pattern (per brainstorm Q3 Option D)

**Default manual CLI:**
```bash
EVAL_MACHINE=b python scripts/eval_checkpoint.py \
  --checkpoint checkpoints/odin-flat-wikitext-ddp/step_1869.pt \
  --model models/odin_flat.py --class-name OdinFlat \
  --output-dir docs/perf/eval-scorecards/
# Writes:
#   docs/perf/eval-scorecards/odin-flat-wikitext-ddp-step-1869.json
# Appends summary to:
#   docs/perf/eval-scorecard.jsonl
```

**Optional trainer hook** (Phase 5):
```bash
# Training with auto-eval after each checkpoint save
bash run_remote.sh "python -m halo_training --model ... --auto-eval"
# Or via DDP launcher:
EXTRA_FLAGS="--auto-eval" bash scripts/launch_ddp.sh
```

The trainer fires `scripts/eval_checkpoint.py` as a subprocess after every `save_checkpoint()` call. Training is not blocked — subprocess runs detached, logs to a per-checkpoint file, failures do NOT fail the trainer.

## 5. Component design

### 5.1 Files touched / created

| File | Change | Lines (est.) |
|------|--------|-------------:|
| `scripts/eval_checkpoint.py` **NEW** | Main CLI entry point; dispatches to each evaluator module; writes per-checkpoint JSON, appends JSONL index | ~200 |
| `halo_training/eval/__init__.py` **NEW** | Package init | ~10 |
| `halo_training/eval/per_domain_bpb.py` **NEW** | Loads validation splits; computes BPB per domain | ~100 |
| `halo_training/eval/int4_quant.py` **NEW** | Runtime int4 quantization wrapper + quantized BPB calculator | ~120 |
| `halo_training/eval/sampling.py` **NEW** | Wraps `ablate_odin_flat_sampling.py` to extract metrics at winning config | ~80 |
| `halo_training/eval/inference_profile.py` **NEW** | tok/s + memory profiling at fixed seq lengths | ~100 |
| `halo_training/eval/sample_pack.py` **NEW** | 20-prompt generation; diffs against prior checkpoint's outputs | ~120 |
| `halo_training/eval/activation_stats.py` **NEW** | Forward hooks for per-layer kurtosis, RMS norms, attention entropy | ~150 |
| `halo_training/eval/common.py` **NEW** | Shared: checkpoint loading, `_orig_mod.` stripping, tokenizer setup | ~60 |
| `evals/sample_pack.txt` **NEW** | 20 curated prompts (one per line) for regression tracking | — |
| `evals/validation_splits/README.md` **NEW** | Describes which validation slices are used per domain | — |
| `halo_training/cli.py` | Add `--auto-eval` flag to trainer | +10 |
| `scripts/train_ddp.py` | Hook for `--auto-eval` (fires post-checkpoint-save as detached subprocess) | +25 |
| `scripts/launch_ddp.sh` | No change (`--auto-eval` flows through via existing `EXTRA_FLAGS`) | +0 |

**Total: ~975 LoC across 11 new files + 2 edits** (11 new = 9 code modules + 2 data/doc files; 2 edits to existing code; `launch_ddp.sh` gets no change since `--auto-eval` flows through the existing `EXTRA_FLAGS`). Slightly higher than the brainstorm estimate due to `common.py` deduplication.

### 5.2 Architecture — modular evaluator design

```
scripts/eval_checkpoint.py
    ├── parse_args() + select_eval_machine()
    ├── load_checkpoint(path) + load_model(file, class)  [via eval/common.py]
    ├── run each evaluator (selectable via --skip-*):
    │   ├── per_domain_bpb.run(model, validation_splits)    →  {wikitext_val, gpt_small_val, ...}
    │   ├── int4_quant.run(model, validation_splits)         →  {int4_wikitext_val, ...}
    │   ├── sampling.run(model, tokenizer)                   →  {distinct_2, distinct_3, self_ppl}
    │   ├── inference_profile.run(model)                     →  {tok_s per seq, peak_mem per seq}
    │   ├── sample_pack.run(model, prompts, prior_ckpt)      →  {outputs, hash, diff_vs_prior}
    │   └── activation_stats.run(model, probe_batches=10)    →  {layer_kurtosis, layer_rms, attn_entropy}
    ├── assemble_scorecard(all results, metadata)            →  scorecard dict
    ├── write_per_checkpoint_json(scorecard, output_dir)     →  docs/perf/eval-scorecards/<name>.json
    └── append_jsonl_index(scorecard, jsonl_path)            →  docs/perf/eval-scorecard.jsonl
```

Each evaluator is an independent module with a single `run(model, ...)` entry point returning a dict. This makes them:
- **Unit-testable** in isolation
- **Skippable** via `--skip-<name>` CLI flags
- **Extensible** — adding a new evaluator means adding a new module, not editing existing ones
- **Failure-isolated** — if `activation_stats` throws, other evaluators still complete

### 5.3 Per-domain BPB (`halo_training/eval/per_domain_bpb.py`)

**Validation splits:** each domain gets a held-out slice (last 2% of the tokenized dataset, not used in training). Splits live at `evals/validation_splits/<domain>.bin` (or point into the existing .bin via offset markers to save storage).

Available splits per current dataset inventory:
- `wikitext_val`: derived from `datasets/wikitext-103-odin32k.bin` (last ~2.5M tokens)
- `gpt_small_val`: from `datasets/gpt-training-small-odin32k.bin` (last ~6M tokens)
- `stem_crawl_val`: from `datasets/stem-crawl-solo-odin32k.bin` (last ~10M tokens)
- `dolma_val`: from `datasets/dolma-10b-odin32k.bin` (last ~140M tokens)

**Implementation:**

```python
# halo_training/eval/per_domain_bpb.py
@torch.no_grad()
def run(model, tokenizer, validation_splits, batch_size=8, max_tokens=50_000):
    """Compute BPB per domain.
    
    Loads up to max_tokens from each split, runs forward passes, averages CE loss,
    converts to BPB = (mean_ce / log(2)) / avg_bytes_per_token.
    """
    results = {}
    for domain, path in validation_splits.items():
        if not os.path.exists(path):
            results[domain] = None  # gracefully skip unavailable splits
            continue
        tokens = np.memmap(path, dtype=np.uint16, mode="r")[:max_tokens]
        mean_ce = compute_mean_ce(model, tokens, batch_size)
        bytes_per_token = estimate_bytes_per_token(tokenizer, tokens[:10_000])
        results[domain] = (mean_ce / math.log(2)) / bytes_per_token
    return results
```

**Budget:** ~1-2 minutes total for all 4 domains at 50K tokens each.

### 5.4 Int4 quantization (`halo_training/eval/int4_quant.py`)

**Strategy:** runtime per-tensor symmetric int4 quantization for the model's 2D weight matrices; embeddings/lm_head stay fp16.

```python
# halo_training/eval/int4_quant.py
@torch.no_grad()
def quantize_model_int4(model):
    """Returns a deepcopy of the model with 2D weights int4-quantized.
    
    Symmetric per-tensor: scale = max(|w|) / 7; round to [-8, 7]; dequantize
    at forward via torch.dequantize. Falls back to fp16 if tensor has ndim < 2
    or contains 'embed' / 'lm_head' in its qualified name.
    """
    quantized = copy.deepcopy(model)
    for name, module in quantized.named_modules():
        if isinstance(module, nn.Linear):
            if "embed" in name or "lm_head" in name:
                continue
            w = module.weight.data
            scale = w.abs().max() / 7.0
            q = torch.round(w / scale).clamp(-8, 7).to(torch.int8)
            # Store quantized + scale; replace forward to dequantize on-the-fly
            module.weight = nn.Parameter(q.to(torch.int8), requires_grad=False)
            module.scale = scale
            module.forward = make_int4_forward(module)
    return quantized

def run(model, tokenizer, validation_splits, batch_size=8, max_tokens=50_000):
    """Compute int4-quantized BPB per domain for deployment-readiness delta."""
    int4_model = quantize_model_int4(model)
    return per_domain_bpb.run(int4_model, tokenizer, validation_splits, batch_size, max_tokens)
```

**Budget:** ~1-2 minutes (second forward pass over same splits).

**Caveat:** per-tensor symmetric int4 is a crude lower bound on deployment quality. Real int4 deployment would use per-channel + asymmetric + group quantization. Sprint 2's int4 is a "deployment-readiness indicator," not a production quantization path.

### 5.5 Sampling integration (`halo_training/eval/sampling.py`)

Wraps the existing `scripts/ablate_odin_flat_sampling.py` — does NOT duplicate the sampling harness. Uses its functions directly:

```python
# halo_training/eval/sampling.py
from scripts.ablate_odin_flat_sampling import (
    run_ablation,            # runs the 3-stage sampling ablation
    select_winning_config,   # picks the winning config from ablation results
)

def run(model, tokenizer, num_samples=3, max_tokens=120):
    """Run sampling ablation, extract metrics at winning config."""
    ablation_results = run_ablation(
        model, tokenizer,
        prompts=["The history of the Roman Empire", "In the field of physics,"],
        num_samples=num_samples,
        max_tokens=max_tokens,
    )
    winner = select_winning_config(ablation_results)
    return {
        "winning_config": winner["config"],
        "distinct_2": winner["metrics"]["distinct_2"],
        "distinct_3": winner["metrics"].get("distinct_3", None),
        "self_ppl": winner["metrics"]["self_ppl"],
        "sample_preview": winner["metrics"]["sample_preview"][:100],
    }
```

Refactor may be needed in `scripts/ablate_odin_flat_sampling.py` to expose these functions cleanly — 30-60 minutes of work depending on how tightly the current script's main loop is coupled. The existing script keeps working; we just extract the reusable bits.

**Budget:** ~1-2 minutes (same as running `ablate_odin_flat_sampling.py` manually).

### 5.6 Inference profile (`halo_training/eval/inference_profile.py`)

```python
# halo_training/eval/inference_profile.py
@torch.no_grad()
def run(model, tokenizer, seq_lengths=(256, 512, 1024), batch_size=1, warmup=10, measure=30):
    """Measure tok/s and peak memory at each seq length via autoregressive generation."""
    results = {}
    for seq_len in seq_lengths:
        # Warmup
        prompt = torch.randint(0, tokenizer.vocab_size, (batch_size, seq_len), device="cuda")
        for _ in range(warmup):
            _ = model(prompt)
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        
        # Measure
        t0 = time.time()
        for _ in range(measure):
            _ = model(prompt)
        torch.cuda.synchronize()
        elapsed = time.time() - t0
        tok_s = (batch_size * seq_len * measure) / elapsed
        peak_mem = torch.cuda.max_memory_allocated() / 1e9
        results[f"tok_s_seq{seq_len}_bs{batch_size}"] = tok_s
        results[f"peak_mem_gb_seq{seq_len}"] = peak_mem
    return results
```

**Budget:** ~1 minute (3 seq lengths × 40 forwards each, lightweight).

**Note:** This measures **full-seq forward** throughput, not autoregressive decode (which is much slower due to KV-cache reuse overhead). For Sprint 4 (post-training) we'd add true autoregressive decode; Sprint 2's forward-only is a deployment-readiness ceiling.

### 5.7 Sample-pack regression (`halo_training/eval/sample_pack.py`)

**Prompts file:** `evals/sample_pack.txt` — 20 hand-curated prompts covering:
- Domain diversity: history, science, fiction, code, math, dialogue, instruction-following
- Failure-mode probes: long-range coherence, repetition-prone topics, rare entities
- Length diversity: short (1-line) to medium (paragraph setup)

Example curation (to be finalized during implementation — 2-3 hours of prompt work):
```
The history of the Roman Empire began
In the field of particle physics,
def fibonacci(n):
The following recipe yields a delicious
Q: What is the capital of Mongolia?
A:
Once upon a time, in a small village,
... (20 total)
```

**Diff logic:**
- For each prompt: generate at fixed seed + `temperature=0.8, top_p=0.95, max_tokens=100`
- Hash the 20-sample output set (sha256 of concatenated outputs)
- Auto-detect prior checkpoint: most recent sibling `step_*.pt` in same directory, or explicit `--prior-checkpoint` override
- Compare hashes: identical → `diff_vs_prior=null`; different → run diff ratio, emit first 3 changed prompts inline in scorecard JSON

```python
# halo_training/eval/sample_pack.py
def run(model, tokenizer, prompts_file, prior_ckpt=None, seed=42, max_tokens=100):
    prompts = load_prompts(prompts_file)  # list of 20 strings
    samples = []
    for i, prompt in enumerate(prompts):
        torch.manual_seed(seed + i)  # deterministic per-prompt
        sample = generate(model, tokenizer, prompt, max_tokens, 
                         temperature=0.8, top_p=0.95)
        samples.append(sample)
    
    combined = "\n---\n".join(samples)
    output_hash = hashlib.sha256(combined.encode()).hexdigest()
    
    diff_info = None
    if prior_ckpt is not None:
        prior_samples = load_prior_samples(prior_ckpt, prompts)  # re-run on prior if cached not found
        diff_count, diff_preview = compute_diff(samples, prior_samples)
        diff_info = {"diff_count": diff_count, "preview": diff_preview[:3]}
    
    return {
        "prompts_file": prompts_file,
        "output_hash": output_hash,
        "diff_vs_prior": diff_info,
    }
```

**Budget:** ~2-3 minutes (20 prompts × 100 tokens, sequential decode).

### 5.8 Activation stats (`halo_training/eval/activation_stats.py`)

Forward hooks on specific modules collect activations during a probe batch; compute kurtosis, RMS norm, and attention head entropy.

```python
# halo_training/eval/activation_stats.py
class ActivationCollector:
    def __init__(self, model):
        self.layer_acts = {}       # layer_idx → list of activation tensors
        self.attn_probs = {}       # layer_idx → attention probs
        self.hooks = []
        self._register_hooks(model)
    
    def _register_hooks(self, model):
        for i, layer in enumerate(model.layers):
            h = layer.register_forward_hook(
                lambda m, inp, out, i=i: self._capture_layer_out(i, out))
            self.hooks.append(h)
            # For attention layers, also hook the softmax (if exposed)
            # If not exposed, skip attn_entropy for that layer
    
    def remove(self):
        for h in self.hooks:
            h.remove()

@torch.no_grad()
def run(model, tokenizer, validation_splits, num_batches=10, seq_len=512, batch_size=8):
    """Compute activation statistics averaged over num_batches random validation batches."""
    collector = ActivationCollector(model)
    try:
        for _ in range(num_batches):
            batch = sample_random_batch(validation_splits, seq_len, batch_size)
            _ = model(batch)
        # Aggregate across batches
        return {
            "layer_kurtosis": [mean_kurtosis(collector.layer_acts[i]) for i in range(len(model.layers))],
            "layer_rms_norm": [mean_rms(collector.layer_acts[i]) for i in range(len(model.layers))],
            "attention_head_entropy_mean": mean_attn_entropy(collector.attn_probs),
        }
    finally:
        collector.remove()  # critical: prevent memory leak
```

**Budget:** ~2 minutes (10 batches × 8 × 512 = ~40K tokens).

### 5.9 Scorecard JSON schema

Per-checkpoint JSON output structure (canonical reference):

```json
{
  "schema_version": "1.0",
  "checkpoint": "checkpoints/odin-flat-wikitext-ddp/step_1869.pt",
  "checkpoint_name": "odin-flat-wikitext-ddp-step-1869",
  "model": {
    "file": "models/odin_flat.py",
    "class": "OdinFlat",
    "params": 121700000,
    "d_model": 768,
    "n_layers": 14
  },
  "eval_machine": "b",
  "eval_timestamp": "2026-05-08T14:22:01Z",
  "eval_duration_s": 187,
  "eval_config": {
    "max_tokens_per_domain": 50000,
    "sampling_max_tokens": 120,
    "inference_seq_lengths": [256, 512, 1024],
    "sample_pack_prompts": "evals/sample_pack.txt"
  },
  "per_domain_bpb": {
    "wikitext_val": 1.79,
    "gpt_small_val": 2.03,
    "stem_crawl_val": 2.11,
    "dolma_val": null
  },
  "int4_bpb": {
    "wikitext_val": 1.83,
    "gpt_small_val": 2.08,
    "stem_crawl_val": 2.15,
    "dolma_val": null
  },
  "sampling": {
    "winning_config": {"temperature": 0.6, "rep_pen": 1.0, "top_p": 1.0, "top_k": 0},
    "distinct_2": 0.765,
    "distinct_3": 0.83,
    "self_ppl": 9.84,
    "sample_preview": "The history of the Roman Empire , by the British Revolution..."
  },
  "inference": {
    "tok_s_seq256_bs1": 142.3,
    "tok_s_seq512_bs1": 119.7,
    "tok_s_seq1024_bs1": 88.4,
    "peak_mem_gb_seq256": 0.91,
    "peak_mem_gb_seq512": 1.44,
    "peak_mem_gb_seq1024": 2.81
  },
  "sample_pack": {
    "prompts_file": "evals/sample_pack.txt",
    "output_hash": "sha256:a3b2c4d5...",
    "diff_vs_prior": {"diff_count": 3, "preview": ["prompt 0: 12% token overlap change", ...]}
  },
  "activation_stats": {
    "layer_kurtosis": [3.1, 3.4, 3.2, ..., 2.9],
    "layer_rms_norm": [1.12, 1.34, 1.41, ..., 1.08],
    "attention_head_entropy_mean": 2.31
  }
}
```

**Global JSONL index** (`docs/perf/eval-scorecard.jsonl`):

```jsonl
{"ts":"2026-05-08T14:22Z","ckpt":"odin-flat-wikitext-ddp-step-1869","avg_bpb":1.98,"int4_delta_bpb":0.04,"distinct_2":0.765,"tok_s_512":120,"peak_mem_gb_512":1.44,"full":"docs/perf/eval-scorecards/odin-flat-wikitext-ddp-step-1869.json"}
{"ts":"2026-05-08T14:28Z","ckpt":"odin-halo-wikitext-ddp-step-1869","avg_bpb":2.11,"int4_delta_bpb":0.05,"distinct_2":0.990,"tok_s_512":95,"peak_mem_gb_512":1.21,"full":"docs/perf/eval-scorecards/odin-halo-wikitext-ddp-step-1869.json"}
```

One line per run — cheap to grep, jq, diff.

### 5.10 CLI surface

```
Usage:
  python scripts/eval_checkpoint.py --checkpoint <path> [options]

Required:
  --checkpoint <path>           .pt file
  --model <file.py>             model module path (e.g., models/odin_flat.py)
  --class-name <name>           model class (e.g., OdinFlat)

Optional:
  --tokenizer-path <path>       tokenizer.json (default: tokenizers/odin-32k/tokenizer.json)
  --output-dir <path>           per-checkpoint JSON directory
                                (default: docs/perf/eval-scorecards/)
  --jsonl-index <path>          JSONL index to append summary to
                                (default: docs/perf/eval-scorecard.jsonl)
  --eval-machine {a,b}          target machine label (uses EVAL_MACHINE env var if unset)
  --prior-checkpoint <path>     for sample-pack diff (default: auto-detect most recent sibling)
  --prompts-file <path>         sample-pack prompts (default: evals/sample_pack.txt)
  --max-tokens-per-domain <N>   BPB evaluation depth (default: 50000)

Skip flags (default: all evaluators run; use to opt out):
  --skip-per-domain-bpb
  --skip-int4
  --skip-sampling
  --skip-inference-profile
  --skip-sample-pack
  --skip-activation-stats
```

And on the trainer side (added in Phase 5):
```
--auto-eval                     spawn eval_checkpoint.py as subprocess after each checkpoint save
```

## 6. Success criteria (Gate C → B)

Sprint 2 succeeds if ALL of:

| Criterion | Target | Measured how |
|-----------|--------|--------------|
| **Scorecard runs on representative subset** | Zero runtime errors on 1 OdinFlat + 1 OdinHalo checkpoint | Manual verification |
| **All metric values finite and sensible** | No NaN, no -inf; ranges consistent with scale | Manual review against reference values |
| **`--auto-eval` smoke-tests** | 200-step training with flag on; no trainer regression; subprocess exits cleanly | Manual run |
| **Machine parity** | `EVAL_MACHINE=a` vs `EVAL_MACHINE=b` match within ±5% on same checkpoint | Parity diff |
| **Wall time** | ≤ 15 minutes end-to-end on either machine | End-to-end timing |
| **Documentation complete** | STATUS.md + AGENTS.md explain invocation, output schema, and integration | Manual review |
| **Unit tests pass** | 11/11 tests green | CI-style verification |

### Failure-response table

| Failure | Response |
|---------|----------|
| int4 quantization produces garbage BPB (>5× fp16 BPB) | Mark int4 as "unreliable at this scale" in JSON; ship with a documented caveat |
| Sampling wrapper refactor breaks `ablate_odin_flat_sampling.py` | Refactor more carefully; preserve the existing CLI |
| Machine parity drift > 5% | Investigate determinism; document any aiter-related variation; if persistent, restrict to one machine |
| Wall time > 15 min | Profile each evaluator; gate `activation_stats` behind opt-in flag if it's the bottleneck |
| Activation hooks leak memory | Ensure `collector.remove()` is in a try/finally; add explicit test |
| Auto-eval subprocess silently dies | Capture STDERR to per-checkpoint log file; add explicit "subprocess failed" line in trainer output |
| Validation split unavailable (e.g., dolma-val not yet built) | Gracefully skip; emit `null` in JSON; don't fail the whole scorecard |
| Prior-checkpoint auto-detection fails | Gracefully skip sample-pack diff; emit `diff_vs_prior: null` |

## 7. Execution phases

| Phase | Duration | What |
|-------|----------|------|
| 1 (dev) | 0.5 day | `scripts/eval_checkpoint.py` skeleton, JSON schema, JSONL append, CLI parsing, checkpoint load helper |
| 2 (dev) | 0.75 day | `per_domain_bpb.py` + `int4_quant.py`; unit tests |
| 3 (dev) | 0.75 day | `sampling.py` (with refactor of `ablate_odin_flat_sampling.py`) + `inference_profile.py`; unit tests |
| 4 (dev) | 0.75 day | `sample_pack.py` + 20-prompt curation + `activation_stats.py`; unit tests |
| 5 (dev) | 0.5 day | `--auto-eval` flag in trainer; detached-subprocess integration; subprocess log capture |
| 6 (validation) | 0.25 day | Retroactive run on 1 OdinFlat + 1 OdinHalo checkpoint; verify output; machine parity check |
| 7 (docs + commit) | 0.5 day | STATUS.md, AGENTS.md, `knowledge/INDEX.md` updates; single commit |

**Total: ~4 days dev + validation** (matches brainstorm Q3's "add polish" extension from 3 to 4 days).

## 8. Testing

### 8.1 Unit tests (11 total per brainstorm)

| Test | Verifies | Phase |
|------|----------|:-:|
| `test_scorecard_json_schema` | Output JSON has all required top-level keys; parseable | 1 |
| `test_jsonl_index_append` | Appends one line; previous lines preserved | 1 |
| `test_per_domain_bpb_finite` | Returns finite BPB for all 4 configured splits (or null for unavailable) | 2 |
| `test_int4_quant_roundtrip` | int4-quantize → dequantize preserves ~99% of fp16 output | 2 |
| `test_int4_bpb_sanity` | int4 BPB within 20% of fp16 BPB (bounded degradation) | 2 |
| `test_sampling_metrics_extractable` | Wrapping `ablate_odin_flat_sampling.py` produces dict with distinct_2, self_ppl | 3 |
| `test_inference_profile_tok_s_positive` | tok/s > 0 at all tested seq lengths; monotonic decrease with seq_len | 3 |
| `test_sample_pack_20_prompts_load` | `evals/sample_pack.txt` exists; loads exactly 20 non-empty entries | 4 |
| `test_sample_pack_output_deterministic` | Same model + same seed → same hash | 4 |
| `test_activation_hooks_no_leak` | After `collector.remove()`, no dangling hooks on model modules | 4 |
| `test_auto_eval_subprocess_spawns` | `--auto-eval` fires subprocess that exits cleanly; trainer unblocked | 5 |

### 8.2 Integration tests

- Smoke test: `python scripts/eval_checkpoint.py --checkpoint <sprint1-run1-ckpt>` runs without error on an existing checkpoint; produces valid JSON.
- DDP + `--auto-eval`: `EXTRA_FLAGS="--auto-eval" bash scripts/launch_ddp.sh` on a 200-step training run produces both training logs AND an eval scorecard JSON.

### 8.3 Retroactive validation (Phase 6)

Run scorecard on:
- `checkpoints/odin-flat-wikitext-ddp/step_1869.pt` (OdinFlat baseline)
- `checkpoints/odin-halo-wikitext-ddp/step_1869.pt` (OdinHalo baseline)

Both should produce complete, finite scorecards. Spot-check values against known measurements from earlier analysis (e.g., sampling distinct-2 was 0.765 for OdinFlat wikitext — scorecard should reproduce this).

## 9. Risk register

| Risk | Likelihood | Impact | Mitigation |
|------|:---:|:---:|------|
| int4 quantization produces garbage BPB (per-tensor sym too crude) | Medium | Low | Document as "indicator, not production"; gate behind `--skip-int4` if broken |
| Sampling refactor breaks the existing `ablate_odin_flat_sampling.py` CLI | Low | Medium | Refactor carefully; keep CLI exactly as-is; only extract pure functions |
| Machine parity drift > 5% (aiter vs not) | Medium | Low | Document drift; restrict to one machine if persistent |
| `--auto-eval` subprocess blocks trainer (e.g., shares GPU) | Medium | Medium | Subprocess runs on different machine than trainer via `EVAL_MACHINE`; or on same machine but after step synced |
| Wall time > 15 min | Medium | Low | Profile; gate slowest evaluator behind opt-in flag; activation_stats likely the candidate |
| Activation hook memory leak | Low | Medium | try/finally + unit test |
| Prior-checkpoint auto-detection ambiguous | Low | Low | Require explicit `--prior-checkpoint` if multiple siblings found |
| dolma validation split not yet built | High (until Sprint 3) | Low | Gracefully skip; emit null |
| 20-prompt sample pack becomes stale after post-training | Low | Low | Accept staleness; curate new pack when needed |
| Checkpoint `_orig_mod.` stripping missed (from Sprint 1 fix) | Low | Medium | Use the same strip logic as `sample_odin_flat.py`; unit test |

## 10. Timeline

| Day | Activity |
|:---:|----------|
| 1 | Phase 1 + part of Phase 2 — skeleton + per-domain BPB |
| 2 | Rest of Phase 2 + Phase 3 — int4 + sampling + inference profile |
| 3 | Phase 4 — sample-pack curation + activation stats + tests |
| 3-4 | Phase 5 — auto-eval trainer hook |
| 4 | Phase 6 validation + Phase 7 docs + commit |

**Total elapsed: ~4 days.**

## 11. Dependencies

### External
- `numpy`, `torch` (existing)
- No new pip dependencies for Sprint 2 (lm-evaluation-harness is deferred to Sprint 4)
- `hashlib` (stdlib) for sample-pack hashing

### Internal
- Sprint 1 completion is NOT strictly required (Sprint 2 can proceed in parallel as dev work), but validation needs real checkpoints to run against
- Existing `scripts/ablate_odin_flat_sampling.py` (refactored during Phase 3)
- Existing tokenizer at `tokenizers/odin-32k/tokenizer.json`
- Validation splits derived from existing `.bin` files (last 2% of each)

## 12. End state

1. `scripts/eval_checkpoint.py` + 7 evaluator modules committed
2. 20-prompt sample pack (`evals/sample_pack.txt`) committed
3. JSON schema documented; JSONL index file exists with at least 2 entries (one per validation checkpoint)
4. `--auto-eval` trainer flag works in a smoke test
5. Machine parity verified (±5% accepted)
6. Wall time ≤ 15 min on either machine
7. 11 unit tests passing
8. STATUS.md, AGENTS.md, `knowledge/INDEX.md` updated
9. Single commit — rollback-by-revert compatible
10. Gate C → B cleared — ready to start Sprint 3 with scorecard in place

## 13. Open questions (do not block Sprint 2)

1. **Validation splits — new files or offset markers into existing `.bin`?** Default: offset markers (e.g., last 5M tokens of each .bin) to avoid data duplication. Decide during Phase 2.
2. **int4 per-channel vs per-tensor?** Default: per-tensor (simpler). Revisit in Sprint 4 if deployment path matters.
3. **Sample pack prompt domain coverage** — 20 prompts is small. Revisit expansion in a future sprint if regression signal is weak.

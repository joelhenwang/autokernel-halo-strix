# Sprint 2 Implementation Plan — Evaluation Scorecard

**Companion to:** `docs/superpowers/specs/2026-05-06-sprint2-eval-scorecard-design.md`
**Date:** 2026-05-06
**Sequence:** 7 dev phases over ~4 days; no long compute; retroactive validation + final commit
**Pre-requisite:** None strictly (can proceed in parallel with Sprint 1 training work, as dev-only during Sprint 1 compute)

---

## Work breakdown

Each phase's tasks have:
- **Exit criterion** (what "done" looks like)
- **Test** (or "none" for integration-level)
- **Rollback step** (what to revert if this task fails)

Sprint 2 is dev-heavy (no long compute runs). Structure favors per-module
independence so tasks can be parallelized if dev capacity allows.

---

## Phase 1 — Scaffolding (0.5 day)

Goal: directory structure, entry-point skeleton, JSON schema, JSONL appending.

### Task 1.1 — Package structure

**Files:**
- `halo_training/eval/__init__.py` **NEW** — empty module init
- `halo_training/eval/common.py` **NEW** — shared utilities

`common.py` contains:

```python
# halo_training/eval/common.py
import importlib.util
import sys
from pathlib import Path

import torch
from tokenizers import Tokenizer


def load_model(model_path, class_name):
    """Load a model class from a .py file path."""
    spec = importlib.util.spec_from_file_location("user_model", model_path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["user_model"] = mod
    spec.loader.exec_module(mod)
    cls = getattr(mod, class_name)
    return cls()


def load_checkpoint(model, checkpoint_path, device="cuda"):
    """Load checkpoint with _orig_mod. stripping (Sprint 1 fix)."""
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    state_dict = ckpt.get("model_state_dict", ckpt)
    cleaned = {k.replace("._orig_mod.", "."): v for k, v in state_dict.items()}
    missing, unexpected = model.load_state_dict(cleaned, strict=False)
    if missing or unexpected:
        raise RuntimeError(
            f"Checkpoint key mismatch: {len(missing)} missing, "
            f"{len(unexpected)} unexpected"
        )
    return model, ckpt.get("step", None)


def load_tokenizer(tokenizer_path):
    return Tokenizer.from_file(tokenizer_path)


def default_validation_splits():
    """Return dict of {domain: path} to validation split .bin files."""
    return {
        "wikitext_val": "evals/validation_splits/wikitext_val.bin",
        "gpt_small_val": "evals/validation_splits/gpt_small_val.bin",
        "stem_crawl_val": "evals/validation_splits/stem_crawl_val.bin",
        "dolma_val": "evals/validation_splits/dolma_val.bin",
    }
```

**Exit criterion:** 
- `from halo_training.eval.common import load_model, load_checkpoint, load_tokenizer` succeeds from a Python shell
- `load_model("models/odin_flat.py", "OdinFlat")` returns OdinFlat instance
- `load_checkpoint` correctly strips `_orig_mod.` prefix (reuses Sprint 1 fix)

**Test:** `tests/test_eval_common.py::test_load_model`, `test_load_checkpoint_strips_orig_mod`

**Rollback:** Delete `halo_training/eval/` directory.

### Task 1.2 — `scripts/eval_checkpoint.py` skeleton

**File:** `scripts/eval_checkpoint.py` **NEW**

**Change:** Main CLI entry point with argparse, dispatch-to-evaluators loop (initially stubbed), JSON output, JSONL index append:

```python
"""Sprint 2: per-checkpoint evaluation scorecard."""

import argparse
import datetime
import json
import os
import time
from pathlib import Path

from halo_training.eval.common import load_model, load_checkpoint, load_tokenizer, default_validation_splits
# Evaluator modules loaded lazily; each is a `halo_training.eval.<name>` module

EVALUATORS = [
    "per_domain_bpb",
    "int4_quant",
    "sampling",
    "inference_profile",
    "sample_pack",
    "activation_stats",
]

def main():
    parser = argparse.ArgumentParser(description="Sprint 2 scorecard")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--class-name", required=True)
    parser.add_argument("--tokenizer-path", default="tokenizers/odin-32k/tokenizer.json")
    parser.add_argument("--output-dir", default="docs/perf/eval-scorecards")
    parser.add_argument("--jsonl-index", default="docs/perf/eval-scorecard.jsonl")
    parser.add_argument("--eval-machine", default=os.environ.get("EVAL_MACHINE", "unknown"))
    parser.add_argument("--prior-checkpoint", default=None)
    parser.add_argument("--prompts-file", default="evals/sample_pack.txt")
    parser.add_argument("--max-tokens-per-domain", type=int, default=50_000)
    for name in EVALUATORS:
        parser.add_argument(f"--skip-{name.replace('_','-')}", action="store_true")
    args = parser.parse_args()

    t0 = time.time()

    # Load model + checkpoint
    model = load_model(args.model, args.class_name).to("cuda").eval()
    model, step = load_checkpoint(model, args.checkpoint)
    tokenizer = load_tokenizer(args.tokenizer_path)
    validation_splits = default_validation_splits()

    # Run each evaluator (with skip flags)
    results = {}
    for name in EVALUATORS:
        if getattr(args, f"skip_{name}"):
            results[name] = None
            continue
        mod = __import__(f"halo_training.eval.{name}", fromlist=["run"])
        results[name] = mod.run(model, tokenizer, validation_splits,
                                 args=args)  # each evaluator takes args for its specifics

    # Assemble and emit scorecard
    scorecard = assemble_scorecard(args, step, model, results, t0)
    write_json(scorecard, args.output_dir, args.checkpoint)
    append_jsonl(scorecard, args.jsonl_index)

    print(f"Scorecard written for {args.checkpoint}")
    print(f"  Duration: {time.time() - t0:.1f}s")
    print(f"  Output: {Path(args.output_dir) / checkpoint_basename(args.checkpoint)}.json")


def assemble_scorecard(args, step, model, results, t0):
    return {
        "schema_version": "1.0",
        "checkpoint": args.checkpoint,
        "checkpoint_name": checkpoint_basename(args.checkpoint),
        "model": {
            "file": args.model,
            "class": args.class_name,
            "params": sum(p.numel() for p in model.parameters()),
        },
        "eval_machine": args.eval_machine,
        "eval_timestamp": datetime.datetime.utcnow().isoformat() + "Z",
        "eval_duration_s": round(time.time() - t0, 1),
        **{k: v for k, v in results.items()},
    }


def write_json(scorecard, output_dir, checkpoint_path):
    os.makedirs(output_dir, exist_ok=True)
    name = checkpoint_basename(checkpoint_path)
    out = Path(output_dir) / f"{name}.json"
    with open(out, "w") as f:
        json.dump(scorecard, f, indent=2)


def append_jsonl(scorecard, jsonl_path):
    os.makedirs(Path(jsonl_path).parent, exist_ok=True)
    summary = {
        "ts": scorecard["eval_timestamp"],
        "ckpt": scorecard["checkpoint_name"],
        "avg_bpb": compute_avg_bpb(scorecard.get("per_domain_bpb")),
        "int4_delta_bpb": compute_int4_delta(scorecard),
        "distinct_2": (scorecard.get("sampling") or {}).get("distinct_2"),
        "tok_s_512": (scorecard.get("inference") or {}).get("tok_s_seq512_bs1"),
        "peak_mem_gb_512": (scorecard.get("inference") or {}).get("peak_mem_gb_seq512"),
        "full": f"docs/perf/eval-scorecards/{scorecard['checkpoint_name']}.json",
    }
    with open(jsonl_path, "a") as f:
        f.write(json.dumps(summary) + "\n")


def checkpoint_basename(checkpoint_path):
    """Turn 'checkpoints/odin-flat-wikitext-ddp/step_1869.pt' into 'odin-flat-wikitext-ddp-step-1869'."""
    p = Path(checkpoint_path)
    return f"{p.parent.name}-{p.stem}"


def compute_avg_bpb(per_domain):
    if not per_domain:
        return None
    values = [v for v in per_domain.values() if v is not None]
    return round(sum(values) / len(values), 3) if values else None


def compute_int4_delta(scorecard):
    fp16 = compute_avg_bpb(scorecard.get("per_domain_bpb"))
    int4 = compute_avg_bpb(scorecard.get("int4_quant"))
    if fp16 is None or int4 is None:
        return None
    return round(int4 - fp16, 3)


if __name__ == "__main__":
    main()
```

At this point, evaluators are stubs returning empty dicts — we add them in Phase 2-4.

**Exit criterion:**
- `python scripts/eval_checkpoint.py --checkpoint existing_ckpt --model ... --class-name ...` runs without errors (stubs return {}; JSON produced with empty eval sections)
- JSON is valid parseable structure; JSONL line appended

**Test:** `tests/test_scorecard.py::test_scorecard_json_schema`, `test_jsonl_index_append`

**Rollback:** Delete `scripts/eval_checkpoint.py`.

### Phase 1 exit gate

- `eval_checkpoint.py` runs end-to-end (all evaluators return empty), produces valid JSON + JSONL line
- Unit tests pass (2 tests so far)

---

## Phase 2 — Per-domain BPB + int4 (0.75 day)

### Task 2.1 — Validation splits

**Decision from spec open-question #1:** offset markers into existing `.bin` files (no duplicate data).

**File:** `evals/validation_splits/README.md` **NEW**

Content: describes the 2% tail of each .bin used as validation:

```markdown
# Validation splits

Each domain's validation slice is the last 2% of its `datasets/*.bin` file
(from the pre-tokenized odin-32k corpus).

| Domain | Source .bin | Tail size | Approx tokens |
|--------|-------------|----------:|--------------:|
| wikitext_val | datasets/wikitext-103-odin32k.bin | 2% | 2.5M |
| gpt_small_val | datasets/gpt-training-small-odin32k.bin | 2% | 6.0M |
| stem_crawl_val | datasets/stem-crawl-solo-odin32k.bin | 2% | 10.6M |
| dolma_val | datasets/dolma-10b-odin32k.bin | 2% | 140M |

The `per_domain_bpb` evaluator uses up to `--max-tokens-per-domain`
(default 50,000) tokens from each split. Split boundaries are computed
dynamically from file size — no separate split files are materialized.
```

**File:** `halo_training/eval/common.py`

**Change:** Update `default_validation_splits()` to return (path, offset_bytes, length_bytes) tuples instead of paths, so evaluators know which tail slice to memmap:

```python
def default_validation_splits():
    """Returns {domain: (path, offset_bytes, length_bytes)} for tail slices."""
    specs = {
        "wikitext_val": "datasets/wikitext-103-odin32k.bin",
        "gpt_small_val": "datasets/gpt-training-small-odin32k.bin",
        "stem_crawl_val": "datasets/stem-crawl-solo-odin32k.bin",
        "dolma_val": "datasets/dolma-10b-odin32k.bin",
    }
    result = {}
    for domain, path in specs.items():
        if not os.path.exists(path):
            result[domain] = None  # gracefully absent
            continue
        file_size = os.path.getsize(path)
        tail_bytes = int(file_size * 0.02)
        offset = file_size - tail_bytes
        result[domain] = (path, offset, tail_bytes)
    return result
```

**Exit criterion:**
- `default_validation_splits()` returns a dict with entries for all 4 domains (values are tuples or None for missing datasets)
- `wikitext_val` has tail slice ~2.5M tokens (~5MB at uint16)

**Test:** `tests/test_eval_common.py::test_validation_split_offsets`

**Rollback:** Revert to path-only return signature.

### Task 2.2 — Per-domain BPB evaluator

**File:** `halo_training/eval/per_domain_bpb.py` **NEW**

```python
"""Per-domain BPB evaluator."""

import math

import numpy as np
import torch
import torch.nn.functional as F


@torch.no_grad()
def run(model, tokenizer, validation_splits, args=None):
    """Compute BPB for each configured validation split.
    
    Returns {domain: bpb_float_or_none}.
    """
    max_tokens = getattr(args, "max_tokens_per_domain", 50_000)
    batch_size = 8
    block_size = 512

    results = {}
    for domain, spec in validation_splits.items():
        if spec is None:
            results[domain] = None
            continue
        path, offset, length_bytes = spec
        # memmap the tail slice
        mm = np.memmap(path, dtype=np.uint16, mode="r",
                        offset=offset, shape=(length_bytes // 2,))
        mm = mm[:max_tokens]
        if len(mm) < block_size * 2:
            results[domain] = None  # too small
            continue

        mean_ce = _compute_mean_ce(model, mm, batch_size, block_size)
        bpb = mean_ce / math.log(2) / _avg_bytes_per_token(tokenizer, mm[:5000])
        results[domain] = round(bpb, 4)

    return results


def _compute_mean_ce(model, tokens, batch_size, block_size):
    """Run forward passes over tokens in strided batches."""
    n_chunks = (len(tokens) - 1) // block_size
    losses = []
    for i in range(0, n_chunks, batch_size):
        batch_indices = range(i, min(i + batch_size, n_chunks))
        # Stack a batch of block_size+1 chunks
        batch = torch.stack([
            torch.from_numpy(tokens[j*block_size:(j+1)*block_size+1].astype(np.int64))
            for j in batch_indices
        ]).cuda()
        input_ids = batch[:, :-1]
        targets = batch[:, 1:]
        with torch.amp.autocast("cuda", dtype=torch.float16):
            logits = model(input_ids)
        if isinstance(logits, tuple):
            logits = logits[0]
        loss = F.cross_entropy(
            logits.reshape(-1, logits.size(-1)).float(),
            targets.reshape(-1),
            reduction="mean",
        )
        losses.append(loss.item())
    return sum(losses) / len(losses) if losses else float("nan")


def _avg_bytes_per_token(tokenizer, sample_tokens):
    """Estimate average bytes per token by decoding a sample."""
    decoded = tokenizer.decode(sample_tokens.tolist())
    return len(decoded.encode("utf-8")) / len(sample_tokens)
```

**Exit criterion:**
- `run(model, tokenizer, splits)` returns dict with valid BPB values for available domains, None for missing
- BPB values in reasonable range (1.5-3.5 for our 122M OdinFlat)

**Test:** `tests/test_per_domain_bpb.py::test_run_finite`, `test_handles_missing_split`

**Rollback:** Delete `per_domain_bpb.py`; `eval_checkpoint.py` will substitute a None placeholder.

### Task 2.3 — Int4 quantization evaluator

**File:** `halo_training/eval/int4_quant.py` **NEW**

```python
"""Int4 quantization evaluator — deployment-readiness BPB delta."""

import copy
import math

import torch
import torch.nn as nn

from halo_training.eval import per_domain_bpb


@torch.no_grad()
def run(model, tokenizer, validation_splits, args=None):
    """Quantize model to int4 (per-tensor symmetric), compute BPB per domain."""
    quantized = _quantize_model_int4(model)
    return per_domain_bpb.run(quantized, tokenizer, validation_splits, args=args)


def _quantize_model_int4(model):
    """Per-tensor symmetric int4 quantization of 2D Linear weights.

    Excludes tok_embeddings and lm_head (they stay fp16).
    Stores quantized weight as int8 (range [-8, 7]) plus scalar scale.
    Replaces module forward to dequantize on-the-fly.
    """
    quantized = copy.deepcopy(model)
    for name, module in quantized.named_modules():
        if not isinstance(module, nn.Linear):
            continue
        if "embed" in name or "lm_head" in name:
            continue
        w = module.weight.data
        if w.dim() < 2:
            continue
        scale = w.abs().max() / 7.0 + 1e-6
        q = torch.round(w / scale).clamp(-8, 7).to(torch.int8)
        module.weight = nn.Parameter(q.to(torch.int8), requires_grad=False)
        module.scale = float(scale.item())
        module.forward = _make_int4_forward(module)
    return quantized


def _make_int4_forward(module):
    def forward(x):
        dequant = module.weight.to(x.dtype) * module.scale
        bias = module.bias if module.bias is not None else None
        return torch.nn.functional.linear(x, dequant, bias)
    return forward
```

**Exit criterion:**
- `_quantize_model_int4` produces a model whose Linear weights are int8 with a `scale` attribute
- Quantized model's forward produces finite outputs
- `run()` returns dict with int4 BPBs within 20% of fp16 BPB per domain

**Test:** `tests/test_int4_quant.py::test_quantize_roundtrip`, `test_int4_bpb_sanity`

**Rollback:** Delete `int4_quant.py`; scorecard substitutes None.

### Phase 2 exit gate

- `scripts/eval_checkpoint.py` with `--skip-sampling --skip-inference-profile --skip-sample-pack --skip-activation-stats` produces a JSON with valid `per_domain_bpb` and `int4_quant` sections
- 4 unit tests passing

---

## Phase 3 — Sampling + inference profile (0.75 day)

### Task 3.1 — Refactor `scripts/ablate_odin_flat_sampling.py` to expose functions

**File:** `scripts/ablate_odin_flat_sampling.py`

**Change:** Extract the core sampling-ablation logic into pure functions that can be imported:

```python
# At top of scripts/ablate_odin_flat_sampling.py

def run_ablation(model, tokenizer, prompts, num_samples=3, max_tokens=120, seed=42):
    """Run the 3-stage sampling ablation. Returns all results."""
    # ... existing per-config sampling loop, but as a function ...


def select_winning_config(ablation_results):
    """Pick winner by highest distinct-2 among configs within 1.5x min self-PPL."""
    # ... existing selection logic ...


def main():  # existing CLI entry unchanged
    # ... parse args, call run_ablation, select_winning_config, print reports ...
```

Preserve the CLI entry point exactly as-is. Extract functions idempotently — existing invocations of `python scripts/ablate_odin_flat_sampling.py ...` continue to work.

**Exit criterion:**
- `from scripts.ablate_odin_flat_sampling import run_ablation, select_winning_config` succeeds
- Existing CLI invocation still works: `python scripts/ablate_odin_flat_sampling.py --checkpoint ... --model ...` produces identical output to before

**Test:** Existing manual invocation; `tests/test_sampling_eval.py::test_functions_importable`

**Rollback:** Revert `ablate_odin_flat_sampling.py` to its current state; `sampling.py` evaluator becomes a stub that returns None.

### Task 3.2 — Sampling evaluator

**File:** `halo_training/eval/sampling.py` **NEW**

```python
"""Sampling-quality evaluator."""

import sys
from pathlib import Path

# Ensure scripts/ is importable
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from scripts.ablate_odin_flat_sampling import run_ablation, select_winning_config


def run(model, tokenizer, validation_splits, args=None):
    """Run sampling ablation on a fixed 2-prompt set, extract metrics at winner."""
    prompts = [
        "The history of the Roman Empire",
        "In the field of physics,",
    ]
    ablation_results = run_ablation(
        model, tokenizer, prompts,
        num_samples=3, max_tokens=120, seed=42,
    )
    winner = select_winning_config(ablation_results)
    return {
        "winning_config": winner["config"],
        "distinct_2": winner["metrics"]["distinct_2"],
        "distinct_3": winner["metrics"].get("distinct_3"),
        "self_ppl": winner["metrics"]["self_ppl"],
        "sample_preview": winner["samples"][0][:100] if winner.get("samples") else None,
    }
```

**Exit criterion:** `run()` returns dict with valid distinct_2, self_ppl values matching ad-hoc invocation of the ablation script.

**Test:** `tests/test_sampling_eval.py::test_run_returns_metrics`

**Rollback:** Delete `sampling.py`; scorecard substitutes None.

### Task 3.3 — Inference profile evaluator

**File:** `halo_training/eval/inference_profile.py` **NEW**

```python
"""Inference throughput + peak memory evaluator."""

import time

import torch


@torch.no_grad()
def run(model, tokenizer, validation_splits, args=None):
    """Measure tok/s + peak memory at seq={256, 512, 1024} with batch=1."""
    seq_lengths = [256, 512, 1024]
    batch_size = 1
    warmup = 10
    measure = 30

    results = {}
    vocab_size = tokenizer.get_vocab_size()
    for seq_len in seq_lengths:
        prompt = torch.randint(0, vocab_size, (batch_size, seq_len), device="cuda")

        # Warmup
        for _ in range(warmup):
            with torch.amp.autocast("cuda", dtype=torch.float16):
                _ = model(prompt)
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()

        # Measure
        t0 = time.time()
        for _ in range(measure):
            with torch.amp.autocast("cuda", dtype=torch.float16):
                _ = model(prompt)
        torch.cuda.synchronize()
        elapsed = time.time() - t0

        tok_s = (batch_size * seq_len * measure) / elapsed
        peak_mem_gb = torch.cuda.max_memory_allocated() / 1e9

        results[f"tok_s_seq{seq_len}_bs{batch_size}"] = round(tok_s, 1)
        results[f"peak_mem_gb_seq{seq_len}"] = round(peak_mem_gb, 2)

    return results
```

**Exit criterion:**
- All 6 metrics (3 seq × {tok_s, peak_mem_gb}) finite and positive
- tok/s monotonically decreasing with seq_len (expected for causal attention)

**Test:** `tests/test_inference_profile.py::test_run_positive`, `test_monotonic_seq_len`

**Rollback:** Delete file; scorecard substitutes None.

### Phase 3 exit gate

- `eval_checkpoint.py` with `--skip-sample-pack --skip-activation-stats` produces a scorecard with `per_domain_bpb`, `int4_quant`, `sampling`, `inference` sections all populated
- 4 unit tests passing

---

## Phase 4 — Sample-pack + activation stats (0.75 day)

### Task 4.1 — Curate 20-prompt sample pack

**File:** `evals/sample_pack.txt` **NEW** (2-3 hours of prompt curation)

Target categories:
- History/biography narratives (3): probes long-range coherence
- Science/technical (3): probes domain knowledge
- Code/math (3): probes structural generation
- Dialogue/QA (3): probes instruction-following baseline
- Narrative fiction (3): probes creative generation
- Edge cases (5): rare entities, long lists, repetition-prone topics

Example (exactly 20 lines):
```
The history of the Roman Empire began in
In the field of particle physics,
A Python function to compute the Fibonacci sequence:
Q: What is the capital of Mongolia?
A:
Once upon a time, in a small village at the edge of the forest,
The solar system consists of
def binary_search(arr, target):
The recipe for a classic Margherita pizza calls for
During the Industrial Revolution,
To prove that the square root of 2 is irrational,
In Shakespeare's "Hamlet", the protagonist
The CRISPR-Cas9 system works by
An essay on climate change must address
The fundamental theorem of calculus states
Dialogue between a teacher and student about photosynthesis:
Teacher:
The four forces in nature are
A list of major European capital cities:
1. Paris
2.
To bake a sourdough bread, first
In the novel "1984" by George Orwell,
```

**Exit criterion:** File exists, contains exactly 20 non-empty prompts.

**Test:** `tests/test_sample_pack.py::test_prompts_load_20`

**Rollback:** Delete file; `sample_pack.py` substitutes a 5-prompt fallback.

### Task 4.2 — Sample-pack evaluator

**File:** `halo_training/eval/sample_pack.py` **NEW**

```python
"""Sample-pack regression evaluator."""

import hashlib
import re
from pathlib import Path

import torch

from halo_training.eval.sampling import (
    run_ablation,  # reuse sampling harness for generation primitives
)


def run(model, tokenizer, validation_splits, args=None):
    """Generate 20 samples; hash; optionally diff vs prior checkpoint."""
    prompts_file = getattr(args, "prompts_file", "evals/sample_pack.txt")
    prior_ckpt = getattr(args, "prior_checkpoint", None)
    if prior_ckpt is None and args is not None:
        prior_ckpt = _auto_detect_prior(args.checkpoint)

    prompts = _load_prompts(prompts_file)
    seed = 42
    max_tokens = 100

    samples = []
    for i, prompt in enumerate(prompts):
        torch.manual_seed(seed + i)
        sample = _generate(model, tokenizer, prompt, max_tokens,
                          temperature=0.8, top_p=0.95)
        samples.append(sample)

    combined = "\n---\n".join(samples)
    output_hash = "sha256:" + hashlib.sha256(combined.encode()).hexdigest()[:16]

    diff_info = None
    if prior_ckpt is not None:
        prior_hash, prior_samples = _load_or_regenerate_prior(prior_ckpt, prompts)
        if prior_hash != output_hash:
            diff_info = _compute_diff(samples, prior_samples)

    return {
        "prompts_file": prompts_file,
        "num_prompts": len(prompts),
        "output_hash": output_hash,
        "diff_vs_prior": diff_info,
    }


def _load_prompts(path):
    with open(path) as f:
        return [line.strip() for line in f if line.strip()]


def _auto_detect_prior(checkpoint_path):
    """Find the most recent sibling step_*.pt with smaller step number."""
    p = Path(checkpoint_path)
    m = re.match(r"step_(\d+)\.pt", p.name)
    if not m:
        return None
    current_step = int(m.group(1))
    siblings = sorted(p.parent.glob("step_*.pt"))
    priors = [s for s in siblings
              if (sm := re.match(r"step_(\d+)\.pt", s.name)) and int(sm.group(1)) < current_step]
    return str(priors[-1]) if priors else None


def _generate(model, tokenizer, prompt, max_tokens, temperature, top_p):
    """Deterministic generation given current torch RNG state."""
    # ... standard autoregressive decode using tokenizer + model ...
    # Implementation reuses logic from sample_odin_flat.py
    pass  # (full impl ~40 LoC)


def _load_or_regenerate_prior(prior_ckpt, prompts):
    """Check if prior's samples were cached; if not, generate them on the fly."""
    # Cache path: docs/perf/eval-scorecards/<prior_ckpt_name>-sample-pack.txt
    # If cache exists: return cached hash + samples
    # Otherwise: load prior checkpoint into a fresh model, generate, cache, return
    # This is expensive (second model load) but only happens when diff_vs_prior is requested
    pass  # (full impl ~30 LoC)


def _compute_diff(samples, prior_samples):
    """Compare sample-by-sample, count changed prompts, emit preview."""
    diff_count = sum(1 for a, b in zip(samples, prior_samples) if a != b)
    preview = []
    for i, (a, b) in enumerate(zip(samples, prior_samples)):
        if a != b:
            preview.append(f"prompt {i}: {_diff_preview(a, b)}")
            if len(preview) >= 3:
                break
    return {"diff_count": diff_count, "total": len(samples), "preview": preview}
```

**Exit criterion:**
- `run()` with no prior checkpoint produces valid hash and `diff_vs_prior=None`
- `run()` with a prior checkpoint from same run produces a diff_info dict
- Re-running with same model + same seed produces identical hash

**Test:** `tests/test_sample_pack.py::test_prompts_load_20`, `test_output_deterministic`

**Rollback:** Delete file; scorecard substitutes None.

### Task 4.3 — Activation stats evaluator

**File:** `halo_training/eval/activation_stats.py` **NEW**

```python
"""Per-layer activation diagnostics evaluator."""

from collections import defaultdict

import numpy as np
import torch


@torch.no_grad()
def run(model, tokenizer, validation_splits, args=None, num_batches=10, seq_len=512, batch_size=8):
    """Collect activation stats over num_batches random validation batches."""
    # Select a validation source for batches
    primary_split = validation_splits.get("wikitext_val")
    if primary_split is None:
        # fallback to another domain
        for key in ["gpt_small_val", "stem_crawl_val", "dolma_val"]:
            if validation_splits.get(key) is not None:
                primary_split = validation_splits[key]
                break
    if primary_split is None:
        return None  # no validation data available

    path, offset, length_bytes = primary_split
    mm = np.memmap(path, dtype=np.uint16, mode="r",
                    offset=offset, shape=(length_bytes // 2,))

    collector = _ActivationCollector(model)
    try:
        for batch_idx in range(num_batches):
            # Sample a random start within the split
            start = np.random.randint(0, len(mm) - batch_size * seq_len - 1)
            batch = torch.stack([
                torch.from_numpy(mm[start + i*seq_len:start + (i+1)*seq_len].astype(np.int64))
                for i in range(batch_size)
            ]).cuda()
            with torch.amp.autocast("cuda", dtype=torch.float16):
                _ = model(batch)

        return {
            "layer_kurtosis": [
                _mean_kurtosis(collector.layer_acts[i])
                for i in range(len(model.layers))
            ],
            "layer_rms_norm": [
                _mean_rms(collector.layer_acts[i])
                for i in range(len(model.layers))
            ],
            "attention_head_entropy_mean": _mean_attn_entropy(collector.attn_probs),
        }
    finally:
        collector.remove()  # critical: no dangling hooks


class _ActivationCollector:
    def __init__(self, model):
        self.layer_acts = defaultdict(list)
        self.attn_probs = defaultdict(list)
        self.hooks = []
        for i, layer in enumerate(model.layers):
            h = layer.register_forward_hook(
                lambda m, inp, out, i=i: self._capture_layer_out(i, out)
            )
            self.hooks.append(h)

    def remove(self):
        for h in self.hooks:
            h.remove()
        self.hooks = []

    def _capture_layer_out(self, i, out):
        # out is the layer output tensor or tuple
        if isinstance(out, tuple):
            out = out[0]
        # Store a CPU float copy so GPU memory isn't pinned
        self.layer_acts[i].append(out.detach().float().cpu())


def _mean_kurtosis(acts_list):
    """Compute mean kurtosis across a list of activation tensors."""
    vals = []
    for a in acts_list:
        # Flatten; kurtosis = E[((x - mean)/std)^4] - 3
        flat = a.flatten()
        std = flat.std().item() + 1e-8
        centered = (flat - flat.mean()) / std
        kurt = (centered.pow(4).mean() - 3.0).item()
        vals.append(kurt)
    return round(sum(vals) / len(vals), 3) if vals else None


def _mean_rms(acts_list):
    """Mean RMS norm per layer."""
    vals = [float(a.pow(2).mean().sqrt()) for a in acts_list]
    return round(sum(vals) / len(vals), 3) if vals else None


def _mean_attn_entropy(attn_probs_by_layer):
    """Mean entropy of attention distributions (if captured)."""
    all_entropy = []
    for layer_probs in attn_probs_by_layer.values():
        for probs in layer_probs:
            # entropy per position, then mean
            H = -(probs * probs.clamp_min(1e-9).log()).sum(dim=-1).mean().item()
            all_entropy.append(H)
    return round(sum(all_entropy) / len(all_entropy), 3) if all_entropy else None
```

**Exit criterion:**
- `run()` returns dict with 14-length `layer_kurtosis`, `layer_rms_norm` lists (matching OdinFlat's 14 layers)
- All values finite
- After `run()`, `gc.collect()`; no dangling hook references to the model

**Test:** `tests/test_activation_stats.py::test_hooks_no_leak`

**Rollback:** Delete file; scorecard substitutes None.

### Phase 4 exit gate

- `eval_checkpoint.py` produces a complete scorecard JSON (all sections populated or null-gracefully)
- 8 unit tests total passing

---

## Phase 5 — `--auto-eval` trainer hook (0.5 day)

### Task 5.1 — Add `--auto-eval` flag

**File:** `halo_training/cli.py` + `scripts/train_ddp.py`

**Change:** Add `--auto-eval` flag; when set, after each `save_checkpoint`, spawn `scripts/eval_checkpoint.py` as a detached subprocess.

```python
# In scripts/train_ddp.py::save_checkpoint (and halo_training/trainer.py equivalent)

def save_checkpoint(model, optimizer, step, checkpoint_dir, total_tokens=0, auto_eval=False):
    # ... existing save logic ...
    path = os.path.join(checkpoint_dir, f"step_{step}.pt")
    torch.save({...}, path)
    print(f"[rank 0] Checkpoint saved: {path}")

    if auto_eval:
        _spawn_auto_eval(path, args.model, args.class_name)


def _spawn_auto_eval(checkpoint_path, model_path, class_name):
    """Spawn eval_checkpoint.py as a detached subprocess. Log captured to file."""
    eval_log = f"{checkpoint_path}.eval.log"
    cmd = [
        "python3", "scripts/eval_checkpoint.py",
        "--checkpoint", checkpoint_path,
        "--model", model_path,
        "--class-name", class_name,
    ]
    import subprocess
    with open(eval_log, "w") as lf:
        subprocess.Popen(
            cmd, stdout=lf, stderr=subprocess.STDOUT,
            start_new_session=True,  # detached; doesn't block trainer
        )
    print(f"[rank 0] Auto-eval spawned; log → {eval_log}")
```

**Exit criterion:**
- `--auto-eval` flag visible in `--help`
- When set, subprocess spawns on each checkpoint save
- Trainer continues without waiting
- Subprocess log file appears at `<checkpoint>.eval.log`

**Test:** `tests/test_auto_eval.py::test_subprocess_spawns`

**Rollback:** Remove `--auto-eval` flag; subprocess never spawned; default behavior unchanged.

### Task 5.2 — Smoke test auto-eval integration

**Command:**
```bash
bash run_remote.sh "cd ~/Desktop/ai_lab/autokernel-halo-strix && \
  MODEL=models/odin_flat.py CLASS=OdinFlat \
  DATASET=datasets/wikitext-103-odin32k.bin \
  CKPT_DIR=checkpoints/sprint2-smoke \
  EPOCHS=1 \
  EXTRA_FLAGS='--max-steps 200 --auto-eval' \
  bash scripts/launch_ddp.sh"
```

**Exit criterion:**
- Training runs 200 steps without regression in throughput/loss vs no-auto-eval baseline
- At least one scorecard JSON appears in `docs/perf/eval-scorecards/` (for the checkpoint at step 200)
- `<checkpoint>.eval.log` exists and shows successful eval run

**Test:** Manual (integration test outside unit-test suite).

**Rollback:** Remove `--auto-eval` from invocation; manual scorecard runs unaffected.

### Phase 5 exit gate

- `--auto-eval` works end-to-end
- 9 unit tests total passing (kept counting as we add tests)

---

## Phase 6 — Retroactive validation (0.25 day)

### Task 6.1 — Retroactive run on existing checkpoints

**Command (on Machine B):**
```bash
export EVAL_MACHINE=b
bash run_remote_b.sh "cd ~/Desktop/comfyui-rocm7.12/autokernel-halo-strix && \
  python3 scripts/eval_checkpoint.py \
    --checkpoint checkpoints/odin-flat-wikitext-ddp/step_1869.pt \
    --model models/odin_flat.py --class-name OdinFlat"
```

```bash
bash run_remote_b.sh "cd ~/Desktop/comfyui-rocm7.12/autokernel-halo-strix && \
  python3 scripts/eval_checkpoint.py \
    --checkpoint checkpoints/odin-halo-wikitext-ddp/step_1869.pt \
    --model models/odin_halo.py --class-name OdinHalo"
```

**Exit criterion:**
- Both runs complete without errors
- Both JSON files produced under `docs/perf/eval-scorecards/`
- Both JSONL lines appended
- All metric values finite and sensible:
  - wikitext BPB for OdinFlat ≈ 1.79 (matches training loss 4.47 / log(2))
  - distinct_2 for OdinFlat winning config ≈ 0.765 (matches earlier sampling ablation)
  - inference tok/s > 50 at seq=512 for 122M model
  - peak memory < 3 GB at seq=1024

**Gate (Sprint 2 → ready):** All exit criteria met on both checkpoints.

### Task 6.2 — Machine parity check

Run the same checkpoint on both machines:
```bash
EVAL_MACHINE=a python scripts/eval_checkpoint.py --checkpoint ... --output-dir docs/perf/parity-a/
EVAL_MACHINE=b python scripts/eval_checkpoint.py --checkpoint ... --output-dir docs/perf/parity-b/
# Diff the two JSON files
```

**Exit criterion:**
- Per-domain BPB values match within ±5% between machines
- tok/s values match within ±10% (aiter may cause some drift)
- distinct_2 and self_ppl match within ±5%

**Gate failure response:** Investigate source of drift; document any persistent variation; if drift is >5% systematically, restrict scorecard to one machine and document.

### Phase 6 exit gate

- Retroactive run on 2 representative checkpoints passes
- Machine parity verified
- Wall time measured ≤ 15 min per run on either machine

---

## Phase 7 — Cleanup & commit (0.5 day)

### Task 7.1 — Update STATUS.md

Add a new section:
```
### Sprint 2: Evaluation Scorecard (YYYY-MM-DD)

Spec: docs/superpowers/specs/2026-05-06-sprint2-eval-scorecard-design.md
Plan: docs/superpowers/plans/2026-05-06-sprint2-eval-scorecard-plan.md

Infrastructure delivered:
  - scripts/eval_checkpoint.py (per-checkpoint scorecard CLI)
  - halo_training/eval/ (7 evaluator modules)
  - evals/sample_pack.txt (20 curated prompts)
  - --auto-eval trainer flag

Retroactive scorecard on existing checkpoints:
  OdinFlat wikitext step_1869:
    BPB:    wikitext_val=1.79, gpt_small_val=X.XX, stem_crawl_val=X.XX
    int4 BPB delta: +X.XX
    Sampling: winning config ..., distinct_2=0.765, self_ppl=9.84
    Inference: tok/s @ 512 = XXX, peak_mem_gb @ 1024 = X.XX

  OdinHalo wikitext step_1869:
    ... similar table ...

Machine parity: ±X% drift between A and B
Wall time: X.X min per scorecard
Gate C → B: cleared. Ready for Sprint 3 (dolma T²-optimal training).
```

### Task 7.2 — Update AGENTS.md

"Training gotchas" and new section:
```
- **Checkpoint evaluation:** use `scripts/eval_checkpoint.py --checkpoint <path>` for
  per-checkpoint scorecard. Outputs:
    docs/perf/eval-scorecards/<name>.json  (full metrics)
    docs/perf/eval-scorecard.jsonl         (trend index)

- **Automatic evaluation during training:** pass `--auto-eval` to the trainer or
  `EXTRA_FLAGS="--auto-eval"` to launch_ddp.sh. Fires detached subprocess per
  checkpoint save; does not block training.

- **Machine selection:** set `EVAL_MACHINE={a,b}` env var to label eval machine in
  scorecard output. Use Machine B preferentially if training is running on Machine A.
```

### Task 7.3 — Update knowledge/INDEX.md

Add new entry under `training/` or a new `eval/` section:
```
## eval/

| File | Description |
|------|-------------|
| [../scripts/eval_checkpoint.py](...) | Sprint 2 scorecard CLI; writes per-checkpoint JSON + JSONL index |
| [../halo_training/eval/](...) | 7 evaluator modules: per-domain BPB, int4 quant, sampling, inference profile, sample-pack, activation stats |
```

### Task 7.4 — Single commit

```
Sprint 2: Evaluation scorecard infrastructure

Delivers per-checkpoint multi-dimensional evaluation:
  - Per-domain BPB (wikitext, gpt-small, stem-crawl, dolma validation splits)
  - Int4 quantized BPB (deployment-readiness delta)
  - Sampling quality (distinct-2/3, self-PPL at winning config, via
    wraps of ablate_odin_flat_sampling.py)
  - Inference tok/s + peak memory at seq={256, 512, 1024}
  - Sample-pack regression (20 curated prompts, deterministic hash + diff)
  - Layer-wise activation statistics (kurtosis, RMS, attention entropy)

CLI:
  python scripts/eval_checkpoint.py --checkpoint <path> \
    --model <file.py> --class-name <class>

Auto-trigger:
  python -m halo_training --auto-eval ...           # single-node
  EXTRA_FLAGS='--auto-eval' bash scripts/launch_ddp.sh  # DDP

Outputs:
  docs/perf/eval-scorecards/<name>.json  (per-checkpoint full)
  docs/perf/eval-scorecard.jsonl         (one-line-per-run index)

Retroactive validation on OdinFlat + OdinHalo wikitext checkpoints
produced finite, consistent metrics. Machine parity within ±X%.
Wall time ≤ X min per scorecard.

Does NOT include lm-evaluation-harness benchmarks (HellaSwag, ARC,
MMLU, PIQA, BLiMP). Those are deferred to Sprint 4 when paired with
instruction-following benchmarks (MT-Bench, AlpacaEval) that require
a post-trained model.

Spec: docs/superpowers/specs/2026-05-06-sprint2-eval-scorecard-design.md
Plan: docs/superpowers/plans/2026-05-06-sprint2-eval-scorecard-plan.md
```

---

## Summary timeline

| Day | Phase | Activity |
|:---:|-------|----------|
| 1 | Phase 1 + 2.1-2.2 | Scaffolding + per-domain BPB |
| 2 | Phase 2.3 + Phase 3 | int4 + sampling + inference profile |
| 3 | Phase 4 | Sample-pack curation + activation stats |
| 3.5 | Phase 5 | `--auto-eval` trainer hook |
| 4 | Phase 6 + 7 | Retroactive validation + docs + commit |

**Total: ~4 elapsed days, minimal compute (< 1 hour of eval runs).**

## Dependencies external to Sprint 2

- Existing checkpoints (`odin-flat-wikitext-ddp/step_1869.pt`, `odin-halo-wikitext-ddp/step_1869.pt`) for validation
- Existing `scripts/ablate_odin_flat_sampling.py` (refactored during Phase 3)
- Existing tokenizer at `tokenizers/odin-32k/tokenizer.json`
- Machines A and B both available during Phase 6 (~15 min parity check)

## Handoff after Sprint 2

Sprint 2 completion unlocks:

- **Sprint 1.5 (conditional):** can use scorecard as its ablation measurement harness
- **Sprint 3:** T²-optimal dolma training runs with `--auto-eval` enabled; per-checkpoint scorecards accumulate during the 50-hour training, giving the team visibility long before the run completes
- **Sprint 4:** post-training pipeline extends the scorecard with lm-evaluation-harness + instruction-following benchmarks

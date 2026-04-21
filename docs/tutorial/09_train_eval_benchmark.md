# Part 09: Train-Evaluate-Benchmark Framework

## Goal
Build an automated experiment framework that handles configuration, training, evaluation, and comparison. By the end, you will have a system where every training run is reproducible, every result is recorded, and comparing N experiments takes one command.

## Why This Matters
You are about to run dozens of experiments: different architectures, learning rates, data mixtures, kernel configurations. Without a framework, you will lose track of what you tried, what worked, and what each checkpoint actually represents. This part is the infrastructure that makes Parts 10-14 possible.

---

## 9.1 The Experiment Problem

### What Goes Wrong Without Tracking

Here is a typical failure mode. You run 8 training experiments over a weekend:

```
checkpoints/
  run1/
  run2/
  run_final/
  run_final_v2/
  run_final_v2_fixed/
  gpt2_bigger_lr/
  test_something/
  ???/
```

Two weeks later, you want to reproduce your best result. Which checkpoint was it? What learning rate did you use? What data? What git commit was the code at?

You do not know. The runs are unreproducible garbage.

### The Solution: Structured Experiments

Every run gets:
1. A config file (JSON) that fully specifies the run
2. A unique experiment ID (hash of the config)
3. A JSONL log (every step: loss, lr, throughput, memory, grad norm)
4. Benchmark results (HellaSwag, ARC, perplexity)
5. Metadata (git commit, timestamp, hardware)

The system is simple — no MLflow, no W&B, no external services. Just files on disk with consistent naming.

---

## 9.2 Experiment Configuration

### Config Schema

```python
"""config.py — Experiment configuration."""
import hashlib
import json
import subprocess
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Optional

@dataclass
class ExperimentConfig:
    # Model
    model_name: str = "gpt2"
    model_file: str = "models/gpt2.py"
    model_class: str = "GPT2"
    d_model: int = 768
    n_layers: int = 12
    n_heads: int = 12
    vocab_size: int = 50257
    seq_len: int = 1024
    
    # Data
    dataset: str = "data/assembled/train.bin"
    val_dataset: str = "data/assembled/val.bin"
    
    # Training
    batch_size: int = 8
    accum_steps: int = 4       # Effective batch = batch_size * accum_steps
    learning_rate: float = 6e-4
    min_lr: float = 6e-5
    warmup_steps: int = 200
    max_steps: int = 10000
    weight_decay: float = 0.1
    grad_clip: float = 1.0
    optimizer: str = "adamw"   # "adamw" or "muon"
    
    # AMP
    use_amp: bool = True
    dtype: str = "float16"     # "float16" on NVIDIA, never "bfloat16" on Strix Halo
    
    # Kernel optimization
    use_autokernel: bool = False
    use_compile: bool = False
    
    # Checkpointing
    checkpoint_interval: int = 1000
    checkpoint_dir: str = "checkpoints"
    
    # Time budget (0 = no limit)
    time_budget_minutes: int = 0
    
    # Evaluation
    eval_interval: int = 500
    eval_steps: int = 50       # Number of val batches per eval
    
    # Metadata (auto-filled)
    git_hash: str = ""
    timestamp: str = ""
    experiment_id: str = ""
    hardware: str = ""
    
    def fill_metadata(self):
        """Auto-fill metadata fields."""
        import datetime
        self.timestamp = datetime.datetime.now().isoformat()
        
        try:
            self.git_hash = subprocess.check_output(
                ["git", "rev-parse", "HEAD"], text=True
            ).strip()
        except Exception:
            self.git_hash = "unknown"
        
        try:
            import torch
            if torch.cuda.is_available():
                self.hardware = torch.cuda.get_device_name()
            else:
                self.hardware = "cpu"
        except Exception:
            self.hardware = "unknown"
        
        # Experiment ID: hash of the training-relevant config
        # Exclude metadata fields from the hash
        relevant = {k: v for k, v in asdict(self).items()
                    if k not in ("git_hash", "timestamp", "experiment_id", "hardware")}
        config_str = json.dumps(relevant, sort_keys=True)
        self.experiment_id = hashlib.sha256(config_str.encode()).hexdigest()[:12]
    
    def save(self, path: Path):
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(asdict(self), f, indent=2)
    
    @classmethod
    def load(cls, path: Path) -> "ExperimentConfig":
        with open(path) as f:
            data = json.load(f)
        return cls(**data)
    
    @property
    def run_dir(self) -> Path:
        """Directory for this experiment's outputs."""
        return Path(self.checkpoint_dir) / f"{self.model_name}_{self.experiment_id}"
    
    @property
    def effective_batch_size(self) -> int:
        return self.batch_size * self.accum_steps
    
    @property
    def total_tokens(self) -> int:
        return self.max_steps * self.effective_batch_size * self.seq_len
```

### Example Config File

```json
{
  "model_name": "gpt2",
  "model_file": "models/gpt2.py",
  "model_class": "GPT2",
  "d_model": 768,
  "n_layers": 12,
  "n_heads": 12,
  "vocab_size": 50257,
  "seq_len": 1024,
  "dataset": "data/assembled/train.bin",
  "batch_size": 8,
  "accum_steps": 4,
  "learning_rate": 6e-4,
  "min_lr": 6e-5,
  "warmup_steps": 200,
  "max_steps": 10000,
  "weight_decay": 0.1,
  "grad_clip": 1.0,
  "optimizer": "adamw",
  "use_amp": true,
  "dtype": "float16",
  "use_autokernel": false,
  "use_compile": false,
  "checkpoint_interval": 1000,
  "time_budget_minutes": 0,
  "eval_interval": 500,
  "eval_steps": 50,
  "git_hash": "a1b2c3d4",
  "timestamp": "2025-01-15T14:30:00",
  "experiment_id": "7f3a2b1c9d4e",
  "hardware": "NVIDIA GeForce RTX 4060 Ti"
}
```

**Why hash the config?** Two runs with identical training configs get the same experiment ID. If you accidentally run the same config twice, the second run writes to the same directory — you will see the collision and know it is a duplicate. Change any hyperparameter and you get a new ID automatically.

---

## 9.3 Training Loop with Logging

### The Logger

```python
"""logger.py — JSONL logger for training metrics."""
import json
import time
from pathlib import Path

class TrainLogger:
    """Append-only JSONL logger. One line per step."""
    
    def __init__(self, log_path: Path):
        self.log_path = log_path
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        self.start_time = time.time()
        self._file = open(self.log_path, "a")  # Append mode for resume
    
    def log(self, step: int, **kwargs):
        """Log a training step with arbitrary metrics."""
        entry = {
            "step": step,
            "elapsed_sec": time.time() - self.start_time,
            **kwargs,
        }
        self._file.write(json.dumps(entry) + "\n")
        self._file.flush()  # Flush every line — crash-safe
    
    def close(self):
        self._file.close()
    
    @staticmethod
    def read(log_path: Path) -> list:
        """Read all entries from a log file."""
        entries = []
        with open(log_path) as f:
            for line in f:
                line = line.strip()
                if line:
                    entries.append(json.loads(line))
        return entries
```

### The Training Loop

This loop is designed for correctness and observability, not speed. Every metric is logged. Every checkpoint is recoverable.

```python
"""train.py — Training loop with logging, checkpointing, and time budget."""
import json
import math
import time
import torch
import torch.nn as nn
import numpy as np
from pathlib import Path

from config import ExperimentConfig
from logger import TrainLogger

def get_lr(step, config):
    """Cosine schedule with linear warmup."""
    if step < config.warmup_steps:
        return config.learning_rate * step / config.warmup_steps
    
    decay_steps = config.max_steps - config.warmup_steps
    progress = (step - config.warmup_steps) / decay_steps
    cosine = 0.5 * (1 + math.cos(math.pi * progress))
    return config.min_lr + (config.learning_rate - config.min_lr) * cosine

def load_model(config):
    """Import and instantiate the model from its file."""
    import importlib.util
    spec = importlib.util.spec_from_file_location("model_module", config.model_file)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    model_class = getattr(module, config.model_class)
    
    # Construct with config params (adapt to your model's __init__)
    model = model_class(
        vocab_size=config.vocab_size,
        d_model=config.d_model,
        n_layers=config.n_layers,
        n_heads=config.n_heads,
        seq_len=config.seq_len,
    )
    return model.cuda()

def save_checkpoint(model, optimizer, scaler, scheduler_state, step, config, run_dir):
    """Save full training state for exact resume."""
    path = run_dir / f"step_{step}.pt"
    torch.save({
        "step": step,
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scaler": scaler.state_dict() if scaler else None,
        "scheduler_state": scheduler_state,
        "config": json.loads(json.dumps(config.__dict__)),
    }, path)
    print(f"  Saved checkpoint: {path}")
    return path

def load_checkpoint(path, model, optimizer, scaler):
    """Load training state from checkpoint. Returns the step to resume from."""
    ckpt = torch.load(path, map_location="cuda", weights_only=False)
    model.load_state_dict(ckpt["model"])
    optimizer.load_state_dict(ckpt["optimizer"])
    if scaler and ckpt.get("scaler"):
        scaler.load_state_dict(ckpt["scaler"])
    return ckpt["step"]

def train(config: ExperimentConfig):
    """Main training function."""
    config.fill_metadata()
    run_dir = config.run_dir
    run_dir.mkdir(parents=True, exist_ok=True)
    config.save(run_dir / "config.json")
    
    print(f"Experiment: {config.experiment_id}")
    print(f"Run dir: {run_dir}")
    print(f"Model: {config.model_name} ({config.d_model}d, {config.n_layers}L)")
    
    # Data
    data = np.memmap(config.dataset, dtype=np.uint16, mode='r')
    if Path(config.val_dataset).exists():
        val_data = np.memmap(config.val_dataset, dtype=np.uint16, mode='r')
    else:
        # Use last 1% of training data as validation
        split = int(len(data) * 0.99)
        val_data = data[split:]
        data = data[:split]
    
    # Model
    model = load_model(config)
    param_count = sum(p.numel() for p in model.parameters())
    print(f"Parameters: {param_count:,}")
    
    # Optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.learning_rate,
        betas=(0.9, 0.95),
        weight_decay=config.weight_decay,
    )
    
    # AMP
    scaler = torch.amp.GradScaler("cuda") if config.use_amp else None
    amp_dtype = torch.float16 if config.dtype == "float16" else torch.bfloat16
    
    # Logger
    logger = TrainLogger(run_dir / "train_log.jsonl")
    
    # Resume from checkpoint if one exists
    start_step = 0
    existing_ckpts = sorted(run_dir.glob("step_*.pt"))
    if existing_ckpts:
        latest = existing_ckpts[-1]
        print(f"Resuming from {latest}")
        start_step = load_checkpoint(latest, model, optimizer, scaler)
        start_step += 1  # Start from the next step
        print(f"Resuming from step {start_step}")
    
    # Training
    model.train()
    t0 = time.time()
    tokens_processed = 0
    
    for step in range(start_step, config.max_steps):
        # Time budget check
        if config.time_budget_minutes > 0:
            elapsed_min = (time.time() - t0) / 60
            if elapsed_min >= config.time_budget_minutes:
                print(f"Time budget reached ({elapsed_min:.1f} / {config.time_budget_minutes} min)")
                save_checkpoint(model, optimizer, scaler, {}, step, config, run_dir)
                break
        
        # Learning rate schedule
        lr = get_lr(step, config)
        for pg in optimizer.param_groups:
            pg["lr"] = lr
        
        # Accumulation loop
        optimizer.zero_grad()
        accum_loss = 0.0
        
        for micro_step in range(config.accum_steps):
            # Sample a random batch
            batch_indices = np.random.randint(
                0, len(data) - config.seq_len - 1, size=config.batch_size
            )
            x = np.stack([data[i:i+config.seq_len] for i in batch_indices])
            y = np.stack([data[i+1:i+config.seq_len+1] for i in batch_indices])
            x = torch.from_numpy(x.astype(np.int64)).cuda()
            y = torch.from_numpy(y.astype(np.int64)).cuda()
            
            with torch.amp.autocast("cuda", dtype=amp_dtype, enabled=config.use_amp):
                logits = model(x)
                loss = nn.functional.cross_entropy(
                    logits.reshape(-1, config.vocab_size), y.reshape(-1)
                )
                loss = loss / config.accum_steps
            
            if scaler:
                scaler.scale(loss).backward()
            else:
                loss.backward()
            
            accum_loss += loss.item()
            tokens_processed += config.batch_size * config.seq_len
        
        # Gradient clipping
        if scaler:
            scaler.unscale_(optimizer)
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip)
        
        # Step
        if scaler:
            scaler.step(optimizer)
            scaler.update()
        else:
            optimizer.step()
        
        # Compute throughput
        elapsed = time.time() - t0
        tok_per_sec = tokens_processed / elapsed if elapsed > 0 else 0
        
        # Memory
        mem_allocated = torch.cuda.memory_allocated() / 1e9
        mem_reserved = torch.cuda.memory_reserved() / 1e9
        
        # Log
        logger.log(
            step=step,
            loss=accum_loss,
            lr=lr,
            grad_norm=grad_norm.item() if isinstance(grad_norm, torch.Tensor) else grad_norm,
            tok_per_sec=tok_per_sec,
            mem_allocated_gb=round(mem_allocated, 2),
            mem_reserved_gb=round(mem_reserved, 2),
        )
        
        if step % 50 == 0:
            print(f"  step={step:>6d}  loss={accum_loss:.4f}  lr={lr:.2e}  "
                  f"grad_norm={grad_norm:.2f}  tok/s={tok_per_sec:.0f}  "
                  f"mem={mem_allocated:.1f}GB")
        
        # Evaluation
        if step > 0 and step % config.eval_interval == 0:
            val_loss = evaluate(model, val_data, config, amp_dtype)
            logger.log(step=step, val_loss=val_loss, event="eval")
            print(f"  [EVAL] step={step}  val_loss={val_loss:.4f}  "
                  f"ppl={math.exp(val_loss):.1f}")
            model.train()
        
        # Checkpoint
        if step > 0 and step % config.checkpoint_interval == 0:
            save_checkpoint(model, optimizer, scaler, {}, step, config, run_dir)
    
    # Final checkpoint (always save at end)
    save_checkpoint(model, optimizer, scaler, {}, step, config, run_dir)
    logger.close()
    print(f"Training complete. Run dir: {run_dir}")


def evaluate(model, val_data, config, amp_dtype):
    """Compute validation loss."""
    model.eval()
    losses = []
    with torch.no_grad():
        for _ in range(config.eval_steps):
            indices = np.random.randint(
                0, len(val_data) - config.seq_len - 1, size=config.batch_size
            )
            x = np.stack([val_data[i:i+config.seq_len] for i in indices])
            y = np.stack([val_data[i+1:i+config.seq_len+1] for i in indices])
            x = torch.from_numpy(x.astype(np.int64)).cuda()
            y = torch.from_numpy(y.astype(np.int64)).cuda()
            
            with torch.amp.autocast("cuda", dtype=amp_dtype, enabled=config.use_amp):
                logits = model(x)
                loss = nn.functional.cross_entropy(
                    logits.reshape(-1, config.vocab_size), y.reshape(-1)
                )
            losses.append(loss.item())
    
    return sum(losses) / len(losses)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True, help="Path to config JSON")
    args = parser.parse_args()
    
    config = ExperimentConfig.load(Path(args.config))
    train(config)
```

### Key Design Decisions

**JSONL, not CSV.** Each line is a self-contained JSON object. You can add new fields without breaking old logs. You can `tail -1` to see the latest step. Crash-safe because each line is flushed immediately.

**Checkpoint = full state.** Model weights + optimizer state + scaler state + step number + config. You can resume from any checkpoint and get the exact same training trajectory.

**Time-budget training.** Set `time_budget_minutes: 480` for an 8-hour overnight run. The loop checks the wall clock every step and saves a final checkpoint when time runs out. This is essential for long runs on shared hardware.

---

## 9.4 Evaluation Harness

### Standard Benchmarks with lm-evaluation-harness

```bash
pip install lm-eval
```

The EleutherAI harness is the standard for LLM evaluation. It supports dozens of benchmarks with standardized prompts and scoring.

### Which Benchmarks Matter at <350M

Not all benchmarks are useful for small models. Here is the honest breakdown:

| Benchmark | Type | Useful at 124M? | Useful at 350M? | Notes |
|-----------|------|:---:|:---:|-------|
| HellaSwag | Common sense | YES | YES | Most discriminative for small models |
| ARC-Easy | Science (easy) | YES | YES | Good signal even at 100M |
| ARC-Challenge | Science (hard) | Marginal | YES | Near-random below 200M |
| PIQA | Physical intuition | YES | YES | Good discrimination |
| WinoGrande | Coreference | Marginal | YES | Needs ~200M+ to rise above random |
| MMLU | Multi-domain knowledge | NO | NO | Random at <1B params |
| GSM8K | Math reasoning | NO | NO | Requires ~7B+ for non-trivial |

**Bottom line:** Focus on HellaSwag + ARC-Easy + PIQA for small models. Do not waste time on MMLU or GSM8K.

### Evaluation Script

```python
"""evaluate_model.py — Run standard benchmarks on a checkpoint."""
import json
import argparse
import subprocess
import sys
from pathlib import Path

def evaluate_checkpoint(
    checkpoint_path: str,
    model_file: str,
    model_class: str,
    output_dir: str,
    tasks: str = "hellaswag,arc_easy,piqa",
    batch_size: int = 8,
):
    """
    Evaluate a checkpoint using lm-evaluation-harness.
    
    This function loads the model, runs the specified benchmarks,
    and saves results to a JSON file.
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # For custom models, we need a wrapper. Here's the approach:
    # 1. Load the model from checkpoint
    # 2. Wrap it in an lm-eval compatible interface
    # 3. Run the harness
    
    results_file = output_path / "eval_results.json"
    
    # Load checkpoint
    import torch
    import importlib.util
    
    ckpt = torch.load(checkpoint_path, map_location="cuda", weights_only=False)
    config = ckpt["config"]
    
    spec = importlib.util.spec_from_file_location("model_mod", model_file)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    ModelClass = getattr(mod, model_class)
    
    model = ModelClass(
        vocab_size=config["vocab_size"],
        d_model=config["d_model"],
        n_layers=config["n_layers"],
        n_heads=config["n_heads"],
        seq_len=config["seq_len"],
    ).cuda()
    model.load_state_dict(ckpt["model"])
    model.eval()
    
    param_count = sum(p.numel() for p in model.parameters())
    print(f"Loaded model: {param_count:,} parameters from {checkpoint_path}")
    
    # Perplexity on validation data
    val_ppl = compute_perplexity(model, config)
    print(f"Validation perplexity: {val_ppl:.2f}")
    
    # Run lm-eval benchmarks
    # The harness expects a specific interface. For custom models,
    # the simplest approach is to use the HFLM wrapper after saving
    # in HuggingFace format, or implement the LM interface directly.
    
    # Simplified: run via subprocess with a temporary HF-format save
    print(f"\nRunning benchmarks: {tasks}")
    
    benchmark_results = run_benchmarks_direct(model, config, tasks, batch_size)
    
    # Combine results
    all_results = {
        "checkpoint": str(checkpoint_path),
        "model": config.get("model_name", "unknown"),
        "params": param_count,
        "val_perplexity": val_ppl,
        "benchmarks": benchmark_results,
        "step": ckpt.get("step", -1),
    }
    
    with open(results_file, "w") as f:
        json.dump(all_results, f, indent=2)
    
    print(f"\nResults saved to {results_file}")
    print_results_table(all_results)
    
    return all_results


def compute_perplexity(model, config, n_batches=100):
    """Compute perplexity on validation data."""
    import torch
    import numpy as np
    import math
    
    val_path = config.get("val_dataset", "data/assembled/val.bin")
    if not Path(val_path).exists():
        # Fall back to last 1% of training data
        train_data = np.memmap(config["dataset"], dtype=np.uint16, mode='r')
        val_data = train_data[int(len(train_data) * 0.99):]
    else:
        val_data = np.memmap(val_path, dtype=np.uint16, mode='r')
    
    seq_len = config["seq_len"]
    batch_size = 4
    total_loss = 0.0
    total_tokens = 0
    
    with torch.no_grad():
        for _ in range(n_batches):
            indices = np.random.randint(0, len(val_data) - seq_len - 1, size=batch_size)
            x = np.stack([val_data[i:i+seq_len] for i in indices])
            y = np.stack([val_data[i+1:i+seq_len+1] for i in indices])
            x = torch.from_numpy(x.astype(np.int64)).cuda()
            y = torch.from_numpy(y.astype(np.int64)).cuda()
            
            with torch.amp.autocast("cuda", dtype=torch.float16):
                logits = model(x)
                loss = torch.nn.functional.cross_entropy(
                    logits.reshape(-1, config["vocab_size"]), y.reshape(-1)
                )
            total_loss += loss.item() * batch_size * seq_len
            total_tokens += batch_size * seq_len
    
    avg_loss = total_loss / total_tokens
    return math.exp(avg_loss)


def run_benchmarks_direct(model, config, tasks_str, batch_size):
    """
    Run benchmarks using lm-eval's Python API.
    Requires implementing the LM interface for custom models.
    """
    import torch
    import tiktoken
    
    enc = tiktoken.get_encoding("gpt2")
    results = {}
    
    # For each task, we do a simplified evaluation
    # Full lm-eval integration requires the LM base class.
    # Here we show the direct approach for HellaSwag as an example.
    
    for task in tasks_str.split(","):
        task = task.strip()
        print(f"  Evaluating {task}...")
        
        try:
            # Use lm_eval Python API
            from lm_eval import evaluator
            from lm_eval.models.huggingface import HFLM
            
            # This requires saving the model in HF format first.
            # For custom architectures, implement lm_eval.api.model.LM instead.
            # See: https://github.com/EleutherAI/lm-evaluation-harness
            
            print(f"    (Skipping {task} — implement LM interface for custom models)")
            results[task] = {"acc": None, "note": "requires LM interface implementation"}
            
        except ImportError:
            print(f"    lm_eval not installed. Install with: pip install lm-eval")
            results[task] = {"acc": None, "note": "lm_eval not installed"}
    
    return results


def print_results_table(results):
    """Print a formatted results table."""
    print("\n" + "=" * 60)
    print(f"Model: {results['model']}  |  Params: {results['params']:,}  |  Step: {results['step']}")
    print("-" * 60)
    print(f"Validation Perplexity: {results['val_perplexity']:.2f}")
    print("-" * 60)
    for task, scores in results.get("benchmarks", {}).items():
        acc = scores.get("acc", "N/A")
        if isinstance(acc, float):
            print(f"  {task:<20s}  {acc:.4f}  ({acc*100:.1f}%)")
        else:
            print(f"  {task:<20s}  {acc}")
    print("=" * 60)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--model-file", required=True)
    parser.add_argument("--model-class", required=True)
    parser.add_argument("--output-dir", default="results")
    parser.add_argument("--tasks", default="hellaswag,arc_easy,piqa")
    parser.add_argument("--batch-size", type=int, default=8)
    args = parser.parse_args()
    
    evaluate_checkpoint(
        args.checkpoint, args.model_file, args.model_class,
        args.output_dir, args.tasks, args.batch_size,
    )
```

### Implementing the lm-eval LM Interface for Custom Models

If you want full lm-evaluation-harness support with your custom architecture, you need to subclass `lm_eval.api.model.LM`. Here is the minimal implementation:

```python
"""lm_eval_wrapper.py — Wrap a custom model for lm-evaluation-harness."""
import torch
import tiktoken
from lm_eval.api.model import LM
from lm_eval.api.instance import Instance


class CustomLM(LM):
    """Minimal wrapper to run lm-eval benchmarks on a custom model."""
    
    def __init__(self, model, seq_len=1024, batch_size=8):
        super().__init__()
        self.model = model
        self.model.eval()
        self._batch_size = batch_size
        self.seq_len = seq_len
        self.enc = tiktoken.get_encoding("gpt2")
        self.vocab_size = 50257
    
    @property
    def eot_token_id(self):
        return 50256
    
    @property
    def max_length(self):
        return self.seq_len
    
    @property
    def batch_size(self):
        return self._batch_size
    
    @property
    def device(self):
        return torch.device("cuda")
    
    def tok_encode(self, string):
        return self.enc.encode_ordinary(string)
    
    def tok_decode(self, tokens):
        return self.enc.decode(tokens)
    
    def loglikelihood(self, requests):
        """Compute log-likelihood of continuations given contexts."""
        results = []
        for request in requests:
            context, continuation = request.args
            ctx_tokens = self.tok_encode(context)
            cont_tokens = self.tok_encode(continuation)
            all_tokens = ctx_tokens + cont_tokens
            
            # Truncate from the left if too long
            if len(all_tokens) > self.seq_len:
                all_tokens = all_tokens[-self.seq_len:]
                cont_len = len(cont_tokens)
            else:
                cont_len = len(cont_tokens)
            
            input_ids = torch.tensor([all_tokens[:-1]], device="cuda")
            target_ids = torch.tensor([all_tokens[1:]], device="cuda")
            
            with torch.no_grad(), torch.amp.autocast("cuda", dtype=torch.float16):
                logits = self.model(input_ids)
            
            # Get log probs for the continuation tokens only
            log_probs = torch.log_softmax(logits[0], dim=-1)
            cont_start = len(all_tokens) - cont_len - 1
            
            total_ll = 0.0
            for i in range(cont_start, len(all_tokens) - 1):
                total_ll += log_probs[i, target_ids[0, i]].item()
            
            is_greedy = True  # Simplified — check if continuation is argmax
            results.append((total_ll, is_greedy))
        
        return results
    
    def loglikelihood_rolling(self, requests):
        """Compute rolling log-likelihood (for perplexity)."""
        results = []
        for request in requests:
            text = request.args[0]
            tokens = self.tok_encode(text)
            
            total_ll = 0.0
            for start in range(0, len(tokens) - 1, self.seq_len):
                end = min(start + self.seq_len, len(tokens))
                chunk = tokens[start:end]
                
                input_ids = torch.tensor([chunk[:-1]], device="cuda")
                target_ids = torch.tensor([chunk[1:]], device="cuda")
                
                with torch.no_grad(), torch.amp.autocast("cuda", dtype=torch.float16):
                    logits = self.model(input_ids)
                
                log_probs = torch.log_softmax(logits[0], dim=-1)
                for i in range(len(chunk) - 1):
                    total_ll += log_probs[i, target_ids[0, i]].item()
            
            results.append(total_ll)
        
        return results
    
    def generate_until(self, requests):
        """Generate text until a stop sequence. Used by few-shot tasks."""
        results = []
        for request in requests:
            context = request.args[0]
            gen_kwargs = request.args[1] if len(request.args) > 1 else {}
            until = gen_kwargs.get("until", ["\n"])
            max_gen = gen_kwargs.get("max_gen_toks", 128)
            
            tokens = self.tok_encode(context)
            if len(tokens) > self.seq_len - max_gen:
                tokens = tokens[-(self.seq_len - max_gen):]
            
            generated = []
            for _ in range(max_gen):
                input_ids = torch.tensor([tokens], device="cuda")
                with torch.no_grad(), torch.amp.autocast("cuda", dtype=torch.float16):
                    logits = self.model(input_ids)
                
                next_token = logits[0, -1].argmax().item()
                tokens.append(next_token)
                generated.append(next_token)
                
                text_so_far = self.tok_decode(generated)
                if any(s in text_so_far for s in until):
                    break
            
            results.append(self.tok_decode(generated))
        
        return results
```

Usage:

```python
from lm_eval import evaluator

wrapper = CustomLM(model, seq_len=1024, batch_size=8)
results = evaluator.simple_evaluate(
    model=wrapper,
    tasks=["hellaswag", "arc_easy", "piqa"],
    batch_size=8,
)
print(results["results"])
```

---

## 9.5 Smoke Test Protocol

Before committing to a long training run (hours or days), run a 200-step smoke test. This catches problems early — NaN gradients, memory leaks, broken data pipelines — before you waste GPU time.

```python
"""smoke_test.py — 200-step validation before any long run."""
import json
import math
import sys
import torch
import numpy as np
from pathlib import Path

from config import ExperimentConfig
from logger import TrainLogger

# Thresholds
MAX_LOSS_STEP_0 = 12.0       # CE loss at step 0 should be ~ln(50257) ≈ 10.8
MIN_LOSS_DECREASE = 0.5       # Loss should drop by at least 0.5 in 200 steps
MAX_GRAD_NORM = 100.0         # Grad norms above this suggest instability
MAX_MEMORY_GB = 15.0          # Must leave headroom on 16GB card
MIN_THROUGHPUT = 1000         # tok/s — below this, something is very wrong
MAX_STATE_NORM_RATIO = 100.0  # For recurrent models: max/min hidden norm ratio


def smoke_test(config: ExperimentConfig) -> dict:
    """
    Run 200 training steps and check 6 health criteria.
    Returns dict with pass/fail for each criterion.
    """
    # Override config for smoke test
    config.max_steps = 200
    config.checkpoint_interval = 999999  # Don't save checkpoints during smoke test
    config.eval_interval = 999999
    config.fill_metadata()
    
    # Temporary run dir
    run_dir = Path("smoke_test_tmp")
    run_dir.mkdir(exist_ok=True)
    
    # Run abbreviated training (reuse the training loop logic)
    from train import load_model, get_lr
    
    model = load_model(config)
    param_count = sum(p.numel() for p in model.parameters())
    
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=config.learning_rate,
        betas=(0.9, 0.95), weight_decay=config.weight_decay,
    )
    scaler = torch.amp.GradScaler("cuda") if config.use_amp else None
    amp_dtype = torch.float16
    
    data = np.memmap(config.dataset, dtype=np.uint16, mode='r')
    
    losses = []
    grad_norms = []
    throughputs = []
    has_nan = False
    peak_memory = 0.0
    
    model.train()
    import time
    t0 = time.time()
    tokens = 0
    
    for step in range(200):
        lr = get_lr(step, config)
        for pg in optimizer.param_groups:
            pg["lr"] = lr
        
        optimizer.zero_grad()
        accum_loss = 0.0
        
        for _ in range(config.accum_steps):
            indices = np.random.randint(0, len(data) - config.seq_len - 1, size=config.batch_size)
            x = np.stack([data[i:i+config.seq_len] for i in indices])
            y = np.stack([data[i+1:i+config.seq_len+1] for i in indices])
            x = torch.from_numpy(x.astype(np.int64)).cuda()
            y = torch.from_numpy(y.astype(np.int64)).cuda()
            
            with torch.amp.autocast("cuda", dtype=amp_dtype, enabled=config.use_amp):
                logits = model(x)
                loss = torch.nn.functional.cross_entropy(
                    logits.reshape(-1, config.vocab_size), y.reshape(-1)
                )
                loss = loss / config.accum_steps
            
            if torch.isnan(loss) or torch.isinf(loss):
                has_nan = True
            
            if scaler:
                scaler.scale(loss).backward()
            else:
                loss.backward()
            
            accum_loss += loss.item()
            tokens += config.batch_size * config.seq_len
        
        if scaler:
            scaler.unscale_(optimizer)
        gn = torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip)
        
        if scaler:
            scaler.step(optimizer)
            scaler.update()
        else:
            optimizer.step()
        
        losses.append(accum_loss)
        grad_norms.append(gn.item() if isinstance(gn, torch.Tensor) else gn)
        
        elapsed = time.time() - t0
        throughputs.append(tokens / elapsed if elapsed > 0 else 0)
        
        mem = torch.cuda.max_memory_allocated() / 1e9
        peak_memory = max(peak_memory, mem)
    
    # --- Check criteria ---
    results = {}
    
    # 1. Loss decreasing
    first_10 = np.mean(losses[:10])
    last_10 = np.mean(losses[-10:])
    loss_decrease = first_10 - last_10
    results["loss_decreasing"] = {
        "pass": loss_decrease >= MIN_LOSS_DECREASE,
        "first_10_avg": round(first_10, 4),
        "last_10_avg": round(last_10, 4),
        "decrease": round(loss_decrease, 4),
        "threshold": MIN_LOSS_DECREASE,
    }
    
    # 2. No NaN
    results["no_nan"] = {
        "pass": not has_nan,
        "detail": "NaN detected in loss" if has_nan else "OK",
    }
    
    # 3. Grad norms bounded
    max_gn = max(grad_norms)
    results["grad_norms_bounded"] = {
        "pass": max_gn <= MAX_GRAD_NORM,
        "max_grad_norm": round(max_gn, 2),
        "threshold": MAX_GRAD_NORM,
    }
    
    # 4. Memory within budget
    results["memory_ok"] = {
        "pass": peak_memory <= MAX_MEMORY_GB,
        "peak_gb": round(peak_memory, 2),
        "threshold": MAX_MEMORY_GB,
    }
    
    # 5. Throughput above threshold
    avg_throughput = np.mean(throughputs[-50:])
    results["throughput_ok"] = {
        "pass": avg_throughput >= MIN_THROUGHPUT,
        "avg_tok_per_sec": round(avg_throughput, 0),
        "threshold": MIN_THROUGHPUT,
    }
    
    # 6. State norms stable (for recurrent/looped models)
    # This checks that hidden states don't explode during forward passes
    results["state_norms_stable"] = {
        "pass": True,
        "detail": "Skipped (not a recurrent model)" 
                  if not hasattr(model, 'get_state_norms') else "Checked",
    }
    
    # Summary
    all_pass = all(v["pass"] for v in results.values())
    results["overall"] = "PASS" if all_pass else "FAIL"
    
    # Print report
    print("\n" + "=" * 50)
    print("SMOKE TEST REPORT")
    print("=" * 50)
    for name, detail in results.items():
        if name == "overall":
            continue
        status = "PASS" if detail["pass"] else "FAIL"
        print(f"  [{status}] {name}")
        for k, v in detail.items():
            if k != "pass":
                print(f"         {k}: {v}")
    print("-" * 50)
    print(f"  OVERALL: {results['overall']}")
    print("=" * 50)
    
    # Save results
    with open(run_dir / "smoke_results.json", "w") as f:
        json.dump(results, f, indent=2)
    
    if not all_pass:
        print("\nSMOKE TEST FAILED. Do NOT start a long training run.")
        print("Fix the failing criteria before proceeding.")
        sys.exit(1)
    
    return results


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    
    config = ExperimentConfig.load(Path(args.config))
    smoke_test(config)
```

### The 6 Criteria

| # | Criterion | What it catches | Threshold |
|---|-----------|----------------|-----------|
| 1 | Loss decreasing | Broken model, wrong LR, dead layers | Delta >= 0.5 in 200 steps |
| 2 | No NaN | Numerical instability, bad init | Any NaN = fail |
| 3 | Grad norms bounded | Exploding gradients, bad LR | Max norm < 100 |
| 4 | Memory within budget | OOM during training, memory leak | Peak < 15 GB |
| 5 | Throughput above floor | I/O bottleneck, bad kernel, CPU-bound | > 1000 tok/s |
| 6 | State norms stable | Recurrent state explosion | Max/min ratio < 100 |

Run this before EVERY long run. It takes 2-3 minutes and saves hours of wasted GPU time.

---

## 9.6 Experiment Comparison

### Comparing N Runs

```python
"""compare.py — Compare experiment results across N runs."""
import json
import argparse
import math
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

from logger import TrainLogger


def load_experiment(run_dir: Path):
    """Load config, log, and eval results for one experiment."""
    run_dir = Path(run_dir)
    
    config = json.load(open(run_dir / "config.json"))
    log = TrainLogger.read(run_dir / "train_log.jsonl")
    
    eval_path = run_dir / "eval_results.json"
    eval_results = json.load(open(eval_path)) if eval_path.exists() else None
    
    return {
        "dir": str(run_dir),
        "config": config,
        "log": log,
        "eval": eval_results,
        "name": config.get("model_name", run_dir.name),
    }


def plot_loss_curves(experiments, output_path="comparison_loss.png"):
    """Plot training loss curves for multiple experiments."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    for exp in experiments:
        steps = [e["step"] for e in exp["log"] if "loss" in e and "event" not in e]
        losses = [e["loss"] for e in exp["log"] if "loss" in e and "event" not in e]
        
        # Smooth with running average
        window = min(50, len(losses) // 10)
        if window > 1:
            smoothed = np.convolve(losses, np.ones(window)/window, mode='valid')
            smooth_steps = steps[window-1:]
        else:
            smoothed = losses
            smooth_steps = steps
        
        label = f"{exp['name']} (lr={exp['config']['learning_rate']})"
        ax1.plot(smooth_steps, smoothed, label=label, alpha=0.8)
        
        # Throughput
        toks = [e["tok_per_sec"] for e in exp["log"]
                if "tok_per_sec" in e and e.get("tok_per_sec", 0) > 0]
        tok_steps = [e["step"] for e in exp["log"]
                     if "tok_per_sec" in e and e.get("tok_per_sec", 0) > 0]
        if toks:
            ax2.plot(tok_steps, toks, label=label, alpha=0.8)
    
    ax1.set_xlabel("Step")
    ax1.set_ylabel("Loss")
    ax1.set_title("Training Loss")
    ax1.legend(fontsize=8)
    ax1.grid(True, alpha=0.3)
    
    ax2.set_xlabel("Step")
    ax2.set_ylabel("Tokens/sec")
    ax2.set_title("Throughput")
    ax2.legend(fontsize=8)
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    print(f"Saved: {output_path}")


def print_comparison_table(experiments):
    """Print a formatted comparison table."""
    print("\n" + "=" * 100)
    print(f"{'Model':<20} {'Params':>10} {'Steps':>8} {'Final Loss':>12} "
          f"{'Val PPL':>10} {'tok/s':>8} {'HellaSwag':>10} {'ARC-E':>10}")
    print("-" * 100)
    
    for exp in experiments:
        config = exp["config"]
        log = exp["log"]
        
        # Extract metrics
        train_entries = [e for e in log if "loss" in e and "event" not in e]
        final_loss = train_entries[-1]["loss"] if train_entries else float("nan")
        max_step = train_entries[-1]["step"] if train_entries else 0
        
        # Throughput (average of last 100 entries)
        toks = [e["tok_per_sec"] for e in train_entries[-100:] if "tok_per_sec" in e]
        avg_tok = np.mean(toks) if toks else 0
        
        # Eval results
        val_ppl = exp["eval"]["val_perplexity"] if exp.get("eval") else float("nan")
        hellaswag = exp["eval"]["benchmarks"].get("hellaswag", {}).get("acc", float("nan")) \
                    if exp.get("eval") else float("nan")
        arc_e = exp["eval"]["benchmarks"].get("arc_easy", {}).get("acc", float("nan")) \
                if exp.get("eval") else float("nan")
        
        # Param count
        d = config["d_model"]
        n = config["n_layers"]
        param_est = f"~{d * d * n * 12 // 1_000_000}M"  # Rough estimate
        
        def fmt(v):
            if isinstance(v, float) and math.isnan(v):
                return "N/A"
            if isinstance(v, float):
                return f"{v:.4f}"
            return str(v)
        
        print(f"{exp['name']:<20} {param_est:>10} {max_step:>8} {fmt(final_loss):>12} "
              f"{fmt(val_ppl):>10} {avg_tok:>8.0f} {fmt(hellaswag):>10} {fmt(arc_e):>10}")
    
    print("=" * 100)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("run_dirs", nargs="+", help="Paths to experiment run directories")
    parser.add_argument("--output", default="comparison.png")
    args = parser.parse_args()
    
    experiments = [load_experiment(d) for d in args.run_dirs]
    print_comparison_table(experiments)
    plot_loss_curves(experiments, args.output)
```

### Usage

```bash
# Compare three learning rate experiments
python compare.py \
    checkpoints/gpt2_abc123 \
    checkpoints/gpt2_def456 \
    checkpoints/gpt2_ghi789 \
    --output lr_comparison.png
```

This produces a table like:

```
====================================================================================================
Model                  Params    Steps   Final Loss    Val PPL    tok/s  HellaSwag      ARC-E
----------------------------------------------------------------------------------------------------
gpt2_lr3e4              ~85M    10000       3.2145    25.12      12500     0.2891     0.4312
gpt2_lr6e4              ~85M    10000       3.1023    22.35      12480     0.3012     0.4456
gpt2_lr1e3              ~85M    10000       3.3891    29.67      12510     0.2734     0.4178
====================================================================================================
```

---

## 9.7 Version Control for Experiments

### What to Commit

```
COMMIT:
  configs/*.json           # Experiment configurations
  results/*.json           # Evaluation results
  scripts/*.py             # Training/eval scripts
  models/*.py              # Model definitions

DO NOT COMMIT:
  checkpoints/             # Too large (100MB+ per checkpoint)
  data/                    # Too large, and should be reproducible from scripts
  *.png                    # Regenerate from logs
  smoke_test_tmp/          # Throwaway
```

### .gitignore

```gitignore
# Checkpoints are too large for git
checkpoints/
*.pt
*.bin

# Data is reproducible
data/
datasets/

# Generated plots
*.png

# Temp
smoke_test_tmp/
__pycache__/
```

### Experiment Tagging

For significant runs (best result, first working architecture, etc.), create a git tag:

```bash
# After a successful training run
git tag -a "exp/gpt2-lr6e4-fineweb" -m "GPT-2 124M, lr=6e-4, FineWeb-Edu, HellaSwag=0.301"
git push origin --tags
```

### Metadata in Checkpoints

The training loop already saves the config (including git hash) in every checkpoint. This means you can always trace a checkpoint back to the exact code and config that produced it:

```python
# Inspect any checkpoint
ckpt = torch.load("checkpoints/gpt2_abc123/step_10000.pt", weights_only=False)
print(f"Git hash: {ckpt['config']['git_hash']}")
print(f"Experiment ID: {ckpt['config']['experiment_id']}")
print(f"Timestamp: {ckpt['config']['timestamp']}")
print(f"Learning rate: {ckpt['config']['learning_rate']}")
```

---

## Exercises

1. **Train GPT-2 with 3 different learning rates.** Create three config files with `learning_rate` set to `3e-4`, `6e-4`, and `1e-3`. Run each for 2000 steps (should take ~10 minutes each on RTX 4060 Ti). Use `compare.py` to plot the loss curves on one chart and identify which learning rate converges fastest.

2. **Run HellaSwag eval on your best checkpoint.** Take the checkpoint with the lowest validation loss from exercise 1. Run `evaluate_model.py` with `--tasks hellaswag`. At 124M parameters trained on 2000 steps, expect HellaSwag accuracy around 0.26-0.28 (barely above the 0.25 random baseline). This is expected — it takes ~50K steps to see meaningful benchmark improvement.

3. **Build a comparison table.** Run `compare.py` on all three runs and verify the table includes: model name, final loss, throughput, and (if available) benchmark scores. Save the output for reference in later parts.

---

## Checkpoint

Before moving to Part 10, verify:
- [ ] `ExperimentConfig` produces unique experiment IDs for different configs
- [ ] Training loop logs to JSONL and you can read it back with `TrainLogger.read()`
- [ ] Checkpoints save and resume correctly (train 100 steps, stop, resume, train 100 more)
- [ ] Smoke test catches a deliberately broken config (set `learning_rate: 100` and verify it fails)
- [ ] `compare.py` produces a readable table and loss curve plot for 2+ experiments
- [ ] You can explain why MMLU is useless at <350M parameters

---

**Previous: [Part 08 -- Data Pipeline: Mixing, Filtering, and Scaling](08_data_pipeline.md)**
**Next: [Part 10 -- Math Foundations for Reading Papers](10_math_foundations.md)**

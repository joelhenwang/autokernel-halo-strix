# Part 15: Agent Automation — The Autonomous LLM Lab

## What You Will Learn
- How to design an AI agent that runs the full LLM development pipeline autonomously
- The agent loop: observe → plan → act → verify → iterate
- How to make each pipeline stage machine-readable (structured inputs/outputs, exit codes, JSON logs)
- Guardrails: when the agent should stop and ask a human
- Building the orchestrator that connects research → design → train → evaluate

## Goal
Build an agent system that takes a research question ("Can we beat SmolLM2-135M with a novel architecture?") and autonomously executes the full pipeline: literature review, architecture design, smoke test, profiling, kernel optimization, training, evaluation, and benchmarking. The human reviews results and steers direction; the agent handles execution.

## Why This Matters
You built every piece of the pipeline in Parts 01-14. But running it manually means: you read papers, you design the architecture, you type the training command, you check the logs, you run eval. That is 50+ manual steps per experiment. An agent reduces this to: state the goal, review the results.

This is not about replacing understanding — you need Parts 01-14 to know WHAT the agent should do. This is about removing the tedium of HOW.

---

## 15.1 The Pipeline as a DAG

Every experiment follows the same directed acyclic graph:

```
research ──→ design ──→ implement ──→ smoke_test ──→ profile
                                          │              │
                                          │ fail: fix    │
                                          ←──────────────┘
                                          │
                                          ↓ pass
                                    optimize_kernels ──→ train
                                                          │
                                                          ↓
                                                    checkpoint ──→ evaluate ──→ benchmark
                                                                                   │
                                                                                   ↓
                                                                             compare_to_baselines
                                                                                   │
                                                                                   ↓
                                                                              report_results
```

Each node has:
- **Inputs:** files, configs, checkpoints from previous stages
- **Outputs:** files, metrics, logs for the next stage
- **Success criteria:** machine-checkable conditions (loss < threshold, no NaN, speedup > 1.0)
- **Failure action:** fix and retry, or escalate to human

The agent's job is to traverse this DAG, executing each node, checking success criteria, and deciding the next action.

---

## 15.2 Making Each Stage Machine-Readable

The agent can only automate what it can parse. Every stage must produce structured, machine-readable output.

### Stage Contracts

Each stage is a script with a contract:

```python
"""
Stage contract:
  Input:   JSON config file
  Output:  JSON result file + artifacts (checkpoints, logs, etc.)
  Exit:    0 = success, 1 = failure, 2 = needs human review
  Stdout:  human-readable progress
  Result:  structured JSON with metrics the agent can parse
"""
```

### Example: Smoke Test Contract

```python
# scripts/agent/run_smoke_test.py
"""
Input:  {"model_path": "models/chimera_halo.py", "class_name": "ChimeraHaloMini"}
Output: {"status": "pass"|"fail", "checks": {...}, "metrics": {...}}
"""
import argparse
import json
import sys
import torch

def run_smoke_test(config: dict) -> dict:
    """Run 200-step smoke test, return structured results."""
    from training.train import create_model, create_dataloader
    
    model = create_model(config["model_path"], config["class_name"])
    loader = create_dataloader("datasets/babylm_train.txt", block_size=512, batch_size=8)
    
    results = {
        "status": "pass",
        "checks": {},
        "metrics": {},
    }
    
    losses = []
    grad_norms = []
    peak_memory = 0
    
    model = model.cuda().train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)
    scaler = torch.amp.GradScaler("cuda")
    
    for step, (x, y) in enumerate(loader):
        if step >= 200:
            break
        x, y = x.cuda(), y.cuda()
        
        with torch.amp.autocast("cuda", dtype=torch.float16):
            loss = model(x, y)
        
        if torch.isnan(loss) or torch.isinf(loss):
            results["status"] = "fail"
            results["checks"]["nan_free"] = False
            results["checks"]["failure_step"] = step
            return results
        
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        gn = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad()
        
        losses.append(loss.item())
        grad_norms.append(gn.item())
        peak_memory = max(peak_memory, torch.cuda.max_memory_allocated() / 1e9)
    
    # Check criteria
    results["checks"]["nan_free"] = True
    results["checks"]["loss_decreasing"] = losses[-1] < losses[0] * 0.9
    results["checks"]["grad_norms_bounded"] = max(grad_norms) < 10.0
    results["checks"]["memory_ok"] = peak_memory < 14.0  # 16GB with headroom
    results["checks"]["initial_loss_sane"] = 9.0 < losses[0] < 12.0
    
    results["metrics"]["initial_loss"] = round(losses[0], 4)
    results["metrics"]["final_loss"] = round(losses[-1], 4)
    results["metrics"]["max_grad_norm"] = round(max(grad_norms), 4)
    results["metrics"]["peak_memory_gb"] = round(peak_memory, 2)
    results["metrics"]["loss_reduction"] = round(1 - losses[-1] / losses[0], 4)
    
    # Overall pass/fail
    if not all(results["checks"].values()):
        results["status"] = "fail"
    
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, help="JSON config file")
    args = parser.parse_args()
    
    config = json.loads(open(args.config).read())
    result = run_smoke_test(config)
    
    # Write result file
    result_path = config.get("result_path", "smoke_test_result.json")
    with open(result_path, "w") as f:
        json.dump(result, f, indent=2)
    
    print(json.dumps(result, indent=2))
    sys.exit(0 if result["status"] == "pass" else 1)
```

### All Stage Contracts

| Stage | Input | Output | Success Criteria |
|-------|-------|--------|-----------------|
| **research** | topic string | `research_notes.json` (papers, key findings) | At least 3 relevant papers found |
| **design** | research notes + constraints | `architecture_spec.json` (dims, layers, param budget) | Param count within budget, all dims 128-aligned |
| **implement** | architecture spec | `models/{name}.py` | Python imports without error, Mini variant exists |
| **smoke_test** | model path + class name | `smoke_result.json` (pass/fail + metrics) | All 6 checks pass |
| **profile** | model path + training config | `profile_result.json` (op breakdown, bottlenecks) | Top-5 bottlenecks identified |
| **optimize_kernels** | profile results + model | `kernel_report.json` (patterns matched, speedups) | At least 1 pattern matched, no correctness failures |
| **train** | model + dataset + config | `train_log.jsonl` + checkpoint | Loss decreasing, no NaN, throughput > threshold |
| **evaluate** | checkpoint path | `eval_result.json` (HellaSwag, ARC, etc.) | Scores parsed, no errors |
| **benchmark** | eval results + baselines | `benchmark_report.json` (comparison table) | Report generated |

---

## 15.3 The Agent Loop

The agent follows an observe-plan-act-verify cycle:

```python
# scripts/agent/orchestrator.py
"""
The agent orchestrator. Executes the pipeline DAG, checking results
at each stage and deciding the next action.
"""
import json
import subprocess
import sys
import time
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class ExperimentState:
    """Tracks the current state of an experiment run."""
    experiment_id: str
    goal: str
    stage: str = "research"
    history: list = field(default_factory=list)
    results: dict = field(default_factory=dict)
    retries: dict = field(default_factory=dict)
    max_retries: int = 3
    
    def save(self, path: str = "experiment_state.json"):
        with open(path, "w") as f:
            json.dump(self.__dict__, f, indent=2)
    
    @classmethod
    def load(cls, path: str = "experiment_state.json"):
        data = json.loads(Path(path).read_text())
        return cls(**data)


# Stage execution order
PIPELINE = [
    "research",
    "design", 
    "implement",
    "smoke_test",
    "profile",
    "optimize_kernels",
    "train",
    "evaluate",
    "benchmark",
    "report",
]


def run_stage(stage: str, config: dict, state: ExperimentState) -> dict:
    """Execute a pipeline stage and return structured results."""
    
    config_path = f"agent_work/{state.experiment_id}/{stage}_config.json"
    result_path = f"agent_work/{state.experiment_id}/{stage}_result.json"
    
    Path(config_path).parent.mkdir(parents=True, exist_ok=True)
    config["result_path"] = result_path
    
    with open(config_path, "w") as f:
        json.dump(config, f, indent=2)
    
    script = f"scripts/agent/run_{stage}.py"
    
    print(f"\n{'='*60}")
    print(f"Stage: {stage}")
    print(f"Config: {config_path}")
    print(f"{'='*60}")
    
    t0 = time.time()
    proc = subprocess.run(
        [sys.executable, script, "--config", config_path],
        capture_output=True, text=True, timeout=7200,  # 2 hour max per stage
    )
    elapsed = time.time() - t0
    
    print(proc.stdout[-2000:] if len(proc.stdout) > 2000 else proc.stdout)
    if proc.stderr:
        print(f"STDERR: {proc.stderr[-500:]}")
    
    if Path(result_path).exists():
        result = json.loads(Path(result_path).read_text())
    else:
        result = {
            "status": "fail",
            "error": proc.stderr[-500:] if proc.stderr else "No result file produced",
            "exit_code": proc.returncode,
        }
    
    result["elapsed_seconds"] = round(elapsed, 1)
    
    state.history.append({
        "stage": stage,
        "status": result.get("status", "unknown"),
        "elapsed": result["elapsed_seconds"],
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
    })
    state.results[stage] = result
    state.save()
    
    return result


def decide_next_action(stage: str, result: dict, state: ExperimentState) -> str:
    """Given a stage result, decide what to do next.
    
    Returns one of:
      "continue"  — move to next stage
      "retry"     — retry current stage (after fixing)
      "escalate"  — stop and ask human for help
      "done"      — pipeline complete
    """
    status = result.get("status", "fail")
    
    if status == "pass":
        # Move to next stage
        idx = PIPELINE.index(stage)
        if idx + 1 >= len(PIPELINE):
            return "done"
        return "continue"
    
    # Handle failure
    retries = state.retries.get(stage, 0)
    
    if retries >= state.max_retries:
        print(f"\n*** Stage '{stage}' failed {retries} times. Escalating to human. ***")
        return "escalate"
    
    # Decide if retryable
    error = result.get("error", "")
    checks = result.get("checks", {})
    
    # Some failures are fixable by the agent
    fixable_failures = {
        "smoke_test": {
            "nan_free": False,         # might be LR too high → reduce and retry
            "loss_decreasing": False,  # might need more steps or different init
            "memory_ok": False,        # reduce batch size
        },
        "optimize_kernels": {
            "correctness": False,      # skip the failing pattern and retry
        },
    }
    
    if stage in fixable_failures:
        for check, expected_fail in fixable_failures[stage].items():
            if checks.get(check) == expected_fail:
                print(f"\n  Fixable failure: {stage}.{check}")
                state.retries[stage] = retries + 1
                return "retry"
    
    # Unknown failure — escalate
    return "escalate"


def auto_fix(stage: str, result: dict, config: dict) -> dict:
    """Attempt to fix a failed stage by adjusting config."""
    checks = result.get("checks", {})
    
    if stage == "smoke_test":
        if not checks.get("memory_ok", True):
            config["batch_size"] = config.get("batch_size", 8) // 2
            print(f"  Fix: reduced batch_size to {config['batch_size']}")
        
        if not checks.get("nan_free", True) or not checks.get("loss_decreasing", True):
            config["lr"] = config.get("lr", 3e-4) * 0.3
            print(f"  Fix: reduced lr to {config['lr']}")
        
        if not checks.get("initial_loss_sane", True):
            print(f"  Fix: initial loss out of range — likely model bug, cannot auto-fix")
    
    if stage == "optimize_kernels":
        if not checks.get("correctness", True):
            failed_patterns = result.get("failed_patterns", [])
            config["exclude_patterns"] = config.get("exclude_patterns", []) + failed_patterns
            print(f"  Fix: excluded failing patterns: {failed_patterns}")
    
    return config


def run_pipeline(goal: str, base_config: dict, experiment_id: str = None):
    """Execute the full pipeline from start to finish."""
    
    if experiment_id is None:
        experiment_id = f"exp_{int(time.time())}"
    
    state = ExperimentState(experiment_id=experiment_id, goal=goal)
    config = dict(base_config)
    config["experiment_id"] = experiment_id
    config["goal"] = goal
    
    stage_idx = 0
    
    while stage_idx < len(PIPELINE):
        stage = PIPELINE[stage_idx]
        state.stage = stage
        
        result = run_stage(stage, config, state)
        action = decide_next_action(stage, result, state)
        
        if action == "continue":
            # Pass outputs to next stage config
            config.update(result.get("next_stage_config", {}))
            stage_idx += 1
        
        elif action == "retry":
            config = auto_fix(stage, result, config)
            # Loop continues with same stage_idx
        
        elif action == "escalate":
            print(f"\n{'='*60}")
            print(f"HUMAN REVIEW NEEDED")
            print(f"Stage: {stage}")
            print(f"Issue: {result.get('error', 'Unknown')}")
            print(f"Checks: {json.dumps(result.get('checks', {}), indent=2)}")
            print(f"State saved to: agent_work/{experiment_id}/experiment_state.json")
            print(f"{'='*60}")
            state.save(f"agent_work/{experiment_id}/experiment_state.json")
            return state
        
        elif action == "done":
            print(f"\n{'='*60}")
            print(f"PIPELINE COMPLETE")
            print(f"Experiment: {experiment_id}")
            print(f"Results: agent_work/{experiment_id}/")
            print(f"{'='*60}")
            state.stage = "complete"
            state.save(f"agent_work/{experiment_id}/experiment_state.json")
            return state
    
    return state
```

---

## 15.4 Stage Implementations

Each stage is a standalone script. Here are the key ones:

### Research Stage

```python
# scripts/agent/run_research.py
"""
Research stage: given a topic, find relevant papers and extract key ideas.

In practice, this is where an LLM agent (Claude, GPT-4) shines.
The script provides structure; the LLM provides intelligence.

For a fully automated version, use an LLM API to:
1. Search arxiv for recent papers on the topic
2. Summarize each paper's key contribution
3. Identify actionable techniques for our hardware constraints
"""
import argparse
import json

def run_research(config: dict) -> dict:
    topic = config["goal"]
    constraints = config.get("constraints", {
        "max_params": "350M",
        "gpu": "RTX 4060 Ti 16GB",
        "framework": "CUDA + PyTorch",
    })
    
    # In automated mode, this calls an LLM API
    # In semi-automated mode, the agent writes a research brief
    # and the human fills in findings
    
    result = {
        "status": "pass",
        "research_notes": {
            "topic": topic,
            "constraints": constraints,
            "papers": [],      # Agent fills these via API or human fills manually
            "key_techniques": [],
            "recommended_components": [],
        },
        "next_stage_config": {
            "research_notes_path": f"agent_work/{config['experiment_id']}/research_notes.json",
        },
    }
    
    # Save research notes for human review / agent continuation
    notes_path = result["next_stage_config"]["research_notes_path"]
    from pathlib import Path
    Path(notes_path).parent.mkdir(parents=True, exist_ok=True)
    with open(notes_path, "w") as f:
        json.dump(result["research_notes"], f, indent=2)
    
    return result
```

### Design Stage

```python
# scripts/agent/run_design.py
"""
Design stage: produce an architecture specification from research notes.

Outputs a JSON spec that the implement stage can turn into code.
"""
import argparse
import json
import math

def calculate_param_budget(spec: dict) -> dict:
    """Calculate parameter count per component."""
    d = spec["d_model"]
    v = spec["vocab_size"]
    n_layers = spec["n_unique_layers"]
    n_heads = spec["n_heads"]
    n_kv = spec["n_kv_heads"]
    ffn = spec["ffn_dim"]
    head_dim = d // n_heads
    
    embed_params = v * d  # standard tied embedding
    if spec.get("factorized_rank"):
        r = spec["factorized_rank"]
        embed_params = v * r + r * d + d * r  # embed + proj_up + proj_down
    
    attn_params = d * d + d * (n_kv * head_dim) * 2 + d * d  # wq + wk + wv + wo
    ffn_params = d * ffn * 2 + ffn * d  # w_gate + w_up + w_down (SwiGLU)
    norm_params = d * 2  # 2 RMSNorms per layer
    layer_params = attn_params + ffn_params + norm_params
    
    n_attn_layers = len(spec.get("gqa_positions", []))
    n_conv_layers = n_layers - n_attn_layers
    
    # Conv layers have different param count
    d_conv = spec.get("d_conv", d)
    conv_params = d * 3 * d_conv + d_conv * 3 + d_conv * d + d * ffn * 2 + ffn * d + norm_params
    
    total_unique = embed_params + n_attn_layers * layer_params + n_conv_layers * conv_params
    n_repeat = spec.get("n_repeat", 1)
    total_effective = embed_params + (n_attn_layers * layer_params + n_conv_layers * conv_params) * n_repeat
    
    return {
        "embed_params": embed_params,
        "per_attn_layer": layer_params,
        "per_conv_layer": conv_params,
        "total_unique": total_unique,
        "total_effective": total_effective,
        "unique_M": round(total_unique / 1e6, 1),
        "effective_M": round(total_effective / 1e6, 1),
    }


def validate_spec(spec: dict) -> list:
    """Check spec for common issues. Return list of warnings."""
    warnings = []
    
    d = spec["d_model"]
    if d % 128 != 0:
        warnings.append(f"d_model={d} not 128-aligned (bad for Tensor Cores)")
    
    if spec.get("factorized_rank") and spec["factorized_rank"] % 128 != 0:
        warnings.append(f"factorized_rank={spec['factorized_rank']} not 128-aligned")
    
    if spec.get("ffn_dim") and spec["ffn_dim"] % 128 != 0:
        warnings.append(f"ffn_dim={spec['ffn_dim']} not 128-aligned")
    
    budget = calculate_param_budget(spec)
    if budget["unique_M"] > 350:
        warnings.append(f"Model too large: {budget['unique_M']}M params (max 350M for 16GB VRAM)")
    
    head_dim = d // spec["n_heads"]
    if head_dim % 8 != 0:
        warnings.append(f"head_dim={head_dim} not divisible by 8 (inefficient attention)")
    
    return warnings


def run_design(config: dict) -> dict:
    spec = config.get("architecture_spec", {
        "name": "experiment_model",
        "vocab_size": 50257,
        "d_model": 768,
        "n_unique_layers": 8,
        "gqa_positions": [3, 6],
        "n_repeat": 2,
        "d_conv": 512,
        "ffn_dim": 2816,
        "n_heads": 12,
        "n_kv_heads": 4,
        "factorized_rank": 256,
        "use_xsa": True,
        "use_parcae": True,
        "mean_recurrence": 3,
    })
    
    budget = calculate_param_budget(spec)
    warnings = validate_spec(spec)
    
    result = {
        "status": "pass" if not warnings else "pass",  # warnings are non-fatal
        "spec": spec,
        "budget": budget,
        "warnings": warnings,
        "next_stage_config": {
            "architecture_spec": spec,
            "architecture_spec_path": f"agent_work/{config['experiment_id']}/architecture_spec.json",
        },
    }
    
    if budget["unique_M"] > 350:
        result["status"] = "fail"
        result["error"] = f"Model too large: {budget['unique_M']}M > 350M limit"
    
    return result
```

### Train Stage

```python
# scripts/agent/run_train.py
"""
Train stage: launch training and monitor progress.

Does NOT block until training is complete. Instead:
1. Launches training as a subprocess
2. Polls train_log.jsonl for progress
3. Checks for failure conditions (NaN, loss plateau, memory)
4. Returns when training completes or fails
"""
import argparse
import json
import subprocess
import sys
import time
from pathlib import Path


def monitor_training(log_path: str, timeout_minutes: float = 60,
                     poll_interval: int = 30) -> dict:
    """Monitor a running training job by tailing its log file."""
    
    deadline = time.time() + timeout_minutes * 60
    last_step = 0
    plateau_count = 0
    prev_loss = None
    
    while time.time() < deadline:
        time.sleep(poll_interval)
        
        log_file = Path(log_path)
        if not log_file.exists():
            continue
        
        # Read last few lines
        lines = log_file.read_text().strip().split("\n")
        if not lines:
            continue
        
        latest = json.loads(lines[-1])
        step = latest.get("step", 0)
        loss = latest.get("loss", 999)
        
        if step == last_step:
            continue  # no new data
        
        last_step = step
        
        # Check for NaN
        if loss != loss:  # NaN check
            return {
                "status": "fail",
                "error": f"NaN loss at step {step}",
                "last_step": step,
            }
        
        # Check for plateau (loss not improving over 500 steps)
        if prev_loss is not None and loss >= prev_loss * 0.999:
            plateau_count += 1
        else:
            plateau_count = 0
        prev_loss = loss
        
        if plateau_count > 10:  # ~10 polls without improvement
            return {
                "status": "pass",
                "note": "Loss plateaued, stopping early",
                "last_step": step,
                "final_loss": loss,
            }
        
        print(f"  Step {step}: loss={loss:.4f}")
    
    # Timeout reached
    lines = Path(log_path).read_text().strip().split("\n")
    if lines:
        latest = json.loads(lines[-1])
        return {
            "status": "pass",
            "last_step": latest.get("step", 0),
            "final_loss": latest.get("loss", 999),
        }
    
    return {"status": "fail", "error": "No log data after timeout"}


def run_train(config: dict) -> dict:
    model_path = config["model_path"]
    class_name = config["class_name"]
    dataset = config.get("dataset", "datasets/babylm_train.txt")
    checkpoint_dir = f"agent_work/{config['experiment_id']}/checkpoints"
    
    cmd = [
        sys.executable, "-m", "training.train",
        "--model", model_path,
        "--class-name", class_name,
        "--dataset", dataset,
        "--checkpoint-dir", checkpoint_dir,
        "--compile",
        "--lr", str(config.get("lr", 3e-4)),
        "--batch-size", str(config.get("batch_size", 8)),
        "--max-steps", str(config.get("max_steps", 10000)),
    ]
    
    if config.get("optimize_kernels"):
        cmd.append("--optimize-kernels")
    
    log_path = f"{checkpoint_dir}/train_log.jsonl"
    
    # Launch training in background
    Path(checkpoint_dir).mkdir(parents=True, exist_ok=True)
    proc = subprocess.Popen(cmd, stdout=open(f"{checkpoint_dir}/stdout.log", "w"),
                           stderr=subprocess.STDOUT)
    
    # Monitor
    monitor_result = monitor_training(
        log_path,
        timeout_minutes=config.get("time_budget", 60),
    )
    
    # Wait for process
    proc.wait(timeout=60)
    
    # Find best checkpoint
    ckpt_dir = Path(checkpoint_dir)
    checkpoints = sorted(ckpt_dir.glob("*.pt"))
    best_ckpt = str(checkpoints[-1]) if checkpoints else None
    
    result = {
        "status": monitor_result["status"],
        "metrics": monitor_result,
        "checkpoint_path": best_ckpt,
        "log_path": log_path,
        "next_stage_config": {
            "checkpoint_path": best_ckpt,
            "model_path": model_path,
            "class_name": class_name,
        },
    }
    
    return result
```

### Evaluate Stage

```python
# scripts/agent/run_evaluate.py
"""
Evaluate stage: run lm-evaluation-harness on a checkpoint.

Produces structured benchmark results.
"""
import argparse
import json
import subprocess
import sys


BENCHMARKS_SMALL = ["hellaswag", "arc_easy", "arc_challenge", "piqa", "winogrande"]


def run_evaluate(config: dict) -> dict:
    checkpoint_path = config["checkpoint_path"]
    model_path = config["model_path"]
    class_name = config["class_name"]
    
    results = {
        "status": "pass",
        "benchmarks": {},
    }
    
    # Run each benchmark
    for bench in BENCHMARKS_SMALL:
        try:
            cmd = [
                sys.executable, "scripts/evaluate.py",
                "--checkpoint", checkpoint_path,
                "--model", model_path,
                "--class-name", class_name,
                "--benchmark", bench,
                "--output-json", f"agent_work/{config['experiment_id']}/eval_{bench}.json",
            ]
            proc = subprocess.run(cmd, capture_output=True, text=True, timeout=1800)
            
            result_path = f"agent_work/{config['experiment_id']}/eval_{bench}.json"
            if proc.returncode == 0:
                bench_result = json.loads(open(result_path).read())
                results["benchmarks"][bench] = bench_result.get("accuracy", 0)
            else:
                results["benchmarks"][bench] = None
                results["status"] = "pass"  # individual bench failure is not fatal
        except Exception as e:
            results["benchmarks"][bench] = None
    
    # Compute average over non-None benchmarks
    valid = [v for v in results["benchmarks"].values() if v is not None]
    results["average"] = round(sum(valid) / len(valid), 2) if valid else 0
    
    return results
```

---

## 15.5 Guardrails: When to Stop and Ask

The agent must know its limits. Some decisions require human judgment.

### Hard Stops (Always Escalate)

```python
HARD_STOPS = {
    "model_architecture_change":
        "Changing the architecture (not hyperparameters) requires human review. "
        "The agent can tune LR, batch size, layers — but not add new mechanisms.",
    
    "data_quality_judgment":
        "The agent cannot judge if training data is toxic, biased, or low quality "
        "beyond automated metrics. Flag for human review.",
    
    "unexpected_improvement":
        "If a change produces >20% improvement, verify it is real and not a bug. "
        "Common false positives: evaluation bug, data leakage, wrong baseline.",
    
    "budget_exceeded":
        "If total GPU hours exceed the budget, stop and report what was achieved.",
    
    "three_consecutive_failures":
        "If the same stage fails 3 times with different fixes, escalate. "
        "The agent's auto-fix repertoire is exhausted.",
}
```

### Soft Stops (Log and Continue)

```python
SOFT_STOPS = {
    "diminishing_returns":
        "If kernel optimization gives <2% speedup, skip remaining kernels. "
        "Not worth the correctness risk.",
    
    "plateau_detected":
        "If loss hasn't improved in 1000 steps, try: "
        "(1) reduce LR by 3x, (2) if already reduced, stop training.",
    
    "below_baseline":
        "If eval scores are below random (25% for 4-way MCQ), "
        "the model is broken. Do not continue to benchmark.",
}
```

---

## 15.6 The Config That Drives Everything

A single JSON config file launches the entire pipeline:

```json
{
    "goal": "Beat SmolLM2-135M (HellaSwag 42.1) with a novel architecture on RTX 4060 Ti",
    "constraints": {
        "max_params_M": 350,
        "gpu": "RTX 4060 Ti 16GB",
        "max_gpu_hours": 24,
        "target_benchmarks": ["hellaswag", "arc_easy", "arc_challenge", "piqa"]
    },
    "baselines": {
        "smollm2_135m": {"hellaswag": 42.1, "arc_avg": 43.9},
        "nandi_150m": {"average": 25.63}
    },
    "architecture_spec": {
        "name": "chimera_halo",
        "d_model": 768,
        "n_unique_layers": 8,
        "gqa_positions": [3, 6],
        "n_repeat": 2,
        "d_conv": 512,
        "ffn_dim": 2816,
        "n_heads": 12,
        "n_kv_heads": 4,
        "factorized_rank": 256,
        "use_xsa": true,
        "use_parcae": true,
        "mean_recurrence": 3
    },
    "training": {
        "dataset": "datasets/datamix_state/phase5_final/train.bin",
        "batch_size": 8,
        "block_size": 1024,
        "lr": 3e-4,
        "max_steps": 50000,
        "compile": true,
        "optimize_kernels": true,
        "time_budget_minutes": 360
    },
    "stages_to_run": ["smoke_test", "profile", "optimize_kernels", "train", "evaluate", "benchmark"]
}
```

### Running the Pipeline

```bash
# Full automated pipeline
python scripts/agent/orchestrator.py --config experiments/chimera_v1.json

# Resume from a specific stage (after human fixes)
python scripts/agent/orchestrator.py --config experiments/chimera_v1.json --resume-from train

# Dry run (print what would execute without running)
python scripts/agent/orchestrator.py --config experiments/chimera_v1.json --dry-run
```

---

## 15.7 The Report Generator

After the pipeline completes, generate a structured report:

```python
# scripts/agent/run_report.py
"""Generate a markdown report summarizing the experiment."""
import json
from pathlib import Path


def generate_report(experiment_dir: str) -> str:
    state = json.loads((Path(experiment_dir) / "experiment_state.json").read_text())
    
    lines = []
    lines.append(f"# Experiment Report: {state['experiment_id']}")
    lines.append(f"\n**Goal:** {state['goal']}")
    lines.append(f"\n**Status:** {state['stage']}")
    
    # Timeline
    lines.append("\n## Timeline\n")
    lines.append("| Stage | Status | Time |")
    lines.append("|-------|--------|------|")
    for entry in state["history"]:
        lines.append(f"| {entry['stage']} | {entry['status']} | {entry['elapsed']}s |")
    
    # Results per stage
    for stage, result in state["results"].items():
        lines.append(f"\n## {stage.replace('_', ' ').title()}\n")
        
        if "metrics" in result:
            lines.append("| Metric | Value |")
            lines.append("|--------|-------|")
            for k, v in result["metrics"].items():
                lines.append(f"| {k} | {v} |")
        
        if "benchmarks" in result:
            lines.append("\n| Benchmark | Score |")
            lines.append("|-----------|-------|")
            for k, v in result["benchmarks"].items():
                lines.append(f"| {k} | {v} |")
        
        if "warnings" in result and result["warnings"]:
            lines.append("\n**Warnings:**")
            for w in result["warnings"]:
                lines.append(f"- {w}")
    
    # Comparison to baselines
    if "evaluate" in state["results"]:
        eval_result = state["results"]["evaluate"]
        baselines = state["results"].get("benchmark", {}).get("baselines", {})
        
        lines.append("\n## Baseline Comparison\n")
        lines.append("| Model | HellaSwag | ARC | Average |")
        lines.append("|-------|-----------|-----|---------|")
        
        ours = eval_result.get("benchmarks", {})
        lines.append(f"| **Ours** | {ours.get('hellaswag', '?')} | {ours.get('arc_challenge', '?')} | {eval_result.get('average', '?')} |")
        
        for name, scores in baselines.items():
            lines.append(f"| {name} | {scores.get('hellaswag', '?')} | {scores.get('arc_avg', '?')} | {scores.get('average', '?')} |")
    
    return "\n".join(lines)
```

---

## 15.8 Extending the Agent: LLM-Powered Stages

The research, design, and fix stages benefit enormously from LLM integration. Instead of hardcoded heuristics, use an LLM to reason about failures and propose fixes.

```python
# scripts/agent/llm_helper.py
"""LLM-assisted decision making for the agent pipeline."""
import json


def ask_llm_for_fix(stage: str, result: dict, config: dict, 
                     api_fn=None) -> dict:
    """Ask an LLM to diagnose a failure and propose a config fix.
    
    api_fn: callable that takes a prompt string and returns a response string.
            Can be anthropic, openai, or a local model.
    """
    
    prompt = f"""You are an ML engineer debugging a training pipeline failure.

Stage: {stage}
Result: {json.dumps(result, indent=2)}
Current config: {json.dumps(config, indent=2)}

Diagnose the failure and suggest a specific config change to fix it.
Respond with JSON only: {{"diagnosis": "...", "fix": {{"key": "value", ...}}}}
"""
    
    if api_fn is None:
        # Fallback to heuristic fixes
        return auto_fix(stage, result, config)
    
    response = api_fn(prompt)
    
    try:
        fix = json.loads(response)
        config.update(fix.get("fix", {}))
        print(f"  LLM diagnosis: {fix.get('diagnosis', 'unknown')}")
        print(f"  LLM fix: {fix.get('fix', {})}")
    except json.JSONDecodeError:
        print(f"  LLM response not parseable, falling back to heuristic")
    
    return config
```

---

## 15.9 Practical Walkthrough: Running an Automated Experiment

Here is what a full automated run looks like end-to-end:

```bash
$ python scripts/agent/orchestrator.py --config experiments/chimera_v1.json

============================================================
Stage: smoke_test
Config: agent_work/exp_1713700000/smoke_test_config.json
============================================================
  Step 50: loss=9.2341
  Step 100: loss=7.8923
  Step 150: loss=7.1234
  Step 200: loss=6.8765
{
  "status": "pass",
  "checks": {"nan_free": true, "loss_decreasing": true, ...},
  "metrics": {"initial_loss": 10.82, "final_loss": 6.88, ...}
}

============================================================
Stage: profile
============================================================
  Top bottlenecks:
    1. matmul (62.3%)
    2. softmax (8.1%)
    3. rmsnorm (5.4%)
    4. silu_gate_mul (4.2%)
    5. rotary_embedding (2.8%)
  Optimization targets: rmsnorm, silu_gate_mul, rotary_embedding

============================================================
Stage: optimize_kernels
============================================================
  Pattern: rmsnorm → matched 25 modules, speedup 2.8x, correctness PASS
  Pattern: silu_gate_mul → matched 12 modules, speedup 1.6x, correctness PASS
  Pattern: fused_qkv → matched 4 modules, speedup 1.3x, correctness PASS
  Combined: 1.34x end-to-end speedup

============================================================
Stage: train
============================================================
  Step 1000: loss=5.4321
  Step 5000: loss=4.5678
  Step 10000: loss=4.2345
  Step 25000: loss=4.0123
  Step 50000: loss=3.8765
  Training complete. Checkpoint: agent_work/exp_1713700000/checkpoints/step_50000.pt

============================================================
Stage: evaluate
============================================================
  hellaswag: 38.2
  arc_easy: 45.1
  arc_challenge: 28.3
  piqa: 66.8
  winogrande: 51.2
  Average: 45.9

============================================================
Stage: benchmark
============================================================
  | Model          | HellaSwag | ARC  | Average |
  |----------------|-----------|------|---------|
  | SmolLM2-135M   | 42.1      | 43.9 | 24.05   |
  | Nandi-150M     | 37.2      | —    | 25.63   |
  | **Ours (94M)** | **38.2**  | 45.1 | **45.9**|

============================================================
PIPELINE COMPLETE
Experiment: exp_1713700000
Results: agent_work/exp_1713700000/
============================================================
```

---

## Exercises

1. **Implement the smoke test stage.** Write `scripts/agent/run_smoke_test.py` using the contract from Section 15.2. Test it with your GPT-2 model from Part 02. Verify it returns structured JSON with all 6 checks.

2. **Build a retry loop.** Intentionally set LR=100 in your config (will cause NaN). Verify the agent detects the failure, reduces LR, and retries successfully.

3. **Add a new stage.** Implement `run_profile.py` that wraps `torch.profiler`, identifies the top-5 slowest ops, and returns them as structured JSON. Wire it into the orchestrator DAG.

4. **LLM-assisted fixing.** If you have an API key, implement the `ask_llm_for_fix` function. Intentionally break the smoke test (wrong model class name) and see if the LLM can diagnose and fix it.

5. **Full pipeline dry run.** Create a config JSON for your architecture from Part 11. Run the orchestrator with `--dry-run` to verify the DAG traversal is correct without executing any stages.

---

## Checkpoint

Before considering this series complete:
- [ ] Orchestrator DAG traversal works (stages execute in order)
- [ ] Smoke test stage returns structured JSON with pass/fail
- [ ] Retry logic triggers on NaN (reduces LR) and OOM (reduces batch size)
- [ ] Escalation triggers after 3 consecutive failures
- [ ] Report generator produces readable markdown
- [ ] You can explain the full pipeline: research → design → smoke → profile → optimize → train → eval → benchmark → report

---

## What You Built

Over 15 parts, you built every component of an LLM research lab:

| Part | Component | Status |
|------|-----------|--------|
| 01 | Hardware profiling | Manual |
| 02 | Training stack | Manual |
| 03 | Profiling | Manual |
| 04-05 | CUDA kernels | Manual |
| 06-07 | Autokernel + compile | Automated (one-liner API) |
| 08 | Data pipeline | Automated (CLIMB + scoring) |
| 09 | Eval framework | Automated (harness + comparison) |
| 10 | Math foundations | Your brain |
| 11 | Architecture design | Semi-automated (papers → spec → code) |
| 12 | SFT + alignment | Manual (data-dependent) |
| 13 | Inference | Automated (server) |
| 14 | Integration | Manual |
| **15** | **Agent orchestration** | **Fully automated pipeline** |

The agent does not replace your understanding — it replaces your tedium. You designed the pipeline. The agent runs it.

---

**Previous: [Part 14 -- Putting It All Together](14_putting_it_together.md)**
**Back to: [Tutorial Index](README.md)**

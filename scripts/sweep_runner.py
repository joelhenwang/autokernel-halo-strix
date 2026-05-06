"""Sequential single-node sweep runner for throughput optimization.

Runs a list of training configs through halo_training with different
hyperparameter overrides, measures steady-state throughput, and appends
structured results to a JSONL file.

Usage:
    python3 scripts/sweep_runner.py \\
        --configs scripts/sweep_configs_a.json \\
        --output docs/perf/ddp-sweep-a.jsonl \\
        --log-dir /tmp/sweep-a

Each config must have:
  name           — human-readable label
  block_size     — sequence length
  batch_size     — microbatch size
  accum_steps    — gradient accumulation steps
  num_workers    — dataloader workers
Optional:
  lr             — default 8e-4
  max_steps      — default 100 (50 warmup + 50 measure)
  dataset        — default datasets/wikitext-103-odin32k.bin
"""

import argparse
import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path


def run_config(cfg, output_jsonl, log_dir, repo_root, compile_mode):
    """Run one halo_training config, parse log, append result to JSONL."""
    name = cfg["name"]
    dataset = cfg.get("dataset", "datasets/wikitext-103-odin32k.bin")
    max_steps = cfg.get("max_steps", 100)
    lr = cfg.get("lr", 8e-4)

    cmd = [
        "python3", "-m", "halo_training",
        "--model", "models/odin_flat.py",
        "--class-name", "OdinFlat",
        "--dataset", dataset,
        "--tokenizer-path", cfg.get("tokenizer_path", "tokenizers/odin-32k/tokenizer.json"),
        "--compile",
        "--epochs", "1",
        "--log-interval", "10",
        "--max-steps", str(max_steps),
        "--block-size", str(cfg["block_size"]),
        "--batch-size", str(cfg["batch_size"]),
        "--accum-steps", str(cfg["accum_steps"]),
        "--num-workers", str(cfg["num_workers"]),
        "--lr", str(lr),
    ]

    env = os.environ.copy()
    env["TORCH_COMPILE_MODE"] = compile_mode

    log_file = log_dir / f"{name}.log"
    print(f"\n[{time.strftime('%H:%M:%S')}] Running {name}...")
    print(f"  cmd: {' '.join(cmd)}")
    start = time.time()
    with open(log_file, "w") as lf:
        result = subprocess.run(
            cmd, stdout=lf, stderr=subprocess.STDOUT,
            env=env, cwd=repo_root,
        )
    elapsed = time.time() - start

    # Parse log for per-step metrics
    log_text = log_file.read_text(errors="replace")
    # Match lines like: [step     50] loss=... tok/s=12,345 ... mem=6.1GB
    pattern = re.compile(
        r"\[step\s+(\d+)\]\s+loss=([\d.]+).*?tok/s=([\d,]+).*?mem=([\d.]+)GB"
    )
    steps = pattern.findall(log_text)

    result_dict = {
        "name": name,
        "block_size": cfg["block_size"],
        "batch_size": cfg["batch_size"],
        "accum_steps": cfg["accum_steps"],
        "num_workers": cfg["num_workers"],
        "eff_batch_seqs": cfg["batch_size"] * cfg["accum_steps"],
        "eff_batch_tokens": cfg["batch_size"] * cfg["accum_steps"] * cfg["block_size"],
        "lr": lr,
        "max_steps": max_steps,
        "wall_time_s": round(elapsed, 1),
        "returncode": result.returncode,
    }

    if not steps:
        print(f"  WARN: no step lines found in log (rc={result.returncode})")
        result_dict.update({
            "mean_tok_s": None, "peak_mem_gb": None,
            "final_loss": None, "n_measured": 0,
            "status": "no_steps_found",
        })
    else:
        parsed = [(int(s), float(l), int(t.replace(",", "")), float(m))
                  for s, l, t, m in steps]
        # Use steps past step 30 (warmup / compile)
        measured = [p for p in parsed if p[0] >= 30]
        if not measured:
            measured = parsed[-3:] if len(parsed) >= 3 else parsed
        mean_tok_s = sum(t for _, _, t, _ in measured) / len(measured)
        peak_mem = max(m for _, _, _, m in measured)
        final_loss = measured[-1][1]
        result_dict.update({
            "mean_tok_s": round(mean_tok_s, 0),
            "peak_mem_gb": round(peak_mem, 2),
            "final_loss": round(final_loss, 4),
            "n_measured": len(measured),
            "status": "ok" if result.returncode == 0 else f"rc={result.returncode}",
        })
        print(f"  -> tok/s={mean_tok_s:,.0f}, mem={peak_mem:.1f}GB, "
              f"loss={final_loss:.3f}, rc={result.returncode}, n={len(measured)}")

    with open(output_jsonl, "a") as f:
        f.write(json.dumps(result_dict) + "\n")
    return result_dict


def main():
    parser = argparse.ArgumentParser(description="Sequential sweep runner")
    parser.add_argument("--configs", required=True, help="JSON file with config list")
    parser.add_argument("--output", required=True, help="JSONL output file")
    parser.add_argument("--log-dir", required=True, help="Per-config log directory")
    parser.add_argument("--repo-root", default=".",
                        help="Working directory for halo_training")
    parser.add_argument("--compile-mode", default="max-autotune-no-cudagraphs")
    args = parser.parse_args()

    log_dir = Path(args.log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(args.configs) as f:
        configs = json.load(f)

    print(f"Sweep runner: {len(configs)} configs")
    print(f"  Output:  {output_path}")
    print(f"  Logs:    {log_dir}")
    print(f"  Compile: {args.compile_mode}")
    print(f"  Repo:    {args.repo_root}")

    # Don't truncate; allow resuming / appending
    if not output_path.exists():
        output_path.touch()

    total_start = time.time()
    results = []
    for i, cfg in enumerate(configs, 1):
        print(f"\n=== [{i}/{len(configs)}] ===")
        r = run_config(cfg, output_path, log_dir, args.repo_root, args.compile_mode)
        results.append(r)

    total_elapsed = time.time() - total_start
    print(f"\n=== Sweep done in {total_elapsed:.0f}s "
          f"({total_elapsed/60:.1f} min, {len(configs)} configs) ===")
    print(f"Results in: {output_path}")


if __name__ == "__main__":
    main()

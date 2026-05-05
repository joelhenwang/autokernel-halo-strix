# CLAUDE.md

AutoKernel: autonomous GPU kernel optimization + model training on AMD Strix Halo (gfx1151, RDNA 3.5, ROCm 7.12).

> **ALWAYS read [STATUS.md](STATUS.md) first** — current runs, best checkpoints, corrupted checkpoints, hard rules.
> **All constraints are in [CONSTRAINTS.md](CONSTRAINTS.md)** — 28 checklist items, machine-parseable. Read it before writing HIP or launching training.

## Setup & Commands

```bash
uv sync && uv sync --extra hip
uv run python scripts/validate_env.py

# Kernel optimization: profile.py → extract.py → bench.py (agent loop) → verify.py
# Training: always --epochs N (minimum 1)
python -m halo_training --model models/vidar_halo.py --class-name VidarHalo --compile --optimize-kernels --dataset babylm --epochs 1
```

For remote commands use `bash run_remote.sh` / `bash run_remote_b.sh` — never raw SSH.
Monitor long runs via `bash run_remote.sh "tail -5 checkpoints/<run>/train_log.jsonl"`.

## Workflow

| Phase | Tool | What |
|-------|------|------|
| A | `profile.py` → `extract.py` | Profile model, extract bottleneck kernels |
| B | Agent edits `kernel.py` → `bench.py` | Optimize HIP kernels autonomously (keep/revert loop) |
| C | `verify.py` | Plug kernels into model, e2e correctness + speedup |
| D | `python -m halo_training` | Train with optimized kernels (Mode A <2B, Mode B >2B) |

`kernel.py` is the ONLY file modified during optimization. `bench.py` and `reference.py` are IMMUTABLE.

## Key Directories

| Dir | Purpose |
|-----|---------|
| `models/` | Model definitions (shared components in `models/_components.py`) |
| `kernels/hip/` | HIP kernel types + `_compile.py` (compilation) + `_torch_ops.py` (autograd) |
| `halo_training/` | Training stack: `python -m halo_training` |
| `knowledge/` | Reference docs (hardware/, kernels/, training/, architectures/) |
| `scripts/` | Utilities (training, benchmarking, tokenization) |
| `autokernel/` | Library API: `autokernel.optimize()` |

## Where to Find Everything

| Topic | Location |
|-------|----------|
| Hardware specs | `knowledge/hardware/amd_rdna35_strix_halo.md` |
| Kernel benchmarks + speedups | `knowledge/kernels/kernel_benchmarks.md` |
| Training anti-patterns | `knowledge/training/training_antipatterns.md` |
| Architecture design specs | `docs/superpowers/specs/` |
| Operational runbooks | `docs/halo-strix-apu/` |
| Historical hypothesis plans | `mad_llm_scientist/ARCHIVED.md` |
| Canonical results | `REPORT.md` |
| Doc index | `KNOWLEDGE_GRAPH.md` |

## graphify

This project has a graphify knowledge graph at `graphify-out/`.
- Before answering architecture questions, read `graphify-out/GRAPH_REPORT.md`
- After modifying code files, run `python -c "from graphify.watch import _rebuild_code; from pathlib import Path; _rebuild_code(Path('.'))"`
# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

AutoKernel is an autonomous AI agent system for GPU kernel optimization and model training on AMD Strix Halo (gfx1151, RDNA 3.5 APU, ROCm 7.12). The `halo_training/` package uses these optimized kernels for fast pretraining (3.05x speedup, 54% MFU).

## Hardware (Quick Reference)

GPU: Radeon 8060S (gfx1151), 40 CUs, wave32, **no MFMA**, ~59.4 TFLOPS FP16. Memory: 128 GB LPDDR5X (~240 GB/s), unified CPU+GPU. CPU: 16 Zen 5 cores, AVX-512. LDS: 64 KB/CU. L2: 6 MB. See `knowledge/hardware/amd_rdna35_strix_halo.md` for full reference.

## Commands

### Note: If the Halo Strix machine is the remote machine, use `run_remote.sh` to run commands on the remote machine with the Halo Strix hardware. Quickly go through the script to understand the workspace path and venv in the remote machine.

### Monitoring long training runs
Do NOT rely on SSH stdout for monitoring — SSH sessions timeout on long runs (exit code 255). Instead, check progress by tailing the training log on the remote:
```bash
bash run_remote.sh "tail -5 checkpoints/<run_name>/train_log.jsonl"
bash run_remote.sh "ps aux | grep python3 | grep -v grep"  # check if still running
```

```bash
# Setup (uv package manager, Python 3.10+, PyTorch 2.10.0+rocm7.12)
uv sync                          # core deps
uv sync --extra models           # + HuggingFace
uv sync --extra hip              # + ninja for HIP compilation
uv run python scripts/validate_env.py  # validate ROCm + gfx1151

# Kernel optimization loop
uv run python bench.py --kernel kernels/hip/reduce.py --reference reference.py
uv run python profile.py --model models/llama_7b.py
uv run python extract.py
uv run python orchestrate.py

# End-to-end verification
uv run python verify.py --model models/llama_7b.py --class-name LlamaModel --input-shape 1,512

# Training (Mode A auto for <2B, Mode B for >2B)
python -m halo_training --model models/llama_7b.py --class-name LlamaModel --dataset babylm
python -m halo_training --model models/llama_7b.py --class-name LlamaModel --compile --optimize-kernels
python -m halo_training --model models/argus_prime.py --class-name ArgusPrime --resume-from checkpoints/argus_prime_gpt/step_12000.pt --dataset datasets/wikitext-103-raw-train --compile --optimize-kernels --muon
```

## Workflow

**Phase A:** Human provides model -> `profile.py` -> `extract.py` -> confirm plan.
**Phase B:** Agent modifies `kernel.py` (single file, HIP_SRC string) -> `bench.py` evaluates (5-stage correctness + roofline). `orchestrate.py` schedules by Amdahl's law.
**Phase C:** `verify.py` plugs kernels into model, runs e2e correctness + benchmark.
**Phase D:** `halo_training/` trains with optimized kernels. Mode A (compile, <2B) or Mode B (streaming + checkpointing, >2B).

## Architecture Constraints

- **`kernel.py`** is the ONLY file modified during optimization. Contains `HIP_SRC` string.
- **`reference.py`** and **`bench.py`** are IMMUTABLE.
- **`program.md`** contains full agent instructions for autonomous operation.

## Key Directories

- `kernels/hip/` — 20+ HIP kernel types + `_compile.py` (compilation) + `_torch_ops.py` (torch.library custom ops)
- `models/` — Self-contained model definitions (LLaMA, GPT-2, AMADEUS, TEMPEST, ARGUS-PRIME, etc.)
- `halo_training/` — Composable training stack (Mode A/B), CLI: `python -m halo_training`
- `autokernel/` — Library API (`autokernel.optimize()`) with pattern matching + kernel replacement
- `knowledge/` — Organized reference docs (hardware/, kernels/, training/, architectures/)

## Key Constraints (always remember)

- **No MFMA** — can't beat rocBLAS for matmul/attention. Don't put matmuls in HIP kernels.
- **Unified memory** — gloo matches NCCL for DDP (no GPU-CPU copy penalty)
- **wave32** not wave64
- **fp16 + GradScaler**, NOT bf16 (24% slower, compile crashes)
- **autokernel breaks d<=256** — skip `--optimize-kernels` for small hidden dims
- **Don't break autokernel FusedQKV+RoPE pattern** — manual QKV fusion loses 3.7x speedup
- **torch.compile model only**, never the optimizer (29GB memory blowup)
- **Check train_log.jsonl** for progress, don't rely on SSH stdout for long runs
- **EOS token** (50256, `<|endoftext|>`) inserted between documents in `halo_training/data.py`
- **Autokernel before checkpoint load** — fused QKV keys must exist before `load_state_dict()`

## Training Target

- **Phase 1:** Beat LFM2.5-350M on standard benchmarks (HellaSwag, ARC, MMLU)
- **Phase 2:** Instruction-tune for on-device Strix Halo assistant
- **Dataset funnel:** smoke -> BabyLM -> GPT-training-small -> Dolma 10B -> Dolma 100B

## Where to Find Everything Else

Start with **[KNOWLEDGE_GRAPH.md](KNOWLEDGE_GRAPH.md)** — master index of all documentation.

| Topic | Location |
|-------|----------|
| Full hardware specs, rocBLAS, profiling | `knowledge/hardware/` |
| Kernel benchmarks, external libs, fused ops | `knowledge/kernels/` |
| DDP setup, Muon, ARGUS-PRIME results, anti-patterns | `knowledge/training/` |
| Architecture rankings, hypothesis results | `knowledge/architectures/` |
| Design decisions | `docs/superpowers/specs/` |
| Operational guides, commands, runbooks | `docs/halo-strix-apu/` |
| Architecture plans (30 hypotheses) | `mad_llm_scientist/plans/` |
| Canonical results summary | `REPORT.md` |

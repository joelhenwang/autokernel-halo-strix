# AGENTS.md

AutoKernel: autonomous GPU kernel optimization + halo model training on AMD Strix Halo (gfx1151, RDNA 3.5, ROCm 7.12).

## Before anything else

- **Read [STATUS.md](STATUS.md) first** — active runs, best/corrupted checkpoints, hard rules.
- **Read [CONSTRAINTS.md](CONSTRAINTS.md) before writing HIP or launching training** — 28-item machine-parseable checklist.

## Remote machines

All compute runs on two Linux Strix Halo machines. Local Windows is for editing only.

```bash
bash sync_remote.sh         # sync code to Machine A (clean, dedicated venv)
bash sync_remote_b.sh       # sync code to Machine B (comfyui venv, has aiter)

bash run_remote.sh "..."    # run command on Machine A — NEVER use raw SSH
bash run_remote_b.sh "..."  # run command on Machine B
```

Sync code before running anything on a remote.

## Training commands

```bash
# Smoke test (200 steps, all PASS = model viable)
python -m halo_training --model models/vidar_halo.py --class-name VidarHalo --smoke

# Real training — always --epochs N (minimum 1)
python -m halo_training --model models/odin_halo.py --class-name OdinHalo \
    --compile --optimize-kernels --dataset babylm --epochs 1

# Full production (OdinHalo, 32K tokenizer, dolma-10b, WSD)
python -m halo_training --model models/odin_halo.py --class-name OdinHalo \
    --dataset datasets/dolma-10b-vidar32k.bin --epochs 1 \
    --compile --polar-ns --muon --scheduler wsd --min-lr-ratio 0.1 \
    --ema --z-loss 1e-4 --block-size 256 \
    --tokenizer-path tokenizers/vidar-32k/tokenizer.json

# DDP smoke test (never full epoch)
--max-steps 300 --time-budget 20

# DDP full training (2 machines over Thunderbolt 4)
# Launch from Machine A — it SSHs into Machine B to start rank 1
bash run_remote.sh "cd ~/Desktop/ai_lab/autokernel-halo-strix && bash scripts/launch_ddp.sh"

# OdinFlat on wikitext-103 (one epoch, ~1869 steps, ~39K tok/s aggregate)
TORCH_COMPILE_MODE=max-autotune-no-cudagraphs
torchrun --nnodes=2 --nproc_per_node=1 \
  --master_addr=10.77.0.1 --master_port=29500 \
  scripts/train_ddp.py --model models/odin_flat.py --class-name OdinFlat \
  --dataset datasets/wikitext-103-odin32k.bin --epochs 1 \
  --block-size 256 --batch-size 16 --accum-steps 8 \
  --compile --no-muon --lr 8e-4 --backend gloo
```

Monitor long runs: `bash run_remote.sh "tail -5 checkpoints/<run>/train_log.jsonl"`.

## Hardware constraints (gfx1151)

- **No MFMA** — can't beat rocBLAS for matmul/attention. Never put matmuls in HIP kernels.
- **bf16 NOT supported** — 24% slower, compile crashes. Always fp16 + GradScaler.
- **wave32** not wave64. **LDS: 64 KB/CU**, **L2: 6 MB**.
- Unified memory — gloo matches NCCL for DDP.

## Model architecture

```
models/
  _components.py          low-level primitives (RMSNorm, SwiGLU, GatedConv, RoPE)
  components/             shared halo blocks — import from here, not other .py files:
    attention.py            Attention, CodaAttention (XSA), NoPECodaAttention (HyPE)
    conv_blocks.py          ShortConvBlock, GQABlock, MoDAGQABlock, HyPEShortConvBlock
    embeddings.py           FactorizedEmbedding, FactorizedLMHead
    injection.py            SimpleParcaeInjection
    loop_utils.py           HyperloopHC, DepthMemoryCache
    mtp.py                  MTPHead (single source; was duplicated)
    speculative.py          DraftHeads, ForecastEmbeddings
  odin_halo.py            ODIN-HALO (58M/156M, d=768, 6L×3iter, HyPE, looped)
  odin_flat.py            ODIN-FLAT (122M, d=768, 14L, HyPE, non-looped)
  vidar_halo.py, tyr_halo.py, ...  (7 more halo variants)
```

New models: import from `from models.components import X`, never from other model `.py` files. New shared components go into `models/components/`, not into individual model files.

## Training gotchas

- `--optimize-kernels` is ~3x throughput (35K vs 10K tok/s). Always use for real training. Breaks at `d <= 256` — skip for small hidden dims.
- `--compile` the model only, never the optimizer (29 GB memory blowup).
- Looped models: trainer auto-uses `model.compile_zones()` for per-layer compilation when available. Velocity clamp `±8.0` required for fp16 Parcae loops (prevents NaN at cost of ~22% tok/s).
- `--ema` always: free +7.5% generalization, decay 0.999.
- No momentum in looped models — MoDA + loop_pos + skip carry cross-iter info; momentum blocks break FusedResidualRMSNorm.
- MTP head: ~45% throughput cost, no evidence at sub-100M scale. ODIN-HALO omits it.
- WSD schedule: `--scheduler wsd --min-lr-ratio 0.1` (MIN_LR=10%, don't decay to zero).
- Logit softcap=30: stability in fp16 looped models.
- **`max-autotune` crashes with `accum_steps > 1`** (CUDA graph buffer overwrite). Use `max-autotune-no-cudagraphs` for any training with gradient accumulation. Same Triton autotuning benefit, no graph capture attempt.

## CE + chunked-CE stack (2026-05-05)

- `kernel.py` has **online-softmax fused CE** with `softcap`/`z_loss`/`label_smoothing`/`ignore_index` baked in. Call via `kernel.ce_full(logits, targets, mode="tiny"|"fused")`. `mode="tiny"` wins at small batch (prod scale), `mode="fused"` for isolated benchmarks.
- `--chunked-ce` flag activates `ChunkedLinearCrossEntropyLoss` which avoids materializing `[N, V]` logits tensor. Saves 1-3 GB peak at batch=16-32. Model must have `use_chunked_ce=True` or accept it as ctor arg.
- For OdinHalo: `use_chunked_ce=True` makes `FactorizedLMHead` return `h_low` (rank-dim) instead of logits during training; trainer calls `chunked_ce(h_low, embed_table.weight, targets)`.
- Tied-weight grad accumulates through PyTorch autograd — no special handling needed.

## Compile strategy

- For **looped models (HALO)**: trainer auto-uses `compile_zones()` per-layer. Don't override.
- **Compile lift scales with batch size**:
  - batch=4: ~1.08× (low value)
  - batch=16: ~1.32× (recommended minimum for compile benefit)
  - batch=32: GPU saturated, throughput plateaus
- **`TORCH_COMPILE_MODE=max-autotune`** is the **production-recommended** mode for OdinHalo. Phase 3 WI-B3 measured **+5.17% tok/s** (14,018 → 14,742) with loss parity within fp16 noise. First-compile: ~2 min (autotune search, one-time). Warm-cache: ~9 s. Usage: `TORCH_COMPILE_MODE=max-autotune python -m halo_training --model models/odin_halo.py ...`. Trainer threads the env var through `compile_zones(mode=...)`.
- **`TORCH_COMPILE_MODE=reduce-overhead`** does NOT crash on looped models (Phase 3 WI-A0 debunked the prior claim) but HIP's CUDA-graph backend produces "empty graph" warnings and runs eagerly — net −1.8% throughput, no memory benefit. Trainer auto-redirects to `default` with a NOTE. See `docs/perf/phase3-wi-a0-consolidated.md`.
- HIP `fused_rope_gate_mul` and Inductor-generated RoPE fusion are performance-equivalent under compile; no need to force HIP fusion.
- `HyPEShortConvBlock._compile_friendly = True` swaps HIP kernels for native PyTorch equivalents. 0 graph breaks vs 4 default. But NOT faster — provided for fullgraph compilation only.

## Fused AdamW

- Always use `torch.optim.AdamW(params, lr=..., fused=True)` — this is +12% over default AdamW.
- Default in `halo_training/optimizer.py`.
- Unfused AdamW takes ~19% of step time (see `scripts/profile_step.py`).

## RoPE complex-tensor bug (2026-05-05 fixed)

`freqs_cis.real` and `freqs_cis.imag` return **non-contiguous views** (complex memory stores `[real, imag]` interleaved fp32 pairs → stride 2). HIP kernels that take these as "contiguous float arrays" read interleaved `[real, imag, real, imag, ...]` and treat every other value as cos → silent RoPE miscompute.

**Rule**: ALWAYS call `.contiguous()` before `.float()` when extracting real/imag parts of a complex tensor for pass to a HIP kernel:

```python
# WRONG — non-contig view, HIP reads wrong data
freqs_cos = freqs_cis.real[:T, :pairs].float()

# CORRECT
freqs_cos = freqs_cis.real[:T, :pairs].contiguous().float()
```

Already fixed in `models/components/conv_blocks.py` HyPEShortConvBlock. All other halo models use `apply_rotary_emb` from `models/_components.py` which uses `torch.view_as_complex`/`torch.view_as_real` — not affected by this bug.

## Data

- Custom 32K tokenizer: `tokenizers/vidar-32k/tokenizer.json` (EOS=0, not 50256).
- `datasets/dolma-10b-vidar32k.bin` — 6.9B tokens, 13.8 GB. Loaded via `np.memmap` (zero-copy).
- BabyLM used for smoke tests (`--smoke` or `--dataset babylm`). No bulk allocation for >1GB datasets.

## Kernel optimization workflow

| Phase | Tool | What |
|-------|------|------|
| A | `profile.py` → `extract.py` | Profile model, extract bottleneck kernels |
| B | Agent edits `kernel.py` → `bench.py` | Optimize HIP kernels (keep/revert loop) |
| C | `verify.py` | Plug kernels into model, e2e correctness + speedup |
| D | `python -m halo_training` | Train with optimized kernels |

`kernel.py` is ONLY file modified during optimization. `bench.py` and `reference.py` are IMMUTABLE.

## graphify

Knowledge graph at `graphify-out/`. Read `GRAPH_REPORT.md` before answering architecture questions. Rebuild after code changes:

```bash
python -c "from graphify.watch import _rebuild_code; from pathlib import Path; _rebuild_code(Path('.'))"
```
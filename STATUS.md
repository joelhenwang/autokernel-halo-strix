# Current Training Status

> **Read this file before every training launch or decision.**
> Update after every run completes or fails.

---

## Active Model

**VIDAR-HALO** (`models/vidar_halo.py`, class `VidarHalo`)
- 47.0M unique / 95M effective params
- d=768, 4 shared layers (3 ShortConv + 1 MoDA-GQA) × 2 Parcae iterations
- No momentum (direct residuals) — 22% faster than momentum variants
- Tokenizer: `tokenizers/vidar-32k/tokenizer.json` (EOS=0, PAD=1, vocab=32000)
- Dataset: `datasets/dolma-10b-vidar32k.bin` (6.9B tokens, 13.8 GB, both machines)

**Variants:**
- `VidarHalo` — 32K vocab (production)
- `VidarHaloGPT2` — 50257 vocab (for GPT-2 .bin files like stem-crawl-solo.bin)
- `VidarHaloMini` — d=128 smoke test (~1.7M params)

**Secondary: FENRIR-HALO** (`models/fenrir_halo.py`, class `FenrirHalo`)
- 80.8M unique, d=640, 10 shared layers × Parcae mean=3
- Velocity clamp ±8.0 added (fix for NaN at step 30800)
- GPT-2 tokenizer (50257 vocab)

---

## Active Training Runs

| Machine | Model | Dataset | Config | Step | Loss | tok/s | Started | Status |
|---------|-------|---------|--------|------|------|-------|---------|--------|
| — | — | — | — | — | — | — | — | idle |

## Latest Smoke Test (2026-05-04)

VIDAR-HALO DDP, 32K tokenizer, compile (no autokernel), AdamW, --max-steps 300:

| Step | Loss | BPB | tok/s | Memory |
|------|------|-----|-------|--------|
| 300 | 6.64 | 2.66 | **41,004** | 8.9 GB |

**41K tok/s achieved** — target was 40K. 32K tokenizer saves 3 GB memory vs GPT-2 vocab.

---

## Best Checkpoints (verified clean)

| Model | Path | Loss/BPB | Dataset | Tokens Seen | Notes |
|-------|------|----------|---------|-------------|-------|
| FenrirHalo | `fenrir_halo_babylm/step_2566.pt` | 3.16 / 1.26 | BabyLM (16M) | ~16M | Clean, well-converged |
| FenrirHalo | `fenrir_halo_dolma/step_15000.pt` | ~4.2 / ~1.7 | dolma-10b (7B) | ~500M | Last clean before NaN zone |
| TyrHaloLight | `tyr_light_e1_v3/step_8299.pt` | 5.08 / 2.10 | stem-crawl (544M) | 544M | Epoch 1 winner vs BALDR |
| BaldrHalo | `baldr_halo_e1_lr002/step_8299.pt` | 6.87 / 2.75 | stem-crawl (544M) | 544M | Lost to TyrHaloLight |
| VidarHalo | `vidar_smoke_32k/step_300.pt` | ~25.8 / ~10.4 | dolma-10b-vidar32k | ~10M | Smoke test only (300 steps, eager) |

---

## Corrupted / Unusable Checkpoints

| Path | Reason |
|------|--------|
| `fenrir_halo_dolma/step_25000.pt` | Trained without velocity clamp; NaN on resume |
| `fenrir_halo_dolma/step_30000.pt` | Same — weights damaged by unclamped velocity |
| `fenrir_halo_dolma/step_105000.pt` | Deep NaN; all steps after 30800 are corrupted |
| `fenrir_halo_dolma_r2/*` | Grad NaN at step 3200; short run, not useful |

---

## Known Issues & Hard Rules

1. **Always use `--optimize-kernels`** for real training. Without it: 10K tok/s. With it: 35K tok/s. The difference is autokernel fusing RMSNorm+SwiGLU+QKV.
2. **Always use .sh scripts** for remote commands: `run_remote.sh` (Machine A), `run_remote_b.sh` (Machine B). Never raw SSH.
3. **bf16 is NOT supported** on gfx1151. Always fp16 or fp32. bf16 is 24% slower, compile crashes.
4. **DDP smoke tests**: `--max-steps 300 --time-budget 20`. Never launch full epoch for smoke testing.
5. **Single-machine smoke tests**: `python -m halo_training ... --smoke` (200 steps built-in).
6. **Large .bin files use memmap** — no bulk np.fromfile for datasets >1GB. Both `data.py` and `train_ddp.py` use zero-copy memmap.
7. **FENRIR checkpoints after step_25000 are corrupted** — trained without velocity clamp. Use step_15000 or earlier.
8. **Custom tokenizer EOS=0** (not 50256). Check tokenizer config before hardcoding EOS.

---

## Throughput Reference

| Config | Model | tok/s | Notes |
|--------|-------|-------|-------|
| Single eager | VidarHalo (47M) | 18,929 | No compile, no autokernel |
| Single compiled (fwd+bwd only) | VidarHalo | 31,362 | No optimizer overhead |
| Single AdamW+compile | VidarHalo | ~16,800 | CE on vocab=50257 was bottleneck |
| DDP AdamW (no autokernel) | VidarHaloGPT2 | 34,541 global | 2 machines, TB4, vocab=50257 |
| **DDP AdamW+compile, 32K tok** | **VidarHalo** | **41,004 global** | **2 machines, TB4, vocab=32000** |
| DDP Muon+compile+autokernel | FenrirHalo (80M) | 35,000 global | Original dolma run (reference) |

---

## Machine Info

| Machine | SSH | Venv | Project | TB4 IP | GPU |
|---------|-----|------|---------|--------|-----|
| A (rank 0) | `run_remote.sh` | `~/Desktop/ai_lab/.venv/` | `~/Desktop/ai_lab/autokernel-halo-strix/` | 10.77.0.1 | gfx1151 |
| B (rank 1) | `run_remote_b.sh` | `~/Desktop/comfyui-rocm7.12/.venv/` | `~/Desktop/comfyui-rocm7.12/autokernel-halo-strix/` | 10.77.0.2 | gfx1151 |

DDP: `GLOO_SOCKET_IFNAME=thunderbolt0`, `MASTER_ADDR=10.77.0.1`, backend=gloo.

---

*Last updated: 2026-05-04*

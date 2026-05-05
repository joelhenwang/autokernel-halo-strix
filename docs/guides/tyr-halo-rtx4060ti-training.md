# TYR-HALO: Training Guide for RTX 4060 Ti (16GB)

**Model:** TYR-HALO (~58M unique / ~115M Parcae-equivalent)
**Hardware:** NVIDIA RTX 4060 Ti 16GB (Ada Lovelace, sm_89)

---

## Hardware Profile

| Spec | Value |
|------|-------|
| GPU | AD106, 4352 CUDA cores |
| VRAM | 16 GB GDDR6 |
| Bandwidth | 288 GB/s |
| FP16/BF16 Tensor Core | ~22 TFLOPS |
| Architecture | Ada Lovelace (sm_89) |

### Key Differences from Strix Halo (AMD)

| Setting | Strix Halo (AMD) | RTX 4060 Ti (NVIDIA) |
|---------|-------------------|----------------------|
| Precision | fp16 + GradScaler | **bf16** (no scaler needed) |
| Compile | `compile_zones()` per-layer | `torch.compile(model)` full |
| Autokernel | Yes (HIP custom ops) | **No** — skip `--optimize-kernels` |
| Attention | SDPA (no flash) | **cuDNN flash attention** (auto) |
| Batch size | 16 | **32** (dedicated VRAM) |
| CUDA graphs | No benefit | **Yes** — `mode="reduce-overhead"` |

---

## Installation

```bash
# CUDA PyTorch
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121

# Dependencies
pip install tiktoken lm-eval

# Verify
python -c "import torch; print(torch.__version__, torch.cuda.get_device_name())"
```

---

## Memory Budget

TYR-HALO at 58M params fits easily in 16GB:

```
Model weights (bf16):           ~116 MB
Optimizer states (Muon+AdamW):  ~232 MB
Activations (batch=32, ctx=1024): ~80 MB
Gradients (bf16):               ~116 MB
MTP head:                       ~26 MB
torch.compile overhead:         ~500 MB
CUDA context:                   ~500 MB
────────────────────────────────────────
Total:                          ~1.6 GB
```

Massive headroom. Use batch=32 or even 64.

---

## Quick Start

### Smoke Test

```bash
python -m halo_training --model models/tyr_halo.py --class-name TyrHaloMini --smoke
```

### BabyLM Validation (~5 min)

```bash
python -m halo_training --model models/tyr_halo.py --class-name TyrHalo \
    --dataset datasets/babylm-strict-small \
    --compile --muon --mtp --bf16 \
    --batch-size 32 --block-size 1024 --accum-steps 4 \
    --lr 0.02 --epochs 2
```

### Full Training (12B synthetic tokens, ~2.5 days)

```bash
python -m halo_training --model models/tyr_halo.py --class-name TyrHalo \
    --dataset datasets/synthetic_pretrain/train.bin \
    --compile --muon --mtp --bf16 \
    --batch-size 32 --block-size 1024 --accum-steps 8 \
    --lr 0.02 --time-budget 4320 \
    --checkpoint-interval 2000
```

---

## Precision: bf16

NVIDIA Ada Lovelace has native bf16 tensor cores — same speed as fp16 but no GradScaler needed. Always use `--bf16`.

The model has a bf16 guard for AMD hardware. On NVIDIA this is auto-detected and skipped:
```python
# In tyr_halo.py forward():
is_amd = hasattr(torch.version, 'hip') and torch.version.hip is not None
if dtype == torch.bfloat16 and is_amd:
    raise RuntimeError(...)  # Only triggers on AMD
```

---

## Compile Strategy

### Full Model (preferred on NVIDIA)

```bash
# torch.compile with CUDA graphs
python -m halo_training --model models/tyr_halo.py --class-name TyrHalo \
    --compile ...
```

The training stack applies `torch.compile(model, mode="default")`. For maximum throughput:
```python
model = torch.compile(model, mode="reduce-overhead")  # Enables CUDA graphs
```

### Fallback: Per-Zone

If full-model compile fails (Poisson sampling may cause graph breaks):
```python
model.compile_zones()  # Compiles each layer independently
```

---

## Expected Throughput

| Config | tok/s |
|--------|-------|
| Eager | ~25-30K |
| torch.compile (default) | ~50-60K |
| torch.compile (reduce-overhead) | ~60-80K |

Assumes batch=32, ctx=1024, bf16. Model is small enough that matmuls aren't the bottleneck — element-wise ops and memory bandwidth dominate.

---

## Training Time Estimates

| Dataset | Tokens | Time |
|---------|--------|------|
| BabyLM smoke | 16M | ~5 min |
| BabyLM 2 epochs | 33M | ~10 min |
| GPT-training-small | 585M | ~3 hours |
| Synthetic 1B | 1B | ~5 hours |
| Synthetic 12B | 12B | ~2.5 days |

---

## Ablation Runs

Run all variants on BabyLM to measure each mechanism's contribution:

```bash
# Baseline: single pass, no novel mechanisms
python -m halo_training --model models/tyr_halo.py --class-name TyrHaloNoLoop \
    --dataset babylm --compile --muon --bf16 --batch-size 32 --epochs 2

# Loop only (no MoDA, no mHC)
python -m halo_training --model models/tyr_halo.py --class-name TyrHaloBare \
    --dataset babylm --compile --muon --bf16 --batch-size 32 --epochs 2

# Loop + MoDA (no mHC)
python -m halo_training --model models/tyr_halo.py --class-name TyrHaloFast \
    --dataset babylm --compile --muon --bf16 --batch-size 32 --epochs 2

# Full system
python -m halo_training --model models/tyr_halo.py --class-name TyrHalo \
    --dataset babylm --compile --muon --mtp --bf16 --batch-size 32 --epochs 2
```

---

## Memory Optimization (if needed)

For larger experiments or if VRAM runs tight:

1. **Reduce batch to 16** — halves activation memory
2. **Gradient checkpointing** — add `torch.utils.checkpoint` on shared block
3. **Increase accum_steps to 16** — same effective batch, less VRAM
4. **TyrHaloMini** — d=128, ~2.3M params, fits anywhere

---

## Monitoring

```bash
# GPU utilization
nvidia-smi dmon -s u -d 5

# Training progress
tail -f checkpoints/tyr_halo_*/train_log.jsonl | python -c "
import sys, json
for line in sys.stdin:
    d = json.loads(line)
    print(f\"step={d['step']:>6} loss={d['loss']:.4f} tok/s={d.get('tok_per_sec', 0):.0f}\")
"
```

---

## Post-Training

### ES Alignment (EGGROLL-style)

```bash
python scripts/eggroll_align.py \
    --model checkpoints/tyr_halo/final.pt \
    --judge-model meta-llama/Llama-3-8B-Instruct \
    --population-size 16 --rank 4 --steps 500
```

### Int8 Speculative Drafter

```bash
python scripts/distill_drafter.py \
    --teacher checkpoints/tyr_halo/final.pt \
    --drafter-dim 64 --drafter-layers 2 \
    --int8 --steps 10000
```

RTX 4060 Ti int8 tensor cores handle drafter inference efficiently — expect 2-3x decode speedup.

---

## Evaluation

```bash
lm_eval --model hf --model_args pretrained=checkpoints/tyr_halo/ \
    --tasks hellaswag,piqa,arc_easy,winogrande,mmlu --batch_size 16
```

### Targets

| Benchmark | Portimbria-150M | TYR-HALO Target | SmolLM2-135M (stretch) |
|-----------|----------------|-----------------|------------------------|
| HellaSwag | 27.5% | >30% | 42.1% |
| PIQA | 57.6% | >58% | — |
| ARC-Easy | 33.8% | >36% | 49.0% |
| Winogrande | 52.7% | >53% | — |

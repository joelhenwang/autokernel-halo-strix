#!/bin/bash
# Phase 1: Re-run WikiText-103 at ctx=256 to get final checkpoints
# Phase 2: Continue both on WikiText-103 at ctx=1024

source ~/Desktop/ai_lab/autokernel-halo-strix/.venv/bin/activate
cd ~/Desktop/ai_lab/autokernel-halo-strix

# === PHASE 1: WikiText-103 at ctx=256 (save final checkpoints) ===

echo "========================================"
echo "Phase 1a: XSADC on WikiText-103 ctx=256 at $(date)"
echo "========================================"
rm -rf checkpoints/wt103_ctx256_XSADC
python3 -m halo_training \
    --model models/jormungandr_halo.py \
    --class-name JormungandrHaloXSADC \
    --dataset datasets/wikitext-103-raw.bin \
    --compile --optimize-kernels \
    --muon --lr 0.0012 \
    --epochs 1 \
    --batch-size 16 --block-size 256 \
    --checkpoint-dir checkpoints/wt103_ctx256_XSADC \
    --checkpoint-interval 999999 \
    --time-budget 90 \
    --resume-from checkpoints/ablation_JormungandrHaloXSADC/step_1000.pt
echo "Finished Phase 1a at $(date)"
echo ""

echo "========================================"
echo "Phase 1b: Full on WikiText-103 ctx=256 at $(date)"
echo "========================================"
rm -rf checkpoints/wt103_ctx256_Full
python3 -m halo_training \
    --model models/jormungandr_halo.py \
    --class-name JormungandrHalo \
    --dataset datasets/wikitext-103-raw.bin \
    --compile --optimize-kernels \
    --muon --lr 0.0012 \
    --epochs 1 \
    --batch-size 16 --block-size 256 \
    --checkpoint-dir checkpoints/wt103_ctx256_Full \
    --checkpoint-interval 999999 \
    --time-budget 90 \
    --resume-from checkpoints/ablation_JormungandrHalo/step_1000.pt
echo "Finished Phase 1b at $(date)"
echo ""

# === PHASE 2: WikiText-103 at ctx=1024 (from Phase 1 checkpoints) ===

# Find the final checkpoint from Phase 1
XSADC_CKPT=$(ls -t checkpoints/wt103_ctx256_XSADC/step_*.pt 2>/dev/null | head -1)
FULL_CKPT=$(ls -t checkpoints/wt103_ctx256_Full/step_*.pt 2>/dev/null | head -1)

echo "Phase 1 checkpoints: XSADC=$XSADC_CKPT, Full=$FULL_CKPT"

if [ -z "$XSADC_CKPT" ] || [ -z "$FULL_CKPT" ]; then
    echo "ERROR: Missing Phase 1 checkpoints, aborting Phase 2"
    exit 1
fi

echo "========================================"
echo "Phase 2a: XSADC on WikiText-103 ctx=1024 at $(date)"
echo "========================================"
rm -rf checkpoints/wt103_ctx1024_XSADC
python3 -m halo_training \
    --model models/jormungandr_halo.py \
    --class-name JormungandrHaloXSADC \
    --dataset datasets/wikitext-103-raw.bin \
    --compile --optimize-kernels \
    --muon --lr 0.0012 \
    --epochs 1 \
    --batch-size 4 --block-size 1024 \
    --accum-steps 16 \
    --checkpoint-dir checkpoints/wt103_ctx1024_XSADC \
    --checkpoint-interval 999999 \
    --time-budget 120 \
    --resume-from "$XSADC_CKPT"
echo "Finished Phase 2a at $(date)"
echo ""

echo "========================================"
echo "Phase 2b: Full on WikiText-103 ctx=1024 at $(date)"
echo "========================================"
rm -rf checkpoints/wt103_ctx1024_Full
python3 -m halo_training \
    --model models/jormungandr_halo.py \
    --class-name JormungandrHalo \
    --dataset datasets/wikitext-103-raw.bin \
    --compile --optimize-kernels \
    --muon --lr 0.0012 \
    --epochs 1 \
    --batch-size 4 --block-size 1024 \
    --accum-steps 16 \
    --checkpoint-dir checkpoints/wt103_ctx1024_Full \
    --checkpoint-interval 999999 \
    --time-budget 120 \
    --resume-from "$FULL_CKPT"
echo "Finished Phase 2b at $(date)"

echo "ALL 4 RUNS COMPLETE at $(date)"

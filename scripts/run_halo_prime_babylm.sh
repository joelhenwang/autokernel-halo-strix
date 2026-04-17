#!/bin/bash
# HALO-PRIME: BabyLM 1 epoch + WikiText-103 CPT
# No --compile (Mamba's autokernel HIP backward conflicts with Dynamo)
# autokernel + muon provide the optimizations

source ~/Desktop/ai_lab/autokernel-halo-strix/.venv/bin/activate
cd ~/Desktop/ai_lab/autokernel-halo-strix

echo "========================================"
echo "HALO-PRIME BabyLM (1 epoch, ctx=256) at $(date)"
echo "========================================"
rm -rf checkpoints/halo_prime_babylm
python3 -m halo_training \
    --model models/halo_prime.py \
    --class-name HaloPrime \
    --dataset babylm \
    --optimize-kernels \
    --muon --lr 0.0012 \
    --epochs 1 \
    --batch-size 16 --block-size 256 \
    --accum-steps 4 \
    --checkpoint-dir checkpoints/halo_prime_babylm \
    --checkpoint-interval 999999 \
    --time-budget 60
echo "Finished BabyLM at $(date)"
echo ""

# Get the last checkpoint for WikiText CPT
CKPT=$(ls -t checkpoints/halo_prime_babylm/step_*.pt 2>/dev/null | head -1)
if [ -z "$CKPT" ]; then
    echo "ERROR: No BabyLM checkpoint found"
    exit 1
fi
echo "Using checkpoint: $CKPT"

echo "========================================"
echo "HALO-PRIME WikiText-103 (1 epoch, ctx=1024, CPT lr=0.0004) at $(date)"
echo "========================================"
rm -rf checkpoints/halo_prime_wt103
python3 -m halo_training \
    --model models/halo_prime.py \
    --class-name HaloPrime \
    --dataset ~/Desktop/ai_lab/autokernel-halo-strix/datasets/wikitext-103-raw \
    --optimize-kernels \
    --muon --lr 0.0004 \
    --epochs 1 \
    --batch-size 4 --block-size 1024 \
    --accum-steps 16 \
    --checkpoint-dir checkpoints/halo_prime_wt103 \
    --checkpoint-interval 999999 \
    --time-budget 120 \
    --resume-from "$CKPT"
echo "Finished WikiText-103 at $(date)"

echo "HALO-PRIME TRAINING COMPLETE at $(date)"

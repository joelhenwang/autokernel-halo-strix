#!/bin/bash
# GRIFFIN-HALO Compile Optimization Fixes — A, B, C isolated
# Baseline: R9 (Griffin+DMC, d=768, compile) = loss 3.193, 21.3K tok/s

source ~/Desktop/ai_lab/autokernel-halo-strix/.venv/bin/activate
cd ~/Desktop/ai_lab/autokernel-halo-strix

COMMON="--model models/griffin_halo.py \
    --dataset babylm \
    --compile --optimize-kernels \
    --muon --lr 0.0012 \
    --epochs 1 \
    --batch-size 16 --block-size 256 \
    --accum-steps 4 \
    --checkpoint-interval 999999 \
    --time-budget 60"

# ==============================================================
# Fix A: causal_conv1d bypassed under compile (amadeus.py change)
# Same config as R9, but conv ops now fusable by Inductor
# ==============================================================
echo "========================================"
echo "Fix A: Fusable conv under compile at $(date)"
echo "========================================"
rm -rf checkpoints/griffin_halo_fixA
python3 -m halo_training \
    --class-name GriffinHaloOpt_fixA \
    --checkpoint-dir checkpoints/griffin_halo_fixA \
    $COMMON
echo "Finished Fix A at $(date)"
tail -1 checkpoints/griffin_halo_fixA/train_log.jsonl
echo ""

# ==============================================================
# Fix B: Griffin + 2 ShortConv per iteration (3 blocks, more fusion)
# ==============================================================
echo "========================================"
echo "Fix B: Griffin + 2 ShortConv at $(date)"
echo "========================================"
rm -rf checkpoints/griffin_halo_fixB
python3 -m halo_training \
    --class-name GriffinHaloOpt_fixB \
    --checkpoint-dir checkpoints/griffin_halo_fixB \
    $COMMON
echo "Finished Fix B at $(date)"
tail -1 checkpoints/griffin_halo_fixB/train_log.jsonl
echo ""

# ==============================================================
# Fix C: Sequential Conv→Griffin (single chain, better fusion)
# ==============================================================
echo "========================================"
echo "Fix C: Sequential Conv→Griffin at $(date)"
echo "========================================"
rm -rf checkpoints/griffin_halo_fixC
python3 -m halo_training \
    --class-name GriffinHaloOpt_fixC \
    --checkpoint-dir checkpoints/griffin_halo_fixC \
    $COMMON
echo "Finished Fix C at $(date)"
tail -1 checkpoints/griffin_halo_fixC/train_log.jsonl
echo ""

echo "========================================"
echo "ALL COMPILE FIXES COMPLETE at $(date)"
echo "Compare against R9 baseline: loss=12.770 (actual 3.193), tok/s=21274"
echo "========================================"

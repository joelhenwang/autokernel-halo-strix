#!/bin/bash
# JORMUNGANDR-HALO XSA + Depth MC ablation sweep — remaining 3 configs
# BabyLM 1 epoch, compile + autokernel, Muon

source ~/Desktop/ai_lab/autokernel-halo-strix/.venv/bin/activate
cd ~/Desktop/ai_lab/autokernel-halo-strix

CONFIGS="JormungandrHaloDC JormungandrHaloXSADC JormungandrHalo"

for CFG in $CONFIGS; do
    echo "========================================"
    echo "Starting $CFG at $(date)"
    echo "========================================"
    python3 -m halo_training \
        --model models/jormungandr_halo.py \
        --class-name "$CFG" \
        --dataset datasets/babylm-strict-small \
        --compile --optimize-kernels \
        --muon --lr 0.0012 \
        --epochs 1 \
        --batch-size 16 --block-size 256 \
        --checkpoint-dir "checkpoints/ablation_${CFG}" \
        --checkpoint-interval 999999
    echo "Finished $CFG at $(date)"
    echo ""
done

echo "REMAINING ABLATION RUNS COMPLETE at $(date)"

#!/bin/bash
# XSADC vs Full on WikiText-103-raw (1 epoch), resumed from BabyLM checkpoints
# Sequential runs (not parallel — would contend for GPU)

source ~/Desktop/ai_lab/autokernel-halo-strix/.venv/bin/activate
cd ~/Desktop/ai_lab/autokernel-halo-strix

echo "========================================"
echo "Starting JormungandrHaloXSADC on WikiText-103 at $(date)"
echo "========================================"
python3 -m halo_training \
    --model models/jormungandr_halo.py \
    --class-name JormungandrHaloXSADC \
    --dataset datasets/wikitext-103-raw.bin \
    --compile --optimize-kernels \
    --muon --lr 0.0012 \
    --epochs 1 \
    --batch-size 16 --block-size 256 \
    --checkpoint-dir checkpoints/wt103_JormungandrHaloXSADC \
    --checkpoint-interval 999999 \
    --time-budget 90 \
    --resume-from checkpoints/ablation_JormungandrHaloXSADC/step_1000.pt
echo "Finished JormungandrHaloXSADC at $(date)"
echo ""

echo "========================================"
echo "Starting JormungandrHalo (Full) on WikiText-103 at $(date)"
echo "========================================"
python3 -m halo_training \
    --model models/jormungandr_halo.py \
    --class-name JormungandrHalo \
    --dataset datasets/wikitext-103-raw.bin \
    --compile --optimize-kernels \
    --muon --lr 0.0012 \
    --epochs 1 \
    --batch-size 16 --block-size 256 \
    --checkpoint-dir checkpoints/wt103_JormungandrHalo \
    --checkpoint-interval 999999 \
    --time-budget 90 \
    --resume-from checkpoints/ablation_JormungandrHalo/step_1000.pt
echo "Finished JormungandrHalo at $(date)"

echo "BOTH WIKITEXT-103 RUNS COMPLETE at $(date)"

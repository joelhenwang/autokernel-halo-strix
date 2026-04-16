#!/bin/bash
# XSA+DC vs Full on WikiText-103 at block_size=1024
# Tests whether TTT/FiLM/VE benefit from longer context
# Resumed from BabyLM checkpoints (ctx=256 -> ctx=1024)

source ~/Desktop/ai_lab/autokernel-halo-strix/.venv/bin/activate
cd ~/Desktop/ai_lab/autokernel-halo-strix

echo "========================================"
echo "Starting XSADC on WikiText-103 ctx=1024 at $(date)"
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
    --resume-from checkpoints/ablation_JormungandrHaloXSADC/step_1000.pt
echo "Finished XSADC ctx=1024 at $(date)"
echo ""

echo "========================================"
echo "Starting Full on WikiText-103 ctx=1024 at $(date)"
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
    --resume-from checkpoints/ablation_JormungandrHalo/step_1000.pt
echo "Finished Full ctx=1024 at $(date)"

echo "BOTH CTX=1024 RUNS COMPLETE at $(date)"

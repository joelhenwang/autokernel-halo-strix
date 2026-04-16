#!/bin/bash
# XSA+DC vs Full on WikiText-103 at block_size=512 (2x context)
# Tests whether TTT/FiLM/VE benefit from longer context
# Sequential runs, resumed from BabyLM checkpoints

source ~/Desktop/ai_lab/autokernel-halo-strix/.venv/bin/activate
cd ~/Desktop/ai_lab/autokernel-halo-strix

echo "========================================"
echo "Starting JormungandrHaloXSADC (ctx=512) at $(date)"
echo "========================================"
python3 -m halo_training \
    --model models/jormungandr_halo.py \
    --class-name JormungandrHaloXSADC \
    --dataset datasets/wikitext-103-raw.bin \
    --compile --optimize-kernels \
    --muon --lr 0.0012 \
    --epochs 1 \
    --batch-size 8 --block-size 512 \
    --accum-steps 8 \
    --checkpoint-dir checkpoints/wt103_ctx512_XSADC \
    --checkpoint-interval 999999 \
    --time-budget 120 \
    --resume-from checkpoints/ablation_JormungandrHaloXSADC/step_1000.pt
echo "Finished JormungandrHaloXSADC (ctx=512) at $(date)"
echo ""

echo "========================================"
echo "Starting JormungandrHalo Full (ctx=512) at $(date)"
echo "========================================"
python3 -m halo_training \
    --model models/jormungandr_halo.py \
    --class-name JormungandrHalo \
    --dataset datasets/wikitext-103-raw.bin \
    --compile --optimize-kernels \
    --muon --lr 0.0012 \
    --epochs 1 \
    --batch-size 8 --block-size 512 \
    --accum-steps 8 \
    --checkpoint-dir checkpoints/wt103_ctx512_Full \
    --checkpoint-interval 999999 \
    --time-budget 120 \
    --resume-from checkpoints/ablation_JormungandrHalo/step_1000.pt
echo "Finished JormungandrHalo Full (ctx=512) at $(date)"

echo "BOTH CTX=512 RUNS COMPLETE at $(date)"

#!/bin/bash
# Full vs XSA+DC on gpt-training-small, 2 epochs, lower LR for CPT
# ctx=1024, resumed from WikiText-103 ctx=1024 checkpoints

source ~/Desktop/ai_lab/autokernel-halo-strix/.venv/bin/activate
cd ~/Desktop/ai_lab/autokernel-halo-strix

echo "========================================"
echo "Starting Full on gpt-training-small (2ep, lr=0.0004) at $(date)"
echo "========================================"
rm -rf checkpoints/gpt_small_2ep_Full
python3 -m halo_training \
    --model models/jormungandr_halo.py \
    --class-name JormungandrHalo \
    --dataset datasets/gpt-training-small.bin \
    --compile --optimize-kernels \
    --muon --lr 0.0004 \
    --epochs 2 \
    --batch-size 4 --block-size 1024 \
    --accum-steps 16 \
    --checkpoint-dir checkpoints/gpt_small_2ep_Full \
    --checkpoint-interval 999999 \
    --time-budget 360 \
    --resume-from checkpoints/wt103_ctx1024_Full/step_1815.pt
echo "Finished Full at $(date)"
echo ""

echo "========================================"
echo "Starting XSA+DC on gpt-training-small (2ep, lr=0.0004) at $(date)"
echo "========================================"
rm -rf checkpoints/gpt_small_2ep_XSADC
python3 -m halo_training \
    --model models/jormungandr_halo.py \
    --class-name JormungandrHaloXSADC \
    --dataset datasets/gpt-training-small.bin \
    --compile --optimize-kernels \
    --muon --lr 0.0004 \
    --epochs 2 \
    --batch-size 4 --block-size 1024 \
    --accum-steps 16 \
    --checkpoint-dir checkpoints/gpt_small_2ep_XSADC \
    --checkpoint-interval 999999 \
    --time-budget 360 \
    --resume-from checkpoints/wt103_ctx1024_XSADC/step_1815.pt
echo "Finished XSA+DC at $(date)"

echo "BOTH GPT-SMALL 2EP RUNS COMPLETE at $(date)"

#!/bin/bash
# XSA+DC on gpt-training-small (1 epoch, ctx=1024)
# Runs AFTER Full finishes — launched by the monitor script

source ~/Desktop/ai_lab/autokernel-halo-strix/.venv/bin/activate
cd ~/Desktop/ai_lab/autokernel-halo-strix

echo "Starting JormungandrHaloXSADC on gpt-training-small at $(date)"
python3 -m halo_training \
    --model models/jormungandr_halo.py \
    --class-name JormungandrHaloXSADC \
    --dataset datasets/gpt-training-small.bin \
    --compile --optimize-kernels \
    --muon --lr 0.0012 \
    --epochs 1 \
    --batch-size 4 --block-size 1024 \
    --accum-steps 16 \
    --checkpoint-dir checkpoints/gpt_small_XSADC \
    --checkpoint-interval 999999 \
    --time-budget 180 \
    --resume-from checkpoints/wt103_ctx1024_XSADC/step_1815.pt
echo "Finished JormungandrHaloXSADC at $(date)"

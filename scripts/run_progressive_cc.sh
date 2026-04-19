#!/bin/bash
# Progressive Narrowing on Common Crawl sample (2.37B tokens, 1 epoch)
# CPT from WikiText-103 checkpoint, ctx=1024, checkpoints every 1/5

source ~/Desktop/ai_lab/autokernel-halo-strix/.venv/bin/activate
cd ~/Desktop/ai_lab/autokernel-halo-strix

CKPT=$(ls -t checkpoints/progressive_wt103_2ep/step_*.pt 2>/dev/null | head -1)
if [ -z "$CKPT" ]; then
    echo "ERROR: No WikiText checkpoint found"
    exit 1
fi
echo "Resuming from: $CKPT"

echo "========================================"
echo "Progressive Common Crawl (1 epoch, 2.37B tokens) at $(date)"
echo "========================================"
rm -rf checkpoints/progressive_cc_1ep
python3 -m halo_training \
    --model models/griffin_halo.py \
    --class-name GriffinHaloProgressive \
    --dataset datasets/common_crawl_sample.bin \
    --compile --optimize-kernels \
    --muon --lr 0.0004 \
    --epochs 1 \
    --batch-size 4 --block-size 1024 \
    --accum-steps 16 \
    --checkpoint-dir checkpoints/progressive_cc_1ep \
    --checkpoint-interval 7236 \
    --time-budget 1200 \
    --resume-from "$CKPT"
echo "Finished at $(date)"
ls -la checkpoints/progressive_cc_1ep/step_*.pt 2>/dev/null

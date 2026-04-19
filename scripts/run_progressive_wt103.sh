#!/bin/bash
# Progressive Narrowing on WikiText-103 combined (raw + raw-train)
# 2 epochs, ctx=1024, checkpoints at every 1/5 of max steps (364)
# CPT from BabyLM checkpoint with lower LR

source ~/Desktop/ai_lab/autokernel-halo-strix/.venv/bin/activate
cd ~/Desktop/ai_lab/autokernel-halo-strix

# Find the BabyLM checkpoint
CKPT=$(ls -t checkpoints/griffin_halo_progressive/step_*.pt 2>/dev/null | head -1)
if [ -z "$CKPT" ]; then
    echo "ERROR: No Progressive BabyLM checkpoint found"
    exit 1
fi
echo "Resuming from: $CKPT"

echo "========================================"
echo "Progressive WikiText-103 Combined (2 epochs, ctx=1024) at $(date)"
echo "Dataset: wikitext-103-combined (raw + raw-train symlinked)"
echo "========================================"
rm -rf checkpoints/progressive_wt103_2ep
python3 -m halo_training \
    --model models/griffin_halo.py \
    --class-name GriffinHaloProgressive \
    --dataset datasets/wikitext-103-combined \
    --compile --optimize-kernels \
    --muon --lr 0.0004 \
    --epochs 2 \
    --batch-size 4 --block-size 1024 \
    --accum-steps 16 \
    --checkpoint-dir checkpoints/progressive_wt103_2ep \
    --checkpoint-interval 364 \
    --time-budget 360 \
    --resume-from "$CKPT"
echo "Finished at $(date)"

echo "Checkpoints saved at steps: 364, 728, 1092, 1456, 1820 + final"
ls -la checkpoints/progressive_wt103_2ep/step_*.pt 2>/dev/null

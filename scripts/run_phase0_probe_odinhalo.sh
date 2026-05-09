#!/bin/bash
# Phase 0 probe: OdinHalo 200 steps with --optimize-kernels at lr_2d=2e-3.
# Compare against S1.3b baseline (25K tok/s, maxabs 15.13 at step 200).
# Accept if tok/s lift >= 15% AND loss matches S1.3b within 0.5 abs.

set -euo pipefail

cd ~/Desktop/ai_lab/autokernel-halo-strix

CKPT=checkpoints/phase0-probe-odinhalo-optkern
rm -rf "$CKPT"
mkdir -p "$CKPT"

echo "Phase 0.3: OdinHalo dolma lr_2d=2e-3 + --optimize-kernels, 200 steps"
echo "Checkpoint dir: $CKPT"
echo

CKPT_DIR="$CKPT" \
MODEL=models/odin_halo.py \
CLASS=OdinHalo \
DATASET=datasets/dolma-10b-odin32k.bin \
EPOCHS=1 \
LR=8e-4 \
BLOCK=512 \
BATCH=16 \
ACCUM=8 \
WARMUP_STEPS=100 \
CHECKPOINT_INTERVAL=200 \
MAX_GRAD_NORM=0.8 \
EXTRA_FLAGS='--max-steps 200 --imu1-groups --normuon --lr-2d 2e-3 --lr-1d 8e-4 --intra-doc-mask --value-residuals --head-gating --z-loss 1e-4 --z-loss-fraction 1.0 --attn-softcap 50.0 --activation-monitor --activation-monitor-interval 50 --optimize-kernels' \
bash scripts/launch_ddp.sh > "$CKPT/launch.log" 2>&1

echo "Launch returned; tail of launch.log:"
tail -5 "$CKPT/launch.log" || true

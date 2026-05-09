#!/bin/bash
# Phase 0 probe: OdinFlat 200 steps with --optimize-kernels at lr_2d=5e-3.
# Compare against S1.5 baseline (30.5K tok/s, layers.13 maxabs ~17 at step 200).
# Accept if tok/s lift >= 15% AND loss matches S1.5 within 0.5 abs.

set -euo pipefail

cd ~/Desktop/ai_lab/autokernel-halo-strix

CKPT=checkpoints/phase0-probe-odinflat-optkern
rm -rf "$CKPT"
mkdir -p "$CKPT"

echo "Phase 0.4: OdinFlat dolma lr_2d=5e-3 + --optimize-kernels, 200 steps"
echo "Checkpoint dir: $CKPT"
echo

CKPT_DIR="$CKPT" \
MODEL=models/odin_flat.py \
CLASS=OdinFlat \
DATASET=datasets/dolma-10b-odin32k.bin \
EPOCHS=1 \
LR=8e-4 \
BLOCK=512 \
BATCH=16 \
ACCUM=8 \
WARMUP_STEPS=100 \
CHECKPOINT_INTERVAL=200 \
MAX_GRAD_NORM=1.0 \
EXTRA_FLAGS='--max-steps 200 --imu1-groups --normuon --lr-2d 5e-3 --lr-1d 8e-4 --intra-doc-mask --value-residuals --head-gating --z-loss 1e-4 --z-loss-fraction 1.0 --attn-softcap 50.0 --activation-monitor --activation-monitor-interval 50 --optimize-kernels' \
bash scripts/launch_ddp.sh > "$CKPT/launch.log" 2>&1

echo "Launch returned; tail of launch.log:"
tail -5 "$CKPT/launch.log" || true

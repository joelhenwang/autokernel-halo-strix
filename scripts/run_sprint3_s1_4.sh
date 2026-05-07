#!/bin/bash
# Sprint 3 Stage 1 S1.4: --optimize-kernels A/B test.
# Treatment run: lr_2d=2e-3 (Stage 1 winner per S1.3 halt rule) + --optimize-kernels
# for 400 steps. Compare against S1.1 baseline (same config w/o flag).
#
# Gate: tok/s lift >= 15% AND loss matches S1.1 within 0.5 (abs).
# S1.1 step 400: loss=4.7152, tok/s=25,243 aggregate.

set -euo pipefail

cd ~/Desktop/ai_lab/autokernel-halo-strix

CKPT=checkpoints/sprint3-s1_4-optkern-lr2e3
rm -rf "$CKPT"
mkdir -p "$CKPT"

echo "Sprint 3 Stage 1 S1.4: OdinHalo dolma lr_2d=2e-3 + --optimize-kernels, 400 steps"
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
EXTRA_FLAGS='--max-steps 400 --imu1-groups --normuon --lr-2d 2e-3 --lr-1d 8e-4 --intra-doc-mask --value-residuals --head-gating --z-loss 1e-4 --z-loss-fraction 1.0 --attn-softcap 50.0 --activation-monitor --activation-monitor-interval 50 --optimize-kernels --auto-eval' \
bash scripts/launch_ddp.sh > "$CKPT/launch.log" 2>&1

echo "Launch returned; tail of launch.log:"
tail -5 "$CKPT/launch.log" || true

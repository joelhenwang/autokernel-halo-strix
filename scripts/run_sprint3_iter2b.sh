#!/bin/bash
# Sprint 3 Stage 1, Iteration 2b: OdinHalo dolma LR probe at lr_2d=2.5e-3
# Middle-ground point between iter 1 (2e-3) and iter 2 (3e-3); provides
# a 3-point curve to pick the S1.3 extended-confirmation winner.
#
# Pass criteria (same as iter 1/2):
#   1. Loss monotonically decreasing from step 50 to 400
#   2. shared_layers.5 maxabs at step 400 < 100
#   3. fp16_headroom at step 400 > 500x
#   4. Zero scaler backoffs
#   5. No NaN microstep messages
#   6. grad=<finite> on final log line

set -euo pipefail

cd ~/Desktop/ai_lab/autokernel-halo-strix

CKPT=checkpoints/sprint3-iter2b-lr2_5e3
rm -rf "$CKPT"
mkdir -p "$CKPT"

echo "Sprint 3 Stage 1 Iter 2b: OdinHalo dolma lr_2d=2.5e-3, 400 steps"
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
EXTRA_FLAGS='--max-steps 400 --imu1-groups --normuon --lr-2d 2.5e-3 --lr-1d 8e-4 --intra-doc-mask --value-residuals --head-gating --z-loss 1e-4 --z-loss-fraction 1.0 --attn-softcap 50.0 --activation-monitor --activation-monitor-interval 50 --auto-eval' \
bash scripts/launch_ddp.sh > "$CKPT/launch.log" 2>&1

echo "Launch returned; tail of launch.log:"
tail -5 "$CKPT/launch.log" || true

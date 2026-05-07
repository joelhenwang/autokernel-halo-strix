#!/bin/bash
# Sprint 3 Stage 1, Iteration 1: OdinHalo dolma LR calibration probe
# lr_2d=2e-3 (conservative), 400 steps, all fp16-stability guards active.
#
# Per docs/superpowers/plans/2026-05-07-stage1-sprint3-execution-plan.md §1.1
#
# Pass criteria:
#   1. Loss monotonically decreasing from step 50 to 400
#   2. shared_layers.5 maxabs at step 400 < 100
#   3. fp16_headroom at step 400 > 500x
#   4. Zero scaler backoffs
#   5. No NaN microstep messages
#   6. grad=<finite> on final log line

set -euo pipefail

cd ~/Desktop/ai_lab/autokernel-halo-strix

CKPT=checkpoints/sprint3-iter1-lr2e3
rm -rf "$CKPT"
mkdir -p "$CKPT"

echo "Sprint 3 Stage 1 Iter 1: OdinHalo dolma lr_2d=2e-3, 400 steps"
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
EXTRA_FLAGS='--max-steps 400 --imu1-groups --normuon --lr-2d 2e-3 --lr-1d 8e-4 --intra-doc-mask --value-residuals --head-gating --z-loss 1e-4 --z-loss-fraction 1.0 --attn-softcap 50.0 --activation-monitor --activation-monitor-interval 50 --auto-eval' \
bash scripts/launch_ddp.sh > "$CKPT/launch.log" 2>&1

echo "Launch returned; tail of launch.log:"
tail -5 "$CKPT/launch.log" || true

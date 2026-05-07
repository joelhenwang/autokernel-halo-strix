#!/bin/bash
# Sprint 3 Stage 1 S1.3b: OdinHalo dolma 700-step confirmation at lr_2d=2e-3.
# The fallback LR chosen after S1.3 (lr_2d=2.5e-3) failed gate #2.
# Confirms empirically what S1.1's 400-step trajectory projected:
# step-700 maxabs ~89 (gate <200).
#
# Pass criteria (same as S1.3):
#   1. Loss monotonically decreasing from step 50 to 700
#   2. shared_layers.5 maxabs at step 700 < 200
#   3. fp16_headroom at step 700 > 300x
#   4. Zero scaler backoffs
#   5. No NaN microstep messages
#   6. grad=<finite> on final log line
#   7. Late-window deceleration: 600->700 growth factor < 650->700 growth factor

set -euo pipefail

cd ~/Desktop/ai_lab/autokernel-halo-strix

CKPT=checkpoints/sprint3-s1_3b-lr2e3-700
rm -rf "$CKPT"
mkdir -p "$CKPT"

echo "Sprint 3 Stage 1 S1.3b: OdinHalo dolma lr_2d=2e-3, 700 steps"
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
EXTRA_FLAGS='--max-steps 700 --imu1-groups --normuon --lr-2d 2e-3 --lr-1d 8e-4 --intra-doc-mask --value-residuals --head-gating --z-loss 1e-4 --z-loss-fraction 1.0 --attn-softcap 50.0 --activation-monitor --activation-monitor-interval 50 --auto-eval' \
bash scripts/launch_ddp.sh > "$CKPT/launch.log" 2>&1

echo "Launch returned; tail of launch.log:"
tail -5 "$CKPT/launch.log" || true

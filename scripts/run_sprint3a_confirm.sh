#!/bin/bash
# Sprint 3A pre-launch confirmation: 2000-step run with the LOCKED recipe
# (C3 combined: μP + SPECTRA) on OdinFlat + dolma-10B via DDP.
#
# Purpose: catch anything that fails to generalize from Phase 1.C's
# 500-step factorial before the 52h full-epoch commitment.
#
# Runtime: ~2h 20m wall at ~31K tok/s × 2000 steps.
# Checkpoint at 500, 1000, 1500, 2000 so auto-eval fires 4 times.
#
# Pass criteria:
#   1. Loss monotonic from step 50 to 2000 (last 100-step window is
#      within noise of overall descent).
#   2. No NaN, no scaler collapse (scale >= 1.0 throughout).
#   3. fp16_headroom at step 2000 > 100x.
#   4. Late-window deceleration: 1900->2000 growth factor < 1.50x.
#   5. Per-domain BPB (from auto-eval at step 2000) shows improvement
#      vs Sprint 1 baseline across all 4 domains.

set -euo pipefail

cd ~/Desktop/ai_lab/autokernel-halo-strix

CKPT=checkpoints/sprint3a-confirm-2000
rm -rf "$CKPT"
mkdir -p "$CKPT"

echo "=== Sprint 3A 2000-step confirmation ==="
echo "Checkpoint dir: $CKPT"
echo "Config: C3 combined (μP + SPECTRA) on OdinFlat + dolma-10B"
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
WARMUP_STEPS=300 \
CHECKPOINT_INTERVAL=500 \
MAX_GRAD_NORM=1.0 \
EXTRA_FLAGS='--max-steps 2000 --imu1-groups --normuon --lr-2d 5e-3 --lr-1d 8e-4 --intra-doc-mask --value-residuals --head-gating --z-loss 1e-4 --z-loss-fraction 1.0 --attn-softcap 50.0 --activation-monitor --activation-monitor-interval 100 --mup --mup-base-width 256 --spectra-post --spectra-clip-norm 1.0 --auto-eval' \
bash scripts/launch_ddp.sh > "$CKPT/launch.log" 2>&1

echo "Launch returned; tail of launch.log:"
tail -5 "$CKPT/launch.log" || true

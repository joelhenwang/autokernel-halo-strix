#!/bin/bash
# Sprint 3 Stage 1 S1.5: OdinFlat dolma sanity check at lr_2d=5e-3.
# Validates that the Sprint 1 recipe transfers from wikitext to dolma on
# the flat (non-looped) model.
#
# Pass criteria (lighter than OdinHalo since flat is generally more stable):
#   1. Loss monotonically decreasing from step 50 to 400
#   2. No NaN / no scaler backoff
#   3. Grad finite every log line
#   4. Final loss comparable to OdinFlat wikitext final loss (4.47) + margin
#
# lr_2d=5e-3 is the Sprint 1 Run 2b winner for OdinFlat on wikitext.

set -euo pipefail

cd ~/Desktop/ai_lab/autokernel-halo-strix

CKPT=checkpoints/sprint3-s1_5-odinflat-dolma
rm -rf "$CKPT"
mkdir -p "$CKPT"

echo "Sprint 3 Stage 1 S1.5: OdinFlat dolma lr_2d=5e-3, 400 steps"
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
EXTRA_FLAGS='--max-steps 400 --imu1-groups --normuon --lr-2d 5e-3 --lr-1d 8e-4 --intra-doc-mask --value-residuals --head-gating --z-loss 1e-4 --z-loss-fraction 1.0 --attn-softcap 50.0 --activation-monitor --activation-monitor-interval 50 --auto-eval' \
bash scripts/launch_ddp.sh > "$CKPT/launch.log" 2>&1

echo "Launch returned; tail of launch.log:"
tail -5 "$CKPT/launch.log" || true

#!/bin/bash
# Phase G: OdinHalo Sprint 3B 1000-step verification probe.
#
# Validates whether Phase B post-fix code + --optimize-kernels is stable
# on OdinHalo at Sprint 3B's locked recipe (lr_2d=2e-3). If passes, Sprint
# 3B ships with --optimize-kernels ON. If fails, fall back to pre-fix code
# or drop --optimize-kernels.
#
# Ship gate: loss@step_1000 < 3.55 (B4 reached ~3.40 at step 1000 pre-fix
# with 14 frozen params; post-fix should be at-least-as-good).
#
# DDP across both machines, same config as Sprint 3B locked recipe.
# Expected: ~45-60 min compute.

set -eo pipefail

cd ~/Desktop/ai_lab/autokernel-halo-strix
source .venv/bin/activate

CKPT="${CKPT:-checkpoints/phase-g-odinhalo-sprint3b-verify}"
rm -rf "$CKPT"
mkdir -p "$CKPT"

ssh joelwang-ai-1@10.77.0.2 "mkdir -p ~/Desktop/comfyui-rocm7.12/autokernel-halo-strix/$CKPT"

echo "=== Phase G: OdinHalo 1000-step Sprint 3B verification (DDP) ==="
echo "  CKPT: $CKPT"
echo "  commit: $(git log --oneline -1)"
echo "  target: loss<=3.55 @ step 1000"
echo ""

MODEL=models/odin_halo.py \
CLASS=OdinHalo \
DATASET=datasets/dolma-10b-odin32k.bin \
EPOCHS=1 \
LR=1e-3 BLOCK=256 BATCH=16 ACCUM=8 \
WARMUP_STEPS=300 CHECKPOINT_INTERVAL=500 MAX_GRAD_NORM=0.8 \
NUM_WORKERS=12 \
CKPT_DIR="$CKPT" \
MASTER_PORT=29530 \
EXTRA_FLAGS='--max-steps 1000 --optimize-kernels --imu1-groups --normuon --lr-2d 2e-3 --lr-1d 8e-4 --intra-doc-mask --value-residuals --head-gating --z-loss 1e-4 --z-loss-fraction 1.0 --attn-softcap 50.0 --activation-monitor --activation-monitor-interval 200 --diag-frozen-params '"$CKPT"'/diag.jsonl' \
bash scripts/launch_ddp.sh > "$CKPT/launch.log" 2>&1

echo "Launched; rank0 log: $CKPT/rank0.log"

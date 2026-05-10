#!/bin/bash
# T-6 Sprint 3B launch: OdinHalo (58M looped) on dolma-10B with best-passing
# throughput stack. Same stack selection as Sprint 3A.
#
# Differences from 3A:
#   - Model: OdinHalo (58M unique, ~156M effective looped)
#   - LR: lr_2d=2e-3 (lower, locked per Phase G findings pre-v3)
#   - --polar-ns (stabilizes Parcae velocity clamp)
#
# Launch from Machine A after Sprint 3A completes (or overlapping).
#   bash scripts/launch_sprint3b.sh

set -e

STACK="${STACK:-A}"

EXTRA_FLAGS_BASE="--imu1-groups --normuon --lr-2d 2e-3 --lr-1d 3e-4 \
  --intra-doc-mask --value-residuals --head-gating \
  --z-loss 1e-4 --z-loss-fraction 1.0 --attn-softcap 50.0 \
  --activation-monitor --activation-monitor-interval 200 \
  --mup --mup-base-width 256 \
  --spectra-post --spectra-clip-norm 1.0 \
  --polar-ns --ema --auto-eval \
  --ak-spectra-branchless \
  --ak-sync-cleanup \
  --use-fused-zloss --ak-loss-zloss \
  --ak-normuon-telemetry"

BATCH_OVERRIDE="16"
ACCUM_OVERRIDE="8"

case "$STACK" in
  A)
    EXTRA_FLAGS="$EXTRA_FLAGS_BASE"
    echo "Sprint 3B Stack A: loss + DDP + sync (conservative)"
    ;;
  B)
    EXTRA_FLAGS="$EXTRA_FLAGS_BASE --ak-ddp-tune"
    BATCH_OVERRIDE="32"
    ACCUM_OVERRIDE="4"
    echo "Sprint 3B Stack B: A + batch=32/accum=4 + DDP tune"
    ;;
  C)
    EXTRA_FLAGS="$EXTRA_FLAGS_BASE --ak-ddp-tune --ak-compiled-autograd"
    BATCH_OVERRIDE="32"
    ACCUM_OVERRIDE="4"
    echo "Sprint 3B Stack C: B + compiled autograd"
    ;;
  D)
    EXTRA_FLAGS="$EXTRA_FLAGS_BASE --ak-ddp-tune --ak-compiled-autograd \
      --ak-fix-rope-gate-op \
      --ak-trust-cap 0.02 --ak-trust-cap-scope w_gate_up \
      --ak-w-gate-up-scale 0.25 --ak-w-gate-up-ramp-steps 1000"
    BATCH_OVERRIDE="32"
    ACCUM_OVERRIDE="4"
    echo "Sprint 3B Stack D: C + hidden kernel recovery (experimental)"
    ;;
  *)
    echo "Unknown STACK=$STACK (must be A, B, C, or D)"
    exit 1
    ;;
esac

echo "  batch=${BATCH_OVERRIDE} accum=${ACCUM_OVERRIDE} effective_batch=$((BATCH_OVERRIDE * ACCUM_OVERRIDE * 2))"
echo ""

CKPT_DIR="${CKPT_DIR:-checkpoints/sprint3b-stack-${STACK}-dolma10b}"

MODEL=models/odin_halo.py \
CLASS=OdinHalo \
DATASET=datasets/dolma-10b-odin32k.bin \
EPOCHS=1 \
LR=8e-4 \
BLOCK=512 \
BATCH="$BATCH_OVERRIDE" \
ACCUM="$ACCUM_OVERRIDE" \
WARMUP_STEPS=300 \
CHECKPOINT_INTERVAL=500 \
MAX_GRAD_NORM=1.0 \
NUM_WORKERS=12 \
CKPT_DIR="$CKPT_DIR" \
EXTRA_FLAGS="$EXTRA_FLAGS" \
bash scripts/launch_ddp.sh

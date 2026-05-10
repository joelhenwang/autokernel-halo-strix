#!/bin/bash
# T-6 Sprint 3A launch: OdinFlat 1 epoch on dolma-10B with best-passing
# throughput stack. Based on T-1 through T-5 findings.
#
# Current Stack A composition (as of T-1 in-progress):
#   - branchless SPECTRA          (--ak-spectra-branchless)
#   - fused z-loss                (--use-fused-zloss + --ak-loss-zloss)
#   - deferred loss sync          (--ak-sync-cleanup)
#   - DDP tuned                   (--ak-ddp-tune)   [conditional on T-1.4]
#   - NorMuon telemetry on        (--ak-normuon-telemetry)
#
# To upgrade to Stack B (after T-2 batch=32 probe passes):
#   Add BATCH=32 ACCUM=4 environment overrides.
#
# To upgrade to Stack C (after T-4 compiled autograd gate passes):
#   Add --ak-compiled-autograd to EXTRA_FLAGS.
#
# To upgrade to Stack D (after T-5 hidden-kernel recovery + T-3.2 test):
#   Add --ak-fix-rope-gate-op, --ak-trust-cap 0.02, --ak-trust-cap-scope
#   w_gate_up, --ak-w-gate-up-scale 0.25, --ak-w-gate-up-ramp-steps 1000.
#
# Launch from Machine A:
#   bash scripts/launch_sprint3a.sh

set -e

# Stack selection via env var; default to Stack A (locked).
STACK="${STACK:-A}"

EXTRA_FLAGS_BASE="--imu1-groups --normuon --lr-2d 5e-3 --lr-1d 8e-4 \
  --intra-doc-mask --value-residuals --head-gating \
  --z-loss 1e-4 --z-loss-fraction 1.0 --attn-softcap 50.0 \
  --activation-monitor --activation-monitor-interval 200 \
  --mup --mup-base-width 256 \
  --spectra-post --spectra-clip-norm 1.0 \
  --ema --auto-eval \
  --ak-spectra-branchless \
  --ak-sync-cleanup \
  --use-fused-zloss --ak-loss-zloss \
  --ak-normuon-telemetry"

BATCH_OVERRIDE="16"
ACCUM_OVERRIDE="8"

case "$STACK" in
  A)
    # Baseline v3 safe stack — no compiled autograd, no hidden kernels.
    EXTRA_FLAGS="$EXTRA_FLAGS_BASE"
    echo "Sprint 3A Stack A: loss + DDP + sync (conservative)"
    ;;
  B)
    # Stack A + batch=32 (requires T-2.1 gate pass).
    EXTRA_FLAGS="$EXTRA_FLAGS_BASE --ak-ddp-tune"
    BATCH_OVERRIDE="32"
    ACCUM_OVERRIDE="4"
    echo "Sprint 3A Stack B: A + batch=32/accum=4 + DDP tune"
    ;;
  C)
    # Stack B + compiled autograd (requires T-4 gate pass).
    EXTRA_FLAGS="$EXTRA_FLAGS_BASE --ak-ddp-tune --ak-compiled-autograd"
    BATCH_OVERRIDE="32"
    ACCUM_OVERRIDE="4"
    echo "Sprint 3A Stack C: B + compiled autograd"
    ;;
  D)
    # Stack C + hidden kernel recovery (requires T-3.2 + T-5 gates).
    EXTRA_FLAGS="$EXTRA_FLAGS_BASE --ak-ddp-tune --ak-compiled-autograd \
      --ak-fix-rope-gate-op \
      --ak-trust-cap 0.02 --ak-trust-cap-scope w_gate_up \
      --ak-w-gate-up-scale 0.25 --ak-w-gate-up-ramp-steps 1000"
    BATCH_OVERRIDE="32"
    ACCUM_OVERRIDE="4"
    echo "Sprint 3A Stack D: C + hidden kernel recovery (experimental)"
    ;;
  *)
    echo "Unknown STACK=$STACK (must be A, B, C, or D)"
    exit 1
    ;;
esac

echo "  batch=${BATCH_OVERRIDE} accum=${ACCUM_OVERRIDE} effective_batch=$((BATCH_OVERRIDE * ACCUM_OVERRIDE * 2))"
echo ""

CKPT_DIR="${CKPT_DIR:-checkpoints/sprint3a-stack-${STACK}-dolma10b}"

MODEL=models/odin_flat.py \
CLASS=OdinFlat \
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

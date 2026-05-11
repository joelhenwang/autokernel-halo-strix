#!/bin/bash
# T-5 C.1.c/d result: Stack E two-stage launcher (delayed enable).
#
# Based on T-5 C.1.c finding: native Sprint 3A-confirm recipe for 1000 steps
# + Stack D resume with preserved optimizer state → final loss 3.1065 at
# total step 2000 vs Stack D from-scratch's 3.1384 (better by -0.033).
# C.1.d adds a loss-only intermediate for slightly better loss (3.0152) but
# more operator complexity; C.1.c is chosen as the canonical Stack E.
#
# Tradeoff vs Stack D:
#   + Slightly better final loss at matching total step count (single-run
#     evidence; delta 0.03 is within fp16+RNG noise — not a conclusive win).
#   - Two-stage operator overhead: Stage 1 checkpoint must exist on BOTH
#     machines before Stage 2 launch (script handles this via scp).
#   - Aggregate tok/s is slightly lower because Stage 1 uses native recipe
#     (~31k tok/s) vs Stack D (~34.7k tok/s). Over 1 epoch, this nets out to
#     ~1-2% aggregate throughput loss compared to pure Stack D.
#
# Configuration:
#   STAGE1_STEPS    number of steps in native Stage 1 (default 1000)
#   STAGE2_STEPS    additional steps in Stack D Stage 2 (default: run to epoch end)
#
# Launch from Machine A:
#   STACK=E bash scripts/launch_sprint3a.sh
#   # or directly:
#   bash scripts/launch_sprint3a_stackE.sh

set -e

cd "$(dirname "$0")/.."

STAGE1_STEPS="${STAGE1_STEPS:-1000}"
STAGE2_STEPS="${STAGE2_STEPS:-}"   # empty = run full epoch after Stage 1
CKPT_DIR_BASE="${CKPT_DIR:-checkpoints/sprint3a-stackE-dolma10b}"
STAGE1_DIR="${CKPT_DIR_BASE}/stage1-native"
STAGE2_DIR="${CKPT_DIR_BASE}/stage2-stackd"

mkdir -p "$STAGE1_DIR"
mkdir -p "$STAGE2_DIR"

# Shared base flags
BASE_FLAGS="--imu1-groups --normuon --lr-2d 5e-3 --lr-1d 8e-4 \
  --intra-doc-mask --value-residuals --head-gating \
  --z-loss 1e-4 --z-loss-fraction 1.0 --attn-softcap 50.0 \
  --activation-monitor --activation-monitor-interval 200 \
  --mup --mup-base-width 256 \
  --spectra-post --spectra-clip-norm 1.0 \
  --ema --auto-eval"

echo "=== Sprint 3A Stack E (delayed-enable) ==="
echo "Stage 1: native for ${STAGE1_STEPS} steps -> ${STAGE1_DIR}"
echo "Stage 2: Stack D resume + preserved optim -> ${STAGE2_DIR}"
echo ""

# ===========================================================================
# STAGE 1: native recipe for STAGE1_STEPS steps
# ===========================================================================
echo "[Stage 1] Launching native for ${STAGE1_STEPS} steps ..."

MODEL=models/odin_flat.py \
CLASS=OdinFlat \
DATASET=datasets/dolma-10b-odin32k.bin \
EPOCHS=1 \
LR=8e-4 \
BLOCK=512 \
BATCH=16 \
ACCUM=8 \
WARMUP_STEPS=300 \
CHECKPOINT_INTERVAL="$STAGE1_STEPS" \
MAX_GRAD_NORM=1.0 \
NUM_WORKERS=12 \
CKPT_DIR="$STAGE1_DIR" \
MASTER_PORT=29500 \
EXTRA_FLAGS="--max-steps $STAGE1_STEPS $BASE_FLAGS" \
bash scripts/launch_ddp.sh

# Wait for Stage 1 to complete
echo "[Stage 1] Waiting for training to finish..."
SECS=0
while pgrep -f "train_ddp.py" > /dev/null; do
  sleep 30
  SECS=$((SECS + 30))
  if [ $((SECS % 300)) -eq 0 ]; then
    if [ -f "$STAGE1_DIR/rank0.log" ]; then
      echo "  [Stage 1 @ ${SECS}s] $(tail -1 "$STAGE1_DIR/rank0.log" 2>/dev/null)"
    fi
  fi
  # 4h ceiling for Stage 1 (normally ~40 min for 1000 steps)
  if [ $SECS -gt 14400 ]; then
    echo "[Stage 1] TIMEOUT at 4h"
    pkill -f "train_ddp.py" || true
    ssh joelwang-ai-1@10.77.0.2 "pkill -f 'train_ddp.py'" || true
    exit 1
  fi
done

STAGE1_CKPT="${STAGE1_DIR}/step_${STAGE1_STEPS}.pt"
if [ ! -f "$STAGE1_CKPT" ]; then
  echo "[Stage 1] FAIL: no checkpoint at $STAGE1_CKPT"
  exit 1
fi

echo "[Stage 1] Done. Checkpoint: $STAGE1_CKPT"
echo ""

# ===========================================================================
# Copy Stage 1 checkpoint to Machine B (DDP requires ckpt on both nodes)
# ===========================================================================
echo "[Stage 1->2] Copying checkpoint to Machine B..."
ssh joelwang-ai-1@10.77.0.2 "mkdir -p ~/Desktop/comfyui-rocm7.12/autokernel-halo-strix/${STAGE1_DIR}"
scp "${STAGE1_CKPT}" "joelwang-ai-1@10.77.0.2:~/Desktop/comfyui-rocm7.12/autokernel-halo-strix/${STAGE1_CKPT}"
echo "[Stage 1->2] Copy complete. Letting gloo ports drain..."
sleep 60

# ===========================================================================
# STAGE 2: Stack D with preserved optimizer state, for remainder of epoch
# ===========================================================================
STACK_D_FLAGS="--use-fused-zloss --ak-loss-zloss \
  --ak-fix-rope-gate-op --ak-causal-conv-shim \
  --ak-sync-cleanup --ak-spectra-branchless \
  --ak-normuon-telemetry"

STAGE2_MAX_FLAG=""
if [ -n "$STAGE2_STEPS" ]; then
  STAGE2_MAX_FLAG="--max-steps $STAGE2_STEPS"
fi

echo "[Stage 2] Launching Stack D resume (preserved optim) ..."
echo "  Resume: $STAGE1_CKPT"
echo "  Max steps: ${STAGE2_STEPS:-full epoch}"
echo ""

MODEL=models/odin_flat.py \
CLASS=OdinFlat \
DATASET=datasets/dolma-10b-odin32k.bin \
EPOCHS=1 \
LR=8e-4 \
BLOCK=512 \
BATCH=16 \
ACCUM=8 \
WARMUP_STEPS=0 \
CHECKPOINT_INTERVAL=500 \
MAX_GRAD_NORM=0.8 \
NUM_WORKERS=12 \
CKPT_DIR="$STAGE2_DIR" \
MASTER_PORT=29510 \
EXTRA_FLAGS="$STAGE2_MAX_FLAG $BASE_FLAGS $STACK_D_FLAGS --resume-from $STAGE1_CKPT --resume-preserve-optimizer" \
bash scripts/launch_ddp.sh

echo ""
echo "=== Stack E launch complete (both stages kicked off) ==="
echo "Monitor Stage 2: tail -f $STAGE2_DIR/rank0.log"

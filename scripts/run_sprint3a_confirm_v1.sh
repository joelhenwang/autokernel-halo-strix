#!/bin/bash
# Phase V: 2000-step validation of the V1 recipe (F.rms_norm + silu HIP
# + autokernel-exclude rmsnorm) before Sprint 3A full epoch launch.
#
# Config: Sprint 1.5 C3 + Option 3 V1 throughput tweaks.
# Expected wall: ~2h 10m at 41K tok/s × 2000 steps × 131K tokens/step.
#
# Pass criteria (same as previous sprint3a-confirm gates):
#   1. Loss monotonic from step 50 to 2000
#   2. No NaN, no scaler collapse
#   3. fp16_headroom at step 2000 > 100x
#   4. Late-window (1900->2000) growth factor < 1.50x
#   5. Per-domain BPB auto-eval shows improvement vs Sprint 1 baseline

set -euo pipefail
cd ~/Desktop/ai_lab/autokernel-halo-strix

CKPT=checkpoints/sprint3a-confirm-v1-2000
rm -rf "$CKPT"
mkdir -p "$CKPT"

echo "=== Phase V: V1 recipe 2000-step validation ==="
echo "Checkpoint dir: $CKPT"
echo "Expected wall: ~2h 10m at 41K tok/s"
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
EXTRA_FLAGS='--max-steps 2000 --imu1-groups --normuon --lr-2d 5e-3 --lr-1d 8e-4 --intra-doc-mask --value-residuals --head-gating --z-loss 1e-4 --z-loss-fraction 1.0 --attn-softcap 50.0 --activation-monitor --activation-monitor-interval 100 --mup --mup-base-width 256 --spectra-post --spectra-clip-norm 1.0 --optimize-kernels --autokernel-exclude rmsnorm --auto-eval' \
bash scripts/launch_ddp.sh > "$CKPT/launch.log" 2>&1

echo "Launched; tail of launch.log:"
tail -5 "$CKPT/launch.log" || true

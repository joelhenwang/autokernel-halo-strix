#!/bin/bash
# Phase III validation: after autograd fix, re-run the rmsnorm probe
# to confirm loss matches baseline and throughput lift is preserved.
set -euo pipefail
cd ~/Desktop/ai_lab/autokernel-halo-strix

CKPT=checkpoints/sprint15-bisect-P3-FIXED
rm -rf "$CKPT"
mkdir -p "$CKPT"

echo "=== Phase III validation: optimize-kernels ON with autograd fix ==="
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
CHECKPOINT_INTERVAL=9999 \
MAX_GRAD_NORM=1.0 \
EXTRA_FLAGS='--max-steps 200 --imu1-groups --normuon --lr-2d 5e-3 --lr-1d 8e-4 --intra-doc-mask --value-residuals --head-gating --z-loss 1e-4 --z-loss-fraction 1.0 --attn-softcap 50.0 --activation-monitor --activation-monitor-interval 50 --mup --mup-base-width 256 --spectra-post --spectra-clip-norm 1.0 --optimize-kernels' \
bash scripts/launch_ddp.sh > "$CKPT/launch.log" 2>&1

echo "Launched. Waiting for Done: marker..."
while true; do
    sleep 30
    if [ -f "$CKPT/rank0.log" ] && grep -q "^Done:" "$CKPT/rank0.log"; then
        break
    fi
    if [ -f "$CKPT/rank0.log" ] && grep -q "^Traceback" "$CKPT/rank0.log" 2>/dev/null; then
        echo "FAIL: run crashed"
        tail -20 "$CKPT/rank0.log"
        exit 1
    fi
done

echo
echo "=== Results ==="
grep -E '^\[step|^Done:|\[autokernel\]' "$CKPT/rank0.log" | head -10
echo "--- layers.13 maxabs ---"
grep '"layer": "layers.13"' "$CKPT/activation_stats.jsonl" | tail -4

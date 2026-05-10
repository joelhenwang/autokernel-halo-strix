#!/bin/bash
# Option 3 V2: F.rms_norm + silu HIP WITH autograd fix re-enabled
# (excluded rmsnorm since it uses F.rms_norm).
# Quick 200-step check — does training stay stable this time?
set -euo pipefail
cd ~/Desktop/ai_lab/autokernel-halo-strix

CKPT=checkpoints/sprint15-opt3-V2-silu-autograd
rm -rf "$CKPT"; mkdir -p "$CKPT"

echo "=== V2: F.rms_norm + HIP silu (autograd-registered) ==="

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
EXTRA_FLAGS='--max-steps 200 --imu1-groups --normuon --lr-2d 5e-3 --lr-1d 8e-4 --intra-doc-mask --value-residuals --head-gating --z-loss 1e-4 --z-loss-fraction 1.0 --attn-softcap 50.0 --activation-monitor --activation-monitor-interval 50 --mup --mup-base-width 256 --spectra-post --spectra-clip-norm 1.0 --optimize-kernels --autokernel-exclude rmsnorm' \
bash scripts/launch_ddp.sh > "$CKPT/launch.log" 2>&1

while true; do
    sleep 30
    if [ -f "$CKPT/rank0.log" ] && grep -q "^Done:" "$CKPT/rank0.log"; then break; fi
    if [ -f "$CKPT/rank0.log" ] && grep -q "^Traceback" "$CKPT/rank0.log" 2>/dev/null; then
        echo "FAIL:"; tail -15 "$CKPT/rank0.log"; exit 1
    fi
done

echo
echo "=== V2 result ==="
grep -E '^Done:|^\[step' "$CKPT/rank0.log" | tail -6
echo "--- maxabs ---"
grep 'layers.13' "$CKPT/activation_stats.jsonl" | tail -5

#!/bin/bash
# Option 3 extension: try max-autotune with cudagraphs now that RMSNorm
# uses native F.rms_norm (previously cudagraphs crashed on HIP rmsnorm).
set -euo pipefail
cd ~/Desktop/ai_lab/autokernel-halo-strix

CKPT=checkpoints/sprint15-opt3-V3-cudagraphs
rm -rf "$CKPT"; mkdir -p "$CKPT"

echo "=== V3: F.rms_norm + silu HIP + max-autotune (cudagraphs) ==="

TORCH_COMPILE_MODE=max-autotune \
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
echo "=== V3 result ==="
grep -E '^Done:|^\[step    200\]' "$CKPT/rank0.log" | head -2

#!/bin/bash
# Option 3 probe: F.rms_norm (native PyTorch, Inductor-fusable) vs HIP
# and vs baseline.
#
# Variants:
#   V0: native F.rms_norm, no HIP kernels     <- new  (Option 3)
#   V1: native F.rms_norm + silu_gate_mul HIP  <- new
#   V2: HIP rmsnorm (autograd-fixed) + silu    <- Phase III result (=40.9K reference)

set -euo pipefail
cd ~/Desktop/ai_lab/autokernel-halo-strix

wait_for_done() {
    local log="$1"
    while true; do
        sleep 30
        if [ -f "$log" ] && grep -q "^Done:" "$log"; then return 0; fi
        if [ -f "$log" ] && grep -q "^Traceback" "$log" 2>/dev/null; then
            echo "FAIL: $log"; tail -10 "$log"; return 1
        fi
    done
}

launch() {
    local name="$1"
    local extra_optkern="$2"

    local ckpt="checkpoints/sprint15-opt3-${name}"
    rm -rf "$ckpt"
    mkdir -p "$ckpt"

    echo
    echo "=== ${name}  |  ${ckpt} ==="
    echo "  optkern flag: ${extra_optkern:-none}"
    echo

    local extra_flags='--max-steps 200 --imu1-groups --normuon --lr-2d 5e-3 --lr-1d 8e-4 --intra-doc-mask --value-residuals --head-gating --z-loss 1e-4 --z-loss-fraction 1.0 --attn-softcap 50.0 --activation-monitor --activation-monitor-interval 50 --mup --mup-base-width 256 --spectra-post --spectra-clip-norm 1.0'
    if [ -n "$extra_optkern" ]; then
        extra_flags="$extra_flags $extra_optkern"
    fi

    CKPT_DIR="$ckpt" \
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
    EXTRA_FLAGS="$extra_flags" \
    bash scripts/launch_ddp.sh > "$ckpt/launch.log" 2>&1

    echo "  launched; waiting..."
    if ! wait_for_done "$ckpt/rank0.log"; then
        echo "  FAIL: ${name}"; return 1
    fi
    echo "  ${name} done."
}

# V0: F.rms_norm native, no autokernel (pure PyTorch + Inductor)
launch "V0-native-no-hip" ""

# V1: F.rms_norm native + silu_gate_mul HIP (exclude rmsnorm pattern)
launch "V1-native-plus-silu-hip" "--optimize-kernels --autokernel-exclude rmsnorm"

echo
echo "=== Option 3 results ==="
for p in V0-native-no-hip V1-native-plus-silu-hip; do
    ckpt="checkpoints/sprint15-opt3-${p}"
    log="$ckpt/rank0.log"
    [ ! -f "$log" ] && { echo "$p: MISSING"; continue; }
    echo "--- $p ---"
    grep -E '^Done:' "$log" | head -1
    grep -E '^\[step    200\]' "$log" | head -1
done

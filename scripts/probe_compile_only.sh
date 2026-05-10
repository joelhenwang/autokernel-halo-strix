#!/bin/bash
# Phase IV: torch.compile interaction study for OdinFlat.
# Tests whether Inductor / compile mode can lift throughput beyond the
# Phase III rmsnorm-fixed +31% path.

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
    local compile_mode="$2"
    local no_bwd="$3"

    local ckpt="checkpoints/sprint15-compile-${name}"
    rm -rf "$ckpt"
    mkdir -p "$ckpt"

    echo
    echo "=== ${name}  |  ${ckpt} ==="
    echo "  compile_mode: ${compile_mode}"
    [ -n "$no_bwd" ] && echo "  AUTOKERNEL_NO_BWD_HIP=1"
    echo

    local extra_flags='--max-steps 200 --imu1-groups --normuon --lr-2d 5e-3 --lr-1d 8e-4 --intra-doc-mask --value-residuals --head-gating --z-loss 1e-4 --z-loss-fraction 1.0 --attn-softcap 50.0 --activation-monitor --activation-monitor-interval 50 --mup --mup-base-width 256 --spectra-post --spectra-clip-norm 1.0 --optimize-kernels'

    TORCH_COMPILE_MODE="$compile_mode" \
    AUTOKERNEL_NO_BWD_HIP="${no_bwd:-0}" \
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
        echo "  FAIL: ${name}"
        return 1
    fi
    echo "  ${name} done."
}

launch "C2-no-hip-bwd" "max-autotune-no-cudagraphs" "1"
launch "C3-cudagraphs" "max-autotune" ""

echo
echo "=== Phase IV summary ==="
for p in C2-no-hip-bwd C3-cudagraphs; do
    ckpt="checkpoints/sprint15-compile-${p}"
    log="$ckpt/rank0.log"
    [ ! -f "$log" ] && { echo "$p: MISSING"; continue; }
    echo "--- $p ---"
    grep -E '^Done:' "$log" | head -2
    grep -E '^\[step    200\]' "$log" | head -1
    echo "  layers.13 @200:"
    grep '"step": 200' "$ckpt/activation_stats.jsonl" 2>/dev/null | grep 'layers.13' || echo "    (none)"
done

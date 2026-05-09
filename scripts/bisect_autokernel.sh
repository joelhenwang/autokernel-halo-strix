#!/bin/bash
# Phase II: OdinFlat bisect probes -- identify which autokernel pattern
# causes the 65x activation collapse.
#
# 2x2 matrix (rmsnorm x fused_silu_gate_mul):
#
#   P0  neither enabled  (Phase 0 baseline; optimize-kernels off)
#   P1  rmsnorm only
#   P2  fused_silu_gate_mul only
#   P3  both enabled     (Phase 0.4 repro)
#
# Each probe: 200 steps, DDP, OdinFlat + Sprint 1.5 C3 recipe + dolma.
# Runtime: ~15 min per probe * 4 = ~1h total.

set -euo pipefail

cd ~/Desktop/ai_lab/autokernel-halo-strix

wait_for_done() {
    local log="$1"
    while true; do
        sleep 30
        if [ -f "$log" ] && grep -q "^Done:" "$log"; then
            return 0
        fi
        if [ -f "$log" ] && grep -q "^Traceback" "$log" 2>/dev/null; then
            echo "FAIL: $log"
            return 1
        fi
    done
}

launch_probe() {
    local name="$1"
    local optkern_flag="$2"     # "--optimize-kernels" or ""
    local include="$3"          # comma list or empty
    local exclude="$4"          # comma list or empty

    local ckpt="checkpoints/sprint15-bisect-${name}"
    rm -rf "$ckpt"
    mkdir -p "$ckpt"

    echo
    echo "=== ${name}  |  ${ckpt} ==="
    [ -n "$optkern_flag" ] && echo "  optimize-kernels ON"
    [ -n "$include" ] && echo "  include: $include"
    [ -n "$exclude" ] && echo "  exclude: $exclude"
    echo

    local extra_flags="--max-steps 200 --imu1-groups --normuon --lr-2d 5e-3 --lr-1d 8e-4 --intra-doc-mask --value-residuals --head-gating --z-loss 1e-4 --z-loss-fraction 1.0 --attn-softcap 50.0 --activation-monitor --activation-monitor-interval 50 --mup --mup-base-width 256 --spectra-post --spectra-clip-norm 1.0"
    if [ -n "$optkern_flag" ]; then
        extra_flags="$extra_flags $optkern_flag"
    fi
    if [ -n "$include" ]; then
        extra_flags="$extra_flags --autokernel-include $include"
    fi
    if [ -n "$exclude" ]; then
        extra_flags="$extra_flags --autokernel-exclude $exclude"
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

    echo "  launched; waiting for Done: marker..."
    if ! wait_for_done "$ckpt/rank0.log"; then
        echo "  FAIL: ${name}"
        return 1
    fi
    echo "  ${name} done."
}

# P0: baseline (no autokernel at all)
launch_probe "P0-baseline" "" "" ""
# P1: rmsnorm only
launch_probe "P1-rmsnorm-only" "--optimize-kernels" "rmsnorm" ""
# P2: fused_silu_gate_mul only
launch_probe "P2-silu-only" "--optimize-kernels" "fused_silu_gate_mul" ""
# P3: both (Phase 0.4 repro)
launch_probe "P3-both" "--optimize-kernels" "" ""

echo
echo "=== Phase II results ==="
echo
printf "%-22s %-14s %-14s %-14s %-14s %s\n" "probe" "tok/s (agg)" "loss @ 200" "maxabs.13 @200" "scale @200" "patterns"
for p in P0-baseline P1-rmsnorm-only P2-silu-only P3-both; do
    ckpt="checkpoints/sprint15-bisect-${p}"
    log="$ckpt/rank0.log"
    [ ! -f "$log" ] && { echo "$p: MISSING"; continue; }

    TOKS=$(grep -E '^Done:' "$log" 2>/dev/null | sed -E 's/.*\(([0-9,]+) tok.*/\1/' | tr -d ',')
    LOSS=$(grep -E '^\[step    200\]' "$log" | sed -E 's/.*loss=([0-9.]+).*/\1/')
    SCALE=$(grep -E '^\[step    200\]' "$log" | sed -E 's/.*scale=([^ ]+).*/\1/')
    MAXABS=$(grep '"step": 200' "$ckpt/activation_stats.jsonl" 2>/dev/null | grep 'layers.13' | sed -E 's/.*"maxabs": ([0-9.]+).*/\1/')
    PATTERNS=$(grep '\[autokernel\]' "$log" 2>/dev/null | sed -E 's/.*\[autokernel\] //' | tr '\n' ';' | sed 's/;$//')
    printf "%-22s %-14s %-14s %-14s %-14s %s\n" "$p" "${TOKS:--}" "${LOSS:--}" "${MAXABS:--}" "${SCALE:--}" "${PATTERNS:-none}"
done

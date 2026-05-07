#!/bin/bash
# Sprint 1.1 Phase B: run B4 combo (fp16 NS + no-CWD).
# Called on Machine A only.
set -e

CKPT="checkpoints/sprint1.1-B4"
pkill -f torchrun 2>/dev/null || true
ssh joelwang-ai-1@10.77.0.2 "pkill -f torchrun 2>/dev/null || true" 2>/dev/null || true
sleep 4

rm -rf "$CKPT"
mkdir -p "$CKPT"

echo "=== Launching B4 (fp16 NS + no-CWD) ==="

CKPT_DIR="$CKPT" \
MODEL="models/odin_flat.py" \
CLASS="OdinFlat" \
DATASET="datasets/wikitext-103-odin32k.bin" \
EPOCHS="1" \
LR="8e-4" \
EXTRA_FLAGS="--max-steps 200 --imu1-groups --normuon --lr-2d 5e-3 --lr-1d 8e-4 --intra-doc-mask --value-residuals --head-gating --no-muon --auto-eval --ns-dtype fp16 --no-cautious-wd" \
bash scripts/launch_ddp.sh > "$CKPT/launch.log" 2>&1

echo "B4 launched, waiting..."
T0=$(date +%s)
while true; do
    if grep -q '^Done:' "$CKPT/rank0.log" 2>/dev/null; then
        echo "B4 DONE after $(( $(date +%s) - T0 ))s"
        break
    fi
    if grep -qE 'Traceback|RuntimeError|Error' "$CKPT/rank0.log" 2>/dev/null; then
        echo "B4 FATAL"
        tail -30 "$CKPT/rank0.log"
        exit 1
    fi
    if [ $(( $(date +%s) - T0 )) -gt 1500 ]; then
        echo "B4 TIMEOUT after $(( $(date +%s) - T0 ))s"
        tail -20 "$CKPT/rank0.log"
        exit 2
    fi
    sleep 15
done

echo "=== B4 SUMMARY ==="
grep -E 'Done:|^\[step' "$CKPT/rank0.log" | tail -3

echo ""
echo "Waiting 60s for scorecard..."
sleep 60

if [ -f docs/perf/eval-scorecards/sprint1.1-B4-step-200.json ]; then
    echo "Scorecard: docs/perf/eval-scorecards/sprint1.1-B4-step-200.json"
else
    echo "WARNING: scorecard missing"
    tail -10 "$CKPT/step_200.pt.eval.log" 2>/dev/null || true
fi

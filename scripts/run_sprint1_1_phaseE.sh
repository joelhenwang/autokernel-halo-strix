#!/bin/bash
# Sprint 1.1 Phase E: Run 2b final validation.
# Full 1-epoch DDP training on wikitext-103 with B1's fp16 NS config.
# Target: match Run 2's quality (loss 4.4736, wiki_bpb 1.893) with +17-20%
# throughput.

set -e

CKPT="checkpoints/sprint1-run2b"

# Clean kill of any stragglers
pkill -f torchrun 2>/dev/null || true
ssh joelwang-ai-1@10.77.0.2 "pkill -f torchrun 2>/dev/null || true" 2>/dev/null || true
sleep 4

rm -rf "$CKPT"
mkdir -p "$CKPT"

echo "=== Run 2b: OdinFlat wikitext-103 1-epoch with fp16 NS ==="
echo "Expected: ~936 opt steps, ~50-60 min wall, target >=35,000 tok/s"
echo ""

CKPT_DIR="$CKPT" \
MODEL="models/odin_flat.py" \
CLASS="OdinFlat" \
DATASET="datasets/wikitext-103-odin32k.bin" \
EPOCHS="1" \
LR="8e-4" \
CHECKPOINT_INTERVAL="500" \
EXTRA_FLAGS="--imu1-groups --normuon --lr-2d 5e-3 --lr-1d 8e-4 --intra-doc-mask --value-residuals --head-gating --no-muon --auto-eval --ns-dtype fp16" \
bash scripts/launch_ddp.sh > "$CKPT/launch.log" 2>&1

echo "Launched. Waiting for completion (up to 75 min)..."
T0=$(date +%s)

while true; do
    if grep -q '^Done:' "$CKPT/rank0.log" 2>/dev/null; then
        echo "DONE after $(( $(date +%s) - T0 ))s"
        break
    fi
    if grep -qE '^Traceback|^RuntimeError|StabilityGuard.*unrecoverable' "$CKPT/rank0.log" 2>/dev/null; then
        echo "FATAL"
        tail -30 "$CKPT/rank0.log"
        exit 1
    fi
    local_elapsed=$(( $(date +%s) - T0 ))
    if [ "$local_elapsed" -gt 4500 ]; then
        echo "TIMEOUT"
        tail -20 "$CKPT/rank0.log"
        exit 2
    fi
    # Occasional progress hint
    if [ $((local_elapsed % 600)) -lt 20 ] && [ "$local_elapsed" -gt 100 ]; then
        LAST_STEP=$(grep -E '^\[step' "$CKPT/rank0.log" 2>/dev/null | tail -1)
        if [ -n "$LAST_STEP" ]; then
            echo "  [${local_elapsed}s] $LAST_STEP"
        fi
    fi
    sleep 30
done

echo ""
echo "=== Run 2b final log lines ==="
grep -E 'Done:|^\[step' "$CKPT/rank0.log" | tail -5

echo ""
echo "Waiting 90s for scorecards to complete..."
sleep 90

for step in 500 936; do
    f=docs/perf/eval-scorecards/sprint1-run2b-step-${step}.json
    if [ -f "$f" ]; then
        echo "Scorecard step $step: $f"
    else
        echo "WARNING: scorecard step $step missing"
        tail -10 "$CKPT/step_${step}.pt.eval.log" 2>/dev/null || true
    fi
done

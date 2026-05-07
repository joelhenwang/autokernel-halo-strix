#!/bin/bash
# Sprint 1.1 Phase B ablation runner.
#
# Runs B0-B3 sequentially (200 steps DDP each). Each ablation uses the
# Run 2 baseline config + one toggle. Auto-eval fires a step_200 scorecard
# via --auto-eval (Sprint 2 detached subprocess).
#
# B0: reference (Run 2 config, no toggles)
# B1: + --ns-dtype fp16
# B2: + --neuron-norm-min-dim 512
# B3: + --no-cautious-wd
# B4: computed separately after B1-B3 results by hand.
#
# Machine A orchestrates; Machine B runs rank 1 via SSH from launch_ddp.sh.
# This script must be launched FROM Machine A (via run_remote.sh or ssh).
#
# Usage:
#   bash run_remote.sh "cd ~/Desktop/ai_lab/autokernel-halo-strix && bash scripts/run_sprint1_1_phaseB.sh"
#
# Monitors with `grep Done: $CKPT_DIR/rank0.log` rather than polling PIDs —
# launch_ddp.sh detaches so PIDs vanish. Done line is written at normal exit.

set -e

BASE_EXTRA="--max-steps 200 --imu1-groups --normuon \
--lr-2d 5e-3 --lr-1d 8e-4 \
--intra-doc-mask --value-residuals --head-gating \
--no-muon --auto-eval"

wait_for_done() {
    local CKPT="$1"
    local name="$2"
    local max_wait_sec="${3:-1200}"  # 20 min default
    local t0=$(date +%s)
    echo "[$name] Waiting for completion (max ${max_wait_sec}s)..."
    while true; do
        # Check if rank0 printed the Done line
        if grep -q "^Done:" "$CKPT/rank0.log" 2>/dev/null; then
            echo "[$name] DONE after $(( $(date +%s) - t0 ))s"
            return 0
        fi
        # Check if rank0 hit a fatal error
        if grep -qE "Traceback|RuntimeError" "$CKPT/rank0.log" 2>/dev/null; then
            echo "[$name] FATAL ERROR in $CKPT/rank0.log"
            tail -30 "$CKPT/rank0.log"
            return 1
        fi
        # Safety timeout
        local elapsed=$(( $(date +%s) - t0 ))
        if [ "$elapsed" -gt "$max_wait_sec" ]; then
            echo "[$name] TIMEOUT after ${elapsed}s — aborting"
            tail -20 "$CKPT/rank0.log"
            return 2
        fi
        sleep 15
    done
}

run_ablation() {
    local NAME="$1"
    local EXTRA="$2"
    local CKPT="checkpoints/sprint1.1-${NAME}"

    echo ""
    echo "============================================================"
    echo "=== $NAME: $CKPT"
    echo "=== Extra flags: $EXTRA"
    echo "============================================================"

    # Belt-and-braces: kill any leftover torchrun on both machines
    pkill -f torchrun 2>/dev/null || true
    ssh joelwang-ai-1@10.77.0.2 "pkill -f torchrun 2>/dev/null || true" 2>/dev/null || true
    sleep 4

    rm -rf "$CKPT"
    mkdir -p "$CKPT"

    CKPT_DIR="$CKPT" \
    MODEL="models/odin_flat.py" \
    CLASS="OdinFlat" \
    DATASET="datasets/wikitext-103-odin32k.bin" \
    EPOCHS="1" \
    LR="8e-4" \
    EXTRA_FLAGS="$EXTRA" \
    bash scripts/launch_ddp.sh > "$CKPT/launch.log" 2>&1

    wait_for_done "$CKPT" "$NAME" 1500 || return $?

    # Print summary line
    echo ""
    echo "=== $NAME SUMMARY ==="
    grep -E "Done:|^\[step " "$CKPT/rank0.log" | tail -3
    echo ""
}

START=$(date +%s)

run_ablation "B0" "$BASE_EXTRA"
run_ablation "B1" "$BASE_EXTRA --ns-dtype fp16"
run_ablation "B2" "$BASE_EXTRA --neuron-norm-min-dim 512"
run_ablation "B3" "$BASE_EXTRA --no-cautious-wd"

ELAPSED=$(( $(date +%s) - START ))
echo ""
echo "============================================================"
echo "=== All 4 ablations complete in ${ELAPSED}s ($((ELAPSED/60))m) ==="
echo "============================================================"

echo ""
echo "Scorecards spawned via --auto-eval; waiting 60s for them to complete..."
sleep 60

for NAME in B0 B1 B2 B3; do
    CKPT="checkpoints/sprint1.1-${NAME}"
    EVAL_JSON="docs/perf/eval-scorecards/sprint1.1-${NAME}-step-200.json"
    # The auto-eval names scorecards based on the checkpoint directory name
    # (see scripts/eval_checkpoint.py for the naming convention). Glob to find.
    FOUND=$(ls docs/perf/eval-scorecards/*sprint1.1-${NAME}*step_200*.json 2>/dev/null | head -1)
    if [ -n "$FOUND" ]; then
        echo "[$NAME] scorecard: $FOUND"
    else
        echo "[$NAME] WARNING: scorecard not yet found. eval.log tail:"
        tail -5 "$CKPT/step_200.pt.eval.log" 2>/dev/null || echo "  (no eval.log)"
    fi
done

echo ""
echo "=== DONE. Extract throughput + BPB from each run's rank0.log and scorecard ==="

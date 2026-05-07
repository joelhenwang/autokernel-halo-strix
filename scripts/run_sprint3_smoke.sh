#!/bin/bash
# Sprint 3 smoke run — de-risks the full dolma-10B training before the 50h
# commitment. Target: OdinHalo, dolma-10b-odin32k, 1000 opt steps (~30 min),
# all the new fp16 stability guards active.
#
# Success criteria:
#   - No NaN / rollback across the whole run
#   - activation_stats.jsonl shows fp16_headroom > 5 for all layers
#   - scaler.scale stays below 4096 throughout
#   - Loss trajectory decreases monotonically (allowing noise)
#   - Final step_1000 scorecard: wiki_bpb + dolma_bpb both finite + sensible
#
# If any gate trips: read the nan_dump_step_N.pt via
# knowledge/training/fp16_stability_gfx1151.md's playbook and decide.

set -e

CKPT="checkpoints/sprint3-smoke-dolma"

pkill -f torchrun 2>/dev/null || true
ssh joelwang-ai-1@10.77.0.2 "pkill -f torchrun 2>/dev/null || true" 2>/dev/null || true
sleep 4

rm -rf "$CKPT"
mkdir -p "$CKPT"

echo "=== Sprint 3 SMOKE — OdinHalo dolma-10b, 1000 opt steps ==="
echo "Guards: --z-loss 1e-4 --attn-softcap 50.0 --activation-monitor"
echo "Default guards (auto): iter_scales.clamp, fp16 NS, growth_interval=500"
echo ""

CKPT_DIR="$CKPT" \
MODEL="models/odin_halo.py" \
CLASS="OdinHalo" \
DATASET="datasets/dolma-10b-odin32k.bin" \
EPOCHS="1" \
LR="8e-4" \
BLOCK="512" \
BATCH="16" \
ACCUM="8" \
CHECKPOINT_INTERVAL="250" \
EXTRA_FLAGS="--max-steps 1000 --imu1-groups --normuon \
--lr-2d 5e-3 --lr-1d 8e-4 --no-muon \
--intra-doc-mask --value-residuals --head-gating \
--z-loss 1e-4 --z-loss-fraction 0.4 \
--attn-softcap 50.0 \
--activation-monitor --activation-monitor-interval 100 \
--auto-eval" \
bash scripts/launch_ddp.sh > "$CKPT/launch.log" 2>&1

echo "Launched. Waiting for completion (up to 50 min)..."
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
    if [ "$local_elapsed" -gt 3000 ]; then
        echo "TIMEOUT after ${local_elapsed}s"
        tail -20 "$CKPT/rank0.log"
        exit 2
    fi
    # Periodic progress hint
    if [ $((local_elapsed % 300)) -lt 30 ] && [ "$local_elapsed" -gt 60 ]; then
        LAST=$(grep -E '^\[step' "$CKPT/rank0.log" 2>/dev/null | tail -1)
        if [ -n "$LAST" ]; then
            echo "  [${local_elapsed}s] $LAST"
        fi
    fi
    sleep 30
done

echo ""
echo "=== Summary ==="
grep -E '^\[step|Done:|StabilityGuard|fp16-stability|activation monitor|NaN' "$CKPT/rank0.log" | tail -15

echo ""
echo "=== activation_stats.jsonl tail (most recent per layer) ==="
if [ -f "$CKPT/activation_stats.jsonl" ]; then
    # Show stats from the last sampled step
    LAST_STEP=$(tail -50 "$CKPT/activation_stats.jsonl" | python3 -c "
import json, sys
lines = [json.loads(l) for l in sys.stdin if l.strip()]
if not lines:
    print('(empty)')
else:
    last = max(l['step'] for l in lines)
    print(f'Last sample step: {last}')
    for l in lines:
        if l['step'] == last:
            print(f'  {l[\"layer\"]:<30s} maxabs={l[\"maxabs\"]:10.3f} headroom={l[\"fp16_headroom\"]:>8.1f} {l[\"dtype\"]}')
" 2>&1)
    echo "$LAST_STEP"
else
    echo "WARNING: activation_stats.jsonl not written"
fi

echo ""
echo "=== nan dumps (should be empty) ==="
ls -la "$CKPT"/nan_dump_step_*.pt 2>/dev/null || echo "  (no NaN dumps — good)"

echo ""
echo "=== scorecards ==="
for step in 250 500 750 1000; do
    f=docs/perf/eval-scorecards/sprint3-smoke-dolma-step-${step}.json
    if [ -f "$f" ]; then
        echo "  step $step: $f"
    fi
done

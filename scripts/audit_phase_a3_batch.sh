#!/bin/bash
# Phase A.3 batch runner: 14-model × 3-config diagnostic probe suite.
#
# For each model, runs three 50-step probes:
#   V0: baseline (no --optimize-kernels)
#   V1: --optimize-kernels (all patterns active)
#   V3: --optimize-kernels --autokernel-exclude fused_silu_gate_mul
#
# Each probe writes a diag.jsonl to checkpoints/audit-{MODEL}-{CONFIG}/ which
# scripts/analyze_diag_frozen_params.py can aggregate.
#
# Usage:
#   bash scripts/audit_phase_a3_batch.sh
#
# Expected wall: ~3-4 hours on a single machine at 50 steps/probe.
#
# Plan: Phase A.3 of master remediation plan.

set -eo pipefail

# Path resolver + venv activation.
if [ -d ~/Desktop/comfyui-rocm7.12/autokernel-halo-strix ]; then
    cd ~/Desktop/comfyui-rocm7.12/autokernel-halo-strix
    source ~/Desktop/comfyui-rocm7.12/.venv/bin/activate
elif [ -d ~/Desktop/ai_lab/autokernel-halo-strix ]; then
    cd ~/Desktop/ai_lab/autokernel-halo-strix
    source .venv/bin/activate
fi

export HSA_OVERRIDE_GFX_VERSION=11.5.1
export TORCH_COMPILE_MODE="${TORCH_COMPILE_MODE:-max-autotune-no-cudagraphs}"

# Audit root: consolidates all output under docs/perf/ for later sync-back.
AUDIT_DIR="checkpoints/audit-phase-a3"
mkdir -p "$AUDIT_DIR"

# Model catalog: (label, model_file, class_name, block_size)
# block_size picked to respect each model's architectural tolerance.
# Uses the Sprint-1.5 recipe flags for all (so any model that supports them
# gets the same stress-test; models that don't auto-ignore).
MODELS=(
    "odin_flat           models/odin_flat.py           OdinFlat            512"
    "odin_flat_ablation  models/odin_flat.py           OdinFlatAblation    512"
    "odin_flat_mini      models/odin_flat.py           OdinFlatMini        256"
    "odin_flat_30m       models/odin_flat_30m.py       OdinFlat30M         512"
    "odin_flat_30m_mini  models/odin_flat_30m.py       OdinFlat30MMini     256"
    "odin_halo           models/odin_halo.py           OdinHalo            256"
    "odin_halo_ablation  models/odin_halo.py           OdinHaloAblation    256"
    "odin_halo_mini      models/odin_halo.py           OdinHaloMini        256"
    "vidar_halo          models/vidar_halo.py          VidarHalo           256"
    "baldr_halo          models/baldr_halo.py          BaldrHalo           256"
    "chimera_halo        models/chimera_halo.py        ChimeraHalo         256"
    "fenrir_halo         models/fenrir_halo.py         FenrirHalo          256"
    "tyr_halo            models/tyr_halo.py            TyrHalo             256"
    "jormungandr_halo    models/jormungandr_halo.py    JormungandrHalo     256"
)

CONFIGS=("V0" "V1" "V3")

run_probe() {
    local LABEL=$1 MODEL_FILE=$2 CLASS=$3 BLOCK=$4 CONFIG=$5
    local CKPT="$AUDIT_DIR/$LABEL-$CONFIG"

    case "$CONFIG" in
        V0) CONFIG_FLAGS="" ;;
        V1) CONFIG_FLAGS="--optimize-kernels" ;;
        V3) CONFIG_FLAGS="--optimize-kernels --autokernel-exclude fused_silu_gate_mul" ;;
        *)  echo "bad CONFIG=$CONFIG" >&2; return 1 ;;
    esac

    echo ""
    echo "=========================================="
    echo "=== $LABEL / $CONFIG (class=$CLASS, block=$BLOCK)"
    echo "=========================================="

    rm -rf "$CKPT"
    mkdir -p "$CKPT"

    # Fixed small batch to fit memory uniformly; the point is grad flow, not tok/s.
    # num-workers=2 minimizes dataloader startup overhead.
    # warmup-steps=10 is short because we only run 50 steps.
    # Pass Sprint 1 feature flags so head_gate / v_res_scale / doc_mask
    # paths are actually exercised in forward — otherwise preflight
    # false-positives on these unused-parameter cases (2026-05-11 fix).
    # Auto-cleared by models that don't implement the flag via
    # getattr(args, ...) checks in train_ddp.py.
    set +e
    torchrun --nproc_per_node=1 --nnodes=1 --node_rank=0 \
      --master_addr=127.0.0.1 --master_port=29512 \
      scripts/train_ddp.py \
      --model "$MODEL_FILE" --class-name "$CLASS" \
      --dataset datasets/dolma-10b-odin32k.bin --epochs 1 \
      --block-size "$BLOCK" --batch-size 4 --accum-steps 2 \
      --compile --no-muon --lr 1e-3 --backend gloo \
      --warmup-steps 10 --num-workers 2 \
      --max-grad-norm 1.0 \
      --checkpoint-dir "$CKPT" --checkpoint-interval 9999 --log-interval 10 \
      --max-steps 50 \
      --intra-doc-mask --value-residuals --head-gating \
      --diag-frozen-params "$CKPT/diag.jsonl" \
      $CONFIG_FLAGS \
      > "$CKPT/run.log" 2>&1
    rc=$?
    set -e

    if [ $rc -ne 0 ]; then
        echo "  FAIL rc=$rc; last 5 lines of log:"
        tail -5 "$CKPT/run.log" 2>/dev/null | sed 's/^/    /'
        echo "$LABEL $CONFIG FAIL rc=$rc" >> "$AUDIT_DIR/_failures.log"
        # Give GPU a moment to settle after any HIP-level failure.
        sleep 5
        return 1
    else
        # Emit a one-line summary.
        STEPS_RECORDED=$(wc -l < "$CKPT/diag.jsonl" 2>/dev/null || echo 0)
        LAST_LOSS=$(grep -oE 'best loss=[0-9.]+' "$CKPT/run.log" 2>/dev/null | tail -1)
        echo "  OK steps=$STEPS_RECORDED $LAST_LOSS"
        return 0
    fi
}

FAILURES=0
SUCCESSES=0
for entry in "${MODELS[@]}"; do
    read -r LABEL MODEL_FILE CLASS BLOCK <<< "$entry"
    for CONFIG in "${CONFIGS[@]}"; do
        if run_probe "$LABEL" "$MODEL_FILE" "$CLASS" "$BLOCK" "$CONFIG"; then
            SUCCESSES=$((SUCCESSES + 1))
        else
            FAILURES=$((FAILURES + 1))
        fi
    done
done

echo ""
echo "=========================================="
echo "=== Phase A.3 batch complete: $SUCCESSES ok / $FAILURES fail (of $((SUCCESSES+FAILURES)) total)"
echo "=========================================="

# Consolidated file list for later analysis
find "$AUDIT_DIR" -name "diag.jsonl" -type f | sort > "$AUDIT_DIR/_all_diag_files.txt"
echo ""
echo "Diag files listed in $AUDIT_DIR/_all_diag_files.txt"
echo "Next: python scripts/analyze_audit_phase_a3.py"

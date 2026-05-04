#!/bin/bash
# VIDAR-HALO Ablation Runner
#
# Parameterized by environment variables:
#   TIER=s|m|v    Ablation tier (screening / scale-confirm / validation)
#   CONFIG=...    Technique configuration name
#   SEED=42       Random seed (default 42)
#
# Usage:
#   TIER=s CONFIG=baseline bash scripts/run_vidar_ablation.sh
#   TIER=s CONFIG=p1a bash scripts/run_vidar_ablation.sh
#   TIER=m CONFIG=p1a_p1b bash scripts/run_vidar_ablation.sh

set -euo pipefail

TIER="${TIER:-s}"
CONFIG="${CONFIG:-baseline}"
SEED="${SEED:-42}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
LOG_DIR="${PROJECT_DIR}/logs/vidar_abl"
mkdir -p "$LOG_DIR"

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOG_FILE="$LOG_DIR/vidar_abl_${TIER}_${CONFIG}_seed${SEED}_${TIMESTAMP}.log"

# --- Tier-specific defaults ---
if [ "$TIER" = "s" ]; then
    MODEL_CLASS="VidarHaloAblation"
    DATASET="datasets/babylm-vidar32k.bin"
    TOKENIZER="--tokenizer-path tokenizers/vidar-32k/tokenizer.json"
    BLOCK_SIZE=512
    BATCH_SIZE=32
    ACCUM=2
    LR=0.003
    WARMUP=50
    COMPILE_FLAG=""
    OPTIMIZE="--optimize-kernels"
    CKPT_INT="--checkpoint-interval 999999"
elif [ "$TIER" = "m" ]; then
    MODEL_CLASS="VidarHaloGPT2"
    DATASET="datasets/wikitext-103-raw.bin"
    TOKENIZER=""
    BLOCK_SIZE=512
    BATCH_SIZE=16
    ACCUM=4
    LR=0.002
    WARMUP=150
    COMPILE_FLAG="--compile"
    OPTIMIZE="--optimize-kernels"
    CKPT_INT="--checkpoint-interval 999999"
elif [ "$TIER" = "v" ]; then
    MODEL_CLASS="VidarHaloGPT2"
    DATASET="datasets/stem-crawl-solo.bin"
    TOKENIZER=""
    BLOCK_SIZE=1024
    BATCH_SIZE=8
    ACCUM=8
    LR=0.002
    WARMUP=300
    COMPILE_FLAG="--compile"
    OPTIMIZE="--optimize-kernels"
    CKPT_INT=""
else
    echo "ERROR: TIER must be s, m, or v (got: $TIER)"
    exit 1
fi

# --- Config-specific technique flags ---
TECHNIQUE_FLAGS=""
POLAR_NS=""
MIN_LR=""

case "$CONFIG" in
    baseline)
        ;;
    p1a)
        POLAR_NS="--polar-ns"
        ;;
    p1b)
        MIN_LR="--min-lr-ratio 0.1"
        ;;
    p1c)
        TECHNIQUE_FLAGS="--model-kwarg iter_scales_enabled=true"
        ;;
    p1a_p1b)
        POLAR_NS="--polar-ns"
        MIN_LR="--min-lr-ratio 0.1"
        ;;
    p1a_p1b_p1c)
        POLAR_NS="--polar-ns"
        MIN_LR="--min-lr-ratio 0.1"
        TECHNIQUE_FLAGS="--model-kwarg iter_scales_enabled=true"
        ;;
    p2a)
        TECHNIQUE_FLAGS="--model-kwarg softcap=true"
        ;;
    p3a)
        TECHNIQUE_FLAGS="--model-kwarg delayed_recurrence=true"
        ;;
    p3a_soft)
        TECHNIQUE_FLAGS="--model-kwarg delayed_recurrence=true --model-kwarg delayed_soft=true"
        ;;
    p4a)
        TECHNIQUE_FLAGS="--model-kwarg parallel_residuals=true"
        ;;
    p5a)
        TECHNIQUE_FLAGS="--model-kwarg skip_connection=true"
        ;;
    p3a_p4a)
        TECHNIQUE_FLAGS="--model-kwarg delayed_recurrence=true --model-kwarg parallel_residuals=true"
        ;;
    p3a_p5a)
        TECHNIQUE_FLAGS="--model-kwarg delayed_recurrence=true --model-kwarg skip_connection=true"
        ;;
    p4a_p5a)
        TECHNIQUE_FLAGS="--model-kwarg parallel_residuals=true --model-kwarg skip_connection=true"
        ;;
    full_stack)
        POLAR_NS="--polar-ns"
        MIN_LR="--min-lr-ratio 0.1"
        TECHNIQUE_FLAGS="--model-kwarg iter_scales_enabled=true --model-kwarg softcap=true --model-kwarg delayed_recurrence=true --model-kwarg delayed_soft=true --model-kwarg parallel_residuals=true --model-kwarg skip_connection=true"
        ;;
    *)
        echo "ERROR: Unknown CONFIG: $CONFIG"
        echo "Valid: baseline p1a p1b p1c p1a_p1b p1a_p1b_p1c p2a p3a p3a_soft p4a p5a"
        echo "       p3a_p4a p3a_p5a p4a_p5a full_stack"
        exit 1
        ;;
esac

echo "========================================"
echo "VIDAR ABLATION: tier=$TIER config=$CONFIG seed=$SEED"
echo "Model: $MODEL_CLASS | Dataset: $DATASET"
echo "Started: $(date)"
echo "Log: $LOG_FILE"
echo "========================================"

python3 -m halo_training \
    --model models/vidar_halo.py \
    --class-name "$MODEL_CLASS" \
    --dataset "$DATASET" \
    $TOKENIZER \
    $COMPILE_FLAG $OPTIMIZE \
    --muon --ema --scheduler wsd --z-loss 1e-4 \
    $POLAR_NS $MIN_LR \
    --lr "$LR" --warmup-steps "$WARMUP" \
    --block-size "$BLOCK_SIZE" --batch-size "$BATCH_SIZE" --accum-steps "$ACCUM" \
    --epochs 1 $CKPT_INT \
    --checkpoint-dir "checkpoints/vidar_abl_${TIER}_${CONFIG}" \
    $TECHNIQUE_FLAGS \
    2>&1 | tee "$LOG_FILE"

echo "========================================"
echo "FINISHED: tier=$TIER config=$CONFIG seed=$SEED"
echo "Ended: $(date)"
echo "========================================"

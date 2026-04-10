#!/bin/bash
# PLE + MatFormer 8-Run Ablation Matrix
#
# Runs 6 training configurations sequentially, logs output, and prints comparison.
#
# Usage:
#   bash scripts/run_ple_ablation.sh [--smoke|--full|--results]
#
#   --smoke   Run 200-step smoke tests only (~2 min each)
#   --full    Run full 45-min training runs (~4.5 hours total)
#   --results Parse existing logs and print comparison table

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
LOG_DIR="${PROJECT_DIR}/logs"
CKPT_DIR="${PROJECT_DIR}/checkpoints"

mkdir -p "$LOG_DIR" "$CKPT_DIR"

# Configuration matrix: (name, model_file, class_name, description)
CONFIGS=(
    "01_base|models/tempest.py|Tempest|Base control (no PLE, no MatFormer)"
    "02_ple_a|models/virtuoso.py|VirtuosoPleA|PLE Path A only (+2.1M params)"
    "03_ple_b|models/virtuoso.py|VirtuosoPleB|PLE Path B only (+2.7M params)"
    "04_ple_ab|models/virtuoso.py|VirtuosoPleAB|PLE A+B combined (+4.8M params)"
    "05_matformer|models/virtuoso.py|VirtuosoMatFormer|MatFormer only (+0 params)"
    "06_full|models/virtuoso.py|VirtuosoFull|PLE A+B + MatFormer (full)"
)

# Common training arguments
COMMON_ARGS="--dataset babylm --batch-size 16 --accum-steps 4 --lr 8e-4 --log-interval 10 --optimize-kernels"

run_smoke() {
    echo "=============================================="
    echo "  PLE ABLATION — SMOKE TESTS (200 steps each)"
    echo "=============================================="
    echo ""

    local passed=0
    local failed=0

    for config in "${CONFIGS[@]}"; do
        IFS='|' read -r name model_file class_name desc <<< "$config"
        log_file="${LOG_DIR}/smoke_${name}.log"

        echo "--- [$name] $desc ---"
        echo "    Model: $model_file :: $class_name"
        echo "    Log:   $log_file"

        if python -m halo_training \
            --model "$model_file" \
            --class-name "$class_name" \
            --smoke \
            2>&1 | tee "$log_file"; then
            echo "    Result: PASSED"
            ((passed++))
        else
            echo "    Result: FAILED (see $log_file)"
            ((failed++))
        fi
        echo ""
    done

    echo "=============================================="
    echo "  SMOKE TEST SUMMARY: $passed passed, $failed failed"
    echo "=============================================="

    if [ "$failed" -gt 0 ]; then
        echo "Fix failures before running full training!"
        exit 1
    fi
}

run_full() {
    echo "=============================================="
    echo "  PLE ABLATION — FULL 45-MIN TRAINING RUNS"
    echo "=============================================="
    echo "  Total estimated time: ~1 hour (10 min each)"
    echo ""

    local run_num=0
    local total=${#CONFIGS[@]}

    for config in "${CONFIGS[@]}"; do
        IFS='|' read -r name model_file class_name desc <<< "$config"
        ((run_num++))
        log_file="${LOG_DIR}/ablation_${name}.log"
        ckpt_dir="${CKPT_DIR}/ablation_${name}"

        echo "=============================================="
        echo "  RUN $run_num/$total: [$name] $desc"
        echo "  Started: $(date '+%Y-%m-%d %H:%M:%S')"
        echo "=============================================="

        mkdir -p "$ckpt_dir"

        python -m halo_training \
            --model "$model_file" \
            --class-name "$class_name" \
            $COMMON_ARGS \
            --time-budget 10 \
            --checkpoint-dir "$ckpt_dir" \
            2>&1 | tee "$log_file"

        echo ""
        echo "  Finished: $(date '+%Y-%m-%d %H:%M:%S')"
        echo ""
    done

    echo "=============================================="
    echo "  ALL RUNS COMPLETE — Parsing results..."
    echo "=============================================="
    echo ""
    print_results
}

print_results() {
    echo "=============================================="
    echo "  PLE ABLATION RESULTS COMPARISON"
    echo "=============================================="
    echo ""
    printf "%-20s %-35s %10s %10s %10s\n" "Run" "Config" "Loss" "BPB" "tok/s"
    printf "%-20s %-35s %10s %10s %10s\n" "----" "------" "----" "---" "-----"

    for config in "${CONFIGS[@]}"; do
        IFS='|' read -r name model_file class_name desc <<< "$config"
        log_file="${LOG_DIR}/ablation_${name}.log"

        if [ ! -f "$log_file" ]; then
            printf "%-20s %-35s %10s %10s %10s\n" "$name" "$class_name" "N/A" "N/A" "N/A"
            continue
        fi

        # Extract final metrics from log (look for last occurrence)
        local loss=$(grep -oP 'loss[=: ]+\K[0-9]+\.[0-9]+' "$log_file" | tail -1)
        local bpb=$(grep -oP '[Bb][Pp][Bb][=: ]+\K[0-9]+\.[0-9]+' "$log_file" | tail -1)
        local toks=$(grep -oP 'tok/s[=: ]+\K[0-9,]+' "$log_file" | tail -1 | tr -d ',')

        printf "%-20s %-35s %10s %10s %10s\n" \
            "$name" \
            "$class_name" \
            "${loss:-N/A}" \
            "${bpb:-N/A}" \
            "${toks:-N/A}"
    done

    echo ""
    echo "Log files: ${LOG_DIR}/ablation_*.log"
    echo "Checkpoints: ${CKPT_DIR}/ablation_*/"
}

# Parse arguments
case "${1:---smoke}" in
    --smoke)   run_smoke ;;
    --full)    run_full ;;
    --results) print_results ;;
    *)
        echo "Usage: $0 [--smoke|--full|--results]"
        echo ""
        echo "  --smoke    200-step smoke tests (~2 min each)"
        echo "  --full     Full 45-min training runs (~4.5 hours)"
        echo "  --results  Parse existing logs and print comparison"
        exit 1
        ;;
esac

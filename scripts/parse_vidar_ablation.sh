#!/bin/bash
# Parse VIDAR-HALO ablation logs and print comparison table
#
# Usage:
#   bash scripts/parse_vidar_ablation.sh          # all logs
#   bash scripts/parse_vidar_ablation.sh s         # tier S only
#   bash scripts/parse_vidar_ablation.sh m         # tier M only

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LOG_DIR="$(dirname "$SCRIPT_DIR")/logs/vidar_abl"
TIER_FILTER="${1:-}"

if [ ! -d "$LOG_DIR" ]; then
    echo "No ablation logs found at $LOG_DIR"
    exit 1
fi

echo "========================================"
echo "  VIDAR-HALO ABLATION RESULTS"
echo "========================================"
echo ""
printf "%-18s %-5s %10s %10s %10s %8s\n" "Config" "Tier" "FinalLoss" "BPB" "tok/s" "Steps"
printf "%-18s %-5s %10s %10s %10s %8s\n" "------" "----" "---------" "---" "-----" "-----"

for log in "$LOG_DIR"/vidar_abl_*.log; do
    [ -f "$log" ] || continue
    fname=$(basename "$log" .log)

    # Extract tier and config from filename: vidar_abl_{tier}_{config}_seed{N}_{timestamp}
    tier=$(echo "$fname" | sed 's/vidar_abl_\([^_]*\)_.*/\1/')
    config=$(echo "$fname" | sed 's/vidar_abl_[^_]*_\(.*\)_seed.*/\1/')

    if [ -n "$TIER_FILTER" ] && [ "$tier" != "$TIER_FILTER" ]; then
        continue
    fi

    loss=$(grep -oP 'loss[=: ]+\K[0-9]+\.[0-9]+' "$log" 2>/dev/null | tail -1 || echo "")
    bpb=$(grep -oP '[Bb][Pp][Bb][=: ]+\K[0-9]+\.[0-9]+' "$log" 2>/dev/null | tail -1 || echo "")
    toks=$(grep -oP 'tok/s[=: ]+\K[0-9,]+' "$log" 2>/dev/null | tail -1 | tr -d ',' || echo "")
    steps=$(grep -oP 'step[=: ]+\K[0-9]+' "$log" 2>/dev/null | tail -1 || echo "")

    printf "%-18s %-5s %10s %10s %10s %8s\n" \
        "$config" "$tier" "${loss:-N/A}" "${bpb:-N/A}" "${toks:-N/A}" "${steps:-N/A}"
done | sort -k2,2 -k3,3n

echo ""
echo "Logs: $LOG_DIR/"

# Show baseline comparison if baseline exists
BASELINE_LOSS=""
for log in "$LOG_DIR"/vidar_abl_s_baseline_*.log; do
    [ -f "$log" ] || continue
    BASELINE_LOSS=$(grep -oP 'loss[=: ]+\K[0-9]+\.[0-9]+' "$log" 2>/dev/null | tail -1 || echo "")
    break
done

if [ -n "$BASELINE_LOSS" ]; then
    echo ""
    echo "Tier S baseline loss: $BASELINE_LOSS"
    echo "Keep threshold: delta >= -0.05 | Maybe: -0.02 to -0.05 | Drop: > -0.02"
fi

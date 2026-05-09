#!/bin/bash
# Summarize Phase II bisect results
set -euo pipefail
cd ~/Desktop/ai_lab/autokernel-halo-strix

for p in P0-baseline P1-rmsnorm-only P2-silu-only P3-both; do
    ckpt="checkpoints/sprint15-bisect-${p}"
    log="$ckpt/rank0.log"
    echo "=== ${p} ==="
    if [ -f "$log" ]; then
        grep -E '^\[step|^Done:|\[autokernel\]' "$log" 2>/dev/null | head -10
        echo "--- activation_stats layers.13 @ steps 50/100/150/200 ---"
        grep '"layer": "layers.13"' "$ckpt/activation_stats.jsonl" 2>/dev/null | tail -4
    else
        echo "  log missing"
    fi
    echo
done

#!/bin/bash
# SWE Specialization Training Pipeline for ARGUS-PRIME
# Machine B: joelwang-ai-1@192.168.1.145
#
# Curriculum:
#   Stage B: Bug Localization SFT   (~50M tokens, ~1 hour)
#   Stage C: Code Repair SFT        (~100M tokens, ~2 hours)
#   Stage D: Bug Explanation SFT     (~130M tokens, ~2.5 hours)
#
# Usage:
#   bash scripts/run_swe_training.sh prepare   # Run data preprocessing (on local machine)
#   bash scripts/run_swe_training.sh stage_b   # Launch localization SFT on Machine B
#   bash scripts/run_swe_training.sh stage_c   # Launch code repair SFT on Machine B
#   bash scripts/run_swe_training.sh stage_d   # Launch bug explanation SFT on Machine B
#   bash scripts/run_swe_training.sh all       # Run full pipeline sequentially

set -e

REMOTE="joelwang-ai-1@192.168.1.145"
REMOTE_DIR="~/Desktop/comfyui-rocm7.12/autokernel-halo-strix"
VENV="~/Desktop/comfyui-rocm7.12/.venv"

# Base checkpoint: OpenHermes SFT (best general instruction-following)
BASE_CKPT="checkpoints/argus_prime_sft_openhermes/step_14710.pt"

# Common training args
MODEL="models/argus_prime.py"
CLASS="ArgusPrime"
COMMON="--compile --optimize-kernels --muon --block-size 1536 --batch-size 16 --accum-steps 4 --log-interval 100 --warmup-steps 50"

run_remote() {
    ssh -o ConnectTimeout=10 "$REMOTE" "
        source $VENV/bin/activate
        cd $REMOTE_DIR
        $@
    "
}

case "${1:-help}" in
    prepare)
        echo "=== Step 1: Preprocessing SWE data (streaming from HuggingFace) ==="
        echo "This will take ~30 minutes for the full 391K rows..."
        python scripts/prepare_swe_data.py \
            --output-dir datasets/swe_prepared \
            --max-tokens 1536

        echo ""
        echo "=== Step 2: Syncing to Machine B ==="
        bash sync_remote_b.sh
        echo "Done! Data is ready on Machine B."
        ;;

    stage_b)
        echo "=== Stage B: Bug Localization SFT ==="
        echo "Short examples (avg 255 tokens), fast convergence"
        run_remote "nohup python -m halo_training \
            --model $MODEL --class-name $CLASS \
            --phase sft --sft-format chatml \
            --sft-dataset datasets/swe_prepared/swe_localize.jsonl \
            --system-prompt 'You are an expert software engineer.' \
            --resume-from $BASE_CKPT \
            --checkpoint-dir checkpoints/argus_prime_swe_localize \
            --checkpoint-interval 3000 \
            --lr 3e-5 --epochs 3 --time-budget 120 \
            $COMMON \
            > checkpoints/argus_prime_swe_localize/stage_b.log 2>&1 &"
        echo "Launched! Monitor with:"
        echo "  bash run_remote_b.sh 'tail -5 checkpoints/argus_prime_swe_localize/train_log.jsonl'"
        ;;

    stage_c)
        echo "=== Stage C: Code Repair SFT ==="
        echo "Core skill: generate SEARCH/REPLACE patches"

        # Use localization checkpoint if available, otherwise base
        RESUME="$BASE_CKPT"
        run_remote "test -f checkpoints/argus_prime_swe_localize/train_log.jsonl" 2>/dev/null && \
            RESUME="checkpoints/argus_prime_swe_localize/\$(ls -t checkpoints/argus_prime_swe_localize/step_*.pt 2>/dev/null | head -1 | xargs basename 2>/dev/null)" || true

        echo "Resuming from: $RESUME"

        run_remote "mkdir -p checkpoints/argus_prime_swe_repair && nohup python -m halo_training \
            --model $MODEL --class-name $CLASS \
            --phase sft --sft-format chatml \
            --sft-dataset datasets/swe_prepared/swe_code_repair.jsonl \
            --system-prompt 'You are an expert software engineer.' \
            --resume-from $RESUME \
            --checkpoint-dir checkpoints/argus_prime_swe_repair \
            --checkpoint-interval 5000 \
            --lr 2e-5 --epochs 2 --time-budget 240 \
            $COMMON \
            > checkpoints/argus_prime_swe_repair/stage_c.log 2>&1 &"
        echo "Launched! Monitor with:"
        echo "  bash run_remote_b.sh 'tail -5 checkpoints/argus_prime_swe_repair/train_log.jsonl'"
        ;;

    stage_d)
        echo "=== Stage D: Bug Explanation SFT ==="
        echo "Chain-of-thought reasoning about bugs"

        RESUME="$BASE_CKPT"
        run_remote "test -f checkpoints/argus_prime_swe_repair/train_log.jsonl" 2>/dev/null && \
            RESUME="checkpoints/argus_prime_swe_repair/\$(ls -t checkpoints/argus_prime_swe_repair/step_*.pt 2>/dev/null | head -1 | xargs basename 2>/dev/null)" || true

        echo "Resuming from: $RESUME"

        run_remote "mkdir -p checkpoints/argus_prime_swe_explain && nohup python -m halo_training \
            --model $MODEL --class-name $CLASS \
            --phase sft --sft-format chatml \
            --sft-dataset datasets/swe_prepared/swe_bug_explain.jsonl \
            --system-prompt 'You are an expert software engineer.' \
            --resume-from $RESUME \
            --checkpoint-dir checkpoints/argus_prime_swe_explain \
            --checkpoint-interval 5000 \
            --lr 1e-5 --epochs 2 --time-budget 300 \
            $COMMON \
            > checkpoints/argus_prime_swe_explain/stage_d.log 2>&1 &"
        echo "Launched! Monitor with:"
        echo "  bash run_remote_b.sh 'tail -5 checkpoints/argus_prime_swe_explain/train_log.jsonl'"
        ;;

    all)
        echo "=== Full SWE Pipeline ==="
        echo "WARNING: This runs all stages sequentially. Total ~6 hours."
        echo ""
        $0 prepare
        echo ""
        echo "Starting Stage B..."
        $0 stage_b
        echo ""
        echo "Stages C and D must be launched manually after B completes."
        echo "  bash scripts/run_swe_training.sh stage_c"
        echo "  bash scripts/run_swe_training.sh stage_d"
        ;;

    status)
        echo "=== SWE Training Status ==="
        for stage in swe_localize swe_repair swe_explain; do
            dir="checkpoints/argus_prime_$stage"
            echo ""
            echo "--- $stage ---"
            run_remote "if [ -f $dir/train_log.jsonl ]; then tail -3 $dir/train_log.jsonl; else echo 'Not started'; fi" 2>/dev/null || echo "  (unreachable)"
        done
        echo ""
        run_remote "ps aux | grep python | grep halo_training | grep -v grep" 2>/dev/null || echo "No training process running"
        ;;

    *)
        echo "Usage: bash scripts/run_swe_training.sh {prepare|stage_b|stage_c|stage_d|all|status}"
        echo ""
        echo "  prepare  - Preprocess SWE data and sync to Machine B"
        echo "  stage_b  - Bug Localization SFT"
        echo "  stage_c  - Code Repair SFT"
        echo "  stage_d  - Bug Explanation SFT"
        echo "  all      - Run prepare + start stage_b"
        echo "  status   - Check training progress on Machine B"
        ;;
esac

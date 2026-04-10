#!/bin/bash
# rocBLAS per-problem GEMM tuning for Strix Halo (gfx1151)
#
# This script profiles the GEMM shapes used by our models, runs rocblas-gemm-tune
# to find optimal Tensile kernels, and applies the tuned configuration.
#
# Usage:
#   bash scripts/tune_rocblas_gemm.sh [--profile|--tune|--apply|--benchmark]
#
# Steps:
#   1. --profile: Collect GEMM shapes from a smoke training run
#   2. --tune: Run rocblas-gemm-tune on collected shapes
#   3. --apply: Set ROCBLAS_TENSILE_GEMM_OVERRIDE_PATH and verify
#   4. --benchmark: Compare tuned vs baseline training throughput
#
# Prerequisites:
#   - ROCm 7.12 installed at /opt/rocm
#   - rocblas-gemm-tune available (part of rocBLAS package)
#   - venv activated with PyTorch + halo_training

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
TUNE_DIR="${PROJECT_DIR}/rocblas_tuning"

mkdir -p "$TUNE_DIR"

# Common GEMM shapes for our ~250M models (d=1024, ffn=2560, vocab=50257)
# Format: M N K (where C = A @ B, A is M×K, B is K×N)
GEMM_SHAPES=(
    # SwiGLU gate+up: (BT, 2*ffn, d) = (4096, 5120, 1024) — every layer
    "4096 5120 1024"
    # SwiGLU down: (BT, d, ffn) = (4096, 1024, 2560) — every layer
    "4096 1024 2560"
    # QKV/out_proj: (BT, d, d) = (4096, 1024, 1024) — every layer
    "4096 1024 1024"
    # LM head: (BT, vocab, d) = (4096, 50257, 1024) — once
    "4096 50257 1024"
    # GatedConv proj: (BT, 3*d_conv, d) = (4096, 1920, 1024) — every layer
    "4096 1920 1024"
)

profile_gemms() {
    echo "=== Step 1: Profile GEMM shapes ==="
    echo "Collecting GEMM shapes from smoke training run..."

    # Method 1: Use ROCBLAS_LAYER to capture all GEMM calls
    ROCBLAS_LAYER=4 \
    ROCBLAS_LOG_PROFILE_PATH="${TUNE_DIR}/rocblas_profile.yaml" \
    python -m halo_training \
        --model models/amadeus.py --class-name Amadeus \
        --smoke --max-steps 10 2>&1 | tee "${TUNE_DIR}/profile_log.txt"

    echo "Profile saved to ${TUNE_DIR}/rocblas_profile.yaml"

    # Method 2: Also write known shapes to a YAML file for manual tuning
    echo "# Known GEMM shapes for ~250M models" > "${TUNE_DIR}/gemm_shapes.yaml"
    echo "# Format: transA transB M N K alpha lda ldb beta ldc" >> "${TUNE_DIR}/gemm_shapes.yaml"
    for shape in "${GEMM_SHAPES[@]}"; do
        read -r M N K <<< "$shape"
        echo "N N $M $N $K 1.0 $K $N 0.0 $N" >> "${TUNE_DIR}/gemm_shapes.yaml"
    done
    echo "Known shapes written to ${TUNE_DIR}/gemm_shapes.yaml"
}

tune_gemms() {
    echo "=== Step 2: Tune GEMM kernels ==="

    if ! command -v rocblas-gemm-tune &>/dev/null; then
        echo "ERROR: rocblas-gemm-tune not found. Install from ROCm rocBLAS package."
        echo "Try: /opt/rocm/bin/rocblas-gemm-tune or install via apt."
        exit 1
    fi

    INPUT="${TUNE_DIR}/gemm_shapes.yaml"
    OUTPUT="${TUNE_DIR}/tuned_gemms.yaml"

    if [ -f "${TUNE_DIR}/rocblas_profile.yaml" ]; then
        INPUT="${TUNE_DIR}/rocblas_profile.yaml"
        echo "Using profiled shapes from training run"
    fi

    echo "Running rocblas-gemm-tune (this may take 10-30 minutes)..."
    rocblas-gemm-tune \
        --input "$INPUT" \
        --output "$OUTPUT" \
        --precision h \
        --device 0 \
        2>&1 | tee "${TUNE_DIR}/tune_log.txt"

    echo "Tuned configuration saved to $OUTPUT"
}

apply_tuning() {
    echo "=== Step 3: Apply tuned GEMM configuration ==="

    TUNED="${TUNE_DIR}/tuned_gemms.yaml"
    if [ ! -f "$TUNED" ]; then
        echo "ERROR: No tuned config found at $TUNED"
        echo "Run: bash scripts/tune_rocblas_gemm.sh --tune first"
        exit 1
    fi

    echo "To apply tuning, set this env var before training:"
    echo ""
    echo "  export ROCBLAS_TENSILE_GEMM_OVERRIDE_PATH=${TUNED}"
    echo ""
    echo "Add to .venv/bin/activate for persistent use."
}

benchmark() {
    echo "=== Step 4: Benchmark tuned vs baseline ==="

    TUNED="${TUNE_DIR}/tuned_gemms.yaml"

    echo "--- Baseline (no tuning) ---"
    python -m halo_training \
        --model models/amadeus.py --class-name Amadeus \
        --smoke --max-steps 50 2>&1 | grep -E "tok/s|step"

    if [ -f "$TUNED" ]; then
        echo ""
        echo "--- With rocBLAS tuning ---"
        ROCBLAS_TENSILE_GEMM_OVERRIDE_PATH="$TUNED" \
        python -m halo_training \
            --model models/amadeus.py --class-name Amadeus \
            --smoke --max-steps 50 2>&1 | grep -E "tok/s|step"
    fi

    echo ""
    echo "--- hipBLASLt env vars ---"
    echo "Testing Stream-K + hipBLASLt epilogue fusion..."
    TENSILE_SOLUTION_SELECTION_METHOD=2 \
    ROCBLAS_USE_HIPBLASLT=1 \
    python -m halo_training \
        --model models/amadeus.py --class-name Amadeus \
        --smoke --max-steps 50 2>&1 | grep -E "tok/s|step"
}

# Parse arguments
case "${1:-all}" in
    --profile)  profile_gemms ;;
    --tune)     tune_gemms ;;
    --apply)    apply_tuning ;;
    --benchmark) benchmark ;;
    all|--all)
        profile_gemms
        tune_gemms
        apply_tuning
        benchmark
        ;;
    *)
        echo "Usage: $0 [--profile|--tune|--apply|--benchmark|--all]"
        exit 1
        ;;
esac

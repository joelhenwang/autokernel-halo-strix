#!/bin/bash
# Run bench.py for a given kernel type inside the autokernel container.
# Usage: run_kernel_bench.sh <kernel_type> [--quick]
set -e

KERNEL_TYPE=$1
QUICK_FLAG=$2

cd /workspace/autokernel

echo "=== Benchmarking kernel: $KERNEL_TYPE ==="
cp "kernels/${KERNEL_TYPE}.py" kernel.py
echo "Copied kernels/${KERNEL_TYPE}.py -> kernel.py"

if [ "$QUICK_FLAG" = "--quick" ]; then
    echo "Running quick (smoke) test..."
    python bench.py --kernel "$KERNEL_TYPE" --quick 2>&1
else
    echo "Running full 5-stage benchmark..."
    python bench.py --kernel "$KERNEL_TYPE" 2>&1
fi
echo "=== Done: $KERNEL_TYPE ==="

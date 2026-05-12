"""
AutoKernel -- Optimized cross_entropy kernel (facade).

Implementation lives in kernels/ce_fused.py. This file re-exports the public
API at the repo root for backward compatibility with bench.py (IMMUTABLE) and
all scripts that do ``import kernel``.
"""
# Re-export everything from the implementation module.
from kernels.ce_fused import *  # noqa: F401,F403

# Explicit re-exports for IDE support and documentation.
from kernels.ce_fused import (  # noqa: F811
    KERNEL_TYPE,
    BACKEND,
    MODEL_SHAPES,
    TEST_SIZES,
    TOLERANCES,
    FLOPS_FN,
    BYTES_FN,
    kernel_fn,
    ce_full,
    _get_fwd_module,
    _get_bwd_module,
    _CrossEntropyHIP,
    HIP_FWD_SRC,
    HIP_BWD_SRC,
)

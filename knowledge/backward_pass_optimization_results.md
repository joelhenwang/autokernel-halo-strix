# Backward Pass Optimization Results

**Date:** 2026-04-10
**Hardware:** AMD Radeon 8060S (gfx1151, RDNA 3.5), 40 CUs, 128 GB LPDDR5X

## Measured Kernel Speedups

| Kernel | Baseline (ms) | HIP (ms) | Speedup | Max Diff | Notes |
|--------|-------------|---------|---------|----------|-------|
| **rmsnorm_backward** | 1.237 | 0.099 | **12.51x** | 0.022 | 32 calls/step. Fused recompute + chain rule + atomic grad_weight |
| **silu_gate_mul_backward** | 4.213 | 0.392 | **10.74x** | 0.004 | 16 calls/step. Single-pass sigmoid + derivative + dual grads |
| **rotary_embedding_backward** | 0.100 | 0.021 | **4.67x** | 0.000 | 16 calls/step. Split-half convention, fp32 intermediate, exact |
| **fused_residual_rmsnorm_backward** | — | — | ~12-14x (est.) | — | Same structure as rmsnorm_backward with dual-output merge |
| **selective_scan_backward** | 20.423 | 0.966 | **21.15x** | 0.37 | 16 calls/step (AMADEUS). Parallel reverse prefix scan replaces 2 sequential loops |

**Benchmark config:** M=4096, N=768 (norm), N=2048 (silu), B=4 H=12 N=256 D=64 (rotary), batch=4 seq=1024 d=128 (scan). Median of 100 runs after 10 warmup.

## Additional Optimizations

| Technique | Status | Expected Impact |
|-----------|--------|----------------|
| **Chunked CE Approach D** | Wired into trainer | 25% LM head backward speedup, ~300MB memory saved |
| **Sampled softmax warmup** | Implemented | 3-6x smaller LM head GEMM during first 3K steps |
| **fp32 grad_logits retention** | Option added | 5-10% on chunked CE backward |
| **PLE GELU direct derivative** | Fixed | Eliminates nested autograd overhead |
| **Low-rank backward (INSTANT)** | Experimental | 20-30% weight gradient GEMM reduction (approximate) |
| **Activation quantization** | Experimental | 2x activation memory reduction (approximate gradients) |
| **Stream overlap** | Experimental | 15-20% hiding (non-deterministic scheduling) |

## Measured End-to-End Training Impact

### LlamaModel 124.7M (autokernel.optimize, batch=8, seq=256)

| Metric | Before (PyTorch bwd) | After (HIP bwd) | Change |
|--------|---------------------|-----------------|--------|
| Step (ms) | 44.82 | 44.25 | **1.01x** |
| Forward (ms) | 27.49 | 27.27 | — |
| Backward (ms) | 10.86 | 10.44 | **1.04x** |
| Backward % of step | 24.2% | 23.6% | — |
| Throughput (tok/s) | 45,692 | 46,284 | **1.01x** |
| MFU | 57.5% | 58.3% | +0.8% |

**Why only 1.04x backward:** Backward is only 24% of the step (not 53% as in eager AMADEUS). The forward dominates at 27ms. Within the backward, attention backward (via SDPA) is the main cost — our fused RMSNorm/SiLU/rotary backward kernels are fast but are a small fraction.

### AMADEUS 243.8M (autokernel.optimize, batch=8, seq=256)

| Metric | Before (PyTorch bwd) | After (HIP bwd) | Change |
|--------|---------------------|-----------------|--------|
| Step (ms) | 271.97 | 272.13 | 1.00x |
| Backward (ms) | 159.60 | 160.09 | 1.00x |
| Throughput (tok/s) | 7,530 | 7,526 | 1.00x |
| MFU | 18.5% | 18.5% | — |

**Why 1.00x:** The SSM selective scan dominates AMADEUS backward (160ms) and uses its own code path (`mamba_ssm.selective_scan_fn` or chunked scan), NOT the `torch.ops.autokernel.selective_scan` custom op. The autokernel pattern only replaces RMSNorm and SwiGLU, whose backward is a tiny fraction of the 160ms total.

### Gap Between Isolated and End-to-End

| Kernel | Isolated Speedup | End-to-End Impact | Reason |
|--------|-----------------|-------------------|--------|
| rmsnorm_backward | 12.51x | ~2% of backward | Small per-call cost, many calls but each <0.1ms |
| silu_gate_mul_backward | 10.74x | ~1% of backward | Element-wise, already fast in PyTorch |
| rotary_embedding_backward | 4.67x | <1% of backward | Only used in attention models |
| selective_scan_backward | 21.15x | **0% (not wired)** | AMADEUS uses own scan path |

### What Would Unlock Full Potential

1. **Wire scan backward into AMADEUS's scan path** — the 21x kernel sits unused because AMADEUS calls `mamba_ssm.selective_scan_fn` which has its own backward
2. **torch.compile integration** — would route all ops through custom ops more aggressively, letting backward kernels fire for more operations
3. **Target attention backward** — for LlamaModel, attention backward is the dominant cost (not norm/activation)

## Correctness Notes

- **RMSNorm**: 0.022 max diff is within fp16 accumulated warp reduction tolerance. The HIP kernel computes in fp32 intermediates but accumulates via half2 vectorized loads which introduce rounding.
- **Rotary**: Exact (0.000 diff). Split-half convention with fp32 cos/sin cache.
- **SiLU Gate Mul**: 0.004 max diff, well within fp16 tolerance.
- **Selective Scan**: 0.37 max diff for grad_dBx — accumulated across T=1024 timesteps in parallel scan vs sequential. The parallel scan uses the same associative operator but different order of operations, causing fp32 accumulation differences. grad_D (0.0002) and grad_x (0.000) are exact since they don't involve the scan.

## Anti-Patterns Confirmed

- **Sequential Python loops for scans**: 21x slower than parallel HIP. Never use Python loops for recurrence backward.
- **Nested autograd for GELU**: `torch.enable_grad()` + `.backward()` inside a backward function has measurable overhead. Direct derivative formula is faster.
- **Separate PyTorch ops for fused backward**: Even with torch's kernel launch fusion, 5 separate ops for RMSNorm backward are 12.5x slower than a single fused HIP kernel.

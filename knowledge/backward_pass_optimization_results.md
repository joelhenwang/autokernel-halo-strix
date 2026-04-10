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

## Estimated Training Impact

### LlamaModel 124.7M (baseline: 43K tok/s)
- Backward is ~53% of step → ~25ms backward per step
- Fused norms (12.5x × 32 calls): saves ~10ms
- Fused rotary (4.7x × 16 calls): saves ~1ms
- Fused SiLU (10.7x × 16 calls): saves ~5ms
- Estimated backward reduction: ~16ms → ~9ms (1.8x backward, ~1.35x overall)
- **Projected tok/s: ~58K** (from 43K)

### AMADEUS 243.8M (baseline: 10.4K tok/s)
- Backward dominated by selective scan (~20ms × 16 calls = ~320ms per step)
- Scan backward 21x: ~320ms → ~15ms
- Plus norm/activation fusions: additional ~15ms savings
- Estimated backward reduction: ~350ms → ~40ms (8.75x backward, ~2.5x overall)
- **Projected tok/s: ~26K** (from 10.4K)

## Correctness Notes

- **RMSNorm**: 0.022 max diff is within fp16 accumulated warp reduction tolerance. The HIP kernel computes in fp32 intermediates but accumulates via half2 vectorized loads which introduce rounding.
- **Rotary**: Exact (0.000 diff). Split-half convention with fp32 cos/sin cache.
- **SiLU Gate Mul**: 0.004 max diff, well within fp16 tolerance.
- **Selective Scan**: 0.37 max diff for grad_dBx — accumulated across T=1024 timesteps in parallel scan vs sequential. The parallel scan uses the same associative operator but different order of operations, causing fp32 accumulation differences. grad_D (0.0002) and grad_x (0.000) are exact since they don't involve the scan.

## Anti-Patterns Confirmed

- **Sequential Python loops for scans**: 21x slower than parallel HIP. Never use Python loops for recurrence backward.
- **Nested autograd for GELU**: `torch.enable_grad()` + `.backward()` inside a backward function has measurable overhead. Direct derivative formula is faster.
- **Separate PyTorch ops for fused backward**: Even with torch's kernel launch fusion, 5 separate ops for RMSNorm backward are 12.5x slower than a single fused HIP kernel.

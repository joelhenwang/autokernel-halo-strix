# Phase I: Triton fused SwiGLU Ship-Gate Bench (2026-05-11)

Production shape: (16, 512, 2048), dtype: torch.float16.

Three bench pairs (each 200 timed iters + 50 warmup):

| Comparison | fwd speedup | fwd+bwd speedup |
|---|---:|---:|
| eager-vs-autograd-hip | 1.693x | 1.450x |
| eager-vs-triton | 1.634x | 1.434x |
| autograd-hip-vs-triton | 0.977x | 0.986x |

## Ship-gate decision

**FAIL** — Triton fused_swiglu fwd+bwd speedup **0.986x** over autograd-safe HIP. Below 1.05x threshold.

**Action**: skip Phase H. Sprint 3A ships without `--optimize-kernels`. Triton path has no clear throughput advantage on OdinFlat's SwiGLU block.

## Per-bench detail

### eager-vs-autograd-hip

- ref_fwd_us: 731.4
- triton_fwd_us: 432.0
- speedup_fwd: 1.693x
- ref_fwd_bwd_us: 2730.6
- triton_fwd_bwd_us: 1882.9
- speedup_fwd_bwd: 1.450x

### eager-vs-triton

- ref_fwd_us: 732.9
- triton_fwd_us: 448.4
- speedup_fwd: 1.634x
- ref_fwd_bwd_us: 2731.0
- triton_fwd_bwd_us: 1905.1
- speedup_fwd_bwd: 1.434x

### autograd-hip-vs-triton

- ref_fwd_us: 437.4
- triton_fwd_us: 447.8
- speedup_fwd: 0.977x
- ref_fwd_bwd_us: 1880.8
- triton_fwd_bwd_us: 1906.6
- speedup_fwd_bwd: 0.986x


# Option 3: F.rms_norm native swap + one-more-probe (2026-05-10)

## Swap applied

`models/_components.py::RMSNorm.forward` replaced manual
`x * rsqrt(mean(x^2)+eps) * weight` with `F.rms_norm(x, shape, weight, eps)`.

This is the PyTorch native functional RMSNorm, which Inductor recognizes
as a single op and can attempt to fuse with neighbors under compile_zones.

## Probes (200 steps, C3 recipe)

| Variant | rmsnorm path | silu path | Steady tok/s | Loss @ 200 | vs baseline |
|---|---|---|---:|---:|---:|
| P0 baseline (manual rmsnorm) | manual | raw | 31,331 | 4.7013 | 0% |
| **V0 native no-HIP** | **F.rms_norm** | raw | **31,736** | 4.7269 | +1.3% |
| **V1 native + silu HIP** | **F.rms_norm** | HIP | **41,198** | **4.6691** | **+31.5%** |
| Phase III reference | HIP (fixed) | HIP | 40,962 | 4.8184 | +30.7% |
| V3 cudagraphs | F.rms_norm | HIP | **CRASH** | — | — |

## Findings

### 1. F.rms_norm ≈ manual rsqrt under Inductor (V0 vs P0)

Switching from manual `x.pow(2).mean()...` to `F.rms_norm` is a **1.3%
change within noise**. Inductor fuses the manual code pattern equivalently
to the native op. **The rmsnorm kernel is NOT the bottleneck, whether
expressed as manual ops or native op.**

### 2. silu HIP is the entire +31% lift (V1 vs V0)

Going from V0 (both native) to V1 (rmsnorm native + silu HIP) gains +30%
throughput. All of the autokernel win comes from the silu_gate_mul HIP
kernel. The rmsnorm HIP kernel path is essentially dead weight.

### 3. V1 is the cleanest production config

- Uses `F.rms_norm` (native, maintained, well-tested)
- Uses silu_gate_mul HIP only (+30% from +28% Phase II P2 matches)
- NO autograd plumbing bugs (rmsnorm never goes through autokernel)
- Slightly BETTER loss than Phase III fixed path (4.67 vs 4.82 at step 200)

### 4. cudagraphs still broken

`TORCH_COMPILE_MODE=max-autotune` with `accum_steps=8` crashes with
"accessing tensor output of CUDAGraphs that has been overwritten".
Known issue per AGENTS.md.

## Recommended config for Sprint 3A

```bash
# Use NATIVE F.rms_norm (already committed via models/_components.py change)
# + autokernel with rmsnorm PATTERN EXCLUDED
EXTRA_FLAGS='... --optimize-kernels --autokernel-exclude rmsnorm'
TORCH_COMPILE_MODE=max-autotune-no-cudagraphs
```

Equivalent throughput to Phase III (+31%, wall ~47h) but simpler code
path and marginally better loss numerics.

## Ceiling assessment

**+31% is the empirical ceiling.** Three independent paths reach it:
- Phase III: HIP rmsnorm (autograd-fixed) + HIP silu
- Phase IV C2: same + AUTOKERNEL_NO_BWD_HIP=1
- Option 3 V1: F.rms_norm + HIP silu

All converge to ~41K tok/s steady. The silu_gate_mul HIP kernel provides
the entire lift. Everything else is marginal.

**A Triton RMSNorm rewrite is unlikely to help.** V0 shows Inductor
already fuses rmsnorm ops at ~baseline throughput. A custom Triton
kernel might be marginally faster in isolation but Inductor won't see
through it for neighboring-op fusion (it's an opaque custom op).
Estimated upside: 0-10% (speculative).

## Artifacts

```
models/_components.py                       swapped to F.rms_norm
scripts/probe_option3_native_rmsnorm.sh     V0, V1 probes
scripts/probe_option3_v3_cudagraphs.sh      V3 probe (crashed)
docs/perf/phase5-option3-logs.tgz           rank0.log + activation_stats
```

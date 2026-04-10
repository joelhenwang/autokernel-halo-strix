# Aiter (AMD AI Tensor Engine Runtime) on ROCm 7.12 / gfx1151

This documents the procedure to install and patch `aiter` for flash_attn backward support on Strix Halo.

- Python `3.12`
- PyTorch `2.10.0+rocm7.12.0`
- ROCm `7.12`
- `gfx1151` / Halo Strix
- `flash_attn 2.8.4`

## Overview

`flash_attn` 2.8.4 on ROCm uses `aiter` for its Triton-based attention kernels. Without aiter, flash_attn's backward pass fails with "CK backward is not supported on gfx11." With aiter installed, both forward and backward work via Triton — but the CK (Composable Kernel) JIT also needs patching for full compatibility.

## 1. Install aiter

aiter is installed as a local editable package from source:

```bash
cd ~/Desktop/ai_lab/autokernel-halo-strix
git clone https://github.com/ROCm/aiter.git
cd aiter
pip install -e .
```

## 2. Symlink hipcc

aiter's JIT build hardcodes `/opt/rocm/bin/hipcc`. On ROCm 7.12, hipcc lives at `/opt/rocm/core-7.12/bin/hipcc`:

```bash
sudo ln -sf /opt/rocm/core-7.12/bin/hipcc /opt/rocm/bin/hipcc
```

## 3. Patch CK headers for gfx1151

aiter bundles Composable Kernel (CK) headers that use bare C math functions (`exp2f`, `expf`, `powf`, `expm1f`, `__logf`, `sincosf`, `log1pf`) in device code. ROCm 7.12's HIP compiler rejects these as host-only on gfx1151. The fix is identical to the causal-conv1d and mamba-ssm patches: replace with `__builtin_` equivalents.

```bash
cd ~/Desktop/ai_lab/autokernel-halo-strix
bash scripts/patch_aiter_ck_rocm.sh aiter
```

This patches ~24 files across `aiter/3rdparty/composable_kernel/include/` and `aiter/csrc/` and clears the JIT build cache.

**What the script does:**
- Replaces `exp2f(` → `__builtin_exp2f(`, `expf(` → `__builtin_expf(`, etc.
- Preserves `__builtin_amdgcn_exp2f` and other already-prefixed calls via regex negative lookbehind
- Handles both `std::expf(` and bare `expf(` forms

## 4. Activate environment variables

The venv activate script must export ROCm paths. Add to `.venv/bin/activate`:

```bash
export ROCM_HOME=/opt/rocm
export ROCM_PATH=/opt/rocm/core-7.12
export HIP_PATH=/opt/rocm/core-7.12
export HIP_PLATFORM=amd
export FLASH_ATTENTION_TRITON_AMD_ENABLE=TRUE
export CPLUS_INCLUDE_PATH=/opt/rocm/core-7.12/include:${CPLUS_INCLUDE_PATH:-}
export LD_LIBRARY_PATH=/opt/rocm/core-7.12/lib:/opt/rocm/core-7.12/lib64:${LD_LIBRARY_PATH:-}
export LIBRARY_PATH=/opt/rocm/core-7.12/lib:/opt/rocm/core-7.12/lib64:${LIBRARY_PATH:-}
```

**Critical:** `FLASH_ATTENTION_TRITON_AMD_ENABLE=TRUE` routes flash_attn through aiter's Triton path (which supports backward). Setting it to `FALSE` uses the AOTriton HIP path (forward-only on gfx11, backward fails with "CK backward is not supported on gfx11").

## 5. Verification

```bash
source .venv/bin/activate
python scripts/bench_external_kernels.py --test attention
```

Expected output:
```
[aiter] finish build [module_aiter_core], cost 7.8s
[aiter] import [module_aiter_core]
[PASS] flash_attn vs SDPA: max_diff=0.000977
Backward: OK
flash_attn forward: 0.25ms
SDPA forward: 1.06ms
flash_attn fwd+bwd: 4.81ms
SDPA fwd+bwd: 4.14ms
```

## 6. Performance findings

| Path | Forward | Backward | Fwd+Bwd | vs SDPA |
|------|---------|----------|---------|---------|
| **hybrid_attention** (flash fwd + SDPA bwd) | **0.25ms** | **2.92ms** | **3.50ms** | **8.9% faster** |
| SDPA (PyTorch built-in) | 1.07ms | 2.64ms | 3.84ms | baseline |
| flash_attn (aiter Triton) | 0.25ms | 4.39ms | 4.84ms | 26% slower |
| flash_attn (AOTriton, no aiter) | **0.17ms** | N/A | N/A | fwd-only, no bwd on gfx11 |

**Recommendation:**
- **Training:** Use `hybrid_flash_sdpa_attention` from `kernels/hip/hybrid_attention.py` — combines flash_attn's fast forward with SDPA's fast backward by passing softmax logsumexp directly to `torch.ops.aten._flash_attention_backward`. **8.9% faster than pure SDPA.**
- **Inference/decode:** Use flash_attn with `FLASH_ATTENTION_TRITON_AMD_ENABLE=FALSE` for fastest forward (0.17ms via AOTriton)
- **Avoid for training:** Pure flash_attn — its Triton backward is 66% slower than SDPA on gfx11

### Why the hybrid works

flash_attn and SDPA both compute a `softmax_lse` (log-sum-exp) tensor of shape `(B, H, T)` in float32 during the forward pass. The backward uses this logsumexp to recompute the attention probability matrix without storing the full N×N attention matrix. By passing flash_attn's logsumexp directly to PyTorch's aten-level `_flash_attention_backward`, we skip the forward recompute entirely. Gradient accuracy: max_diff=0.002 (within fp16 tolerance).

## 7. Triton backward tuning results (exhaustive)

We tested every viable optimization path for the flash_attn backward on gfx1151:

| Approach | Fwd+Bwd | vs SDPA | Verdict |
|----------|---------|---------|---------|
| **hybrid_v2** (flash fwd + SDPA aten bwd) | **3.51ms** | **8.6% faster** | Best for training |
| SDPA | 3.84ms | baseline | Simple fallback |
| flash_attn (Triton, autotuned 4 configs) | 4.81ms | 25% slower | 32x32 confirmed optimal for RDNA |
| hybrid_v1 (SDPA recompute bwd) | 4.52ms | 18% slower | Recompute cost too high |
| flash_attn (split mode) | 15.39ms | 4x slower | 3 kernel launches catastrophic |

**Triton autotune confirmed:** The original 32x32x32x32 RDNA config is already optimal. Larger tiles (64x64, 32x128, 128x64) were tested via `scripts/patch_aiter_bwd_rdna.py` and all performed worse — RDNA 3.5 without MFMA can't exploit larger tiles effectively. The `exp2` trick is already in use (3 calls, 0 bare exp calls). Split backward mode launches 3 separate kernels with 3x overhead.

## 8. Known issues

- **CK JIT still reports CK ops disabled at runtime.** The `module_aiter_core` builds successfully but CK attention backward is fundamentally not supported on gfx11 architecture. aiter correctly falls back to Triton for attention ops. The CK JIT build enables other aiter features (fused RMSNorm, quantization, RoPE, MoE gating).

- **`-mllvm -amdgpu-coerce-illegal-types=1` skipped.** aiter reports this flag is not supported by the current hipcc. This is informational — the build proceeds without it.

- **First import is slow (~8s).** The CK JIT compiles on first use. Subsequent imports use the cached `.so` at `aiter/aiter/jit/module_aiter_core.so`.

## 9. Files

- `scripts/patch_aiter_ck_rocm.sh` — Automated CK header patch for ROCm 7.12 math builtins
- `scripts/patch_aiter_bwd_rdna.py` — RDNA backward config tuning (edit bwd.py configs + BWD_MODE)
- `kernels/hip/hybrid_attention.py` — Hybrid flash fwd + SDPA bwd (best training backend)
- `scripts/bench_attention_backward.py` — Isolated fwd/bwd/fwd+bwd benchmark for all backends
- `aiter/` — aiter source (git repo, CK patches applied in-place)
- `aiter/aiter/jit/build/module_aiter_core/` — JIT build cache (auto-generated)

---
title: "Install mamba-ssm on ROCm"
domain: operations
type: guide
status: active
related:
  - knowledge/kernels/external_kernels.md
tags: [%install, %mamba-ssm, %rocm]
---

# Mamba SSM ROCm Install Notes

This documents the local-source procedure used in this workspace to patch and install `mamba-ssm` on:

- Python `3.12`
- PyTorch `2.10.0+rocm7.12.0`
- ROCm `7.12`
- `gfx1151` / Halo Strix

The key failure was during HIP compilation of the selective-scan extension. The upstream build fell back from a missing prebuilt wheel and then failed on float math calls such as `expf`, `exp2f`, `log1pf`, and `sincosf` being resolved as host-only libc entrypoints inside device code. The local fix is to build from a patched checkout and replace those calls with Clang builtins.

## 1. Activate the project venv

From the repo root:

```bash
source .venv/bin/activate
```

In this environment, `.venv/bin/activate` already exports the ROCm/HIP variables used by the working local builds in this repo.

## 2. Use the helper installer

The reproducible installer is:

[scripts/install_mamba_ssm_rocm.sh](/home/joelwang-ai-1/Desktop/comfyui-rocm7.12/scripts/install_mamba_ssm_rocm.sh)

Run:

```bash
./scripts/install_mamba_ssm_rocm.sh
```

What it does:

1. Clones `state-spaces/mamba`
2. Checks out `v2.3.0`
3. Patches these files in the local checkout:
   - `csrc/selective_scan/selective_scan_common.h`
   - `csrc/selective_scan/selective_scan_fwd_kernel.cuh`
   - `csrc/selective_scan/selective_scan_bwd_kernel.cuh`
   - `csrc/selective_scan/reverse_scan.cuh`
4. Adds `#include <cmath>` to the common selective-scan header
5. Rewrites the device math calls:
   - `expf(` -> `__builtin_expf(`
   - `exp2f(` -> `__builtin_exp2f(`
   - softplus `log1pf(expf(x))` -> `__builtin_logf(1.0f + __builtin_expf(x))`
   - `sincosf(` -> `__builtin_sincosf(`
6. Pins the ROCm reverse-scan header to `#define WARP_THREADS 32` so the hipified build does not depend on a non-constexpr `HIPCUB_WARP_THREADS` definition under ROCm `7.12`
7. Forces a local source build and installs from the patched tree into `./.venv`

## 3. Important build detail

The build must be forced to compile locally instead of probing for a prebuilt wheel.

The successful workflow uses:

```bash
MAMBA_FORCE_BUILD=TRUE
```

Without that, `setup.py` first guesses a GitHub release wheel URL for your exact Python, torch, ABI, and ROCm matrix. In this environment that URL path falls through and the install proceeds to a source build anyway.

## 4. Manual procedure

If you do not want to use the helper script, this is the same workflow manually:

```bash
source .venv/bin/activate
mkdir -p external
git clone --branch v2.3.0 --depth 1 https://github.com/state-spaces/mamba.git external/mamba
cd external/mamba
```

Then patch the three selective-scan source files so that the float-valued device math calls use the builtin forms listed above, and install with:

```bash
MAMBA_FORCE_BUILD=TRUE ../../.venv/bin/python -m pip install --no-build-isolation --no-cache-dir -v .
```

## 5. Verification

After install, verify with:

```bash
./.venv/bin/python -c "import mamba_ssm; print(mamba_ssm.__file__)"
```

If the selective-scan extension imports cleanly, the `mamba-ssm` package should load successfully from the workspace venv.

## 6. Files created during this fix

- [scripts/install_mamba_ssm_rocm.sh](/home/joelwang-ai-1/Desktop/comfyui-rocm7.12/scripts/install_mamba_ssm_rocm.sh)
- [external/mamba](/home/joelwang-ai-1/Desktop/comfyui-rocm7.12/external/mamba)
- [INSTALL_MAMBA_SSM.md](/home/joelwang-ai-1/Desktop/comfyui-rocm7.12/INSTALL_MAMBA_SSM.md)

# Halo Strix ROCm Install Notes

This documents the exact procedure that worked in this workspace to install `causal-conv1d` on:

- Python `3.12`
- PyTorch `2.10.0+rocm7.12.0`
- ROCm `7.12`
- `gfx1151` / Halo Strix

The key issue was that the upstream source build failed during HIP compilation on `expf(...)` inside device code. The working fix was to build from a local patched checkout and replace those specific float-valued SiLU exponential calls with `__builtin_expf(...)`.

## 1. Activate the project venv

From the repo root:

```bash
source .venv/bin/activate
```

In this environment, `.venv/bin/activate` already exports the ROCm/HIP variables that were used for the successful build:

```bash
export TRITON_HIP_LLD_PATH=/opt/rocm/core-7.12/lib/llvm/bin/ld.lld
export FLASH_ATTENTION_TRITON_AMD_ENABLE=TRUE
export ROCM_HOME=/opt/rocm
export ROCM_PATH=/opt/rocm/core-7.12/
export HIP_PATH=/opt/rocm
export CPLUS_INCLUDE_PATH=/opt/rocm/core-7.12/include:$CPLUS_INCLUDE_PATH
export C_INCLUDE_PATH=/opt/rocm/core-7.12/include:$C_INCLUDE_PATH
export PATH=/opt/rocm/bin:$PATH
export LD_LIBRARY_PATH=/opt/rocm/core-7.12/lib:/opt/rocm/lib:/opt/rocm/lib64:$LD_LIBRARY_PATH
export LIBRARY_PATH=/opt/rocm/core-7.12/lib:/opt/rocm/lib:/opt/rocm/lib64:$LIBRARY_PATH
export CMAKE_PREFIX_PATH=/opt/rocm:$CMAKE_PREFIX_PATH
export HIP_PLATFORM=amd
export HIPCC_VERBOSE=1
export TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
```

These variables matter. The successful build in this repo used them exactly as provided by the activate script.

## 2. Use the helper installer

The reproducible installer is:

[scripts/install_causal_conv1d_rocm.sh](/home/joelwang-ai-1/Desktop/comfyui-rocm7.12/scripts/install_causal_conv1d_rocm.sh)

Run:

```bash
./scripts/install_causal_conv1d_rocm.sh
```

What it does:

1. Clones `Dao-AILab/causal-conv1d`
2. Checks out `v1.6.1`
3. Patches these files:
   - `csrc/causal_conv1d_fwd.cu`
   - `csrc/causal_conv1d_bwd.cu`
   - `csrc/causal_conv1d_update.cu`
4. Forces the source to include:
   - `#include <hip/hip_runtime.h>`
   - `#include <cmath>`
5. Rewrites the float SiLU exponential calls in those files:
   - `expf(` -> `__builtin_expf(`
   - `exp(` -> `__builtin_expf(`
6. Installs from the local source tree into `./.venv`

## 3. Important build detail

The build must be forced to compile locally instead of probing for a prebuilt wheel.

The successful install used:

```bash
CAUSAL_CONV1D_FORCE_BUILD=TRUE
```

Without that, `setup.py` first tries a GitHub release wheel URL for your exact matrix, and that path is not reliable here.

## 4. Manual procedure

If you do not want to use the helper script, this is the same workflow manually:

```bash
source .venv/bin/activate
mkdir -p external
git clone --branch v1.6.1 --depth 1 https://github.com/Dao-AILab/causal-conv1d.git external/causal-conv1d
cd external/causal-conv1d
```

Then patch the three source files so that:

- `#include <hip/hip_runtime.h>` is the first include
- `#include <cmath>` is also present
- every float-valued `expf(` and `exp(` in the SiLU paths becomes `__builtin_expf(`

Then install:

```bash
CAUSAL_CONV1D_FORCE_BUILD=TRUE ../../.venv/bin/python -m pip install --no-build-isolation --no-cache-dir -v .
```

## 5. Verification

The install was verified with:

```bash
./.venv/bin/python -c "import causal_conv1d, causal_conv1d_cuda; print(causal_conv1d.__file__)"
```

That succeeded and imported both:

- `causal_conv1d`
- `causal_conv1d_cuda`

## 6. Files created during this fix

- [scripts/install_causal_conv1d_rocm.sh](/home/joelwang-ai-1/Desktop/comfyui-rocm7.12/scripts/install_causal_conv1d_rocm.sh)
- [external/causal-conv1d](/home/joelwang-ai-1/Desktop/comfyui-rocm7.12/external/causal-conv1d)
- [causal_conv1d_build.log](/home/joelwang-ai-1/Desktop/comfyui-rocm7.12/causal_conv1d_build.log)

## 7. Result

`causal-conv1d 1.6.1` is installed successfully in:

`/home/joelwang-ai-1/Desktop/comfyui-rocm7.12/.venv`

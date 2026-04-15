---
title: "Building RCCL from Source for gfx1151 (Strix Halo)"
domain: training
type: guide
status: active
related:
  - knowledge/training/ddp_setup_guide.md
tags: [%rccl, %nccl, %gfx1151, %build-from-source]
---

# Building RCCL from Source for gfx1151 (Strix Halo)

## Why

The bundled RCCL in the pip PyTorch wheel (`_rocm_sdk_libraries_gfx1151/lib/librccl.so.1`) and the system RCCL at `/opt/rocm/core-7.12/lib/` both fail with `HIP failure: 'invalid kernel file'` when trying to run allreduce GPU kernels on gfx1151. The RCCL init succeeds (TCP sockets work) but the actual collective kernels aren't compiled for our GPU.

Building from source with explicit `GPU_TARGETS=gfx1151` fixes this.

## Prerequisites

- ROCm 7.12 installed at `/opt/rocm/core-7.12/`
- hipcc available
- cmake >= 3.16
- Build tools: `sudo apt install build-essential cmake git`

## Steps (run on EACH machine)

### 1. Set ROCm environment

```bash
export ROCM_PATH=/opt/rocm/core-7.12
export HIP_PATH=/opt/rocm/core-7.12
export HIP_PLATFORM=amd
export CPLUS_INCLUDE_PATH=/opt/rocm/core-7.12/include:${CPLUS_INCLUDE_PATH:-}
export LD_LIBRARY_PATH=/opt/rocm/core-7.12/lib:/opt/rocm/core-7.12/lib64:${LD_LIBRARY_PATH:-}
export LIBRARY_PATH=/opt/rocm/core-7.12/lib:/opt/rocm/core-7.12/lib64:${LIBRARY_PATH:-}
export HSA_OVERRIDE_GFX_VERSION=11.5.1
```

Verify hipcc:
```bash
/opt/rocm/core-7.12/bin/hipcc --version
# If hipcc is not at /opt/rocm/bin/hipcc:
sudo ln -sf /opt/rocm/core-7.12/bin/hipcc /opt/rocm/bin/hipcc
```

### 2. Clone RCCL

```bash
cd ~/Desktop
git clone https://github.com/ROCm/rccl.git
cd rccl
git checkout develop    # or: git checkout therock-7.11
git submodule update --init --recursive
```

### 3. Check if math builtin patches are needed

```bash
grep -rn "expf\|exp2f\|powf\|__logf" src/ --include="*.cpp" --include="*.cu" --include="*.h" \
  | grep -v "__builtin_" | grep -v "amdgcn_" | grep -v "//.*expf" | head -20
```

If matches found, apply the patch:

```bash
find src/ -name "*.cu" -o -name "*.cpp" -o -name "*.h" | while read f; do
    sed -i 's/\bstd::expf\b/__builtin_expf/g' "$f"
    sed -i 's/\bstd::exp2f\b/__builtin_exp2f/g' "$f"
    sed -i 's/\bstd::powf\b/__builtin_powf/g' "$f"
    # Be careful not to double-patch __builtin_ prefixed calls
done
```

### 4. Build

```bash
mkdir -p build && cd build

cmake -DCMAKE_BUILD_TYPE=Release \
      -DCMAKE_INSTALL_PREFIX=$HOME/rccl-gfx1151 \
      -DGPU_TARGETS="gfx1151" \
      -DCMAKE_CXX_COMPILER=/opt/rocm/core-7.12/bin/hipcc \
      -DCMAKE_C_COMPILER=/opt/rocm/core-7.12/bin/hipcc \
      -DROCM_PATH=/opt/rocm/core-7.12 \
      ..

# Build with all cores (takes 10-30 minutes)
make -j$(nproc)
make install
```

**Alternatively**, use RCCL's install.sh:
```bash
cd ~/Desktop/rccl
./install.sh --amdgpu_targets "gfx1151" \
             --prefix $HOME/rccl-gfx1151
```

### 5. Verify the build

```bash
# Check that librccl was built
ls -la $HOME/rccl-gfx1151/lib/librccl*

# Check gfx1151 code objects are present
strings $HOME/rccl-gfx1151/lib/librccl.so | grep gfx1151

# Quick smoke test
LD_PRELOAD=$HOME/rccl-gfx1151/lib/librccl.so \
python3 -c "import torch; torch.distributed.init_process_group('nccl', init_method='tcp://127.0.0.1:29500', rank=0, world_size=1); print('NCCL/RCCL init OK'); torch.distributed.destroy_process_group()"
```

### 6. Update ddp_env.sh

Add to `scripts/ddp_env.sh` on BOTH machines:

```bash
# Force the custom-built RCCL with gfx1151 kernels
export LD_PRELOAD=$HOME/rccl-gfx1151/lib/librccl.so
```

This overrides the bundled RCCL in the pip wheel.

### 7. Test DDP with NCCL backend

```bash
# Machine 0:
source scripts/ddp_env.sh
torchrun --nproc_per_node=1 --nnodes=2 --node_rank=0 \
    --master_addr=10.77.0.1 --master_port=29500 \
    scripts/train_ddp.py --backend nccl \
    --model models/argus_prime.py --class-name ArgusPrime \
    --dataset datasets/common_crawl_sample.bin \
    --epochs 1 --compile --optimize-kernels --lr 0.0012 \
    --batch-size 16 --block-size 256 --accum-steps 4 \
    --checkpoint-dir checkpoints/argus_prime_cc_ddp_nccl \
    --log-interval 10 --time-budget 5

# Machine 1: same but --node_rank=1
```

With NCCL backend + fp16 gradient compression, expected: **~33K tok/s**.

## Troubleshooting

| Issue | Fix |
|-------|-----|
| `cmake can't find HIP` | Set `CMAKE_CXX_COMPILER=/opt/rocm/core-7.12/bin/hipcc` explicitly |
| `undefined reference to __builtin_*` | Apply math builtin patches (step 3) |
| Build takes >1 hour | Use `-DGPU_TARGETS="gfx1151"` only (don't build for other GPUs) |
| `LD_PRELOAD` doesn't override | Check: `ldd $(python3 -c "import torch; print(torch.__file__)")` shows which librccl is loaded. May need to also set `LD_LIBRARY_PATH` |
| Still `invalid kernel file` | Verify: `strings $HOME/rccl-gfx1151/lib/librccl.so | grep gfx1151` shows matches |
| `pfn_hsa_system_get_info failed` | This is a warning, not fatal. Set `HSA_OVERRIDE_GFX_VERSION=11.5.1` |

## Expected Performance (NCCL vs Gloo)

| Backend | Gradient sync (168M) | Compression | Expected tok/s | Speedup |
|---------|---------------------|-------------|----------------|---------|
| Gloo (current) | ~610ms (fp32, TCP) | None | 31K | 1.85x |
| NCCL + fp16 hook | ~33ms (fp16, GPU-side) | Yes | ~33K | ~1.97x |

The NCCL backend runs allreduce directly on the GPU, and the fp16 compress hook halves the payload. Combined with 9 Gbps TB4 bandwidth, this should hit near-theoretical 2x scaling.

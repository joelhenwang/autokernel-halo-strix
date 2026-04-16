#!/bin/bash
# RCCL environment for DDP over Thunderbolt 4 (2x Strix Halo)
# Source this on BOTH machines before running torchrun:
#   source scripts/ddp_env.sh

export HSA_OVERRIDE_GFX_VERSION=11.5.1
export LIBRARY_PATH=/opt/rocm/core-7.12/lib:$LIBRARY_PATH
export LD_LIBRARY_PATH=/opt/rocm/core-7.12/lib:$LD_LIBRARY_PATH

# Custom RCCL built with gfx1151 GPU kernels (overrides bundled RCCL)
# Machine 0: ~/Desktop/ai_lab/autokernel-halo-strix/external/rccl/build/release/
# Machine 1: needs its own build or a copy of librccl.so.1.0
export LD_PRELOAD=$HOME/Desktop/ai_lab/autokernel-halo-strix/external/rccl/build/release/librccl.so.1.0

export NCCL_SOCKET_IFNAME=thunderbolt0
export GLOO_SOCKET_IFNAME=thunderbolt0
export NCCL_P2P_DISABLE=1
export NCCL_IB_DISABLE=1
export NCCL_SOCKET_NTHREADS=4
export NCCL_NSOCKS_PERTHREAD=4
export NCCL_TIMEOUT=1800
export MASTER_ADDR=10.77.0.1
export MASTER_PORT=29500

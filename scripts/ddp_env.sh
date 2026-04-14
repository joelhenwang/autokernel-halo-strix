#!/bin/bash
# RCCL environment for DDP over Thunderbolt 4 (2x Strix Halo)
# Source this on BOTH machines before running torchrun:
#   source scripts/ddp_env.sh

export NCCL_SOCKET_IFNAME=tb-ddp
export NCCL_P2P_DISABLE=1
export NCCL_IB_DISABLE=1
export NCCL_SOCKET_NTHREADS=4
export NCCL_NSOCKS_PERTHREAD=4
export NCCL_TIMEOUT=1800
export MASTER_ADDR=10.77.0.1
export MASTER_PORT=29500

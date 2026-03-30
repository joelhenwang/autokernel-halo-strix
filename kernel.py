"""
AutoKernel -- Fused SwiGLU MLP kernel.

Exp-F1: Apply matmul optimizations (autotune, 3-arg tl.dot, grouped ordering,
         128x128 tiles, num_warps=8) to fused gate+up kernel.
"""

KERNEL_TYPE = "fused_mlp"

import torch
import triton
import triton.language as tl


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 64, 'GROUP_SIZE_M': 8}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 64, 'GROUP_SIZE_M': 8}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8}, num_warps=4, num_stages=2),
    ],
    key=['M', 'N', 'K'],
)
@triton.jit
def fused_gate_up_kernel(
    X_ptr,
    W_gate_ptr,
    W_up_ptr,
    Out_ptr,
    M, N, K,
    stride_xm, stride_xk,
    stride_wgk, stride_wgn,
    stride_wuk, stride_wun,
    stride_om, stride_on,
    USE_SILU: tl.constexpr,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr,
):
    pid = tl.program_id(0)
    num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)

    num_pid_in_group = GROUP_SIZE_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_SIZE_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
    pid_m = first_pid_m + ((pid % num_pid_in_group) % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m

    offs_am = (pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)) % M
    offs_bn = (pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)) % N
    offs_k = tl.arange(0, BLOCK_SIZE_K)

    x_ptrs = X_ptr + offs_am[:, None] * stride_xm + offs_k[None, :] * stride_xk
    wg_ptrs = W_gate_ptr + offs_k[:, None] * stride_wgk + offs_bn[None, :] * stride_wgn
    wu_ptrs = W_up_ptr + offs_k[:, None] * stride_wuk + offs_bn[None, :] * stride_wun

    acc_gate = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    acc_up = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)

    for k in range(0, tl.cdiv(K, BLOCK_SIZE_K)):
        x = tl.load(x_ptrs, mask=offs_k[None, :] < K - k * BLOCK_SIZE_K, other=0.0)
        wg = tl.load(wg_ptrs, mask=offs_k[:, None] < K - k * BLOCK_SIZE_K, other=0.0)
        wu = tl.load(wu_ptrs, mask=offs_k[:, None] < K - k * BLOCK_SIZE_K, other=0.0)

        acc_gate = tl.dot(x, wg, acc_gate)
        acc_up = tl.dot(x, wu, acc_up)

        x_ptrs += BLOCK_SIZE_K * stride_xk
        wg_ptrs += BLOCK_SIZE_K * stride_wgk
        wu_ptrs += BLOCK_SIZE_K * stride_wuk

    if USE_SILU:
        gate_activated = acc_gate * tl.sigmoid(acc_gate)
    else:
        tanh_arg = 0.7978845608 * (acc_gate + 0.044715 * acc_gate * acc_gate * acc_gate)
        tanh_val = 2.0 * tl.sigmoid(2.0 * tanh_arg) - 1.0
        gate_activated = 0.5 * acc_gate * (1.0 + tanh_val)

    result = gate_activated * acc_up

    offs_cm = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_cn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    out_ptrs = Out_ptr + offs_cm[:, None] * stride_om + offs_cn[None, :] * stride_on
    out_mask = (offs_cm[:, None] < M) & (offs_cn[None, :] < N)
    tl.store(out_ptrs, result.to(Out_ptr.dtype.element_ty), mask=out_mask)


def kernel_fn(
    x: torch.Tensor,
    w_gate: torch.Tensor,
    w_up: torch.Tensor,
    w_down: torch.Tensor,
    activation: str = "silu",
) -> torch.Tensor:
    """Entry point called by bench.py."""
    assert x.is_cuda

    orig_shape = x.shape
    if x.ndim > 2:
        x = x.view(-1, x.shape[-1])

    M, K = x.shape
    N, K2 = w_gate.shape
    assert K == K2
    assert w_up.shape == (N, K)

    hidden = torch.empty((M, N), device=x.device, dtype=x.dtype)

    grid = lambda META: (triton.cdiv(M, META['BLOCK_SIZE_M']) * triton.cdiv(N, META['BLOCK_SIZE_N']),)

    fused_gate_up_kernel[grid](
        x, w_gate, w_up, hidden,
        M, N, K,
        x.stride(0), x.stride(1),
        w_gate.stride(1), w_gate.stride(0),
        w_up.stride(1), w_up.stride(0),
        hidden.stride(0), hidden.stride(1),
        USE_SILU=(activation == "silu"),
    )

    out = hidden @ w_down.t()

    if len(orig_shape) > 2:
        out = out.view(*orig_shape[:-1], out.shape[-1])

    return out

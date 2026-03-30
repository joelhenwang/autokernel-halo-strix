"""
AutoKernel -- Softmax kernel.

Exp-S1: Autotune num_warps + BLOCK_SIZE, multi-row processing for small rows.
"""

KERNEL_TYPE = "softmax"

import torch
import triton
import triton.language as tl


@triton.jit
def softmax_kernel(
    input_ptr,
    output_ptr,
    n_cols,
    stride_input_row,
    stride_output_row,
    BLOCK_SIZE: tl.constexpr,
    NUM_ROWS_PER_PROGRAM: tl.constexpr,
):
    pid = tl.program_id(0)
    row_start = pid * NUM_ROWS_PER_PROGRAM

    for row_off in range(NUM_ROWS_PER_PROGRAM):
        row_idx = row_start + row_off

        row_start_input = input_ptr + row_idx * stride_input_row
        row_start_output = output_ptr + row_idx * stride_output_row

        col_offsets = tl.arange(0, BLOCK_SIZE)
        mask = col_offsets < n_cols

        row = tl.load(row_start_input + col_offsets, mask=mask, other=float("-inf"))
        row_max = tl.max(row, axis=0)
        row = row - row_max
        numerator = tl.exp(row)
        denominator = tl.sum(numerator, axis=0)
        result = numerator / denominator
        tl.store(row_start_output + col_offsets, result, mask=mask)


def kernel_fn(x: torch.Tensor) -> torch.Tensor:
    """Entry point called by bench.py."""
    assert x.is_cuda

    orig_shape = x.shape
    if x.ndim == 1:
        x = x.unsqueeze(0)
    elif x.ndim > 2:
        x = x.view(-1, x.shape[-1])

    n_rows, n_cols = x.shape
    output = torch.empty_like(x)

    BLOCK_SIZE = triton.next_power_of_2(n_cols)

    if n_cols <= 2048:
        num_warps = 4
    elif n_cols <= 8192:
        num_warps = 8
    else:
        num_warps = 16

    ROWS_PER_PROG = 4
    grid = (triton.cdiv(n_rows, ROWS_PER_PROG),)

    softmax_kernel[grid](
        x, output,
        n_cols,
        x.stride(0),
        output.stride(0),
        BLOCK_SIZE=BLOCK_SIZE,
        NUM_ROWS_PER_PROGRAM=ROWS_PER_PROG,
        num_warps=num_warps,
        num_stages=2,
    )

    return output.view(orig_shape)

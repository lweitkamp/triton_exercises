import pytest
import torch
import triton
import triton.language as tl


@triton.jit
def sum_row_blocked_iterative_kernel(
    A_ptr: tl.tensor, outputs_ptr: tl.tensor,
    M: tl.constexpr, N: tl.constexpr,
    BLOCK_N: tl.constexpr,
    A_strides_x, A_strides_y,
):
    """Calculate the sum of a row of the input tensor, storing the result in
    the output. We assume the input row fits into SRAM.

    Args:
        A_ptr: Pointer to the input tensor.
        outputs_ptr: Pointer to the output tensor.
        M: Number of rows in the input tensor.
        N: Number of columns in the input tensor.
        BLOCK_N: Block size of each row we load.
        input_stride_x: Stride of the input tensor along the row dim.
        input_stride_y: Stride of the input tensor along the column dim.
    """
    program_id = tl.program_id(axis=0)

    input_block_ptr = tl.make_block_ptr(
        base=A_ptr,
        shape=(M, N),
        strides=(A_strides_x, A_strides_y),
        offsets=(program_id, 0),
        block_shape=(1, BLOCK_N),
        order=(1, 0),
    )
    output_block_ptr = tl.make_block_ptr(
        base=outputs_ptr,
        shape=(M, ),
        strides=(1, ),
        offsets=(program_id, ),
        block_shape=(1, ),
        order=(0, ),
    )

    accumulator = tl.zeros((1, ), dtype=tl.float32)
    for _ in range(0, N, BLOCK_N):
        input_block = tl.load(input_block_ptr, boundary_check=(0, 1))
        accumulator += tl.sum(input_block, axis=1)
        input_block_ptr = tl.advance(input_block_ptr, (0, BLOCK_N))

    tl.store(output_block_ptr, accumulator)


def sum_row_blocked_iterative(A: torch.Tensor) -> torch.Tensor:
    """Calculate the sum of a tensor A along the final dim.

    Args:
        A: Tensor of shape (M, N) containing the input values.

    Returns:
        Tensor of shape (M, ) containing the summed values.
    """
    M, N = A.shape
    outputs = torch.empty((M,), dtype=A.dtype, device=A.device)

    sum_row_blocked_iterative_kernel[(M, )](
        A_ptr=A, outputs_ptr=outputs,
        M=M, N=N,
        A_strides_x=A.stride(0), A_strides_y=A.stride(1),
        BLOCK_N=8,
    )

    return outputs


@pytest.mark.parametrize("M, N", [(16, 16), (32, 16)])
def test_sum_row(M: int, N: int):
    inputs = torch.randn((M, N), device='cuda')
    outputs = sum_row_blocked_iterative(inputs)
    torch.testing.assert_close(inputs.sum(axis=1), outputs)

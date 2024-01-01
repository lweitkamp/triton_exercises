import pytest
import torch
import triton
import triton.language as tl


@triton.jit
def sum_row_kernel(
    A_ptr: tl.tensor, outputs_ptr: tl.tensor,
    M: tl.constexpr, N: tl.constexpr,
    input_stride_x, input_stride_y,
):
    """Calculate the sum of a row of the input tensor, storing the result in
    the output. We assume the input row fits into SRAM.

    Args:
        input_ptr: Pointer to the input tensor.
        output_ptr: Pointer to the output tensor.
        M: Number of rows in the input tensor.
        N: Number of columns in the input tensor.
        input_stride_x: Stride of the input tensor along the row dim.
        input_stride_y: Stride of the input tensor along the column dim.
    """
    program_id = tl.program_id(axis=0)

    input_block_ptr = tl.make_block_ptr(
        base=A_ptr,
        shape=(M, N),
        strides=(input_stride_x, input_stride_y),
        offsets=(program_id, 0),
        block_shape=(1, N),
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

    input_block = tl.load(input_block_ptr)

    tl.store(output_block_ptr, tl.sum(input_block))


def sum_row(inputs: torch.Tensor) -> torch.Tensor:
    """Calculate the sum of a tensor along the final dim.

    Args:
        inputs: Tensor of shape (M, N) containing the input values.

    Returns:
        Tensor of shape (M, ) containing the summed values.
    """
    M, N = inputs.shape
    outputs = torch.empty((M,), dtype=inputs.dtype, device=inputs.device)

    sum_row_kernel[(M, )](
        A_ptr=inputs, outputs_ptr=outputs,
        M=M, N=N,
        input_stride_x=inputs.stride(0), input_stride_y=inputs.stride(1),
    )

    return outputs


@pytest.mark.parametrize("M, N", [(16, 16), (32, 16)])
def test_sum_row(M: int, N: int):
    inputs = torch.randn((M, N), device='cuda')
    outputs = sum_row(inputs)
    torch.testing.assert_close(inputs.sum(dim=1), outputs)

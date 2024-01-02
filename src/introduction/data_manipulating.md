# Manipulating Data
## Loading and Storing Data

With block pointers ready we can start loading data from global memory to much faster shared memory. Triton provides the [`tl.load`](https://triton-lang.org/main/python-api/generated/triton.language.load.html) and [`tl.store`](https://triton-lang.org/main/python-api/generated/triton.language.store.html) functions for this. Since we use block pointers we can ignore most arguments in both loading and storing, but the docs do give us some information:

> pointer could be a block pointer defined by make_block_ptr, in which case:
> - mask and other must be None
> - boundary_check and padding_option can be specified to control the behavior of out-of-bound access 

Ignoring the mask argument, we are left with `boundary_check` and `padding_option`. If `boundary_check` is enabled, out-of-bound memory can be set to a static value using `padding_option`. 


In most cases we are dealing with some multiple of 8 or power of 2 that is happily divisible by whatever block size we use.
If there is a mismatch between the tensor length and the block size (e.g. `N % BLOCK_N != 0`) we will need to have `boundary_check` enabled.
This padding is not very versatile though, since padding options are only `zero` and `nan`. For the purpose of our sum kernel the `zero` works fine, but we will need to use some additional masking for a iterative softmax kernel.

## Tensors
We need to keep track of an accumulator variable that stores the intermediate sum.
It will essentially be a scalar per program but we have to initiate it as a Triton tensor `tl.Tensor`.
As in Torch, Triton has a variety of ways to start a tensor. We could use `tl.full(shape=(1, ), value=0, dtype=tf.float32)`, which would be the same as `tl.zeros(shape=(1, ), dtype=tf.float32)`. What *is* different is that the data type is not optional, you **have** to set it.

What you might often see is that regardless of the precision that comes in (if \\(A\\) is `torch.bfloat` or `torch.float16`), accumulation will be done in `tl.float32` to achieve the highest precision available.
There is not much downside here since the accumulation is typically small in shape and the data requires is already loaded, so no memory bandwidth wasted here. 


## The Final Iterative Sum Kernel
We can now load and store data and we have a way to keep track of accumulative sums of the blocks of a row. In case `N` is not divible by `BLOCK_N` we can load using a `boundary_check`. We can add this check in any case and see later what the cost of this check is during the optimization section. The final kernel is then as follows:

```python
:::import torch
:::import triton
:::import triton.language as tl
:::
:::
def sum_row_blocked_iterative(A: torch.Tensor) -> torch.Tensor:
:::    """Calculate the sum of a tensor A along the final dim.
:::
:::    Args:
:::        A: Tensor of shape (M, N) containing the input values.
:::
:::    Returns:
:::        Tensor of shape (M, ) containing the summed values.
:::    """
    M, N = A.shape
    outputs = torch.empty((M,), dtype=A.dtype, device=A.device)

    sum_row_blocked_iterative_kernel[(M, )](
        A_ptr=A, outputs_ptr=outputs,
        M=M, N=N,
        A_strides_x=A.stride(0), A_strides_y=A.stride(1),
        BLOCK_N=8,
    )

    return outputs


@triton.jit
def sum_row_blocked_iterative_kernel(
    A_ptr: tl.tensor, outputs_ptr: tl.tensor,
    M: tl.constexpr, N: tl.constexpr,
    BLOCK_N: tl.constexpr,
    A_strides_x, A_strides_y,
):
:::    """Calculate the sum of a row of the input tensor, storing the result in
:::    the output. We assume the input row fits into SRAM.
:::
:::    Args:
:::        A_ptr: Pointer to the input tensor.
:::        outputs_ptr: Pointer to the output tensor.
:::        M: Number of rows in the input tensor.
:::        N: Number of columns in the input tensor.
:::        BLOCK_N: Block size of each row we load.
:::        input_stride_x: Stride of the input tensor along the row dim.
:::        input_stride_y: Stride of the input tensor along the column dim.
:::    """
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
```

I hope this was a useful primer on Triton so far. The next chapter will delve into benchmarking and optimizing kernels. For complete code to the sum kernels you can check the [code](https://github.com/lweitkamp/triton_exercises/tree/main/code) folder, there will be three versions: simple row sum, blocked row sum and iterative row sum, each discussed in some parts in this chapter.
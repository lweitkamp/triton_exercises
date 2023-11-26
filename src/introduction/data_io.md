# Loading and Storing Data

With block pointers ready we can start loading data from global memory to much faster shared memory. Triton provides the [`tl.load`](https://triton-lang.org/main/python-api/generated/triton.language.load.html) and [`tl.store`](https://triton-lang.org/main/python-api/generated/triton.language.store.html) functions for this. Since we use block pointers we can ignore most arguments in both loading and storing, but the docs do give us some information:

> pointer could be a block pointer defined by make_block_ptr, in which case:
> - mask and other must be None
> - boundary_check and padding_option can be specified to control the behavior of out-of-bound access 

Ignoring the mask argument, we are left with `boundary_check` and `padding_option`. If `boundary_check` is enabled, out-of-bound memory can be set to a static value using `padding_option`. This approach is unfortunately not as versatile as the non-block pointer approach, since padding options are only `zero` and `nan`. As an example of where this hurts, the softmax tutorial in the documentation uses the old loading approach and here you can simply set out-of-bound values to `-inf`.

We will continue the 2D example above and expand the intention to a max-pool operation for each block. That is, each program loads a block of 16 by 16 and calculates its maximum value. The output will be a 2 by 2 matrix. Because we are strictly loading quarters of the original tensor we probably don't need a boundary check and a padding value, but we will add both as an example.

```python
@triton.jit
def maxpool_kernel(
    input_ptr: tl.tensor, output_ptr: tl.tensor,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr,    
    input_stride_x, input_stride_y,
    output_stride_x, output_stride_y,
):
    pid_M = tl.program_id(axis=0)
    pid_N = tl.program_id(axis=1)

    input_block_ptr = tl.make_block_ptr(
        base=input_ptr,
        shape=(32, 32),
        strides=(input_stride_x, input_stride_y),
        offsets=(pid_M * BLOCK_M, pid_N * BLOCK_N),
        block_shape=(BLOCK_M, BLOCK_N),
        order=(1, 0),
    )
    output_block_ptr = tl.make_block_ptr(
        base=output_ptr,
        shape=(2, 2),
        strides=(output_stride_x, output_stride_y),
        offsets=(pid_M, pid_N),
        block_shape=(1, 1),
        order=(1, 0),
    )

    input_block = tl.load(
        input_block_ptr,
        boundary_check=(0, 1),
        padding_option="zero",
    )

    tl.store(output_block_ptr, tl.max(input_block))
```

Instead of [`tl.max`](https://triton-lang.org/main/python-api/generated/triton.language.max.html#triton.language.max) we could just as easily have used [`tl.sum`](https://triton-lang.org/main/python-api/generated/triton.language.sum.html#triton.language.sum). If we had two tensors we could've [`tl.dot`](https://triton-lang.org/main/python-api/generated/triton.language.sum.html#triton.language.sum)'ed them too. Check out the whole of [`triton.language`](https://triton-lang.org/main/python-api/triton.language.html), we will use a lot of these operations in due time.

To run the kernel will require a sort of wrapper function that takes the input tensor(s), defines the output tensor in memory and runs the kernel with a proper launch grid. An example can be seen below.


```python
def maxpool(inputs: torch.Tensor) -> torch.Tensor:
    outputs = torch.empty((2, 2), dtype=inputs.dtype, device=inputs.device)

    maxpool_kernel[(2, 2)](
        input_ptr=inputs, output_ptr=outputs,
        BLOCK_M=16, BLOCK_N=16,
        input_stride_x=inputs.stride(0), input_stride_y=inputs.stride(1),
    )

    return outputs
```

# Block Pointers

Pointer arithmetic can be tedious work and it's easy to mess up.
Not to mention that we have not worked with loading blocks of 2D data, introducing multidimensional pointer blocks.
From the official [Triton tutorial on matrix multiplications](https://triton-lang.org/main/getting-started/tutorials/03-matrix-multiplication.html), the following snippet shows how this 2D arithmetic works:

```python
# Program ID
pid = tl.program_id(axis=0)
# Number of program ids along the M axis
num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
# Number of programs ids along the N axis
num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
# Number of programs in group
num_pid_in_group = GROUP_SIZE_M * num_pid_n
# Id of the group this program is in
group_id = pid // num_pid_in_group
# Row-id of the first program in the group
first_pid_m = group_id * GROUP_SIZE_M
# If `num_pid_m` isn't divisible by `GROUP_SIZE_M`, the last group is smaller
group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
# *Within groups*, programs are ordered in a column-major order
# Row-id of the program in the *launch grid*
pid_m = first_pid_m + (pid % group_size_m)
# Col-id of the program in the *launch grid*
pid_n = (pid % num_pid_in_group) // group_size_m
```

Let's steer clear of all that, and start using the block pointer functionality that is still an experimental feature. It changes the setup from this:

```python
:::import triton
:::import triton.language as tl
:::import torch
:::
:::def sum_row(A: torch.Tensor) -> torch.Tensor:
:::    """Calculate the sum of a tensor A along the final dim.
:::
:::    Args:
:::        A: Tensor of shape (M, N) containing the input values.
:::
:::    Returns:
:::        Tensor of shape (M, ) containing the summed values.
:::    """
:::    M, N = A.shape
:::    outputs = torch.empty((M,), dtype=A.dtype, device=A.device)
:::
:::    launch_grid = (M, )
:::
:::    sum_kernel[launch_grid](
:::        A, outputs,
:::        N=N,
:::        A_strides_x=A.strides(0), A_strides_y=A.strides(1),
:::    )
:::
:::    return outputs
:::
@triton.jit
def sum_kernel(
    A_ptr, outputs_ptr,
    N,
    A_strides_x, A_strides_y,
):
    program_id = tl.program_id(axis=0)
    offsets = tl.arange(0, N) + A_ptr + program_id * A_stride_y
```

To this:

```python
:::import triton
:::import triton.language as tl
:::import torch
:::
:::def sum_row(A: torch.Tensor) -> torch.Tensor:
:::    """Calculate the sum of a tensor A along the final dim.
:::
:::    Args:
:::        A: Tensor of shape (M, N) containing the input values.
:::
:::    Returns:
:::        Tensor of shape (M, ) containing the summed values.
:::    """
:::    M, N = A.shape
:::    outputs = torch.empty((M,), dtype=A.dtype, device=A.device)
:::
:::    launch_grid = (M, )
:::
:::    sum_row_kernel[launch_grid](
:::        A, outputs,
:::        M=M, N=N,
:::        A_strides_x=A.stride(0), A_strides_y=A.stride(1),
:::    )
:::
:::    return outputs
:::
@triton.jit
def sum_row_kernel(
    A_ptr, outputs_ptr,
    M, N,
    input_stride_x, input_stride_y,
):
    program_id = tl.program_id(axis=0)
    offsets = tl.make_block_ptr(
        base=A_ptr,
        shape=(M, N),
        strides=(input_stride_x, A_stride_y),
        offsets=(program_id, 0),
        block_shape=(1, N),
        order=(1, 0),
    )
```
A little bit more work and more added arguments, but this allows us to load 1D and 2D blocks with ease, and we can also skip any masking for out-of-bounds memory access. The table below gives a brief description per argument[^1].

| abc         | def |
|-------------|-----|
| base        | The data pointer from which you want to load a block |
| shape       | Shape of the base tensor |
| strides     | Strides of the base tensor |
| offsets     | From what location do you want to start loading data |
| block_shape | What is the shape of the data block to load |
| order       | The memory layout of the base tensor |

## Block Pointers in 2D and Dynamic Launch Grids
We mentioned earlier that block pointers make 2D loading easier too. As an example, let's transform the block pointer to load not one row, but 2 or potentially more as the following figure indicates:

![With block pointers we can also switch easily to blocks of rows.](images/block-offsets.svg)

This has consequences for the launch grid, though. We would essentially need half as much programs to be launched if we load 2 rows each time. But what if we load 4 rows each? That would reduce the number of programs by half again. Instead of statically changing the launch grid each time, we can make it ***dynamic***.

The launch grid is not defined only to be a tuple of integers, it can also be a callable that *returns* a tuple of integers. This callable has as input the parameters of the kernel so we can dynamically select the number of programs to be launched as a function of the number of rows we process:

```python
:::import triton
:::import triton.language as tl
:::import torch
:::
def sum_row_blocked(A: torch.Tensor) -> torch.Tensor:
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

    dynamic_launch_grid = lambda params: (triton.cdiv(M, params["BLOCK_M"]), )
    sum_row_blocked_kernel[dynamic_launch_grid](
        A_ptr=A, outputs_ptr=outputs,
        M=M, N=N,
        A_strides_x=A.stride(0), A_strides_y=A.stride(1),
        BLOCK_M=2,
    )

    return outputs


@triton.jit
def sum_row_blocked_kernel(
    A_ptr, outputs_ptr,
    M, N,
    BLOCK_M,
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
:::        input_stride_x: Stride of the input tensor along the row dim.
:::        input_stride_y: Stride of the input tensor along the column dim.
:::    """
    program_id = tl.program_id(axis=0)
:::
    input_block_ptr = tl.make_block_ptr(
        base=A_ptr,
        shape=(M, N),
        strides=(A_strides_x, A_strides_y),
        offsets=(program_id * BLOCK_M, 0),
        block_shape=(BLOCK_M, N),
        order=(1, 0),
    )
:::    output_block_ptr = tl.make_block_ptr(
:::        base=outputs_ptr,
:::        shape=(M, ),
:::        strides=(1, ),
:::        offsets=(program_id * BLOCK_M, ),
:::        block_shape=(BLOCK_M, ),
:::        order=(0, ),
:::    )
:::
:::    input_block = tl.load(input_block_ptr)
:::
:::    tl.store(output_block_ptr, tl.sum(input_block, axis=1))
```

It's impressive how little code we had to change to switch from 1D to 2D, so block pointers are definitely my go-to for getting data offsets.

## Advancing Block Pointers
In most situations we can easily load the whole row into memory and process a row or even a set of rows per program. But imagine we are not capable of loading the entire row into memory - it's too big for our cache! What *can* do, is iterate over the row in blocks. 

the iterative part is where [`tl.advance`](https://github.com/openai/triton/blob/f107df16a07dda3001b466d764ed87a69a56c60e/python/triton/language/core.py#L1179) comes into play.
Each program will load a block of size `BLOCK_N` << `N` and iterate untill it has seen the full row.

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
    A_ptr, outputs_pt,
    M, N,
    BLOCK_N,
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
:::    output_block_ptr = tl.make_block_ptr(
:::        base=outputs_ptr,
:::        shape=(M, ),
:::        strides=(1, ),
:::        offsets=(program_id, ),
:::        block_shape=(1, ),
:::        order=(0, ),
:::    )

:::    accumulator = tl.zeros((1, ), dtype=tl.float32)
    for _ in range(0, N, BLOCK_N):
        input_block = tl.load(input_block_ptr)
:::        accumulator += tl.sum(input_block, axis=1)
        input_block_ptr = tl.advance(input_block_ptr, (0, BLOCK_N))
:::
:::    tl.store(output_block_ptr, accumulator)
```

There are some consequences in terms of out-of-bounds memory access checking, but we will cover this is the next section.

[^1]: There is more information available in the official Triton docs for the [Blocked Pointer Matrix Multiplication](https://triton-lang.org/main/getting-started/tutorials/08-experimental-block-pointer.html#make-a-block-pointer) tutorial.
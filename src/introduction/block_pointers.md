# Block Pointers

Pointer arithmetic can be tedious work and it's easy to mess up.
Not to mention that we have not worked with loading blocks of 2D data, introducing multidimensional pointer blocks.

Let's steer clear of all that, and start using the block pointer functionality that is still an experimental feature.

It changes the setup from this:

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
:::        BLOCK=N,
:::        A_strides_x=A.strides(0), A_strides_y=A.strides(1),
:::    )
:::
:::    return outputs
:::
@triton.jit
def sum_kernel(A_ptr, outputs_ptr, BLOCK, A_strides_x, A_strides_y):
    program_id = tl.program_id(axis=0)
    offsets = tl.arange(0, BLOCK) + A_ptr + program_id * A_stride_y    
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
:::    sum_kernel[launch_grid](
:::        A, outputs,
:::        BLOCK=N,
:::        A_strides_x=A.strides(0), A_strides_y=A.strides(1),
:::    )
:::
:::    return outputs
:::
@triton.jit
def sum_kernel(A_ptr, outputs_ptr, M, N, A_strides_x, A_strides_y):
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
A little bit more work and more added arguments, but this allows us to load 1D and 2D blocks with ease, and we can also skip any masking for out-of-bounds memory access. 

| abc         | def |
|-------------|-----|
| base        | The data pointer from which you want to load a block |
| shape       | Shape of the base tensor |
| strides     | Strides of the base tensor |
| offsets     | From what location do you want to start loading data |
| block_shape | What is the shape of the data block to load |
| order       | ... |

For some common access patterns, see the figure below.


## Advancing Block Pointers
This section is more for reference since we will not have to advance any block pointers for the row-sum kernel. Imagine we are not capable of loading the entire row into memory - it's too big for our cache! What *can* do, is iterate over the row in blocks. the iterative part is where [`tl.advance`]()
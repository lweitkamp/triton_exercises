# The Launch Grid
A Triton kernel will launch a number of *programs* to distribute the work over blocks of data.
The number of programs to run is a variable we can change, it depends on both the hardware present and the complexity of the algorithm.
We can control the number of programs in the *launch grid*.

As an example, lets try to calculate the sum of rows of an \\(6 \times 4\\) matrix \\(A\\). A possible kernel here would be one that launches as many programs as there are rows, and lets each program essentially perform a vector sum. Since the launch grid is a tuple in Python, it would correspond to `(6, )`. This will launch 6 distinct programs, each with a row of the data. Each program is denoted a program identifier (PID for short) that is accessible inside the kernel with `triton.language.program_id()`. A visualization of this setup can be seen below:

![A 1 dimensional launch grid of 6 programs.](images/launch-grid-1d.svg)

And it's not much work in python as you can see in the code snippet below:

```python
:::import torch
:::
:::
def sum_row(inputs: torch.Tensor) -> torch.Tensor:
:::    """Calculate the sum of a tensor along the final dim.
:::
:::    Args:
:::        inputs: Tensor of shape (M, N) containing the input values.
:::
:::    Returns:
:::        Tensor of shape (M, ) containing the summed values.
:::    """
    M, N = inputs.shape
    outputs = torch.empty((M,), dtype=inputs.dtype, device=inputs.device)

    launch_grid = (M, )

    sum_kernel[launch_grid](
        input_ptr=inputs, output_ptr=outputs,
        M=M, N=N,
        input_stride_x=inputs.stride(0), input_stride_y=inputs.stride(1),
    )

    return outputs
```

For now, assume the kernel `sum_kernel` is a valid Triton kernel. A valid triton kernel is called with the funky `kernel[launch_grid]()` syntax to denote **which version of the kernel you want to launch**. Think of it as a python dictionary where keys are different launch grid configurations and the values are the compiled kernels related to configuration.

We can also divide the work into sets of rows *and* columns. If we keep the number of programs equal to 6, each program can also process two half rows. This will require a multidimensional launch grid `(2, 3)`:

```python
:::import torch
:::
:::
:::def sum_row(inputs: torch.Tensor) -> torch.Tensor:
:::    """Calculate the sum of a tensor along the final dim.
:::
:::    Args:
:::        inputs: Tensor of shape (M, N) containing the input values.
:::
:::    Returns:
:::        Tensor of shape (M, ) containing the summed values.
:::    """
:::    M, N = inputs.shape
:::    outputs = torch.empty((M,), dtype=inputs.dtype, device=inputs.device)
:::
    launch_grid = (M // 3, N // 2)
:::
:::    sum_kernel[launch_grid](
:::        input_ptr=inputs, output_ptr=outputs,
:::        M=M, N=N,
:::        input_stride_x=inputs.stride(0), input_stride_y=inputs.stride(1),
:::    )
:::
:::    return outputs
```
And again for a higher overview we can look at the figure below. Notice the topological layout.

![A two dimensional launch grid of 3 times 2 programs.](images/launch-grid-2d.svg)

Since we have a two dimensional launch grid, we have programs that have corresponding \\(x\\) and \\(y\\) identfiers. To identify the current program working we would have to get both identifiers: `pid_x = triton.language.program_id(axis=0)` and `pid_y = triton.language.program_id(axis=1)`.

The change to 2D can have an effect on performance since we are no longer loading blocks of contiguous memory. Multidimensional launch grids are, however, not very common, or at least not from what I've seen. In the exercises we will stick to 1D grids.

We will revisit the launch grid later in the optimization section, where we use the fact that the launch grid [can be either Tuple[int], or Callable(metaparameters) -> Tuple[int]](https://github.com/openai/triton/blob/7d3f045045eb8d1a36e01d5b9ba26644304c02cf/python/tutorials/01-vector-add.py#L66C44-L66C116).
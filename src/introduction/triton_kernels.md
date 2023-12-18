# Triton Kernels
A Triton kernel is written in the Triton domain specific language (DSL) and is compiled just-in-time using the `@jtriton.jit` decorator.
Inside the kernel you can use specific Triton DSL operations that look surpisingly like Python. Let's look at the vector addition example from the tutorial site:

```python
:::import torch
:::import triton
:::import triton.language as tl
:::
:::
@triton.jit
def add_kernel(x_ptr,  # *Pointer* to first input vector.
               y_ptr,  # *Pointer* to second input vector.
               output_ptr,  # *Pointer* to output vector.
               n_elements,  # Size of the vector.
               BLOCK_SIZE: tl.constexpr,  # Number of elements each program should process.
               # NOTE: `constexpr` so it can be used as a shape value.
               ):
:::    # There are multiple 'programs' processing different data. We identify which program
:::    # we are here:
    pid = tl.program_id(axis=0)  # We use a 1D launch grid so axis is 0.
:::    # This program will process inputs that are offset from the initial data.
:::    # For instance, if you had a vector of length 256 and block_size of 64, the programs
:::    # would each access the elements [0:64, 64:128, 128:192, 192:256].
:::    # Note that offsets is a list of pointers:
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
:::    # Create a mask to guard memory operations against out-of-bounds accesses.
    mask = offsets < n_elements
:::    # Load x and y from DRAM, masking out any extra elements in case the input is not a
:::    # multiple of the block size.
    x = tl.load(x_ptr + offsets, mask=mask)
    y = tl.load(y_ptr + offsets, mask=mask)
    output = x + y
    # Write x + y back to DRAM.
    tl.store(output_ptr + offsets, output, mask=mask)
```

## `triton.jit`
Functions to be decorated with the just in time compilation decorator have some requirements. 

> talk about requirements

Additionally, any `torch.Tensor` that is passed to the function is converted to a pointer towards its first value in memory. In the example above we simply pass some vector `x` and expect the value to be a pointer `x_ptr` inside of the kernel.

There are a lot of parameters that come with the jit decorator, most of them not well documented. You can run into all of them if you accidentally forget to add arguments to the kernel and run it:

```python
:::import triton
:::
@triton.jit
def do_nothing():
   pass

do_nothing[(1, )]()
>>> def do_nothing( , grid=None, num_warps=4, num_stages=3, extern_libs=None, stream=None, warmup=False, device=None, device_type=None):
                   ^
SyntaxError: invalid syntax
```

I've briefly documented the parameters in the table below.

## Triton's Domain Specific Language
Inside a kernel you can do...

<!-- 
We can divide most Triton implementations into two parts: the *kernel* and the *launch grid*. The kernel is where the computation happens and the launch grid is where we define how many programs will be launched and how they will be distributed over the data. We will tackle it in reverse order, discussing the launch grid first, using the vector addition from the [Triton Tutorials](https://triton-lang.org/main/getting-started/tutorials/01-vector-add.html) as an example. -->


<!-- 


Before running a Triton kernel we need to define specific parameters that define how the compiler will utilize the GPU.
For example, we need to define how many *programs* will be launched, and depending on how many programs we launch we might have to change the size of the *block(s)* that each program will work on. To make it a bit easier we typically wrap the launch grid and the kernel launch itself in a helper function. For the vector addition example it would make sense to call the kernel `add_kernel` and the helper function `add` (although you might want to post/pre-fix it with `triton`).

The launch grid is set as a parameter when you launch the kernel: `kernel[grid](...)`. It is a tuple of integers (or a callable that returns a tuple of integers) where each value defines how many programs will be launched on that axis. In general we will define 1D launch grids and figure out how to map the data to the programs inside the kernel, but it is possible to define multidimensional launch grids if that makes more sense for your problem. Think of the grid as an abstraction to help you map the data to the programs.

Below is the launch grid from the tutorial, I've hidden the comments since I thought they could be more confusing at this point. -->
<!-- 
```python
import torch
import triton


def add(x: torch.Tensor, y: torch.Tensor):
:::    # We need to preallocate the output.
    output = torch.empty_like(x)
    assert x.is_cuda and y.is_cuda and output.is_cuda
    n_elements = output.numel()
:::    # The SPMD launch grid denotes the number of kernel instances that run in parallel.
:::    # It is analogous to CUDA launch grids. It can be either Tuple[int], or Callable(metaparameters) -> Tuple[int].
:::    # In this case, we use a 1D grid where the size is the number of blocks:

    grid = lambda meta: (triton.cdiv(n_elements, meta['BLOCK_SIZE']), )
:::    # NOTE:
:::    #  - Each torch.tensor object is implicitly converted into a pointer to its first element.
:::    #  - `triton.jit`'ed functions can be indexed with a launch grid to obtain a callable GPU kernel.
:::    #  - Don't forget to pass meta-parameters as keywords arguments.
    add_kernel[grid](x, y, output, n_elements, BLOCK_SIZE=1024)
:::    # We return a handle to z but, since `torch.cuda.synchronize()` hasn't been called, the kernel is still
:::    # running asynchronously at this point.
    return output
```

The launch grid here is a callable that returns a tuple of a single integer, which is the number of elements divided by the block size.
In the tutorial the length of the vector ranges from \\( 2^{12} \\) through \\( 2^{28} \\), if we take the first value of the launch grid we get \\( 4096 / 1024 = 4 \\) programs launched and if we take the last value we get \\( \frac{2^{28}}{1024} = 262144 \\) programs launched.
How many of those programs will actually be launched concurrently depends on the capacity of your accelerator.

As mentioned, the launch grid could be hard-coded as a tuple directly, but this is less flexible and not recommended - we want the program
to scale up when the input size increases. We will discuss further ways of optimizing the launch of kernels based on input size in the [Optimization](../benchmark_and_optimization/optimization.md) section.

It is important to know that any `torch.tensor` input to the kernel is transformed into a **pointer to its first value in memory**.



## The Kernel
The kernel is written in the Triton domain specific language (DSL) and is compiled just-in-time using the `@jtriton.jit` decorator.
Inside the kernel you can use specific Triton DSL operations that look surpisingly like Python. 


```python
:::import torch
:::
:::import triton
:::import triton.language as tl
:::
:::
@triton.jit
def add_kernel(x_ptr,  # *Pointer* to first input vector.
               y_ptr,  # *Pointer* to second input vector.
               output_ptr,  # *Pointer* to output vector.
               n_elements,  # Size of the vector.
               BLOCK_SIZE: tl.constexpr,  # Number of elements each program should process.
               # NOTE: `constexpr` so it can be used as a shape value.
               ):
:::    # There are multiple 'programs' processing different data. We identify which program
:::    # we are here:
    pid = tl.program_id(axis=0)  # We use a 1D launch grid so axis is 0.
:::    # This program will process inputs that are offset from the initial data.
:::    # For instance, if you had a vector of length 256 and block_size of 64, the programs
:::    # would each access the elements [0:64, 64:128, 128:192, 192:256].
:::    # Note that offsets is a list of pointers:
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
:::    # Create a mask to guard memory operations against out-of-bounds accesses.
    mask = offsets < n_elements
:::    # Load x and y from DRAM, masking out any extra elements in case the input is not a
:::    # multiple of the block size.
    x = tl.load(x_ptr + offsets, mask=mask)
    y = tl.load(y_ptr + offsets, mask=mask)
    output = x + y
:::    # Write x + y back to DRAM.
    tl.store(output_ptr + offsets, output, mask=mask)
```

The program identifier that marks which program is running is retrieved with the [`tl.program_id`](https://triton-lang.org/main/python-api/generated/triton.language.program_id.html) function. If we take the vector length to be  \\( 4096 \\) we would get one out of `(0, 1, 2, 3)` values for the program identifier. We need the identifier to figure out what block of data to work. This tutorial example is a bit outdated and we will introduce a better mechanism for figuring out what block to work on later, but we essentially need to create a **list of pointers** for each value that we want to load in the block.

Let's make this a bit more clear with the table below:

| Program ID | Block Start | Offsets |
| ---------- | ----------- | ------- |
| 0          | 0           | (0, ..., 1023) |
| 1          | 1024        | (1024, ..., 2047) |
| 2          | 2048        | (2048, ..., 3071) |
| 3          | 3072        | (3072, ..., 4095) |

A quick addition of the `offset` with the data pointer will give us the correct block to load. A masking mechanism ensures that we don't load data outside of the vector length. 

The main setup of the kernel will typically be something along the lines of (1) load data, (2) transform data, and (3) store data. Let's take a look at a new way to load and store data in the next section. -->
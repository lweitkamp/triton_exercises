# `triton.jit`
With the launch grid defined, we can finally start working on our sum kernel.
The first step towards this is writing a function decorated using the Triton just-in-time compilaton decorator, [`@triton.jit`]((https://triton-lang.org/main/python-api/generated/triton.jit.html)).
A function that has the decorator can make use of the triton domain specific language inside of it, but will have some limitations:

> This function will be compiled and run on the GPU. It will only have access to:
> - python primitives,
> - builtins within the triton package,
> - arguments to this function,
> - other jit’d functions

Let's do just that and run it:

```python
:::import triton
:::
@triton.jit
def do_nothing():
   pass

do_nothing[(1, )]()
>>> def do_nothing(, grid=None, num_warps=4, num_stages=3, extern_libs=None, stream=None, warmup=False, device=None, device_type=None):
                   ^
SyntaxError: invalid syntax
```

Interesting!
It's not totally unexpected that we get an error because the kernel needs input arguments to work.
But this reveals a lot of arguments that get added after the jitting.
I've briefly documented the parameters in the table below.

| Arg name    | Arg description |
|-------------|-----------------|
| grid        | |
| num_warps   | A warp is a set of 32 threads. How many warps should be ran for this kernel?|
| num_stages  | |
| extern_libs | |
| stream      | |
| warmup      | |
| device      | |
| device_type | |

> Table is Work In Progress

But we are here trying to write a sum-row kernel imitating `A.sum(axis=1)`! So the first thing we do is add \\(A\\) as an input to the kernel. The result is that A is turned into a *pointer towards its first element*. Everything related to data loading and storing is done through pointers, so its good to get comfortable with some minor pointer arithmetic. We also have to add the pre-defined output vector `outputs`.

```python
:::import triton
:::import torch
:::
def sum_row(A: torch.Tensor) -> torch.Tensor:
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
:::
    launch_grid = (M, )
:::
    sum_row_kernel[launch_grid](A, outputs)
:::
    return outputs


@triton.jit
def sum_row_kernel(A_ptr, outputs_ptr):
    # A_ptr is now a pointer towards its first element, similar for outputs_ptr.
```

We now have a kernel that will run \\(M\\) different programs each with a pointer towards the first element in \\(A\\). What we don't have is some way to distinguish these programs to access different points in the data.
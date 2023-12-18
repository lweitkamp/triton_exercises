# Block Pointers
This guide relies heavily on a newly added and still experimental feature in Triton called Block Pointers. In the original blog post and in most of the (current) tutorials, data is loaded through pointer arithmetic. Take a look at the compute kernel from the vector addition tutorial, truncated here to highlight only the pointer arithmetic:

```python
:::import torch
:::
:::import triton
:::import triton.language as tl
:::
:::
:::@triton.jit
:::def add_kernel(x_ptr,  # *Pointer* to first input vector.
:::               y_ptr,  # *Pointer* to second input vector.
:::               output_ptr,  # *Pointer* to output vector.
:::               n_elements,  # Size of the vector.
:::               BLOCK_SIZE: tl.constexpr,  # Number of elements each program should process.
:::               # NOTE: `constexpr` so it can be used as a shape value.
:::               ):
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

Here, we use the program identifier `pid` to create a list of offset pointers that locate the block this program has to work on.
It does not look complex in 1D, but extend it to 2D and you will have to deal with a lot of pointer arithmetic that complicates the logic of the kernel.

With the block pointer feature we can simplify this logic by creating a block pointer that locates the block we want to work on. This avoids pointer arithmetic, simplifies masking and makes loading and storing data easier. Here is an example of loading `x`

```python
:::@triton.jit
:::def vector_do_nothing(
:::    x_ptr: tl.tensor,
:::    n_elements: tl.constexpr,
:::    BLOCK_SIZE: tl.constexpr,
:::):
:::    """Load a block from a vector, do nothing.
:::    """
:::    program_id = tl.program_id(axis=0)
:::
    x_block_ptr = tl.make_block_ptr(
      base=x_ptr,
      shape=(n_elements, ),
      strides=(1, ),
      offsets=(program_id * BLOCK_SIZE, ),
      block_shape=(BLOCK_SIZE, ),
      order=(0, ),
    )
    x = tl.load(x_block_ptr)
:::
```

We do add more lines (you can squeeze it into three lines if you'd like), but I would argue it also makes reading the code easier.
Let's break down the arguments of `tl.make_block_ptr`, given a torch matrix `X` of size `N` by `M`:

- `base` should point to the tensor's first value location in memory, so typically just `X`.
- `shape` should be the shape of the tensor that defines `base` (`(N, M)`).
- `strides` should be the strides of the base tensor, and are typically just fed as `X.strides(0), X.strides(1), ..., X.strides(n - 1)`. For vectors this is just `(1, )`.
- `order` is a tricky one. In general, you will always set it to `(n - 1, n - 2, ..., 0)`[^4], but an explanation is given in the [block pointer tutorial](https://triton-lang.org/main/getting-started/tutorials/08-experimental-block-pointer.html#sphx-glr-getting-started-tutorials-08-experimental-block-pointer-py) regarding 2D block pointers:
   > Note that the order argument is set to (1, 0), which means the second axis is the inner dimension in terms of storage, and the first axis is the outer dimension. This information may sound redundant, but it is necessary for some hardware backends to optimize for better performance.

As for `offsets` and `block_shape`, these two parameters are interrelated. If we want each program to process an entire row each we would have `offsets=(program_id, )` and `block_shape=(1, M)`.
If we want each program to load a block of `BLOCK_SIZE` rows, we would have `offsets=(program_id * BLOCK_SIZE, )` and `block_shape=(BLOCK_SIZE, M)`.


## 2D Block Pointers

Block pointers can be expanded to higher dimension by expanding shape, strides, offset, block shape and order parameters. 
If we choose a two dimensional grid with program identifiers `pid_x` and `pid_y` and want to process quarters of the matrix, `offsets=(pid_x * (N // 2), pid_y * (M // 2))` and `block_shape=(N // 2, M // 2)`.

...

I'm sure block pointers can be extended to higher dimensions, but I have not tried it yet - most operations are 2D with a batch or sequence dimension added which you parallelize over.

## Advancing Block Pointers

...


[^4]: This is because Triton is row-major, so the last axis is the inner dimension in terms of storage.

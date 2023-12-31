# Manipulating Data
## Loading and Storing Data

With block pointers ready we can start loading data from global memory to much faster shared memory. Triton provides the [`tl.load`](https://triton-lang.org/main/python-api/generated/triton.language.load.html) and [`tl.store`](https://triton-lang.org/main/python-api/generated/triton.language.store.html) functions for this. Since we use block pointers we can ignore most arguments in both loading and storing, but the docs do give us some information:

> pointer could be a block pointer defined by make_block_ptr, in which case:
> - mask and other must be None
> - boundary_check and padding_option can be specified to control the behavior of out-of-bound access 

Ignoring the mask argument, we are left with `boundary_check` and `padding_option`. If `boundary_check` is enabled, out-of-bound memory can be set to a static value using `padding_option`. This approach is unfortunately not as versatile as the non-block pointer approach, since padding options are only `zero` and `nan`. As an example of where this hurts, the softmax tutorial in the documentation uses the old loading approach and here you can simply set out-of-bound values to `-inf`. That's fixed with a simple masking using `tl.where()`

# Getting Started
Before we can start programming some elementary kernels, we need to create a foundation of knowledge on how Triton works. In the following chapters we will discuss how to launch a kernel in Python, how to create a kernel, and commonly used operations inside of kernels. 


## Blocked Algorithms
Triton implements what are called **blocked algorithms**, and most algorithms of interest (this is the Triton assumption) can be written as a blocked algorithm.
Think of matrix multiplication, LU factorization, even element-wise operations can be done in blocks.
We can write the same blocked algorithms in CUDA and in Triton, but it's much easier to optimize a Triton kernel than it is to optimize a CUDA kernel.
This is because Triton depends on the fact that each program being launched will process a block of data instead of scalars (in CUDA) and automatically tunes memory needs accordingly.
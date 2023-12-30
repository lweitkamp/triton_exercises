# Blocked Algorithms
Triton is designed to implement what are called **blocked algorithms**.
Blocking (or tiling) can drastically increase locality of reference for a variety of problems.
It is the de-facto way to perform fast matrix multiplications in CUDA, and it is also an easy way to perform elementwise operations on a large set of values.
Additionally, some other linear algebra calculations such as LU Factorization, SVD(!), and Cholesky Factorization can also be expressed as blocked algorithms, quite nice!


As mentioned, blocked algorithms can also be implemented in CUDA. So how does Triton do it different? It's precisely because Triton is **built** for blocked algorithms
that allow the user to bypass a lot of memory and warp level optimizations or make it trivial to do so by changing key-word arguments during compilation of kernels.

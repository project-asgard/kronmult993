# Kronmult

This library implements the `kronmult_batched` function which computes `output[K] += kron(matrix_list[K]) * input[K]` (k being an index in a batch) 
which is a batch version of the matrix product of the kronecker product of several matrices and a given vector.

We provide efficient parallel implementations, for both CPU (using OpenMP) and GPU (using CUDA), that have been tuned for the needs of [ASGarD](https://github.com/project-asgard/asgard).
In particular, we expect our inputs to be *col-major* matrices and some output pointers to overlap.

# Theory

We implement a variant of the backward version of algorithm 993 ([Algorithm 993: Efficient Computation with Kronecker Products](https://dl.acm.org/doi/abs/10.1145/3291041)), chosen to perform well on col-major matrices.

We highly recommend reading [ON KRONECKER PRODUCTS, TENSOR PRODUCTS AND MATRIX DIFFERENTIAL CALCULUS by Stephen Pollock](https://www.le.ac.uk/economics/research/RePEc/lec/leecon/dp14-02.pdf) to get more familiar with the algebra and reshaping tricks techniques used in the implementation.

## Installation

You can use either the `kronmult_omp` (CPU paralelism) or the `kronmult_gpu` (GPU paralelism) CMake target to link this library.
See the corresponding folders for further information on both instalation and implementations.

## Usage

Include either `kronmult.hpp` (CPU) or `kronmult.cuh` (GPU) to get access to the `kronmult_batched` function.

```cpp
void kronmult_batched(const int matrix_number, const int matrix_size, T const * const matrix_list_batched[], const int matrix_stride,
                      T* input_batched[], T* output_batched[], T* workspace_batched[], const int nb_batch)
```

### Inputs

- `matrix_list_batched` is an array of `nb_batch`*`matrix_count` pointers to square matrices of size `matrix_size` by `matrix_size` and stride `matrix_stride`
- `input_batched` is an array of `nb_batch` pointers to array of size `matrix_size`^`matrix_count`
- `output_batched` is an array of `nb_batch` pointers to array of size `matrix_size`^`matrix_count`, to which the outputs will be added
- `workspace` is an array of `nb_batch` pointers to array of size `matrix_size`^`matrix_count`, to be used as workspaces

### Warnings

- `input_batched` and `workspace_batched` will be used as temporary workspaces and thus modified
- the matrices are assumed to be stored in col-major order
- the sizes are assumed to be correct
- the gpu version assumes that all the arrays have already been allocated **on GPU** (using `cudaMalloc` for example)

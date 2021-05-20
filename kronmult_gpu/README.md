# Kronmult GPU

This GPU version of kronmult is paralelized with CUDA.
It uses one block per batch elements, paralelizing over the size of the input with the block's threads.

We use atomic instructions during the final addition to insure that it is thread-safe.

## Installation

This is a static *CUDA* library, if using CMake, you can link the `kronmult_gpu` target.

You will need CUDA to compile and use this library.

## Usage

Include `kronmult.cuh` to get access to the `kronmult_batched` function 
which computes `output[K] += kron(matrix_list[K]) * input[K]` for 0 <= k < batchCount 
assuming that some of the output pointers will be equal (thus, requiring a thread-safe addition).

```cpp
#include <kronmult.cuh>

void kronmult_batched(const int matrix_number, const int matrix_size, T const * const matrix_list_batched[], const int matrix_stride,
                      T* input_batched[], T* output_batched[], T* workspace_batched[], const int nb_batch)
```

### Inputs

- `matrix_list_batched` is an array of `nb_batch`*`matrix_count` pointers to square matrices of size `matrix_size` by `matrix_size` and stride `matrix_stride`
- `input_batched` is an array of `nb_batch` pointers to array of size `matrix_size`^`matrix_count`
- `output_batched` is an array of `nb_batch` pointers to array of size `matrix_size`^`matrix_count`, to which the outputs will be added
- `workspace` is an array of `nb_batch` pointers to array of size `matrix_size`^`matrix_count`, to be used as workspaces

### Warnings

 - **we assume that all the arrays have already been allocated *on GPU* (using `cudaMalloc` for example)**
 - `input_batched` and `workspace_batched` will be used as temporary workspaces and thus modified
 - the matrices are assumed to be stored in col-major order
 - the sizes are assumed to be correct
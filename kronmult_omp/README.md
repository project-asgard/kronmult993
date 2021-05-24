# Kronmult CPU

This CPU version of kronmult is paralelized with OpenMP and can use BLAS if available.

OpenMP is used to paralelise over batch elements. We use atomic instructions during the final addition to insure that it
is thread-safe.

BLAS is used as a faster substitute to our matrix product implementation in float and double precision.

## Installation

This is a *header-only* library, including this folder should be enough to get it working. If using CMake, you can link
the `kronmult_omp` target.

To use OpenMP you just need to pass the usual flags (and link targets) to your compiler.

To use BLAS, link a BLAS implementation of your choice (our tests were done with the Intel MKL library but any
implementation should work)
and pass the `KRONMULT_USE_BLAS` flag to your compiler.

## Usage

Include `kronmult.hpp` to get access to the `kronmult_batched` function which
computes `output[K] += kron(matrix_list[K]) * input[K]` for 0 <= k < batchCount assuming that some output pointers will
be equal (thus, requiring a thread-safe addition).

```cpp
#include <kronmult.hpp>

void kronmult_batched(int const matrix_number, int const matrix_size, T const * const matrix_list_batched[], int const matrix_stride,
                      T* input_batched[], T* output_batched[], T* workspace_batched[], int const nb_batch)
```

### Inputs

- `matrix_list_batched` is an array of `nb_batch`*`matrix_count` pointers to square matrices of size `matrix_size`
  by `matrix_size` and stride `matrix_stride`
- `input_batched` is an array of `nb_batch` pointers to array of size `matrix_size`^`matrix_count`
- `output_batched` is an array of `nb_batch` pointers to array of size `matrix_size`^`matrix_count`, to which the
  outputs will be added
- `workspace` is an array of `nb_batch` pointers to array of size `matrix_size`^`matrix_count`, to be used as workspaces

### Warnings

- `input_batched` and `workspace_batched` will be used as temporary workspaces and thus modified
- the matrices are assumed to be stored in col-major order
- the sizes are assumed to be correct
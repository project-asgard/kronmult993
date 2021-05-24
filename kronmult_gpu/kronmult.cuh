#pragma once
#include <cuda_runtime.h>

/*
 * computes number^power for integers
 * does not care about performances
 * does not use std::pow as it does an implicit float conversion that could lead to rounding errors for large
 * numbers
 */
__host__ int pow_int(int const number, int const power);

/*
 * Computes output[K] += kron(matrix_list[K]) * input[K] for 0 <= k < batchCount
 * assuming that some of the output pointers will be equal requiring a thread-safe addition
 *
 * `matrix_list_batched` is an array of `nb_batch`*`matrix_count` pointers to square matrices of size
 * `matrix_size` by `matrix_size` and stride `matrix_stride` `input_batched` is an array of `nb_batch`
 * pointers to array of size `matrix_size`^`matrix_count` `output_batched` is an array of `nb_batch` pointers
 * to array of size `matrix_size`^`matrix_count`, to which the outputs will be added `workspace` is an array
 * of `nb_batch` pointers to array of size `matrix_size`^`matrix_count`, to be used as workspaces
 *
 * WARNINGS:
 * - we assume that all the arrays have already been allocated *on GPU* (using `cudaMalloc` for example)
 * - `input_batched` and `workspace_batched` will be used as temporary workspaces and thus modified
 * - the matrices are assumed to be stored in col-major order
 * - the sizes are assumed to be correct
 */
template<typename T>
__host__ cudaError kronmult_batched(int const matrix_count, int const matrix_size,
                                    T const *const matrix_list_batched[], int const matrix_stride,
                                    T *input_batched[], T *output_batched[], T *workspace_batched[],
                                    int const nb_batch);

#pragma once
#include <cuda.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include "linear_algebra.cuh"

/*
 * computes number^power for integers
 * does not care about performances
 * does not use std::pow as it does an implicit float conversion that could lead to rounding errors for high
 * numbers
 */
__host__ int pow_int(const int number, const int power);

/*
 * Computes output[K] += kron(matrix_list[K]) * input[K] for 0 <= k < batchCount
 *
 * `matrix_list_batched` is an array of `nb_batch`*`matrix_count` pointers to square matrices of size `matrix_size` by `matrix_size` and stride `matrix_stride`
 * `input_batched` is an array of `nb_batch` vectors of size `matrix_size`^`matrix_count`
 * `output_batched` is an array of `nb_batch` vectors of size `matrix_size`^`matrix_count`, where the outputs will be stored
 * `workspace` is an array of `nb_batch` vectors of size `matrix_size`^`matrix_count`, to be used as workspaces
 *
 * WARNINGS:
 * `input_batched` and `workspace_batched` will be used as temporary workspaces and thus modified
 * the matrices should be stored in col-major order
 * the inputs should already be on the GPU.
 */
template<typename T>
__host__ cudaError kronmult_batched(const int matrix_count, const int matrix_size, T const * const matrix_list_batched[], const int matrix_stride,
                                    T* input_batched[], T* output_batched[], T* workspace_batched[], const int nb_batch);

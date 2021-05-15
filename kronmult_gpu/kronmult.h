#pragma once
#include "kronmult.cuh"

template<typename T>
void kronmult_batched(const int matrix_count, const int matrix_size, T const * const matrix_list_batched[], const int matrix_stride,
                      T* input_batched[],
                      T* output_batched[], T* workspace_batched[],
                      const int nb_batch)
{
    cuda_kronmult_batched<T><<<1,1>>>(matrix_count, matrix_size, matrix_list_batched, matrix_stride,
                                      input_batched, output_batched, workspace_batched, nb_batch);
}
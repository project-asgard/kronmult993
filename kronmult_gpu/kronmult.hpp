#pragma once
#include <cuda.h>

cudaError kronmult_batched(const int matrix_count, const int matrix_size, T const * const matrix_list_batched[], const int matrix_stride,
                                    T* input_batched[], T* output_batched[], T* workspace_batched[], const int nb_batch);

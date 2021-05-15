#include "kronmult.cuh"

// declarations to ensure that the templates specializations will be compiled be CUDA
template<>
__global__ void cuda_kronmult_batched<float>(const int matrix_count, const int matrix_size, float const * const matrix_list_batched[], const int matrix_stride,
                                             float* input_batched[], float* output_batched[],float* workspace_batched[], const int nb_batch);
template<>
__global__ void cuda_kronmult_batched<double>(const int matrix_count, const int matrix_size, double const * const matrix_list_batched[], const int matrix_stride,
                                              double* input_batched[], double* output_batched[], double* workspace_batched[], const int nb_batch);
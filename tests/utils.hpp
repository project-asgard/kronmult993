#pragma once
#define USE_GPU
#ifdef USE_GPU
// Generate random numbers with Cuda
#include <curand.h>
// CUDA runtime
#include <cuda_runtime.h>
// Helper functions and utilities to work with CUDA
//#include <helper_cuda.h>
#endif
#include <cstdlib>
#include <iostream>
#include <openmp/linear_algebra.hpp>
namespace utils
{
#ifdef USE_GPU
template <typename T>
void  random_init_flatarray_device(T *A, size_t size);

template <typename T>
void init_array_pointer_kernel(T ** X_p, T* X,
                               size_t outer_size, size_t inner_size);

template <typename T>
__host__ __device__ void init_array_pointer(T ** X_p, T* X,
                                            size_t outer_size, size_t inner_size);

template<typename T>
void initialize_pointers_device( T *** matrix_list_batched_pp, T *** input_batched_pp, T *** output_batched_pp,
                                 T *** workspace_batched_pp, size_t batch_count, size_t matrix_count, size_t dimensions,
                                 size_t size_input, size_t matrix_size, size_t matrix_stride, size_t grid_level);
#else
template <typename T>
void init_array_pointer(T ** X_p, T* X,
                        size_t outer_size, size_t inner_size);

template <typename T>
void initialize_pointers_host( T *** matrix_list_batched_p, T *** input_batched_p, T *** output_batched_p,
                               T *** workspace_batched_p, size_t batch_count, size_t matrix_count, size_t dimensions,
                               size_t size_input, size_t matrix_size, size_t matrix_stride, size_t grid_level);
#endif

void display_debug(size_t degree, size_t size_input, size_t matrix_stride, size_t dimensions,
                   size_t grid_level, size_t batch_count)
{
    std::cerr
        << "Square Matrix Size (skinny) == Degree: " << degree
        << " Tall matrix size == size input: " << size_input
        << " Coefficient matrix stride: " << matrix_stride
        << " Matrix count in kronmult == Dimensions: " << dimensions
        << " grid level: " << grid_level
        << " batch count: " << batch_count
        << std::endl;
}

}
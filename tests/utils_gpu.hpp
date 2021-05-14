#pragma once
#include <cstdlib>
#include <iostream>
#include <gpu/linear_algebra.hpp>

template <typename T>
__host__
void  random_init_flatarray_device(T *A, size_t size);

template <typename T>
__global__
void init_array_pointer_kernel(T ** X_p, T* X,
        size_t outer_size, size_t inner_size);

template <typename T>
__host__
void init_array_pointer(T ** X_p, T* X,
        size_t outer_size, size_t inner_size);

template<typename T>
__host__
void initialize_pointers_device( T *** matrix_list_batched_pp);
       // , T *** input_batched_pp, T *** output_batched_pp,
       // T *** workspace_batched_pp, size_t batch_count, size_t matrix_count, size_t dimensions,
       // size_t size_input, size_t matrix_size, size_t matrix_stride, size_t grid_level);
void test(void);

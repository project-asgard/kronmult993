#pragma once
#include <cstdlib>
#include <iostream>
#include <openmp/linear_algebra.hpp>
namespace utils
{
#include "utils_common.hpp"
    template <typename T>
        void init_array_pointer(T ** X_p, T* X,
                size_t outer_size, size_t inner_size);

    template <typename T>
        void initialize_pointers_host( T *** matrix_list_batched_p, T *** input_batched_p, T *** output_batched_p,
                T *** workspace_batched_p, size_t batch_count, size_t matrix_count, size_t dimensions,
                size_t size_input, size_t matrix_size, size_t matrix_stride, size_t grid_level);

}

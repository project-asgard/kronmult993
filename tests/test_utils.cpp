#include "utils.hpp"
#include <cassert>
#include <cstdlib>

template <typename T>
void test_utils()
{
    T ** matrix_list_batched_p;
    T ** input_batched_p;
    T ** output_batched_p;
    T ** workspace_batched_p;
    size_t batch_count = 1023;
    size_t matrix_count = 4;
    size_t dimensions = 4;
    size_t size_input = 2048;
    size_t matrix_size = 8;
    size_t matrix_stride = 8;
    size_t grid_level = 4;
#ifndef USE_GPU
    utils::initialize_pointers_host( &matrix_list_batched_p, &input_batched_p, &output_batched_p,
                                     &workspace_batched_p, batch_count, matrix_count, dimensions,
                                     size_input, matrix_size, matrix_stride, grid_level);
#else
    utils::initialize_pointers_device( &matrix_list_batched_p, &input_batched_p, &output_batched_p,
                                     &workspace_batched_p, batch_count, matrix_count, dimensions,
                                     size_input, matrix_size, matrix_stride, grid_level);
#endif
    assert(matrix_list_batched_p != nullptr);
    assert(input_batched_p != nullptr);
    assert(output_batched_p != nullptr);
    assert(workspace_batched_p != nullptr);
}
int main(int ac, char * av[]){
    test_utils<double>();
    return 0;
}

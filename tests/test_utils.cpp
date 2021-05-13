#include "utils.hpp"

template <typename T>
void test_utils()
{
    T *** matrix_list_batched_p;
    T *** input_batched_p;
    T *** output_batched_p;
    T *** workspace_batched_p;
    size_t batch_count;
    size_t matrix_count;
    size_t dimensions;
    size_t s;

    initialize_pointers_device( T *** matrix_list_batched_p, T *** input_batched_p, T *** output_batched_p,
                          T *** workspace_batched_p, size_t batch_count, size_t matrix_count, size_t dimensions,
                         size_t size_input, size_t matrix_size, size_t matrix_stride, size_t grid_level)
}
int main(int ac, char * av[]){
    test_utils<double>();
    return 0;
}

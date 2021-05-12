#include <stdexcept>
#include "xbatched/xbatched.hpp"

/*
 * wrapper on top of the various kronmult implementations
 */
void kronmult_batched(const int matrix_count, const int matrix_size, double const * const matrix_list_batched[], const int matrix_stride,
                      double* input_batched[],
                      double* output_batched[], double* workspace_batched[],
                      const int nb_batch)
{
    switch(matrix_count)
    {
    case 1:
        kronmult1_xbatched(matrix_size, matrix_list_batched, matrix_stride, input_batched, output_batched, workspace_batched, nb_batch);
        break;
    case 2:
        kronmult2_xbatched(matrix_size, matrix_list_batched, matrix_stride, input_batched, output_batched, workspace_batched, nb_batch);
        break;
    case 3:
        kronmult3_xbatched(matrix_size, matrix_list_batched, matrix_stride, input_batched, output_batched, workspace_batched, nb_batch);
        break;
    case 4:
        kronmult4_xbatched(matrix_size, matrix_list_batched, matrix_stride, input_batched, output_batched, workspace_batched, nb_batch);
        break;
    case 5:
        kronmult5_xbatched(matrix_size, matrix_list_batched, matrix_stride, input_batched, output_batched, workspace_batched, nb_batch);
        break;
    case 6:
        kronmult6_xbatched(matrix_size, matrix_list_batched, matrix_stride, input_batched, output_batched, workspace_batched, nb_batch);
        break;
    default:
        throw std::runtime_error("Invalid number of matrices passed to `kronmult_batched`!");
    }
}
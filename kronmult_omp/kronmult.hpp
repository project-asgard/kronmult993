#pragma once
#include "linear_algebra.hpp"

/*
 * computes number^power for integers
 * does not care about performances
 * does not use std::pow as it does an implicit float conversion that could lead to rounding errors for large numbers
 */
int pow_int(const int number, const int power)
{
    if(power == 0) return 1;
    return number * pow_int(number, power-1);
}

/*
 * Computes output += kron(matrix_list) * input while insuring that the addition to output is thread-safe
 *
 * `matrix_list` is an array containing pointers to `matrix_number` square matrices of size `matrix_size` by `matrix_size` and stride `matrix_stride`
 * `input` is a `size_input` (`matrix_size`^`matrix_number`) elements vector
 * `output` is a `size_input` elements vector, to which the output of the multiplication will be added
 * `workspace` is a `size_input` elements vector, to be used as workspace
 * `transpose_workspace` is a vector of size `matrix_size`*`matrix_size` to store transposed matrices temporarily
 *
 * WARNINGS:
 * - `input` and `workspace` will be used as temporary workspaces and thus modified
 * - the matrices are assumed to be stored in col-major order
 * - the sizes are assumed to be correct
 */
template<typename T>
void kronmult(const int matrix_count, const int matrix_size, T const * const matrix_list[], const int matrix_stride,
              T input[], const int size_input,
              T output[],
              T workspace[], T transpose_workspace[])
{
    // how many column should `input` have for the multiplications to be legal
    const int nb_col_input = size_input / matrix_size;

    // iterates on the matrices from last to first
    for(int i = matrix_count-1; i >= 0; i--)
    {
        // takes `matrix` into account and put the result in `workspace`
        T const * const matrix = matrix_list[i];
        multiply_transpose<T>(input, nb_col_input, matrix, matrix_size, matrix_stride, workspace, transpose_workspace);
        // swap `input` and `workspace` such that `input` contains, once again, the input
        // note that, while they have the same size flattened, the shapes (numbers of columns and rows) of `input` and `workspace` are different
        // this is on purpose and equivalent to a reshape operation that is actually needed by the algorithm
        std::swap(input, workspace);
    }

    // adds to input in a thread-safe way
    for(int i = 0; i < size_input; i++)
    {
        #pragma omp atomic
        output[i] += input[i];
    }
}

/*
 * Computes output[K] += kron(matrix_list[K]) * input[K] for 0 <= k < batchCount
 * assuming that some of the output pointers will be equal requiring a thread-safe addition
 *
 * `matrix_list_batched` is an array of `nb_batch`*`matrix_count` pointers to square matrices of size `matrix_size` by `matrix_size` and stride `matrix_stride`
 * `input_batched` is an array of `nb_batch` pointers to array of size `matrix_size`^`matrix_count`
 * `output_batched` is an array of `nb_batch` pointers to array of size `matrix_size`^`matrix_count`, to which the outputs will be added
 * `workspace` is an array of `nb_batch` pointers to array of size `matrix_size`^`matrix_count`, to be used as workspaces
 *
 * WARNINGS:
 * - `input_batched` and `workspace_batched` will be used as temporary workspaces and thus modified
 * - the matrices are assumed to be stored in col-major order
 * - the sizes are assumed to be correct
 */
template<typename T>
void kronmult_batched(const int matrix_count, const int matrix_size, T const * const matrix_list_batched[], const int matrix_stride,
                      T* input_batched[],
                      T* output_batched[], T* workspace_batched[],
                      const int nb_batch)
{
    // numbers of elements in the input vector
    int size_input = pow_int(matrix_size, matrix_count);

    // paralelize over batch elements
    #pragma omp parallel
    {
        // workspace that will be used to store matrix transpositions
        // only one, allocated once, per thread
        T* transpose_workspace = new T[matrix_size*matrix_size];

        // computes kronmult for all batch elements
        #pragma omp for
        for(int i=0; i < nb_batch; i++)
        {
            T const * const * matrix_list = &matrix_list_batched[i*matrix_count];
            T* input = input_batched[i];
            T* output = output_batched[i];
            T* workspace = workspace_batched[i];
            kronmult<T>(matrix_count, matrix_size, matrix_list, matrix_stride, input, size_input, output, workspace, transpose_workspace);
        }

        delete[] transpose_workspace;
    }
}

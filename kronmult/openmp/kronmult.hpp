#pragma once
#include "linear_algebra.hpp"

/*
 * computes number^power for integers
 * does not care about performances
 * does not use std::pow as it does an implicit float conversion that could lead to rounding errors for high numbers
 */
int pow_int(const int number, const int power)
{
    if(power == 0) return 1;
    return number * pow_int(number, power-1);
}

/*
 * Computes output[K] += kron(matrix_list[K]) * input[K] for 0 <= k < batchCount
 *
 * `matrix_list_batched` is an array of `nb_batch`*`matrix_count` pointers to square matrices of size `matrix_size` by `matrix_size` and stride `matrix_stride`
 * `input_batched` is an array of `nb_batch` vectors of size `matrix_size`^`matrix_count`
 * `output_batched` is an array of `nb_batch` vectors of size `matrix_size`^`matrix_count`, where the outputs will be stored
 * `workspace` is an array of `nb_batch` vectors of size `matrix_size`^`matrix_count`, to be used as workspaces
 *
 * WARNINGS:
 * `input_batched` and `workspace_batched` will be used as temporary workspaces and thus modified
 * the matrices should be stored in col-major order
 */
template<typename T>
void kronmult_batched(const int matrix_count, const int matrix_size, T const * const matrix_list_batched[], const int matrix_stride,
                      T* input_batched[],
                      T* output_batched[], T* workspace_batched[],
                      const int nb_batch)
{
    // numbers of elements in the input vector
    int size_input = pow_int(matrix_size, matrix_count);
    // how many column should `input` have for the multiplications to be legal
    const int nb_col_input = size_input / matrix_size;

    // put constants in array format
    T const* * matrix_batch_listed = new T const*[nb_batch*matrix_count];
    char* should_transpose_input_batched = new char[nb_batch];
    char* should_transpose_matrix_batched = new char[nb_batch];
    int* nb_col_input_batched = new int[nb_batch];
    int* matrix_size_batched = new int[nb_batch];
    int* matrix_stride_batched = new int[nb_batch];
    #pragma omp parallel for
    for(int b=0; b < nb_batch; b++)
    {
        for(int m = 0; m < matrix_count; m++)
        {
            matrix_batch_listed[nb_batch*m + b] = matrix_list_batched[b*matrix_count + m];
        }
        should_transpose_input_batched[b] = 'T';
        should_transpose_matrix_batched[b] = 'T';
        nb_col_input_batched[b] = nb_col_input;
        matrix_size_batched[b] = matrix_size;
        matrix_stride_batched[b] = matrix_stride;
    }

    // iterates on the matrices from the last to the one just before first
    // puts the result in input_batched
    for(int m = matrix_count-1; m >= 0; m--)
    {
        T const** matrix_batched = &matrix_batch_listed[nb_batch*m];
        T const** input_batch_const = const_cast<T const**>(input_batched);
        multiply_transpose_batched<T>(input_batch_const, nb_col_input_batched, should_transpose_input_batched,
                                      matrix_batched, matrix_size_batched, matrix_stride_batched, should_transpose_matrix_batched,
                                      workspace_batched, nb_batch);
        std::swap(input_batched, workspace_batched);
    }

    // free the memory
    delete[] matrix_batch_listed;
    delete[] should_transpose_input_batched;
    delete[] should_transpose_matrix_batched;
    delete[] nb_col_input_batched;
    delete[] matrix_size_batched;
    delete[] matrix_stride_batched;

    // reduction
    #pragma omp parallel for
    for(int b=0; b < nb_batch; b++)
    {
        T* input = input_batched[b];
        T* output = output_batched[b];
        // reduce in a threadsafe way
        for(int i = 0; i < size_input; i++)
        {
            #pragma omp atomic
            output[i] += input[i];
        }
    }
}

#pragma once
#include "gpu_operations.hpp"
#include <iostream>

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
 * Computes Y = X^T * M^T
 *      <=> Y[i,j] = X[k,i] * M[j,k]
 *
 * M is a `size_M` by `size_M` matrix of stride `matrix_stride`
 * X is a `size_M` by `nb_col_X` matrix
 * Y is a `nb_col_X` by `size_M` matrix
 * should_zero_Y (template argument to eliminate the test at compile time) decides whether the algorithm will add the result to Y or overwrite it
 *
 * WARNING: the matrices should be stored in col-major order
 * NOTE: we assume that `nb_col_X` is very large compared to `size_M`
 *
 * TODO transposing M in advance might be cheaper as it leads to better alignement, we would need  workspace to store it
 */
template<typename T, bool should_zero_Y>
GLOBAL_FUNCTION void multiply_transpose(const T X[], const int nb_col_X,
                                        const T M[], const int size_M, const int stride_M,
                                        T Y[])
{
    #define colmajor(row, col, nb_rows) ((row) + (col) * (nb_rows))

    // this first loop has much more iterations than the inner loops
    #ifndef USE_GPU
    #pragma omp parallel for
    #endif
    for(int colX=0; colX < nb_col_X; colX++)
    {
        for(int rowM=0; rowM < size_M; rowM++)
        {
            if(should_zero_Y)
            {
                Y[colmajor(colX,rowM,nb_col_X)] = 0.;
            }
            for(int k=0; k < size_M; k++)
            {
                Y[colmajor(colX,rowM,nb_col_X)] += X[colmajor(k,colX,size_M)] * M[colmajor(rowM,k,stride_M)];
            }
        }
    }

    #undef colmajor
}

/*
 * Computes output += kron(matrix_list) * input
 *
 * `matrix_list` is an array containing pointers to `matrix_number` square matrices of size `matrix_size` by `matrix_size` and stride `matrix_stride`
 * `input` is a `matrix_size`^`matrix_number` elements vector
 * `output` is a `matrix_size`^`matrix_number` elements vector, where the output will be stored
 * `workspace` is a `matrix_size`^`matrix_number` elements vector, to be used as workspace
 *
 * WARNING: `input` and `workspace` will be used as temporary workspaces and thus modified
 */
template<typename T>
GLOBAL_FUNCTION void kronmult(const int matrix_number, const int matrix_size, T const * const matrix_list[], const int matrix_stride,
                              T input[],
                              T output[], T workspace[])
{
    // how many column should `input` have for the multiplications to be legal
    const int nb_col_input = pow_int(matrix_size, matrix_number - 1);

    // iterates on the matrices from the last to the one just before first
    for(int i = matrix_number-1; i >= 1; i--)
    {
        // takes `matrix` into account and put the result in `workspace` (use `output` as a workspace if needed)
        T const * const matrix = matrix_list[i];
        multiply_transpose<T, true>(input, nb_col_input, matrix, matrix_size, matrix_stride, workspace);
        // swap `input` and `workspace` such that `input` contains once again the input
        // note that, while they have the same size flattened, the shape (nb_columns and nb_rows) of `input` and `workspace` are different
        // this is on purpose and equivalent to a reshape operation that is actually needed by the algorithm
        std::swap(input, workspace);
    }

    // puts the final result in `output` rather than `workspace`
    // the result is *added* to output (rather than zeroing it out)
    T const * const matrix = matrix_list[0];
    multiply_transpose<T, false>(input, nb_col_input, matrix, matrix_size, matrix_stride, output);
}

/*
 * Computes output[K] += kron(matrix_list[K]) * input[K] for 0 <= k < batchCount
 *
 * `matrix_list_batched` is an array of `nb_batch`*`matrix_number` pointers to square matrices of size `matrix_size` by `matrix_size` and stride `matrix_stride`
 * `input_batched` is an array of `nb_batch` vectors of size `matrix_size`^`matrix_number`
 * `output_batched` is an array of `nb_batch` vectors of size `matrix_size`^`matrix_number`, where the outputs will be stored
 * `workspace` is an array of `nb_batch` vectors of size `matrix_size`^`matrix_number`, to be used as workspaces
 *
 * WARNING: `input_batched` and `workspace_batched` will be used as temporary workspaces and thus modified
 */
template<typename T>
GLOBAL_FUNCTION void kronmult_batched(const int matrix_number, const int matrix_size, T const * const matrix_list_batched[], const int matrix_stride,
                                      T* input_batched[],
                                      T* output_batched[], T* workspace_batched[],
                                      const int nb_batch)
{
    // runs kronmult one batch at a time
    // TODO we could remove this loop with batched matrix multiplications
    for(int i=0; i < nb_batch; i++)
    {
        T const * const * matrix_list = &matrix_list_batched[i*matrix_number];
        T* input = input_batched[i];
        T* output = output_batched[i];
        T* workspace = workspace_batched[i];
        kronmult<T>(matrix_number, matrix_size, matrix_list, matrix_stride, input, output, workspace);
    }
}

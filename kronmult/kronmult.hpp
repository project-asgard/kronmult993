#pragma once

#include "gpu_operations.hpp"

/*
 * Computes Y = X^T * M^T
 *      <=> Y[i,j] = X[k,i] * M[j,k]
 *
 * M is a `size_M` by `size_M` matrix
 * X is a `size_M` by `nb_col_X` matrix
 * Y is a `nb_col_X` by `size_M` matrix
 * workspace is a `size_M`*`nb_col_X` vector that can be sued to store intermediate values if needed
 *
 * WARNING: the matrices should be stored in col-major order
 * NOTE: we assume that `nb_col_X` is very large compared to `size_M`
 *
 * TODO transposing M in advance might be cheaper as it leads to better alignement, we could store the transpose in workspace
 */
template<typename T>
GLOBAL_FUNCTION void multiply_transpose(const T X[], const unsigned int nb_col_X,
                                        const T M[], const unsigned int size_M,
                                        T Y[], T workspace[])
{
    #define colmajor(row, col, nb_rows) ((row) + (col) * (nb_rows))

    // this first loop has much more iterations than the inner loops
    #ifndef USE_GPU
    #pragma omp parallel for
    #endif
    for(unsigned int colX=0; colX < nb_col_X; colX++)
    {
        for(unsigned int rowM=0; rowM < size_M; rowM++)
        {
            Y[colmajor(colX,rowM,nb_col_X)] = 0.;
            for(unsigned int k=0; k < size_M; k++)
            {
                Y[colmajor(colX,rowM,nb_col_X)] += X[colmajor(k,colX,size_M)] * M[colmajor(rowM,k,size_M)];
            }
        }
    }

    #undef colmajor
}

/*
 * Computes output = kron(matrix_list) * input
 *
 * `matrix_list` is an array containing pointers to `matrix_number` square matrices of size `matrix_size` by `matrix_size`
 * `input` is a `nb_elements_input` vector (we expect `nb_elements_input` to be matrix_size^matrix_number)
 * `output` is a `nb_elements_input` vector, where the output will be stored
 * `workspace` is a `nb_elements_input` vector, to be used as workspace
 *
 * WARNING: `input` and `workspace` will be used as temporary workspaces and thus modified
 */
template<typename T>
GLOBAL_FUNCTION void kronmult(const unsigned int matrix_number, const unsigned int matrix_size, T const* const matrix_list[],
                              const unsigned int nb_elements_input, T input[],
                              T output[], T workspace[])
{
    // how many column should `input` have for the multiplications to be legal
    const unsigned int nb_col_input = nb_elements_input / matrix_size; // matrix_size^(matrix_number - 1)

    // iterates on the matrices from the last to the one just before first
    for(unsigned int i = matrix_number-1; i >= 1; i--)
    {
        // takes `matrix` into account and put the result in `workspace` (use output as a workspace)
        const T matrix[] = matrix_list[i];
        multiply_transpose<T>(input, nb_col_input, matrix, matrix_size, workspace, output);
        // swap `input` and `workspace` such that `input` contains once again the input
        std::swap(input, workspace);
    }

    // puts the final result in `output` rather than `workspace`
    const T matrix[] = matrix_list[0];
    multiply_transpose<T>(input, nb_col_input, matrix, matrix_size, output, workspace);
}

/*
 * Computes output[K] = kron(matrix_list[K]) * input[K] for 0 <= k < batchCount
 *
 * `matrix_list_batched` is an array of `nb_batch`*`matrix_number` pointers to square matrices of size `matrix_size` by `matrix_size`
 * `input_batched` is an array of `nb_batch` vectors of size `nb_elements_input` (we expect `nb_elements_input` to be matrix_size^matrix_number)
 * `output_batched` is an array of `nb_batch` vectors of size `nb_elements_input`, where the outputs will be stored
 * `workspace` is an array of `nb_batch` vectors of size `nb_elements_input`, to be used as workspaces
 *
 * WARNING: `input_batched` and `workspace_batched` will be used as temporary workspaces and thus modified
 */
template<typename T>
GLOBAL_FUNCTION void kronmult_batched(const unsigned int matrix_number, const unsigned int matrix_size, T const* const matrix_list_batched[],
                                      const unsigned int nb_elements_input, T* input_batched[],
                                      T* output_batched[], T* workspace_batched[],
                                      const unsigned int nb_batch)
{
    // runs kronmult one batch at a time
    // TODO we could remove this loop with batched matrix multiplications
    for(unsigned int i=0; i < nb_batch; i++)
    {
        T const *const matrix_list = matrix_list_batched[i*matrix_number];
        const T* input = input_batched[i];
        const T* output = output_batched[i];
        const T* workspace = workspace_batched[i];
        kronmult<T>(matrix_number, matrix_size, matrix_list, nb_elements_input, input, output, workspace);
    }
}

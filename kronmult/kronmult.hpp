#pragma once
#include "gpu_operations.hpp"
#include "omp_reduction.hpp"

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
 * transpose a square matrix with a given stride into the given output matrix (assume to have a stride of matrix_size)
 * the matrix is assumed to be in column major orientation
 */
template<typename T>
GLOBAL_FUNCTION void transpose(const T input[], T output[], const int matrix_size, const int input_stride)
{
    #define colmajor(row, col, nb_rows) ((row) + (col) * (nb_rows))

    for(int r = 0; r < matrix_size; r++)
    {
        for(int c = 0; c < matrix_size; c++)
        {
            output[colmajor(r, c, matrix_size)] = input[colmajor(c, r, input_stride)];
        }
    }

    #undef colmajor
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
 */
template<typename T>
GLOBAL_FUNCTION void multiply_transpose(const T X[], const int nb_col_X,
                                        const T M[], const int size_M, const int stride_M,
                                        T Y[])
{
    #define colmajor(row, col, nb_rows) ((row) + (col) * (nb_rows))

    // transpose the matrix to get a better alignement
    // TODO would be more efficient if we recycled the transposition workspace between multiplications
    // TODO would be more efficient if asgard feed us transposed matrices
    T* M_transposed = new T[size_M*size_M];
    transpose(M, M_transposed, size_M, stride_M);

    for(int rowM=0; rowM < size_M; rowM++)
    {
        const T* M_transposed_col = &M_transposed[colmajor(0,rowM,size_M)];
        for(int colX=0; colX < nb_col_X; colX++)
        {
            const T* X_transposed_row = &X[colmajor(0,colX,size_M)];
            T dotprod = 0.;
            #pragma omp simd reduction(+:dotprod)
            for(int k=0; k < size_M; k++)
            {
                dotprod += X_transposed_row[k] * M_transposed_col[k];
            }
            Y[colmajor(colX,rowM,nb_col_X)] = dotprod;
        }
    }

    // free the workspace memory
    delete[] M_transposed;

    #undef colmajor
}

/*
 * Computes kron(matrix_list) * input
 * Returns a pointeur to an array containing the result (`workspace` if `matrix_number` is odd and `input` if it is even)
 *
 * `matrix_list` is an array containing pointers to `matrix_number` square matrices of size `matrix_size` by `matrix_size` and stride `matrix_stride`
 * `input` is a `size_input`=`matrix_size`^`matrix_number` elements vector
 * `output` is a `size_input` elements vector, where the output will be stored
 * `workspace` is a `size_input` elements vector, to be used as workspace
 *
 * WARNINGS:
 * `input` and `workspace` will be used as temporary workspaces and thus modified
 * the matrices should be stored in col-major order
 */
template<typename T>
GLOBAL_FUNCTION T* kronmult(const int matrix_number, const int matrix_size, T const * const matrix_list[], const int matrix_stride,
                            T input[], const int size_input,
                            T workspace[])
{
    // how many column should `input` have for the multiplications to be legal
    const int nb_col_input = size_input / matrix_size;

    // iterates on the matrices from the last to the one just before first
    for(int i = matrix_number-1; i >= 0; i--)
    {
        // takes `matrix` into account and put the result in `workspace` (use `output` as a workspace if needed)
        T const * const matrix = matrix_list[i];
        multiply_transpose<T>(input, nb_col_input, matrix, matrix_size, matrix_stride, workspace);
        // swap `input` and `workspace` such that `input` contains once again the input
        // note that, while they have the same size flattened, the shape (nb_columns and nb_rows) of `input` and `workspace` are different
        // this is on purpose and equivalent to a reshape operation that is actually needed by the algorithm
        std::swap(input, workspace);
    }

    return input;
}

/*
 * Computes output[K] += kron(matrix_list[K]) * input[K] for 0 <= k < batchCount
 *
 * `matrix_list_batched` is an array of `nb_batch`*`matrix_number` pointers to square matrices of size `matrix_size` by `matrix_size` and stride `matrix_stride`
 * `input_batched` is an array of `nb_batch` vectors of size `matrix_size`^`matrix_number`
 * `output_batched` is an array of `nb_batch` vectors of size `matrix_size`^`matrix_number`, where the outputs will be stored
 * `workspace` is an array of `nb_batch` vectors of size `matrix_size`^`matrix_number`, to be used as workspaces
 *
 * WARNINGS:
 * `input_batched` and `workspace_batched` will be used as temporary workspaces and thus modified
 * the matrices should be stored in col-major order
 */
template<typename T>
GLOBAL_FUNCTION void kronmult_batched(const int matrix_number, const int matrix_size, T const * const matrix_list_batched[], const int matrix_stride,
                                      T* input_batched[],
                                      T* output_batched[], T* workspace_batched[],
                                      const int nb_batch)
{
    // numbers of elements in the input vector
    int size_input = pow_int(matrix_size, matrix_number);

    // runs kronmult one batch at a time
    #ifndef USE_GPU
    #pragma omp parallel for
    #endif
    for(int i=0; i < nb_batch; i++)
    {
        // computes kronmult
        T const * const * matrix_list = &matrix_list_batched[i*matrix_number];
        T* input = input_batched[i];
        T* workspace = workspace_batched[i];
        // result is stored in `workspace` if `matrix_number` is odd and `input` if it is even
        kronmult<T>(matrix_number, matrix_size, matrix_list, matrix_stride, input, size_input, workspace);
    }

    // adds the results to the outputs in a threadsafe way
    T** intermediate_output_batched = (matrix_number % 2 == 0) ? input_batched : workspace_batched;
    reduction<T>(nb_batch, intermediate_output_batched , output_batched, size_input);
}

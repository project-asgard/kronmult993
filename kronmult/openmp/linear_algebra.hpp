#pragma once

#include <mkl_blas.h>

/*
 * converts row and col indices into a single index for a matrix store in col-major
 * `stride` is usually the number of rows of the matrix
 */
constexpr int colmajor(const int row, const int col, const int stride)
{
    return row + col*stride;
}

/*
 * Computes Y = X^T * M^T
 *      <=> Y[i,j] = X[k,i] * M[j,k]
 *
 * X is a `size_M` by `nb_col_X` matrix
 * M is a `size_M` by `size_M` matrix of stride `matrix_stride`
 * Y is a `nb_col_X` by `size_M` matrix
 * M_transposed is a `size_M` by `size_M` matrix of stride `size_M` to store M^T temporarily
 *
 * WARNING: the matrices are assumed to be stored in col-major order
 */
template<typename T>
void multiply_transpose(const T X[], const int nb_col_X,
                        const T M[], const int size_M, const int stride_M,
                        T Y[])
{
    for(int colX=0; colX < nb_col_X; colX++)
    {
        for(int rowM=0; rowM < size_M; rowM++)
        {
            T dotprod = 0.;
            for(int k=0; k < size_M; k++)
            {
                dotprod += X[colmajor(k,colX,size_M)] * M[colmajor(rowM,k,stride_M)];
            }
            Y[colmajor(colX,rowM,nb_col_X)] = dotprod;
        }
    }
}

/*
 * TODO write doc
 * ignores the should transpose
 */
template<typename T>
void multiply_transpose_batched(const T* input_batched[], const int nb_col_input_batched[], const char[],
                                const T* matrix_batched[], const int matrix_size_batched[], const int matrix_stride_batched[], const char[],
                                T* output_batched[], int nb_batch)
{
    #pragma omp parallel for
    for(int b = 0; b < nb_batch; b++)
    {
        multiply_transpose(input_batched[b], nb_col_input_batched[b], matrix_batched[b], matrix_size_batched[b], matrix_stride_batched[b], output_batched[b]);
    }
}

/*
 * TODO write doc
 * TODO add float version
 */
template<>
void multiply_transpose_batched<double>(const double* input_batched[], const int nb_col_input_batched[], const char should_transpose_input_batched[],
                                        const double* matrix_batched[], const int matrix_size_batched[], const int matrix_stride_batched[], const char should_transpose_matrix_batched[],
                                        double* output_batched[], int nb_batch)
{
    // number of different weights
    int group_count = 1;
    auto weight_product = new double[group_count];
    auto weight_output = new double[group_count];

    // https://scc.ustc.edu.cn/zlsc/tc4600/intel/2017.0.098/mkl/common/mklman_c/GUID-D797E8FA-B0CE-417C-98F1-896CDFB4FC35.htm
    dgemm_batch(should_transpose_input_batched, should_transpose_matrix_batched,
                nb_col_input_batched, matrix_size_batched, matrix_size_batched,
                weight_product,
                input_batched, matrix_size_batched,
                matrix_batched, matrix_stride_batched,
                weight_output, output_batched, nb_col_input_batched,
                &group_count, &nb_batch);

    delete[] weight_product;
    delete[] weight_output;
}

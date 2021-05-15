#pragma once
#include <cuda_runtime.h>

/*
 * converts row and col indices into a single index for a matrix store in col-major
 * `stride` is usually the number of rows of the matrix
 */
__device__ constexpr int colmajor(const int row, const int col, const int stride)
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
// TODO no transpose version, slower but avoid alloc
template<typename T>
__device__ void multiply_transpose(const T X[], const int nb_col_X,
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


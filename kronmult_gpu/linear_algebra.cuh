#pragma once
#include <device_launch_parameters.h>

/*
 * converts row and col indices into a single index for a matrix stored in col-major
 * `stride` is usually the number of rows of the matrix
 */
__device__ __forceinline__ constexpr int colmajor(const int row, const int col, const int stride)
{
    return row + col*stride;
}

/*
 * computes output = input^T
 *
 * `input` is a `matrix_size` by `matrix_size` square matrix of stride `input_stride`
 * `output` is a `matrix_size` by `matrix_size` square matrix of stride `matrix_size`
 *
 * WARNING: the matrices are assumed to be stored in col-major order
 */
template<typename T>
__device__ void transpose(const T input[], T output[], const int matrix_size, const int input_stride)
{
    for(int r = 0; r < matrix_size; r++)
    {
        for(int c = 0; c < matrix_size; c++)
        {
            output[colmajor(r, c, matrix_size)] = input[colmajor(c, r, input_stride)];
        }
    }
}


/*
 * Computes Y = X^T * M^T
 *
 * X is a `size_M` by `nb_col_X` matrix of stride `size_M`
 * M_transposed is a `size_M` by `size_M` matrix of stride `size_M` that contains a precomputed M^T
 * Y is a `nb_col_X` by `size_M` matrix of stride `nb_col_X`
 *
 * WARNING: the matrices are assumed to be stored in col-major order
 */
template<typename T>
__device__ void multiply_transpose(const T X[], const int nb_col_X,
                                   const T M_transposed[], const int size_M,
                                   T Y[])
{
    // strided loop, each thread threadIdx.x manages the inputs i such that threadIdx.x % t==0
    for(int i = threadIdx.x; i < nb_col_X*size_M; i+=blockDim.x)
    {
        // extracts the column and row number for the current thread
        const int colX = i / size_M;
        const int rowM = i - colX*size_M;
        // computes the dot product to fill the [colX,rowM] cell of the matrix
        T dotprod = 0.;
        for(int k=0; k < size_M; k++)
        {
            dotprod += X[colmajor(k,colX,size_M)] * M_transposed[colmajor(k,rowM,size_M)];
        }
        Y[colmajor(colX,rowM,nb_col_X)] = dotprod;
    }
}

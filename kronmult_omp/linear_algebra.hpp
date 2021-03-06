#pragma once
#include <stdexcept>

#ifdef KRONMULT_USE_BLAS

// BLAS function header: call to mkl, lapack, magma or others.
// Col-Major by default.
extern "C"
{
    // matrix multiplication
    // double precision: fp64
    int dgemm_(char *transa, char *transb, int *m, int *n, int *k, double *alpha, double *A, int *lda,
               double *B, int *ldb, double *beta, double *C, int *ldc);
    // single precision: fp32
    int sgemm_(char *transa, char *transb, int *m, int *n, int *k, float *alpha, float *A, int *lda, float *B,
               int *ldb, float *beta, float *C, int *ldc);
}

/*
 * Computes Y = X^T * M^T
 *
 * X is a `size_M` by `nb_col_X` matrix of stride `size_M`
 * M is a `size_M` by `size_M` matrix of stride `stride_M`
 * Y is a `nb_col_X` by `size_M` matrix of stride `nb_col_X`
 * M_transposed is a `size_M` by `size_M` matrix of stride `size_M` (it is ignored in this specialization)
 *
 * WARNING: the matrices are assumed to be stored in col-major order
 */
template<typename T>
void multiply_transpose(T const X_const[], int const nb_col_X_const, T const M_const[],
                        int const size_M_const, int const stride_M_const, T Y[], T[])
{
    // drops some const qualifiers as requested by BLAS
    auto X       = const_cast<T *>(X_const);
    auto M       = const_cast<T *>(M_const);
    int nb_col_X = nb_col_X_const;
    int size_M   = size_M_const;
    int stride_M = stride_M_const;
    // Y = weight_XM * X^T * M^T + weight_Y * Y
    char should_transpose_X = 'T';
    char should_transpose_M = 'T';
    T weight_XM             = 1.0;
    T weight_Y              = 0.0;
    // calls the proper specialization
    if constexpr (std::is_same<T, float>::value)
    {
        sgemm_(&should_transpose_X, &should_transpose_M, &nb_col_X, &size_M, &size_M, &weight_XM, X, &size_M,
               M, &stride_M, &weight_Y, Y, &nb_col_X);
    }
    else if constexpr (std::is_same<T, double>::value)
    {
        dgemm_(&should_transpose_X, &should_transpose_M, &nb_col_X, &size_M, &size_M, &weight_XM, X, &size_M,
               M, &stride_M, &weight_Y, Y, &nb_col_X);
    }
    static_assert(std::is_same<T, double>::value or std::is_same<T, float>::value,
                  "The function `multiply_transpose` is only defined for float and double precision");
}

#else

/*
 * converts row and col indices into a single index for a matrix stored in col-major
 * `stride` is usually the number of rows of the matrix
 */
inline int colmajor(int const row, int const col, int const stride)
{
    return row + col * stride;
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
void transpose(T const input[], T output[], int const matrix_size, int const input_stride)
{
    for (int r = 0; r < matrix_size; r++)
    {
        for (int c = 0; c < matrix_size; c++)
        {
            output[colmajor(r, c, matrix_size)] = input[colmajor(c, r, input_stride)];
        }
    }
}

/*
 * Computes Y = X^T * M^T
 *
 * X is a `size_M` by `nb_col_X` matrix of stride `size_M`
 * M is a `size_M` by `size_M` matrix of stride `stride_M`
 * Y is a `nb_col_X` by `size_M` matrix of stride `nb_col_X`
 * M_transposed is a `size_M` by `size_M` matrix of stride `size_M` that will be used to store M^T temporarily
 *
 * WARNING:
 * the matrices are assumed to be stored in col-major order
 * `transpose_workspace` will be used as temporary workspaces and thus modified
 */
template<typename T>
void multiply_transpose(T const X[], int const nb_col_X, T const M[], int const size_M, int const stride_M,
                        T Y[], T M_transposed[])
{
    // transpose the matrix to get a better cache behaviour
    transpose(M, M_transposed, size_M, stride_M);

    for (int colX = 0; colX < nb_col_X; colX++)
    {
        for (int rowM = 0; rowM < size_M; rowM++)
        {
            T dotprod = 0.;
            for (int k = 0; k < size_M; k++)
            {
                dotprod += X[colmajor(k, colX, size_M)] * M_transposed[colmajor(k, rowM, size_M)];
            }
            Y[colmajor(colX, rowM, nb_col_X)] = dotprod;
        }
    }
}

#endif
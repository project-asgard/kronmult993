#pragma once

/*
 * computes output = input^T
 *
 * `input` is a `matrix_size` by `matrix_size` square matrix of stride `input_stride`
 * `output` is a `matrix_size` by `matrix_size` square matrix of stride `matrix_size`
 *
 * WARNING: the matrices are assumed to be stored in col-major order
 */
template<typename T>
void transpose(const T input[], T output[], const int matrix_size, const int input_stride)
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
                                        T Y[], T M_transposed[])
{
    #define colmajor(row, col, nb_rows) ((row) + (col) * (nb_rows))

    // transpose the matrix to get a better alignement
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

    #undef colmajor
}

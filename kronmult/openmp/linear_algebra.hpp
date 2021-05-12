#pragma once

/*
 * converts row and col indices into a single index for a matrix store in col-major
 * `stride` is usually the number of rows of the matrix
 */
constexpr int colmajor(const int row, const int col, const int stride)
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
void transpose(const T input[], T output[], const int matrix_size, const int input_stride)
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
    std::cerr << "not using blas!" << std::endl;

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
}

#ifdef KRONMULT_USE_BLAS

// BLAS function header: call to mkl, lapack, magma or others.
// Col-Major by default.
extern "C"
{
    // matrix multiplication
    //double precision: fp64
    int dgemm_(char* transa, char* transb, int* m, int* n, int* k, double* alpha, double* A, int* lda, double* B, int* ldb, double* beta, double* C, int* ldc);
    //single precision: fp32
    int sgemm_(char* transa, char* transb, int* m, int* n, int* k, float* alpha, float* A, int* lda, float* B, int* ldb, float* beta, float* C, int* ldc);
}

template<>
void multiply_transpose<float>(const float X_const[], const int nb_col_X_const,
                               const float M_const[], const int size_M_const, const int stride_M_const,
                               float Y[], float M_transposed[])
{
    // drops some const qualifiers as requested by BLAS
    float* X = const_cast<float*>(X_const);
    float* M = const_cast<float*>(M_const);
    int nb_col_X = nb_col_X_const;
    int size_M = size_M_const;
    int stride_M = stride_M_const;
    // Y = weight_XM * X^T * M^T + weight_Y * Y
    char should_transpose_X = 'T';
    char should_transpose_M = 'T';
    float weight_XM = 1.0f;
    float weight_Y = 0.0f;
    int errorCode = sgemm_(&should_transpose_X, &should_transpose_M, &nb_col_X, &size_M, &size_M,
                           &weight_XM, X, &size_M, M, &stride_M, &weight_Y, Y, &nb_col_X);
    // TODO gets a non-zero code but correct result ?!
    //if (errorCode != 0) throw std::runtime_error("BLAS routine 'SGEMM' failed with error code " + std::to_string(errorCode));
}

template<>
void multiply_transpose<double>(const double X_const[], const int nb_col_X_const,
                                const double M_const[], const int size_M_const, const int stride_M_const,
                                double Y[], double M_transposed[])
{
    // drops some const qualifiers as requested by BLAS
    double* X = const_cast<double*>(X_const);
    double* M = const_cast<double*>(M_const);
    int nb_col_X = nb_col_X_const;
    int size_M = size_M_const;
    int stride_M = stride_M_const;
    // Y = weight_XM * X^T * M^T + weight_Y * Y
    char should_transpose_X = 'T';
    char should_transpose_M = 'T';
    double weight_XM = 1.0;
    double weight_Y = 0.0;
    int errorCode = dgemm_(&should_transpose_X, &should_transpose_M, &nb_col_X, &size_M, &size_M,
                           &weight_XM, X, &size_M, M, &stride_M, &weight_Y, Y, &nb_col_X);
    // TODO gets a non-zero code but correct result ?!
    //if (errorCode != 0) throw std::runtime_error("BLAS routine 'DGEMM' failed with error code " + std::to_string(errorCode));
}

#endif

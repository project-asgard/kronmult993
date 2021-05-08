#pragma once
#define CHECK(X) if (X != 0) {cerr << "Error in BLAS GEMM call"<< endl; exit(-1);}

/*
 * converts row and col indices into a single index for a matrix store in col-major
 * `stride` is usually the number of rows of the matrix
 */
inline int colmajor(const int row, const int col, const int stride)
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

// BLAS function header: call to mkl, lapack, magma or others.
// Col-Major by default.
extern "C"
{
    // matrix multiplication
    //double precision: fp64
    int dgemm_(char *transa, char *transb, int *m, int *n, int *k, double *alpha, double *A, int *lda, double *B, int *ldb, double *beta, double *C, int *ldc);
    //single precision: fp32
    int sgemm_(char *transa, char *transb, int *m, int *n, int *k, float *alpha, float *A, int *lda, float *B, int *ldb, float *beta, float *C, int *ldc);
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
void multiply_transpose_blas(const T X[], const int nb_col_X,
                                        const T M[], const int size_M, const int stride_M,
                                        T Y[])
{
    std::cerr << "Error type not supported"<< std::endl;
    exit(-1);
}

template<typename T>
void multiply_transpose_blas_fillargs(int * res, char * transa, char * transb, int * m, int * n, int * k,
        T * one, T * zero, int * lda, int * ldb, int * ldc, const int nb_col_X, const int size_M, const int stride_M,
        T ** A, T ** B, T ** C, const T * X, const T * M, T * Y)
{
    // C = alpha*A*B + beta * C
    // C (m,n), A (m,k), B (k,n)
    // C<>Y A<>X B<>M
    // Y (nbCol,sizeM), X (sizeM,nbCol), M (sizeM,sizeM)
    // m == nbCol
    // n == size_M
    // k == size_M
    *transa = 'T'; 
    *transb = 'T';
    *m = nb_col_X;
    *n = size_M;
    *k = size_M;
    *one = static_cast<T>(1);
    *zero = static_cast<T>(0.0);
    /* lda/b/c are given by number of lines (ColMajor) */
    *lda = size_M; // X(<>A) is a size_M by nb_col_X matrix
    *ldb = stride_M; // M(<>B) is a size_M by size_M matrix
    *ldc = nb_col_X; // Y(<>C) is a nb_col_X by size_M matrix
    *A = const_cast<T*>(X);
    *B = const_cast<T*>(M);
    *C = Y;
    *res = 0;
}

template<>
void multiply_transpose_blas<float>(const float X[], const int nb_col_X,
                                        const float M[], const int size_M, const int stride_M,
                                        float Y[])
{
    float *A, *B, *C, zero, one;
    char transa, transb;
    int m, n, k, lda, ldb, ldc, res;
    multiply_transpose_blas_fillargs<float>(&res, &transa, &transb, &m, &n, &k, &one, &zero, &lda, &ldb,
            &ldc, nb_col_X, size_M, stride_M, &A, &B, &C, X, M, Y);
    res = sgemm_(&transa, &transb, &m, &n, &k, &one, A, &lda, B, &ldb, &zero, C, &ldc);
    CHECK(res);
}

template<>
void multiply_transpose_blas<double>(const double X[], const int nb_col_X,
                                        const double M[], const int size_M, const int stride_M,
                                        double Y[])
{
    double *A, *B, *C, zero, one;
    char transa, transb;
    int m, n, k, lda, ldb, ldc, res;
    multiply_transpose_blas_fillargs<double>(&res, &transa, &transb, &m, &n, &k, &one, &zero, &lda, &ldb,
            &ldc, nb_col_X, size_M, stride_M, &A, &B, &C, X, M, Y);
    res = dgemm_(&transa, &transb, &m, &n, &k, &one, A, &lda, B, &ldb, &zero, C, &ldc);
    CHECK(res);
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

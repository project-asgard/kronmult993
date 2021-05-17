#pragma once

/*
 * converts row and col indices into a single index for a matrix store in col-major
 * `stride` is usually the number of rows of the matrix
 */
constexpr int colmaj(const int row, const int col, const int stride)
{
    return row + col*stride;
}

/*
 * naive implementation of a matrix vector product
 * we suppose that the matrix is square, stored in col-major and of stride one
 * the result is *added* to output
 */
template<typename T>
void matrix_vector_product(const T matrix[], const int size, const int stride,
                           const T vector[], T output[])
{
    for(int row = 0; row < size; row++)
    {
        for(int col = 0; col < size; col++)
        {
            output[row] += matrix[colmaj(row,col,stride)] * vector[col];
        }
    }
}

/*
 * does the kronecker product of two square col-major matrices
 * the output is a square col-major matrix of stride its number of columns
 */
template<typename T>
void kronecker_product(const T matrix1[], const int size1, const int stride1,
                       const T matrix2[], const int size2, const int stride2,
                       T output[])
{
    const int stride_out = size1*size2;
    for(int row1 = 0; row1 < size1; row1++)
    {
        for(int row2 = 0; row2 < size2; row2++)
        {
            for(int col1 = 0; col1 < size1; col1++)
            {
                for(int col2 = 0; col2 < size2; col2++)
                {
                    const int row_out = row1*size2 + row2;
                    const int col_out = col1*size2 + col2;
                    output[colmaj(row_out,col_out,stride_out)] = matrix1[colmaj(row1,col1,stride1)] * matrix2[colmaj(row2,col2,stride2)];
                }
            }
        }
    }
}

/*
 * naive implementation of kronmult
 */
template<typename T>
void kronmult_naive(const int matrix_count, const int matrix_size, T* matrix_list[], const int matrix_stride, const T input[], T output[], const bool gpuAlloc=false)
{
    // computes the kronnecker product
    T* kronmat = matrix_list[0];
    int size_kron = matrix_size;
    int stride_kron = matrix_stride;
    for(int m = 1; m < matrix_count; m++)
    {
        // allocates new kronmat
        T* kronmat_new;
        if(gpuAlloc) kronmat_new = cudaNew<T>(size_kron*matrix_size);
        else kronmat_new = new T[size_kron*matrix_size];
        // does kronecker product
        const T* matrix = matrix_list[m];
        kronecker_product(kronmat, size_kron, stride_kron, matrix, matrix_size, matrix_stride, kronmat_new);
        // replace old kronmat
        // do not delete input matrix
        if(m > 1)
        {
            if(gpuAlloc) cudaFree(kronmat);
            else delete[] kronmat;
        }
        kronmat = kronmat_new;
        size_kron = size_kron*matrix_size;
        stride_kron = size_kron;
    }
    // does the matrix vector product and adds the result to output
    matrix_vector_product(kronmat, size_kron, stride_kron, input, output);
    // frees the memory
    // do not delete input matrix
    if(matrix_count > 1)
    {
        if(gpuAlloc) cudaFree(kronmat);
        else delete[] kronmat;
    }
}

/*
 * naive implementation of batched kronmult
 */
template<typename T>
void kronmult_batched_naive(const int matrix_count, const int matrix_size, T* matrix_list_batched[], const int matrix_stride,
                            T* input_batched[],  T* output_batched[], T* workspace_batched[], const int nb_batch, const bool gpuAlloc=false)
{
    for(int i=0; i < nb_batch; i++)
    {
        T** matrix_list = &matrix_list_batched[i*matrix_count];
        T* input = input_batched[i];
        T* output = output_batched[i];
        kronmult_naive(matrix_count, matrix_size, matrix_list, matrix_stride, input, output, gpuAlloc);
    }
}

#include "kronmult.cuh"
#include <cstdio>
#include <omp.h>

/*
 * computes number^power for integers
 * does not care about performances
 * does not use std::pow as it does an implicit float conversion that could lead to rounding errors for high
 * numbers
 */
__host__ int pow_int(const int number, const int power)
{
    if(power == 0) return 1;
    return number * pow_int(number, power-1);
}

/*
 * Computes output += kron(matrix_list) * input
 *
 * `matrix_list` is an array containing pointers to `matrix_number` square matrices of size `matrix_size` by `matrix_size` and stride `matrix_stride`
 * `input` is a `size_input`=`matrix_size`^`matrix_number` elements vector
 * `output` is a `size_input` elements vector, where the output will be stored
 * `workspace` is a `size_input` elements vector, to be used as workspace
 * `transpose_workspace` is a vector of size `matrix_size`*`matrix_size` to store transposed matrices temporarily
 *
 * WARNINGS:
 * `input` and `workspace` will be used as temporary workspaces and thus modified
 * the matrices should be stored in col-major order
 */
template<typename T>
__device__ void cuda_kronmult(const int matrix_count, const int matrix_size,
                         T const * matrix_list, const int matrix_stride,
                              T * input, const int size_input,
                              T * output, T *workspace, T *transpose_workspace)
{
    // how many column should `input` have for the multiplications to be legal
    const int nb_col_input = size_input / matrix_size;

    // iterates on the matrices from the last to the one just before first
    for(int i = matrix_count-1; i >= 0; i--)
    {
        // transpose the matrix to get a better alignement
        T const * const matrix = &matrix_list[i];
        if(threadIdx.x == 0) transpose(matrix, transpose_workspace, matrix_size, matrix_stride);
        __syncthreads();

        // performs the multiplication to consume the matrix
        multiply_transpose<T>(input, nb_col_input, transpose_workspace, matrix_size, workspace);
        __syncthreads();

        // swap `input` and `workspace` such that `input` contains once again the input
        // note that, while they have the same size flattened, the shape (nb_columns and nb_rows) of `input` and `workspace` are different
        // this is on purpose and equivalent to a reshape operation that is actually needed by the algorithm
        T* temp = input;
        input = workspace;
        workspace = temp;
    }

    // reduce in a threadsafe way
    // each thread t manage the input i such that i%t==0
    for(int i = threadIdx.x; i < size_input; i+=blockDim.x)
    {
        atomicAdd(&output[i], input[i]);
    }
}

/*
 * gets the batch element as a function of the thread and block index
 * calls `cuda_kronmult` with the proper inputs
 */
    template<typename T>
__global__ void cuda_kronmult_thread(const int matrix_count,
        const int matrix_size, T const * matrix_list, const int matrix_stride,
        T* input, const int size_input,
        T* output, T* workspace)
{
    // each block corresponds to a batch element
    //const int batchId = blockIdx.x;

    // gets the inputs for the algorithm
    //T const * const * matrix_list = &matrix_list_batched[batchId*matrix_count];
    //T const * matrix_list = matrix_list_batched;

    // allocates the transpose workspace in shared memory
    __shared__ T* transpose_workspace;
    if(threadIdx.x == 0) transpose_workspace = new T[matrix_size*matrix_size];
    __syncthreads();

    // computes kronmult
    cuda_kronmult<T>(matrix_count, matrix_size, matrix_list, matrix_stride, input, size_input, output, workspace, transpose_workspace);

    // free memory
    __syncthreads();
    if(threadIdx.x == 0) delete[] transpose_workspace;
}

/*
 * Calls the cuda kernel with proper thread parameters.
 * This function expects its inputs to already be on the device (GPU).
 */
template<typename T>
__host__ cudaError cuda_kronmult_batched(const int matrix_count, const int matrix_size, T const * const matrix_list_batched[], const int matrix_stride,
                                         T* input_batched[], T* output_batched[], T* workspace_batched[], const int nb_batch)
{
    // numbers of elements in the input vector
    int size_input = pow_int(matrix_size, matrix_count);

    // each block will be a batch element
    // each thread will be a subset of the lines in input
    int deviceId;
    cudaGetDevice(&deviceId);
    int threadsPerBlock;
    cudaDeviceGetAttribute(&threadsPerBlock, cudaDevAttrMaxThreadsPerBlock, deviceId);
    if(size_input < threadsPerBlock) threadsPerBlock = size_input;
    //printf("threads-per-block:%d nb-blocks:%d\n", threadsPerBlock, nb_batch);
    int nbBlocks = size_input / threadsPerBlock;

    // paralelize on batch elements
    cudaStream_t stream1, stream2, stream3, stream4, stream[4] ;
    cudaStreamCreate(&stream1);
    cudaStreamCreate(&stream2);
    cudaStreamCreate(&stream3);
    cudaStreamCreate(&stream4);
    stream[0] = stream1;
    stream[1] = stream2;
    stream[2] = stream3;
    stream[3] = stream4;
    #pragma omp parallel for
    for(int i=0; i < nb_batch; i++)
    {
        cuda_kronmult_thread<<<1,nbBlocks, threadsPerBlock,stream[i%4]>>>(matrix_count, matrix_size,
                matrix_list_batched[i*matrix_count], matrix_stride,
                input_batched[i], size_input, output_batched[i], workspace_batched[i]);
    }

    // waits for kernel to succeed and returns error code
    return cudaDeviceSynchronize();
}

/*
 * float specialization
 */
template<>
__host__ cudaError kronmult_batched<double>(const int matrix_count, const int matrix_size, double const * const matrix_list_batched[], const int matrix_stride,
                                            double* input_batched[], double* output_batched[], double* workspace_batched[], const int nb_batch)
{
    return cuda_kronmult_batched(matrix_count, matrix_size, matrix_list_batched, matrix_stride, input_batched, output_batched, workspace_batched, nb_batch);
}

/*
 * double specialization
 */
template<>
__host__ cudaError kronmult_batched<float>(const int matrix_count, const int matrix_size, float const * const matrix_list_batched[], const int matrix_stride,
                                           float* input_batched[], float* output_batched[], float* workspace_batched[], const int nb_batch)
{
    return cuda_kronmult_batched(matrix_count, matrix_size, matrix_list_batched, matrix_stride, input_batched, output_batched, workspace_batched, nb_batch);
}

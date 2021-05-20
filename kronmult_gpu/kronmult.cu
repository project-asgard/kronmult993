#include <device_launch_parameters.h>
#include "kronmult.cuh"

/*
 * computes number^power for integers
 * does not care about performances
 * does not use std::pow as it does an implicit float conversion that could lead to rounding errors for large numbers
 */
__host__ int pow_int(const int number, const int power)
{
    if(power == 0) return 1;
    return number * pow_int(number, power-1);
}

/*
 * Computes output += kron(matrix_list) * input while insuring that the addition to output is thread-safe
 *
 * `matrix_list` is an array containing pointers to `matrix_number` square matrices of size `matrix_size` by `matrix_size` and stride `matrix_stride`
 * `input` is a `size_input` (`matrix_size`^`matrix_number`) elements vector
 * `output` is a `size_input` elements vector, to which the output of the multiplication will be added
 * `workspace` is a `size_input` elements vector, to be used as workspace
 * `transpose_workspace` is a vector of size `matrix_size`*`matrix_size` to store transposed matrices temporarily
 *
 * WARNINGS:
 * - `input`, `workspace` and `transpose_workspace` will be used as temporary workspaces and thus modified
 * - the matrices are assumed to be stored in col-major order
 * - the sizes are assumed to be correct
 */
template<typename T>
__device__ void cuda_kronmult(const int matrix_count, const int matrix_size, T const * const matrix_list[], const int matrix_stride,
                              T input[], const int size_input,
                              T output[], T workspace[], T transpose_workspace[])
{
    // how many column should `input` have for the multiplications to be legal
    const int nb_col_input = size_input / matrix_size;

    // iterates on the matrices from last to first
    for(int i = matrix_count-1; i >= 0; i--)
    {
        // transpose the matrix (with a single thead) to get a better memory coalescing
        T const * const matrix = matrix_list[i];
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

    // adds result to output in a thread-safe way
    // strided loop, each thread threadIdx.x manages the input i such that i % threadIdx.x==0
    for(int i = threadIdx.x; i < size_input; i+=blockDim.x)
    {
        atomicAdd(&output[i], input[i]);
    }
}

/*
 * each block gets a single batch element to process
 *
 * computes the current batch element
 * finds the corresponding inputs
 * and calls kronmult on them
 */
template<typename T>
__global__ void cuda_kronmult_batchelement(const int matrix_count, const int matrix_size, T const * const matrix_list_batched[], const int matrix_stride,
                                           T* input_batched[], const int size_input,
                                           T* output_batched[], T* workspace_batched[],
                                           const int nb_batch)
{
    // each block corresponds to a single batch element
    const int batchId = blockIdx.x;
    // gets the inputs for a given batch element
    T const * const * matrix_list = &matrix_list_batched[batchId*matrix_count];
    T* input = input_batched[batchId];
    T* output = output_batched[batchId];
    T* workspace = workspace_batched[batchId];

    // uses a thread to allocates the transpose workspace
    // in shared memory for improved performances
    __shared__ T* transpose_workspace;
    if(threadIdx.x == 0) transpose_workspace = new T[matrix_size*matrix_size];
    __syncthreads();

    // does the kronmult computations
    cuda_kronmult<T>(matrix_count, matrix_size, matrix_list, matrix_stride, input, size_input, output, workspace, transpose_workspace);

    // frees the tranpose workspace memory
    __syncthreads();
    if(threadIdx.x == 0) delete[] transpose_workspace;
}

/*
 * calls the cuda kernel with the proper number of blocks and threads
 * we expect the inputs to already be on the GPU
 */
template<typename T>
__host__ cudaError cuda_kronmult_batched(const int matrix_count, const int matrix_size, T const * const matrix_list_batched[], const int matrix_stride,
                                         T* input_batched[], T* output_batched[], T* workspace_batched[], const int nb_batch)
{
    // numbers of elements in the input vector
    int size_input = pow_int(matrix_size, matrix_count);

    // each block will take care of a single batch element
    // the threads within a block will loop over input_size
    int deviceId;
    cudaGetDevice(&deviceId);
    int threadsPerBlock;
    cudaDeviceGetAttribute(&threadsPerBlock, cudaDevAttrMaxThreadsPerBlock, deviceId);
    if(size_input < threadsPerBlock) threadsPerBlock = size_input;

    // parallelize over batch elements
    cuda_kronmult_batchelement<<<nb_batch, threadsPerBlock>>>(matrix_count, matrix_size, matrix_list_batched, matrix_stride,
                                                              input_batched, size_input, output_batched, workspace_batched, nb_batch);

    // waits for kernel to finish and returns the error code
    return cudaDeviceSynchronize();
}

/*
 * double specialization of kronmult_batched
 */
template<>
__host__ cudaError kronmult_batched<double>(const int matrix_count, const int matrix_size, double const * const matrix_list_batched[], const int matrix_stride,
                                            double* input_batched[], double* output_batched[], double* workspace_batched[], const int nb_batch)
{
    return cuda_kronmult_batched(matrix_count, matrix_size, matrix_list_batched, matrix_stride, input_batched, output_batched, workspace_batched, nb_batch);
}

/*
 * float specialization of kronmult_batched
 */
template<>
__host__ cudaError kronmult_batched<float>(const int matrix_count, const int matrix_size, float const * const matrix_list_batched[], const int matrix_stride,
                                           float* input_batched[], float* output_batched[], float* workspace_batched[], const int nb_batch)
{
    return cuda_kronmult_batched(matrix_count, matrix_size, matrix_list_batched, matrix_stride, input_batched, output_batched, workspace_batched, nb_batch);
}

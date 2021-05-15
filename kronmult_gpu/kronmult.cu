#include "kronmult.cuh"
#include "stdio.h"

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
__device__ void cuda_kronmult(const int matrix_count, const int matrix_size, T const * const matrix_list[], const int matrix_stride,
                              T input[], const int size_input,
                              T output[], T workspace[], T transpose_workspace[])
{
    // how many column should `input` have for the multiplications to be legal
    const int nb_col_input = size_input / matrix_size;

    // iterates on the matrices from the last to the one just before first
    for(int i = matrix_count-1; i >= 0; i--)
    {
        // takes `matrix` into account and put the result in `workspace` (use `output` as a workspace if needed)
        T const * const matrix = matrix_list[i];
        multiply_transpose<T>(input, nb_col_input, matrix, matrix_size, matrix_stride, workspace, transpose_workspace);
        // swap `input` and `workspace` such that `input` contains once again the input
        // note that, while they have the same size flattened, the shape (nb_columns and nb_rows) of `input` and `workspace` are different
        // this is on purpose and equivalent to a reshape operation that is actually needed by the algorithm
        T* temp = input;
        input = workspace;
        workspace = temp;
    }

    // reduce in a threadsafe way
    for(int i = 0; i < size_input; i++)
    {
        atomicAdd(&output[i], input[i]);
    }
}

/*
 * gets the batch element as a function of the thread and block index
 * calls `cuda_kronmult` with the proper inputs
 */
template<typename T>
__global__ void cuda_kronmult_thread(const int matrix_count, const int matrix_size, T const * const matrix_list_batched[], const int matrix_stride,
                                      T* input_batched[], const int size_input,
                                      T* output_batched[], T* workspace_batched[], T transpose_workspace_batched[],
                                      const int nb_batch)
{
    // each thread get a single batch
    const int batchId = blockIdx.x * blockDim.x + threadIdx.x;
    if(batchId < nb_batch)
    {
        // computes kronmult
        T const * const * matrix_list = &matrix_list_batched[batchId*matrix_count];
        T* input = input_batched[batchId];
        T* output = output_batched[batchId];
        T* workspace = workspace_batched[batchId];
        T* transpose_workspace = &transpose_workspace_batched[threadIdx.x*matrix_size*matrix_size];
        // result is stored in `workspace` if `matrix_count` is odd and `input` if it is even
        cuda_kronmult<T>(matrix_count, matrix_size, matrix_list, matrix_stride, input, size_input, output, workspace, transpose_workspace);
    }
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

    // gets maximum number of thread possible per block
    int deviceId;
    cudaGetDevice(&deviceId);
    int threadsPerBlock;
    cudaDeviceGetAttribute(&threadsPerBlock, cudaDevAttrMaxThreadsPerBlock, deviceId);
    // split batches into blocks with a maximum of threads in each
    if(nb_batch < threadsPerBlock) threadsPerBlock = nb_batch;
    unsigned int nbBlocks = (nb_batch + threadsPerBlock - 1) / threadsPerBlock; // ceil(nb_batch/threadsPerBlock)

    // allocate workspace for transposition (just enough for a single block)
    T* transpose_workspace_batched;
    cudaMalloc((void**)&transpose_workspace_batched, sizeof(T) * (matrix_size*matrix_size) * threadsPerBlock);

    // paralelize on batch elements
    cuda_kronmult_thread<<<nbBlocks, threadsPerBlock>>>(matrix_count, matrix_size, matrix_list_batched, matrix_stride,
                                                        input_batched, size_input, output_batched, workspace_batched, transpose_workspace_batched, nb_batch);

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

#include "kronmult.cuh"
#include "stdio.h"

__host__ int pow_int(const int number, const int power)
{
    if(power == 0) return 1;
    return number * pow_int(number, power-1);
}

template<>
__host__ cudaError kronmult_batched<double>(const int matrix_count, const int matrix_size, double const * const matrix_list_batched[], const int matrix_stride,
                                            double* input_batched[], double* output_batched[], double* workspace_batched[], const int nb_batch)
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
    double* transpose_workspace_batched;
    cudaMalloc((void**)&transpose_workspace_batched, sizeof(double) * (matrix_size*matrix_size) * threadsPerBlock);

    // paralelize on batch elements
    cuda_kronmult_batched<<<nbBlocks, threadsPerBlock>>>(matrix_count, matrix_size, matrix_list_batched, matrix_stride,
                                                         input_batched, size_input, output_batched, workspace_batched, transpose_workspace_batched, nb_batch);

    // waits for kernel to succeed and returns error code
    return cudaDeviceSynchronize();
}

template<>
__host__ cudaError kronmult_batched<float>(const int matrix_count, const int matrix_size, float const * const matrix_list_batched[], const int matrix_stride,
                                           float* input_batched[], float* output_batched[], float* workspace_batched[], const int nb_batch)
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
    float* transpose_workspace_batched;
    cudaMalloc((void**)&transpose_workspace_batched, sizeof(float) * (matrix_size*matrix_size) * threadsPerBlock);

    // paralelize on batch elements
    cuda_kronmult_batched<<<nbBlocks, threadsPerBlock>>>(matrix_count, matrix_size, matrix_list_batched, matrix_stride,
                                                         input_batched, size_input, output_batched, workspace_batched, transpose_workspace_batched, nb_batch);

    // waits for kernel to succeed and returns error code
    return cudaDeviceSynchronize();
}

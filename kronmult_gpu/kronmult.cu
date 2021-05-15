#include "kronmult.cuh"
#include "stdio.h"

__host__ int pow_int(const int number, const int power)
{
    if(power == 0) return 1;
    return number * pow_int(number, power-1);
}

// TODO define for floats
template<>
__host__ int kronmult_batched<double>(const int matrix_count, const int matrix_size, double const * const matrix_list_batched[], const int matrix_stride,
                                       double* input_batched[], double* output_batched[], double* workspace_batched[], const int nb_batch)
{
    // numbers of elements in the input vector
    int size_input = pow_int(matrix_size, matrix_count);

    // split batches into blocks with a maximum of threads in each
    unsigned int threadsPerBlock = 1024; // aximum possible TODO automatize
    if(nb_batch < threadsPerBlock) threadsPerBlock = nb_batch;
    unsigned int nbBlocks = (nb_batch + threadsPerBlock - 1) / threadsPerBlock; // ceil(nb_batch/threadsPerBlock)

    // paralelize on batch elements
    cuda_kronmult_batched<<<nbBlocks, threadsPerBlock>>>(matrix_count, matrix_size, matrix_list_batched, matrix_stride,
                                    input_batched, size_input, output_batched, workspace_batched, nb_batch);

    // wait for kernel to succeed and returns error code (!= 0 => problem)
    return cudaDeviceSynchronize();
}
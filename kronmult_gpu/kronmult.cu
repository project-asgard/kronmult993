#include "kronmult.cuh"

__host__ int pow_int(const int number, const int power)
{
    if(power == 0) return 1;
    return number * pow_int(number, power-1);
}

// TODO define for floats
// TODO make parallel with on thread per batch element
template<>
__host__ void kronmult_batched<double>(const int matrix_count, const int matrix_size, double const * const matrix_list_batched[], const int matrix_stride,
                                       double* input_batched[], double* output_batched[], double* workspace_batched[], const int nb_batch)
{
    // numbers of elements in the input vector
    int size_input = pow_int(matrix_size, matrix_count);

    // paralelize on batch elements
    cuda_kronmult_batched<<<1, 1>>>(matrix_count, matrix_size, matrix_list_batched, matrix_stride,
                                    input_batched, size_input, output_batched, workspace_batched, nb_batch);

    // wait for kernel to succeed
    cudaDeviceSynchronize();
}
#include "utils/kronmult_naive.h"
#include "utils/utils_gpu.h"
#include <iostream>
#include <kronmult.cuh>

// change this to run the bench in another precision
using Number = double;

/*
 * runs a test with the given parameters
 * `nb_distinct_outputs` modelizes the fact that most outputs are identical
 */
Number runTest(int const degree, int const dimension, int const grid_level, std::string const benchName,
               int const nb_distinct_outputs = 5)
{
    // Kronmult parameters
    // TODO find proper formula for batch count, current one can generates batches too large to be allocated
    // without the min
    int const batch_count = pow_int(2, grid_level) * pow_int(grid_level, dimension - 1);
    int const matrix_size  = degree;
    int const matrix_count = dimension;
    int const size_input   = pow_int(matrix_size, matrix_count);
    int const matrix_stride = 67; // large prime integer, modelize the fact that columns are not adjascent in memory
    std::cout << benchName << " benchcase"
              << " batch_count:" << batch_count << " matrix_size:" << matrix_size
              << " matrix_count:" << matrix_count << " size_input:" << size_input
              << " nb_distinct_outputs:" << nb_distinct_outputs << std::endl;

    std::cout << "Starting allocation." << std::endl;
    bool const should_initialize_data = true;
    DeviceArrayBatch<Number> matrix_list_batched(matrix_size * matrix_stride, batch_count * matrix_count,
                                                 should_initialize_data);
    DeviceArrayBatch<Number> input_batched(size_input, batch_count, should_initialize_data); // this will only be modified by the second algorithm
    DeviceArrayBatch<Number> workspace_batched(size_input, batch_count, should_initialize_data);
    DeviceArrayBatch_withRepetition<Number> output_batched(size_input, batch_count, nb_distinct_outputs, should_initialize_data);
    DeviceArrayBatch_withRepetition<Number> output_batched2(output_batched); // copy as this will be modified by both algorithms

    std::cout << "Starting Naive Kronmult" << std::endl;
    kronmult_batched_naive(matrix_count, matrix_size, matrix_list_batched.rawPointer, matrix_stride,
                           input_batched.rawPointer, output_batched.rawPointer, workspace_batched.rawPointer,
                           batch_count, cudaNew<Number>, [](void *ptr) { cudaFree(ptr); });

    std::cout << "Starting Kronmult" << std::endl;
    cudaError errorCode = kronmult_batched(
        matrix_count, matrix_size, matrix_list_batched.rawPointer, matrix_stride, input_batched.rawPointer,
        output_batched2.rawPointer, workspace_batched.rawPointer, batch_count);
    checkCudaErrorCode(errorCode, "kronmult_batched");

    std::cout << "Computing the error" << std::endl;
    Number const error = output_batched.distance(output_batched2);
    std::cout << "Error: " << error << std::endl;

    return error;
}

/*
 * Runs tests of increasing sizes and displays the results
 */
int main()
{
    std::cout << "Starting tests." << std::endl;
    // gets basic information on the cuda device
    int deviceId;
    cudaGetDevice(&deviceId);
    int threadsPerBlock;
    cudaDeviceGetAttribute(&threadsPerBlock, cudaDevAttrMaxThreadsPerBlock, deviceId);
    cudaDeviceProp deviceProp;
    cudaError const errorCode = cudaGetDeviceProperties(&deviceProp, deviceId);
    checkCudaErrorCode(errorCode, "cudaSetDevice");
    std::cout << "GPU device:" << deviceId << " threadsAvailable:" << threadsPerBlock
              << " architecture:" << deviceProp.major << '.' << deviceProp.minor << std::endl;

    // running the benchmarks
    auto toy   = runTest(4, 1, 2, "toy");
    auto small = runTest(4, 2, 4, "small");

    // display results
    std::cout << std::endl
              << "Errors:" << std::endl
              << "toy: " << toy << std::endl
              << "small: " << small << std::endl;
}

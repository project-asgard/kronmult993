#include <iostream>
#include <chrono>
#include <kronmult.cuh>
#include "utils/utils_gpu.h"

// change this to run the bench in another precision
using Number = double;

/*
 * runs a benchmark with the given parameters
 * `nb_distinct_outputs` modelizes the fact that most outputs are identical
 */
long runBench(const int degree, const int dimension, const int grid_level, const std::string benchName, const int nb_distinct_outputs = 5)
{
    // Kronmult parameters
    // TODO find proper formula for batch count, current one generates batches too large to be allocated without the min
    // TODO go back to max 2^13 if possible (does not fit on GPU)
    const int batch_count = std::min(pow_int(2,10), pow_int(2, grid_level) * pow_int(grid_level, dimension-1));
    const int matrix_size = degree;
    const int matrix_count = dimension;
    const int size_input = pow_int(matrix_size, matrix_count);
    const int matrix_stride = 67; // large prime integer, modelize the fact that columns are not adjascent in memory
    std::cout << benchName << " benchcase"
              << " batch_count:" << batch_count
              << " matrix_size:" << matrix_size
              << " matrix_count:" << matrix_count
              << " size_input:" << size_input
              << " nb_distinct_outputs:" << nb_distinct_outputs
              << std::endl;

    // allocates a problem
    // we do not put data in the vectors/matrices as it doesn't matter here
    std::cout << "Starting allocation." << std::endl;
    DeviceArrayBatch<Number> matrix_list_batched(matrix_size * matrix_stride, batch_count * matrix_count);
    DeviceArrayBatch<Number> input_batched(size_input, batch_count);
    DeviceArrayBatch<Number> workspace_batched(size_input, batch_count);
    DeviceArrayBatch_withRepetition<Number> output_batched(size_input, batch_count, nb_distinct_outputs);

    // runs kronmult several times and displays the average runtime
    std::cout << "Starting Kronmult" << std::endl;
    auto start = std::chrono::high_resolution_clock::now();
    const cudaError errorCode = kronmult_batched(matrix_count, matrix_size, matrix_list_batched.rawPointer,
                                                 matrix_stride, input_batched.rawPointer, output_batched.rawPointer,
                                                 workspace_batched.rawPointer, batch_count);
    auto stop = std::chrono::high_resolution_clock::now();
    auto milliseconds = std::chrono::duration_cast<std::chrono::milliseconds>(stop - start).count();
    std::cout << "Runtime: " << milliseconds << "ms" << std::endl;
    checkCudaErrorCode(errorCode, "kronmult_batched");

    return milliseconds;
}

/*
 * Runs benchmarks of increasing sizes and displays the results
 */
int main()
{
    std::cout << "Starting benchmark." << std::endl;
    // gets basic information on the cuda device
    int deviceId;
    cudaGetDevice(&deviceId);
    int threadsPerBlock;
    cudaDeviceGetAttribute(&threadsPerBlock, cudaDevAttrMaxThreadsPerBlock, deviceId);
    cudaDeviceProp deviceProp;
    const cudaError errorCode = cudaGetDeviceProperties(&deviceProp, deviceId);
    checkCudaErrorCode(errorCode, "cudaSetDevice");
    std::cout << "GPU device:" << deviceId
              << " threadsAvailable:" << threadsPerBlock
              << " architecture:" << deviceProp.major << '.' << deviceProp.minor
              << std::endl;

    // running the benchmarks
    auto toy = runBench(4, 1, 2, "toy");
    auto small = runBench(4, 2, 4, "small");
    auto medium = runBench(6, 3, 6, "medium");
    auto large = runBench(8, 6, 7, "large");
    auto realistic = runBench(8, 6, 9, "realistic");

    // display results
    std::cout << std::endl << "Results:" << std::endl
              << "toy: " << toy << "ms" << std::endl
              << "small: " << small << "ms" << std::endl
              << "medium: " << medium << "ms" << std::endl
              << "large: " << large << "ms" << std::endl
              << "realistic: " << realistic << "ms" << std::endl;
}

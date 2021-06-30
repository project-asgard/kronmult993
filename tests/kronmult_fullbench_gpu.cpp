#include "utils/utils_gpu.h"
#include <chrono>
#include <iostream>
#include <kronmult.cuh>
#include "utils/batch_size.h"

// change this to run the bench in another precision
using Number = double;

/*
 * runs a benchmark with the given parameters
 * `nb_distinct_outputs` modelizes the fact that most outputs are identical
 */
long runBench(int const degree, int const dimension, int const grid_level, std::string const benchName,
              int const nb_distinct_outputs = 5)
{
    // Kronmult parameters
    int const matrix_size  = degree;
    int const matrix_count = dimension;
    int const size_input   = pow_int(matrix_size, matrix_count);
    int const matrix_stride = 67; // large prime integer, modelize the fact that columns are not adjascent in memory
    int const batch_count = compute_batch_size(degree, dimension, grid_level, nb_distinct_outputs);
    std::cout << benchName << " benchcase"
              << " batch_count:" << batch_count << " matrix_size:" << matrix_size
              << " matrix_count:" << matrix_count << " size_input:" << size_input
              << " nb_distinct_outputs:" << nb_distinct_outputs << std::endl;

    // allocates a problem
    // we do not put data in the vectors/matrices as it doesn't matter here
    std::cout << "Starting allocation." << std::endl;
    DeviceArrayBatch<Number> matrix_list_batched(matrix_size * matrix_stride, batch_count * matrix_count);
    DeviceArrayBatch<Number> input_batched(size_input, batch_count);
    DeviceArrayBatch<Number> workspace_batched(size_input, batch_count);
    DeviceArrayBatch_withRepetition<Number> output_batched(size_input, batch_count, nb_distinct_outputs);

    // runs kronmult several times and displays the average runtime
    std::cout << "Starting Kronmult" << std::endl;
    auto start                = std::chrono::high_resolution_clock::now();
    cudaError const errorCode = kronmult_batched(
        matrix_count, matrix_size, matrix_list_batched.rawPointer, matrix_stride, input_batched.rawPointer,
        output_batched.rawPointer, workspace_batched.rawPointer, batch_count);
    auto stop         = std::chrono::high_resolution_clock::now();
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
    cudaError const errorCode = cudaGetDeviceProperties(&deviceProp, deviceId);
    checkCudaErrorCode(errorCode, "cudaSetDevice");
    std::cout << "GPU device:" << deviceId << " threadsAvailable:" << threadsPerBlock
              << " architecture:" << deviceProp.major << '.' << deviceProp.minor << std::endl;

    // running the benchmarks
    std::vector<std::string> names;
    std::vector<long> times;
    for(int degree = 2; degree <= 10; degree++)
    {
        for(int dimension = 1; dimension <= 6; dimension++)
        {
            for(int level = 2; level <= 9; level++)
            {
                // run bench
                int const nb_distinct_outputs = 5;
                int const batch_count = compute_batch_size(degree, dimension, level, nb_distinct_outputs);
                std::string name = "degree:" + std::to_string(degree) + " dimension:" + std::to_string(dimension)
                                 + " level:" + std::to_string(level) + " batch-size:" + std::to_string(batch_count);
                auto time = runBench(degree, dimension, level, name, nb_distinct_outputs);
                // strore result
                names.push_back(name);
                times.push_back(time);
            }
        }
    }

    // display results
    std::cout << std::endl << "Results:" << std::endl;
    for(unsigned int i=0; i < names.size(); i++)
    {
        std::string name = names[i];
        long time = times[i];
        std::cout << name << ": " << time << "ms" << std::endl;
    }
}

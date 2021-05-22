#include <iostream>
#include <chrono>
#include <omp.h>
#include <kronmult.hpp>
#include "utils/utils_cpu.h"

// change this to run the bench in another precision
using Number = float;

/*
 * computes number^power for integers
 * does not care about performances
 * does not use std::pow as it does an implicit float conversion that could lead to rounding errors for high numbers
 */
int pow_int(const int number, const int power)
{
    if(power == 0) return 1;
    return number * pow_int(number, power-1);
}

/*
 * runs a benchmark with the given parameters
 * `nb_distinct_outputs` modelizes the fact that most outputs are identical
 */
long runBench(const int degree, const int dimension, const int grid_level, const std::string benchName, const int nb_distinct_outputs = 5)
{
    // Kronmult parameters
    // TODO find proper formula for batch count, current one generates batches too large to be allocated without the min
    const int matrix_size = degree;
    const int matrix_count = dimension;
    const int size_input = pow_int(matrix_size, matrix_count);
    const int matrix_stride = 67; // large prime integer, modelize the fact that columns are not adjascent in memory
    // nb_elements = batch_count * size_input * (2 + matrix_count*matrix_size*matrix_size) + nb_distinct_outputs*size_input;
    const long max_element_number = 395000000000;
    const int max_batch_count = (max_element_number - nb_distinct_outputs*size_input) / (size_input * (2 + matrix_count*matrix_size*matrix_size));
    const int batch_count = std::min(max_batch_count, pow_int(2, grid_level) * pow_int(grid_level, dimension-1));
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
    ArrayBatch<Number> matrix_list_batched(matrix_size * matrix_stride, batch_count * matrix_count);
    ArrayBatch<Number> input_batched(size_input, batch_count);
    ArrayBatch<Number> workspace_batched(size_input, batch_count);
    ArrayBatch_withRepetition<Number> output_batched(size_input, batch_count, nb_distinct_outputs);

    // runs kronmult several times and displays the average runtime
    std::cout << "Starting Kronmult" << std::endl;
    auto start = std::chrono::high_resolution_clock::now();
    kronmult_batched(matrix_count, matrix_size, matrix_list_batched.rawPointer,
                     matrix_stride, input_batched.rawPointer, output_batched.rawPointer,
                     workspace_batched.rawPointer, batch_count);
    auto stop = std::chrono::high_resolution_clock::now();
    auto milliseconds = std::chrono::duration_cast<std::chrono::milliseconds>(stop - start).count();
    std::cout << "Runtime: " << milliseconds << "ms" << std::endl;

    return milliseconds;
}

/*
 * Runs benchmarks of increasing sizes and displays the results
 */
int main()
{
    std::cout << "Starting benchmark (" << omp_get_num_procs() << " procs)." << std::endl;
    std::cout << "Using old kronmult version." << std::endl;

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
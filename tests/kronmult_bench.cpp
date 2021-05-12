#include <chrono>
#include <iostream>
#include <omp.h>
#include <openmp/kronmult.hpp>

/*
 * run a test on a single, large, case
 * as those are the one for which computing time matters most
 * TODO ask the asgard team for a *representative* large case
 */
int main()
{
    std::cout << "Starting benchmark." << std::endl;
    // bench parameters
    const int nb_calls = 10;
    // Asgard parameters
    const int dimensions = 6;
    const int grid_level = 1; // goes up to 4 but time goes up quicly and allocs can fail
    const int degree = 8;
    // Kronmult parameters
    const int batch_count_increase = 1; // used to increase batchcount by a given factor to get a larger case size
    const int batch_count = batch_count_increase * degree * pow_int(2, grid_level*dimensions);
    const int matrix_size = degree;
    const int size_input = pow_int(degree, dimensions);
    const int matrix_count = dimensions;
    const int matrix_stride = matrix_size; // TODO use realistic value here
    std::cout << "batch_count:" << batch_count
              << " matrix_size:" << matrix_size
              << " size_input:" << size_input
              << std::endl;

    // allocates a problem
    // we do not put data in memory as it doesn't matter here
    std::cout << "Starting allocation." << std::endl;
    auto matrix_list_batched = new double*[batch_count * matrix_count];
    auto input_batched = new double*[batch_count];
    auto output_batched = new double*[batch_count];
    auto workspace_batched = new double*[batch_count];
    #pragma omp parallel for
    for(int batch = 0; batch < batch_count; batch++)
    {
        for(int mat = 0; mat < matrix_count; mat++)
        {
            matrix_list_batched[batch * matrix_count + mat] = new double[matrix_size * matrix_stride];
        }
        input_batched[batch] = new double[size_input];
        output_batched[batch] = new double[size_input];
        workspace_batched[batch] = new double[size_input];
    }

    // runs kronmult several times and displays the average runtime
    std::cout << "Starting Kronmult (" << omp_get_num_procs() << " procs)." << std::endl;
    #ifdef KRONMULT_USE_BLAS
        std::cout << "BLAS detected properly." << std::endl;
    #endif
    auto start = std::chrono::high_resolution_clock::now();
    for(int i=1; i <= nb_calls; i++)
    {
        kronmult_batched(matrix_count, matrix_size, matrix_list_batched,
                         matrix_stride, input_batched, output_batched,
                         workspace_batched, batch_count);
        std::cerr << i << " / " << nb_calls << std::endl;
    }
    auto stop = std::chrono::high_resolution_clock::now();
    auto milliseconds = std::chrono::duration_cast<std::chrono::milliseconds>((stop - start) / nb_calls);
    std::cout << "Runtime: " << milliseconds.count() << "ms" << std::endl;

    // deallocates the problem
    std::cout << "Starting delete." << std::endl;
    // TODO simulate the fact that the output vector are often identical by using identical pointers
    for(int batch = 0; batch < batch_count; batch++)
    {
        for(int mat = 0; mat < matrix_count; mat++)
        {
            delete[] matrix_list_batched[batch * matrix_count + mat];
        }
        delete[] input_batched[batch];
        delete[] output_batched[batch];
        delete[] workspace_batched[batch];
    }
    delete[] matrix_list_batched;
    delete[] input_batched;
    delete[] output_batched;
    delete[] workspace_batched;
    std::cout << "Done.";
}

/*
 * some measures (login node, not fully reserved but late at night)
 *
 * BLAS
 * min runtime: 1515ms
 * transpose: 1201 (1496)
 * no atomic on reduction (ideal case): 1600
 * reduction parallel on size: 7148 (SUPER SLOW)
 * reduction parallel on batch: 1800
 *
 * No BLAS
 * base: 1200 => seem to be the fastest ?! (should ve validated on dedicated node)
 * no reduction: 1760 (often up to 1917)
 * no transpose no reduction: 1949
 * no transpose but reduction: 1873
 * nothing (simplest implem):1819 (1905) (as low as 1674)
 *
 * the no blas version appears to be the fastest!
 * that would require further, proper, testing
 * if it confirms, we can drop blas which would simplify the code and cmakes
 */
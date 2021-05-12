#include <chrono>
#include <iostream>
#include <openmp/kronmult.hpp>

/*
 * run a test on a single, large, case
 * as those are the one for which computing time matters most
 */
int main()
{
    // bench parameters
    const int nb_calls = 10;
    // asgard parameters
    const int dimensions = 6;
    const int grid_level = 4;
    const int degree = 8;
    // kronmult parameters
    const int batch_count = degree * pow_int(2, grid_level*dimensions);
    const int matrix_size = degree;
    const int size_input = pow_int(degree, dimensions);
    const int matrix_count = dimensions;
    const int matrix_stride = matrix_size; // TODO use realistic value here

    // allocates a problem
    // we do not put data in memory as it doesn't matter here
    auto matrix_list_batched = new double*[batch_count * matrix_count];
    auto input_batched = new double*[batch_count];
    auto output_batched = new double*[batch_count];
    auto workspace_batched = new double*[batch_count];
    for(int batch = 0; batch < batch_count; batch++)
    {
        for(int mat = 0; mat < matrix_count; mat++)
        {
            auto square_matrix = new double[matrix_size * matrix_stride];
            matrix_list_batched[batch * matrix_count + mat] = square_matrix;
        }
        input_batched[batch] = new double[size_input];
        output_batched[batch] = new double[size_input];
        workspace_batched[batch] = new double[size_input];
    }

    // runs kronmult several times and displays the average runtime
    std::cout << "Starting benchark." << std::endl;
    auto start = std::chrono::high_resolution_clock::now();
    for(int j=0; j < nb_calls; j++)
    {
        kronmult_batched(matrix_count, matrix_size, matrix_list_batched,
                         matrix_stride, input_batched, output_batched,
                         workspace_batched, batch_count);
    }
    auto stop = std::chrono::high_resolution_clock::now();
    std::cout << "Runtime: " << ((stop - start)/nb_calls).count() << std::endl;

    // deallocates the problem
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
}
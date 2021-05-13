#include <chrono>
#include <random>
#include <iomanip>
#include <iostream>
#include <openmp/linear_algebra.hpp>
#include <openmp/kronmult.hpp>

#include "utils.hpp"

const size_t TOTAL_ITERATIONS = 1;

template<typename T>
std::pair<double, double> test_kronmult_cpu(size_t size_input, size_t matrix_stride, size_t dimensions,
                                        size_t grid_level, size_t batch_count, size_t matrix_count, size_t matrix_size)
{
    T ** matrix_list_batched; // list of batched matrix lists(kronmult)
    T ** input_batched;// vector of solution before this explicit time advance time step
    T ** output_batched; // vector of solution after this explicit time advance time step
    T ** workspace_batched; // mandatory for local computation
    utils::initialize_pointers_host(&matrix_list_batched, &input_batched, &output_batched, &workspace_batched,
                        batch_count, matrix_count, dimensions, size_input, matrix_size, matrix_stride, grid_level);
// TODO if clock too small, add a loop to make several time the same execution
// TODO Maybe on different data so that there is no cache reuse effect
    auto start = std::chrono::high_resolution_clock::now();
    for(int j=0; j< TOTAL_ITERATIONS; j++)
    {
// kronmult_openmp::kronmult(matrix_count, matrix_size, matrix_list, matrix_stride, input,
//                      size_input, output, workspace, transpose_workspace);
        kronmult_openmp::kronmult_batched(matrix_count, matrix_size, matrix_list_batched,
                                          matrix_stride, input_batched, output_batched,
                                          workspace_batched, batch_count);
    }
    auto stop = std::chrono::high_resolution_clock::now();
    auto start_origin = std::chrono::high_resolution_clock::now();
    auto stop_origin = std::chrono::high_resolution_clock::now();
    switch(dimensions)
    {
    case 1: {
        start_origin = std::chrono::high_resolution_clock::now();
        for (int j = 0; j < TOTAL_ITERATIONS; j++)
        {
            kronmult1_xbatched<T>(matrix_size, matrix_list_batched, matrix_stride,
                                  output_batched, input_batched, workspace_batched,
                                  batch_count);
        }
        stop_origin = std::chrono::high_resolution_clock::now();
        break;
    }
    case 2: {
        start_origin = std::chrono::high_resolution_clock::now();
        for (int j = 0; j < TOTAL_ITERATIONS; j++)
        {
            kronmult2_xbatched<T>(matrix_size, matrix_list_batched, matrix_stride,
                                  output_batched, input_batched, workspace_batched,
                                  batch_count);
        }
        stop_origin = std::chrono::high_resolution_clock::now();
        break;
    }
    case 3: {
        start_origin = std::chrono::high_resolution_clock::now();
        for (int j = 0; j < TOTAL_ITERATIONS; j++)
        {
            kronmult3_xbatched<T>(matrix_size, matrix_list_batched, matrix_stride,
                                  output_batched, input_batched, workspace_batched,
                                  batch_count);
        }
        stop_origin = std::chrono::high_resolution_clock::now();
        break;
    }
    case 4: {
        start_origin = std::chrono::high_resolution_clock::now();
        for (int j = 0; j < TOTAL_ITERATIONS; j++)
        {
            kronmult4_xbatched<T>(matrix_size, matrix_list_batched, matrix_stride,
                                  output_batched, input_batched, workspace_batched,
                                  batch_count);
        }
        stop_origin = std::chrono::high_resolution_clock::now();
        break;
    }
    case 5: {
        start_origin = std::chrono::high_resolution_clock::now();
        for (int j = 0; j < TOTAL_ITERATIONS; j++)
        {
            kronmult5_xbatched<T>(matrix_size, matrix_list_batched, matrix_stride,
                                  output_batched, input_batched, workspace_batched,
                                  batch_count);
        }
        stop_origin = std::chrono::high_resolution_clock::now();
        break;
    }
    case 6: {
        start_origin = std::chrono::high_resolution_clock::now();
        for (int j = 0; j < TOTAL_ITERATIONS; j++)
        {
            kronmult6_xbatched<T>(matrix_size, matrix_list_batched, matrix_stride,
                                  output_batched, input_batched, workspace_batched,
                                  batch_count);
        }
        stop_origin = std::chrono::high_resolution_clock::now();
        break;
    }
    default:
        exit(-1);

    }
    // Matrix_size and degree are the same
// TODO: plot time/FLOPS
    T flops = std::pow(kronmult_openmp::pow_long(matrix_size, dimensions), 3.) * batch_count * TOTAL_ITERATIONS;
    T real_flops = std::pow(matrix_size, dimensions)* std::pow(matrix_size,2) * batch_count * TOTAL_ITERATIONS;
    T orig_flops = 12.*std::pow(matrix_size, dimensions+1) * batch_count * TOTAL_ITERATIONS;
    std::chrono::duration<T> diff = stop-start;
    std::chrono::duration<T> diff_origin = stop_origin-start_origin;
#ifdef DEBUG
    std::cerr << "Time 993: " << diff.count() << std::endl;
    std::cerr << "Time Origin: " << diff_origin.count() << std::endl;
    std::cerr << "Theoretical Flops/sec: " << flops/diff.count() << std::endl;
    std::cerr << "Real Flops/sec: " << real_flops/diff.count() << std::endl;
    std::cerr << "Orig Flops/sec: " << orig_flops/diff.count() << std::endl;
#endif
    for(int i=0; i< matrix_count; i++)
    {
        utils::free_wrapper(input_batched[i]);
        utils::free_wrapper(output_batched[i]);
        utils::free_wrapper(workspace_batched[i]);
        utils::free_wrapper(matrix_list_batched[i]);
    }
    utils::free_wrapper(matrix_list_batched);
    utils::free_wrapper(output_batched);
    utils::free_wrapper(workspace_batched);
    std::pair<double,double> result = {diff.count(), diff_origin.count()};
    return result;
}

#ifdef USE_GPU
template<typename T>
std::pair<double, double> test_kronmult_gpu(size_t size_input, size_t matrix_stride, size_t dimensions,
                   size_t grid_level, size_t batch_count, size_t matrix_count, size_t matrix_size)
{
    T ** matrix_list_batched; // list of batched matrix lists(kronmult)
    T ** input_batched;// vector of solution before this explicit time advance time step
    T ** output_batched; // vector of solution after this explicit time advance time step
    T ** workspace_batched; // mandatory for local computation
    /* Initialize all data on Host */
    utils::initialize_pointers_device(&matrix_list_batched, &input_batched, &output_batched, &workspace_batched,
                        batch_count, matrix_count, dimensions, size_input, matrix_size, matrix_stride, grid_level);
    cudaError_t check;
// TODO if clock too small, add a loop to make several time the same execution
// TODO Maybe on different data so that there is no cache reuse effect
    auto start = std::chrono::high_resolution_clock::now();
    for(int j=0; j< TOTAL_ITERATIONS; j++)
    {
        //TODO should we also measure the time to copy the data from device to host?
        // Should be the Same for both algorithm ...
        kronmult_gpu::kronmult_batched(matrix_count, matrix_size, matrix_list_batched,
                                          matrix_stride, input_batched, output_batched,
                                          workspace_batched, batch_count);
        cudaDeviceSynchronize();
    }
    auto stop = std::chrono::high_resolution_clock::now();
    auto start_origin = std::chrono::high_resolution_clock::now();
    auto stop_origin = std::chrono::high_resolution_clock::now();
    switch(dimensions)
    {
    case 1: {
        start_origin = std::chrono::high_resolution_clock::now();
        for (int j = 0; j < TOTAL_ITERATIONS; j++)
        {
            kronmult1_xbatched<T>(matrix_size, matrix_list_batched, matrix_stride,
                                  output_batched, input_batched, workspace_batched,
                                  batch_count);
        }
        stop_origin = std::chrono::high_resolution_clock::now();
        break;
    }
    case 2: {
        start_origin = std::chrono::high_resolution_clock::now();
        for (int j = 0; j < TOTAL_ITERATIONS; j++)
        {
            kronmult2_xbatched<T>(matrix_size, matrix_list_batched, matrix_stride,
                                  output_batched, input_batched, workspace_batched,
                                  batch_count);
        }
        stop_origin = std::chrono::high_resolution_clock::now();
        break;
    }
    case 3: {
        start_origin = std::chrono::high_resolution_clock::now();
        for (int j = 0; j < TOTAL_ITERATIONS; j++)
        {
            kronmult3_xbatched<T>(matrix_size, matrix_list_batched, matrix_stride,
                                  output_batched, input_batched, workspace_batched,
                                  batch_count);
        }
        stop_origin = std::chrono::high_resolution_clock::now();
        break;
    }
    case 4: {
        start_origin = std::chrono::high_resolution_clock::now();
        for (int j = 0; j < TOTAL_ITERATIONS; j++)
        {
            kronmult4_xbatched<T>(matrix_size, matrix_list_batched, matrix_stride,
                                  output_batched, input_batched, workspace_batched,
                                  batch_count);
        }
        stop_origin = std::chrono::high_resolution_clock::now();
        break;
    }
    case 5: {
        start_origin = std::chrono::high_resolution_clock::now();
        for (int j = 0; j < TOTAL_ITERATIONS; j++)
        {
            kronmult5_xbatched<T>(matrix_size, matrix_list_batched, matrix_stride,
                                  output_batched, input_batched, workspace_batched,
                                  batch_count);
        }
        stop_origin = std::chrono::high_resolution_clock::now();
        break;
    }
    case 6: {
        start_origin = std::chrono::high_resolution_clock::now();
        for (int j = 0; j < TOTAL_ITERATIONS; j++)
        {
            kronmult6_xbatched<T>(matrix_size, matrix_list_batched, matrix_stride,
                                  output_batched, input_batched, workspace_batched,
                                  batch_count);
        }
        stop_origin = std::chrono::high_resolution_clock::now();
        break;
    }
    default:
        exit(-1);

    }
// TODO: plot time/FLOPS
    T flops = std::pow(kronmult_openmp::pow_long(matrix_size, dimensions), 3.) * batch_count * TOTAL_ITERATIONS;
    T real_flops = std::pow(matrix_size, dimensions)* std::pow(matrix_size,2) * batch_count * TOTAL_ITERATIONS;
    T orig_flops = 12.*std::pow(matrix_size, dimensions+1) * batch_count * TOTAL_ITERATIONS;
    std::chrono::duration<T> diff = stop-start;
    std::chrono::duration<T> diff_origin = stop_origin-start_origin;
#ifdef DEBUG
    std::cerr << "Time 993: " << diff.count() << std::endl;
    std::cerr << "Time Origin: " << diff_origin.count() << std::endl;
    std::cerr << "Theoretical Flops/sec: " << flops/diff.count() << std::endl;
    std::cerr << "Real Flops/sec: " << real_flops/diff.count() << std::endl;
    std::cerr << "Orig Flops/sec: " << orig_flops/diff.count() << std::endl;
#endif
    for(int i=0; i< matrix_count; i++)
    {
        utils::free_wrapper(input_batched[i]);
        utils::free_wrapper(output_batched[i]);
        utils::free_wrapper(workspace_batched[i]);
        utils::free_wrapper(matrix_list_batched[i]);
    }
    utils::free_wrapper(matrix_list_batched);
    utils::free_wrapper(output_batched);
    utils::free_wrapper(workspace_batched);
    std::pair<double,double> result = {diff.count(), diff_origin.count()};
    return result;
}
#endif

int main(int ac, char * av[]){
    /* Tests reflects sizes used in ASGARD implemented PDE: degree and dimensions
     * Degrees go from 2 to 10 and dimensions from 1 to 6.
     * Level from 2 to 10.
     * degree == size_M
     * dimensions == number_of_matrices (kronmult)
     * batch_count == degree * pow(2, level*dimension)
     * size_input == pow(degree, dimension)
     * */
std:cerr << "degree" << ","
        << "grid_level" << ","
        << "dimensions" << ","
        << "batch_count" << ","
        << "double_precision" << ","
        << "size_input" << ","
        << "perf_993" << ","
        << "perf_origin"
    << std::endl;
    for(size_t dimensions = 1; dimensions <=6; dimensions++)
    {
        for(size_t grid_level = 2; grid_level <= 4; grid_level++)
        {
            for(size_t degree = 2; degree <=8; degree++)
            {

                size_t batch_count = max(pow_int(2, grid_level*dimensions),pow_int(2, 13));
                size_t matrix_size = degree;
                size_t size_input = pow_int(degree, dimensions);
                size_t matrix_count = dimensions;
                size_t matrix_stride = matrix_size;// TODO: What does it represents in Asgard?
                int double_precision = 1;
#ifdef USE_GPU
                auto perf = test_kronmult_gpu<double>(size_input, matrix_stride, dimensions, grid_level, batch_count,
                                                      matrix_count, matrix_size);
#endif
                auto perf = test_kronmult_cpu<double>(size_input, matrix_stride, dimensions, grid_level, batch_count,
                                      matrix_count, matrix_size);
                std::cerr << degree << ","
                        << grid_level << ","
                        << dimensions << ","
                        << batch_count << ","
                        << double_precision << ","
                        << size_input << ","
                        << perf.first << ","
                         << perf.second
                        << std::endl;
                double_precision = 0;
#ifdef USE_GPU
                perf = test_kronmult_gpu<float>(degree, size_input, matrix_stride, dimensions, grid_level, batch_count,
                                                matrix_count, matrix_size);
#endif
                perf = test_kronmult_cpu<float>(size_input, matrix_stride, dimensions, grid_level, batch_count,
                                     matrix_count, matrix_size);
                std::cerr << degree << ","
                        << grid_level << ","
                        << dimensions << ","
                        << batch_count << ","
                        << double_precision << ","
                        << size_input << ","
                        << perf.first << ","
                         << perf.second
                        << std::endl;
            }

        }
    }

    return 0;
}

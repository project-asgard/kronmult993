#include <iostream>
#include <chrono>
#include <kronmult.cuh>

// change this to run the bench in another precision
using Number = double;

/*
 * allocates an array of the given size on the device
 * similar to `new T[size]`
 */
template<typename T> T* cudaNew(size_t size)
{
    T* pointer;
    cudaMalloc((void**)&pointer, sizeof(T) * size);
    return pointer;
}

/*
 * runs a benchmark with the given parameters
 * `nb_distinct_outputs` modelizes the fact that most outputs are identical
 */
long runBench(const int degree, const int dimension, const int grid_level, const std::string benchName, const int nb_distinct_outputs = 5)
{
    // Kronmult parameters
    // TODO find proper formula for batch count, current one generates batches too large to be allocated without the min
    const int batch_count = std::min(pow_int(2,13), pow_int(2, grid_level) * pow_int(grid_level, dimension-1));
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
    auto matrix_list_batched = cudaNew<Number*>(batch_count * matrix_count);
    auto input_batched = cudaNew<Number*>(batch_count);
    auto output_batched = cudaNew<Number*>(batch_count);
    auto workspace_batched = cudaNew<Number*>(batch_count);
    // outputs that will be used
    auto actual_output_batched = new Number*[nb_distinct_outputs];
    for(int batch = 0; batch < nb_distinct_outputs; batch++)
    {
        actual_output_batched[batch] = cudaNew<Number>(size_input);
    }
    //TODO can you actually ut values in device arrays like that?
    #pragma omp parallel for
    for(int batch = 0; batch < batch_count; batch++)
    {
        for(int mat = 0; mat < matrix_count; mat++)
        {
            matrix_list_batched[batch * matrix_count + mat] = cudaNew<Number>(matrix_size * matrix_stride);
        }
        input_batched[batch] = cudaNew<Number>(size_input);
        workspace_batched[batch] = cudaNew<Number>(size_input);
        // represents output vector reuse
        const int output_number = (batch * nb_distinct_outputs) / batch_count;
        output_batched[batch] = actual_output_batched[output_number];
    }

    // runs kronmult several times and displays the average runtime
    std::cout << "Starting Kronmult" << std::endl;
    auto start = std::chrono::high_resolution_clock::now();
    kronmult_batched(matrix_count, matrix_size, matrix_list_batched,
                     matrix_stride, input_batched, output_batched,
                     workspace_batched, batch_count);
    auto stop = std::chrono::high_resolution_clock::now();
    auto milliseconds = std::chrono::duration_cast<std::chrono::milliseconds>(stop - start).count();
    std::cout << "Runtime: " << milliseconds << "ms" << std::endl;

    // deallocates the problem
    std::cout << "Starting delete." << std::endl;
    for(int batch = 0; batch < batch_count; batch++)
    {
        for(int mat = 0; mat < matrix_count; mat++)
        {
            cudaFree(matrix_list_batched[batch * matrix_count + mat]);
        }
        cudaFree(input_batched[batch]);
        cudaFree(workspace_batched[batch]);
        if(batch < nb_distinct_outputs) cudaFree(actual_output_batched[batch]);
    }
    cudaFree(actual_output_batched);
    cudaFree(matrix_list_batched);
    cudaFree(input_batched);
    cudaFree(output_batched);
    cudaFree(workspace_batched);

    return milliseconds;
}

/*
 * Runs benchmarks of increasing sizes and displays the results
 */
int main()
{
    std::cout << "Starting benchmark (GPU)." << std::endl;

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

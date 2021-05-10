#include <chrono>
#include <malloc.h>
#include <random>

#include "../algo-origin/kronmult/xbatched/xbatched.hpp"
#include <iomanip>
#include <kronmult_openmp.hpp>
#include <openmp/kronmult.hpp>
#define  DEBUG 1
static constexpr size_t TOTAL_ITERATIONS = 10;

template<typename  T>
void random_init(T X[], int nb_row_X, int nb_col_X, int stride)
{
    static bool first = true;
    if (first)
        srand(NULL);
    first = false;
    for (size_t rowindex = 0; rowindex < nb_row_X; rowindex++)
    {
        for(size_t colindex=0 ; colindex<nb_col_X; colindex++)
        {
            X[kronmult_openmp::colmajor(rowindex, colindex, stride)] =
                static_cast<T>(random()) / static_cast<T>(INT64_MAX);
        }
    }
}

template<typename  T>
void value_init(T X[], int nb_row_X, int nb_col_X, int stride, T value)
{
    for (size_t rowindex = 0; rowindex < nb_row_X; rowindex++)
    {
        for(size_t colindex=0 ; colindex<nb_col_X; colindex++)
        {
            X[kronmult_openmp::colmajor(rowindex, colindex, stride)] = value;
        }
    }
}

void display_debug(size_t degree, size_t size_input, size_t matrix_stride, size_t dimensions,
                   size_t grid_level, size_t batch_count)
{
    std::cerr
        << "Square Matrix Size (skinny) == Degree: " << degree
        << " Tall matrix size == size input: " << size_input
        << " Coefficient matrix stride: " << matrix_stride
        << " Matrix count in kronmult == Dimensions: " << dimensions
        << " grid level: " << grid_level
        << " batch count: " << batch_count
        << std::endl;
}

template<typename T>
void test_kronmult(size_t degree, size_t size_input, size_t matrix_stride, size_t dimensions,
                   size_t grid_level, size_t batch_count, size_t matrix_count, size_t matrix_size)
{
    T ** matrix_list_batched; // list of batched matrix lists(kronmult)
    T ** input_batched;// vector of solution before this explicit time advance time step
    T ** output_batched; // vector of solution after this explicit time advance time step
    T ** workspace_batched; // mandatory for local computation
    matrix_list_batched = (T **) malloc(sizeof(T *) * batch_count * matrix_count);
    input_batched = (T **) malloc(sizeof(T*) * batch_count);
    output_batched = (T **) malloc(sizeof(T*) * batch_count);
    workspace_batched = (T **) malloc(sizeof(T*) * batch_count);

    display_debug(degree, size_input, matrix_stride, dimensions, grid_level, batch_count);
    if(NULL == matrix_list_batched
       || NULL == input_batched
       || NULL == output_batched
       || NULL == workspace_batched)
    {
        free(input_batched);
        free(output_batched);
        free(workspace_batched);
        free(matrix_list_batched);
        std::cerr << "Dynamic allocation failed." << std::endl;
        return;
    }
    for(int batchid = 0; batchid< batch_count; batchid++)
    {
        for(int matrix_count_index = 0; matrix_count_index < matrix_count; matrix_count_index++)
        {
            T *square_matrix = (T *)malloc(sizeof(T) * matrix_size * matrix_size);
            random_init(square_matrix, matrix_size, matrix_size, matrix_size);
            matrix_list_batched[batchid * matrix_count + matrix_count_index] = square_matrix;
        }
        input_batched[batchid] = (T*) malloc(sizeof(T) * size_input);
        random_init(input_batched[batchid], size_input, 1, size_input); // tensor matrix_size ^ matrix_number
        output_batched[batchid] = (T*) malloc(sizeof(T) * size_input);
        value_init<T>(output_batched[batchid], size_input, 1, size_input, 0.); // tensor
        workspace_batched[batchid] = (T*) malloc(sizeof(T) * size_input);
        value_init<T>(workspace_batched[batchid], size_input, 1, size_input, 0.); // tensor
    }
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
// TODO: plot time/FLOPS
    T flops = std::pow(kronmult_openmp::pow_long(degree, dimensions), 3.) * batch_count * TOTAL_ITERATIONS;
    T real_flops = std::pow(degree, dimensions)* std::pow(degree,2) * batch_count * TOTAL_ITERATIONS;
    T orig_flops = 12.*std::pow(degree, dimensions+1) * batch_count * TOTAL_ITERATIONS;
    std::chrono::duration<T> diff = stop-start;
    std::chrono::duration<T> diff_origin = stop_origin-start_origin;
    std::cerr << "Time 993: " << diff.count() << std::endl;
    std::cerr << "Time Origin: " << diff_origin.count() << std::endl;
    std::cerr << "Theoretical Flops/sec: " << flops/diff.count() << std::endl;
    std::cerr << "Real Flops/sec: " << real_flops/diff.count() << std::endl;
    std::cerr << "Orig Flops/sec: " << orig_flops/diff.count() << std::endl;
    for(int i=0; i< matrix_count; i++)
    {
        free(input_batched[i]);
        free(output_batched[i]);
        free(workspace_batched[i]);
        free(matrix_list_batched[i]);
    }
    free(matrix_list_batched);
    free(output_batched);
    free(workspace_batched);
//free(matrix_list_batched);
}

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
        << "perf"
        << std::endl;
    for(size_t dimensions = 2; dimensions <=6; dimensions++)
    {
        for(size_t grid_level = 2; grid_level <= 4; grid_level++)
        {
            for(size_t degree = 2; degree <=8; degree++)
            {
                size_t batch_count = degree * kronmult_openmp::pow_long(2, grid_level*dimensions);
                size_t matrix_size = degree;
                size_t size_input = kronmult_openmp::pow_int(degree, dimensions);
                size_t matrix_count = dimensions;
                size_t matrix_stride = matrix_size;// TODO: What does it represents in Asgard?
                int double_precision = 1;
                test_kronmult<double>(degree, size_input, matrix_stride, dimensions, grid_level, batch_count,
                                      matrix_count, matrix_size);
                std:cerr << degree << ","
                        << grid_level << ","
                        << dimensions << ","
                        << batch_count << ","
                        << double_precision << ","
                        << size_input << ","
                        << perf
                        << std::endl;
                double_precision = 0;
                test_kronmult<float>(degree, size_input, matrix_stride, dimensions, grid_level, batch_count,
                                     matrix_count, matrix_size);
                std:cerr << degree << ","
                        << grid_level << ","
                        << dimensions << ","
                        << batch_count << ","
                        << double_precision << ","
                        << size_input << ","
                        << perf
                        << std::endl;
            }

        }
    }

    return 0;
}

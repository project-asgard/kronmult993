#include <chrono>
#include <malloc.h>
#include <random>
#include <iomanip>
#include <iostream>
#include <openmp/linear_algebra.hpp>
#include <openmp/kronmult.hpp>

#define  DEBUG 1
const size_t TOTAL_ITERATIONS = 10;

template<typename  T>
void random_init(T X[], int nb_row_X, int nb_col_X, int stride)
{
    static bool first = true;
    if (first) srand(NULL);
    first = false;
    for (size_t rowindex = 0; rowindex < nb_row_X; rowindex++)
    {
        for(size_t colindex=0 ; colindex<nb_col_X; colindex++)
        {
            X[colmajor(rowindex, colindex, stride)] = static_cast<T>(random()) / static_cast<T>(INT64_MAX);
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
            X[colmajor(rowindex, colindex, stride)] = value;
        }
    }
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
    for(size_t dimensions = 1; dimensions <=6; dimensions++){
        for(size_t grid_level = 2; grid_level <= 4; grid_level++){
            for(size_t degree = 2; degree <=8; degree++){

                size_t batch_count = degree * pow_int(2, int(grid_level*dimensions));
                size_t matrix_size = degree;
                size_t size_input = pow_int(degree, dimensions);
                size_t matrix_count = dimensions;
                size_t matrix_stride = matrix_size;// TODO: What does it represents in Asgard?
                #ifdef DEBUG
                std::cerr
                    << "Square Matrix Size (skinny) == Degree: " << degree
                    << " Tall matrix size == size input: " << size_input
                    << " Coefficient matrix stride: " << matrix_stride
                    << " Matrix count in kronmult == Dimensions: " << dimensions
                    << " grid level: " << grid_level
                    << " batch count: " << batch_count
                    << std::endl;
                #endif
                double ** matrix_list_batched = (double **) malloc(sizeof(double *) * batch_count * matrix_count);
                if(NULL == matrix_list_batched){
                    free(matrix_list_batched);
                    std::cerr << "Dynamic allocation failed." << std::endl;
                    std::cerr
                        << "Square Matrix Size (skinny) == Degree: " << degree
                        << " Tall matrix size == size input: " << size_input
                        << " Coefficient matrix stride: " << matrix_stride
                        << " Matrix count in kronmult == Dimensions: " << dimensions
                        << " grid level: " << grid_level
                        << " batch count: " << batch_count
                        << std::endl;
                    continue;
                }
                double ** input_batched;// vector of solution before this explicit time advance time step
                double ** output_batched; // vector of solution after this explicit time advance time step
                double ** workspace_batched; // mandatory for local computation
                input_batched = (double **) malloc(sizeof(double) * batch_count);
                output_batched = (double **) malloc(sizeof(double) * batch_count);
                workspace_batched = (double **) malloc(sizeof(double) * batch_count);
                if(NULL == input_batched
                   || NULL == output_batched
                   || NULL == workspace_batched){
                    free(input_batched);
                    free(output_batched);
                    free(workspace_batched);
                    std::cerr << "Dynamic allocation failed." << std::endl;
                    std::cerr
                        << "Square Matrix Size (skinny) == Degree: " << degree
                        << " Tall matrix size == size input: " << size_input
                        << " Coefficient matrix stride: " << matrix_stride
                        << " Matrix count in kronmult == Dimensions: " << dimensions
                        << " grid level: " << grid_level
                        << " batch count: " << batch_count
                        << std::endl;
                    continue;
                }
                for(int batchid = 0; batchid< batch_count; batchid++)
                {
                    for(int matrix_count_index = 0; matrix_count_index < matrix_count; matrix_count_index++)
                    {
                        double *square_matrix = (double *)malloc(sizeof(double) * matrix_size * matrix_size);
                        random_init(square_matrix, matrix_size, matrix_size, matrix_size);
                        matrix_list_batched[batchid * matrix_count + matrix_count_index] = square_matrix;
                    }
                    input_batched[batchid] = (double*) malloc(sizeof(double) * size_input);
                    random_init(input_batched[batchid], size_input, 1, size_input); // tensor matrix_size ^ matrix_number
                    output_batched[batchid] = (double*) malloc(sizeof(double) * size_input);
                    value_init<double>(output_batched[batchid], size_input, 1, size_input, 0.); // tensor
                    workspace_batched[batchid] = (double*) malloc(sizeof(double) * size_input);
                    value_init<double>(workspace_batched[batchid], size_input, 1, size_input, 0.); // tensor
                }
                // TODO if clock too small, add a loop to make several time the same execution
                // TODO Maybe on different data so that there is no cache reuse effect
                auto start = std::chrono::high_resolution_clock::now();
                for(int j=0; j < TOTAL_ITERATIONS; j++)
                {
                    // kronmult_openmp::kronmult(matrix_count, matrix_size, matrix_list, matrix_stride, input,
                    //                      size_input, output, workspace, transpose_workspace);
                    kronmult_batched(matrix_count, matrix_size, matrix_list_batched,
                                                      matrix_stride, input_batched, output_batched,
                                                      workspace_batched, batch_count);
                }
                auto stop = std::chrono::high_resolution_clock::now();
                // TODO: plot time/FLOPS
                double flops = pow_int(degree, dimensions * 3) * batch_count * TOTAL_ITERATIONS;
                double real_flops = pow_int(degree, dimensions + 2) * batch_count * TOTAL_ITERATIONS;
                double orig_flops = 12.*pow_int(degree, dimensions+1) * batch_count * TOTAL_ITERATIONS;
                std::chrono::duration<double> diff = stop-start;
                std::cout << "Time: " << diff.count() << std::endl;
                std::cout << "Theoretical Flops/sec: " << flops/diff.count() << std::endl;
                std::cout << "Real Flops/sec: " << real_flops/diff.count() << std::endl;
                std::cout << "Orig Flops/sec: " << orig_flops/diff.count() << std::endl;
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

        }
    }

    return 0;
}

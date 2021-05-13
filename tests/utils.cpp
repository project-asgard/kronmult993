#include <cstdlib>
#include <iostream>
#include "utils.hpp"
#include <openmp/linear_algebra.hpp>

template<typename  T>
void random_init(T X[], int nb_row_X, int nb_col_X, int stride)
{
    static bool first = true;
    if (first)
        srand((size_t)NULL);
    first = false;
    for (size_t rowindex = 0; rowindex < nb_row_X; rowindex++)
    {
        for(size_t colindex=0 ; colindex<nb_col_X; colindex++)
        {
            X[colmajor(rowindex, colindex, stride)] =
                static_cast<T>(random()) / static_cast<T>(RAND_MAX);
        }
    }
}

template<typename  T>
void init_array_pointer(T ** X_p, T* X,
                        size_t outer_size, size_t inner_size)
{
    // CPU code
    for (size_t outer_p_index= 0;  outer_p_index < outer_size; outer_p_index++)
    {
        X_p[outer_p_index] = &(X + outer_p_index*inner_size);
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

template<typename T>
void initialize_pointers_host( T *** matrix_list_batched_p, T *** input_batched_p, T *** output_batched_p,
                               T *** workspace_batched_p, size_t batch_count, size_t matrix_count, size_t dimensions,
                               size_t size_input, size_t matrix_size, size_t matrix_stride, size_t grid_level)
{
    T ** matrix_list_batched = new T*[batch_count * matrix_count];
    T ** input_batched = new T*[batch_count];
    T ** output_batched = new T*[batch_count];
    T ** workspace_batched = new T*[batch_count];
    if(NULL == matrix_list_batched
       || NULL == input_batched
       || NULL == output_batched
       || NULL == workspace_batched)
    {
        display_debug(matrix_size, size_input, matrix_stride, dimensions, grid_level, batch_count);
        delete [] input_batched;
        delete [] output_batched;
        delete [] workspace_batched;
        delete [] matrix_list_batched;
        std::cerr << "Dynamic allocation failed." << std::endl;
        return;
    }
    for(int batchid = 0; batchid< batch_count; batchid++)
    {
        for(int matrix_count_index = 0; matrix_count_index < matrix_count; matrix_count_index++)
        {
            T *square_matrix = new T[matrix_size * matrix_size];
            random_init(square_matrix, matrix_size, matrix_size, matrix_size);
            matrix_list_batched[batchid * matrix_count + matrix_count_index] = square_matrix;
        }
        input_batched[batchid] = new T[size_input];
        random_init(input_batched[batchid], size_input, 1, size_input); // tensor matrix_size ^ matrix_number
        output_batched[batchid] = new T[size_input];
        value_init<T>(output_batched[batchid], size_input, 1, size_input, 0.); // tensor
        workspace_batched[batchid] = new T[size_input];
        value_init<T>(workspace_batched[batchid], size_input, 1, size_input, 0.); // tensor
    }
    *matrix_list_batched_p = matrix_list_batched;
    *input_batched_p       = input_batched;
    *output_batched_p      = output_batched;
    *workspace_batched_p   = workspace_batched;
}
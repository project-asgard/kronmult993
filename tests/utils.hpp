#pragma once
#ifdef USE_GPU
// Generate random numbers with Cuda
#include <curand.h>
// CUDA runtime
#include <cuda_runtime.h>
// Helper functions and utilities to work with CUDA
#include <helper_cuda.h>
#endif
#include <kronmult_openmp.hpp>
namespace utils
{
#ifdef USE_GPU
    /* Generate matrix */
    void GPU_fill_rand(double *A, int N){
        curandGenerator_t prng;
        curandCreateGenerator (&prng, CURAND_RNG_PSEUDO_DEFAULT);
        curandSetPseudoRandomGeneratorSeed(prng, (unsigned long long) clock());
        curandGenerateUniformDouble(prng, A, N);
    }
#endif

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
                        static_cast<T>(random()) / static_cast<T>(INT64_MAX);
                }
            }
        }

    template<typename  T>
        void init_array_pointer(T ** X_p, T* X, 
                size_t outer_size, size_t inner_size)
        {
#ifdef USE_GPU
            // GPU code
            init_array_pointer<<<32>>>(X_p, X, outer_size, inner_size);
#else
            // CPU code
            for (size_t outer_p_index= 0;  outer_p_index < outer_size; outer_p_index++)
            {
                X_p[outer_p_index] = &(X + outer_p_index*inner_size);
            }
#endif
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

#ifdef USE_GPU
    template<typename T>
        void initialize_pointers_device( T *** matrix_list_batched_pp, T *** input_batched_pp, T *** output_batched_pp,
                T *** workspace_batched_pp, size_t batch_count, size_t matrix_count, size_t dimensions,
                size_t size_input, size_t matrix_size, size_t matrix_stride, size_t grid_level)
        {
            // TODO: should bne changed in template (device == true) function.
            T ** matrix_list_batched_p = (T **) malloc_wrapper(sizeof(T *) * batch_count * matrix_count,
                    true);
            T ** input_batched_p = (T **) malloc_wrapper(sizeof(T*) * batch_count, true);
            T * input_batched  =  (T*) malloc_wrapper(sizeof(T) * batch_count * size_input, true);
            T ** output_batched_p = (T **) malloc_wrapper(sizeof(T*) * batch_count, true);
            T * output_batched  =  (T*) malloc_wrapper(sizeof(T) * batch_count * size_input, true);
            T ** workspace_batched_p = (T **) malloc_wrapper(sizeof(T*) * batch_count, true);
            T * workspace_batched  =  (T*) malloc_wrapper(sizeof(T) * batch_count * size_input, true);
            if(NULL == matrix_list_batched_p
                    || NULL == input_batched_p
                    || NULL == output_batched_p
                    || NULL == workspace_batched_p)
            {
                display_debug(matrix_size, size_input, matrix_stride, dimensions, grid_level, batch_count);
                free_wrapper(input_batched_p, true);
                free_wrapper(output_batched_p, true);
                free_wrapper(workspace_batched_p,true);
                free_wrapper(matrix_list_batched_p,true);
                std::cerr << "Dynamic allocation failed." << std::endl;
                return;
            }
            // Initialization of 2dimensions array
            T *square_matrix = (T *) malloc_wrapper(sizeof(T) * matrix_size * matrix_size, true);
            random_init_flatarray_device(matrix_list_batched, batch_count * matrix_count * matrix_size * matrix_size);
            init_array_pointer(matrix_list_batched_p, matrix_list_batched, batch_count * matrix_count, matrix_size * matrix_size);
            // Initializing 1dimension arrays
            random_init_flatarray_device(input_batched, batch_count * size_input); // tensor matrix_size ^ matrix_number
            value_init_flatarray_device(output_batched, batch_count * size_input, 0.); // tensor matrix_size ^ matrix_number
            value_init_flatarray_device(workspace_batched, batch_count * size_input, 0.); // tensor matrix_size ^ matrix_number
            init_array_pointer(input_batched_p, workspace_batched, batch_count, size_input);
            init_array_pointer(output_batched_p, workspace_batched, batch_count, size_input);
            init_array_pointer(workspace_batched_p, workspace_batched, batch_count, size_input);

            *matrix_list_batched_pp = matrix_list_batched_p;
            *input_batched_pp       = input_batched_p;
            *output_batched_pp      = output_batched_p;
            *workspace_batched_pp   = workspace_batched_p;
        }
#endif

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

}

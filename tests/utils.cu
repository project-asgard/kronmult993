#include "utils_gpu.hpp"
// Generate random numbers with Cuda
#include <curand.h>
// CUDA runtime
#include <cuda_runtime.h>
// Helper functions and utilities to work with CUDA
//#include <helper_cuda.h>

void test(void){
}

    template <typename T>
__host__
void random_init_flatarray_device(T *A, size_t size)
{
    curandGenerator_t prng;
    curandCreateGenerator (&prng, CURAND_RNG_PSEUDO_DEFAULT);
    curandSetPseudoRandomGeneratorSeed(prng, (unsigned long long) clock());
    curandGenerateUniformDouble(prng, A, size);
}

    template <typename T>
__global__
void init_array_pointer_kernel(T ** X_p, T* X,
        size_t outer_size, size_t inner_size)
{
    int threadId = blockIdx.x * blockDim.x + threadIdx.x;
    if(threadId < outer_size)
        X_p[threadId] = &(X + threadId*inner_size);
}

    template<typename  T>
__host__
void init_array_pointer_device(T ** X_p, T* X,
        size_t outer_size, size_t inner_size)
{
    // GPU code
    constexpr size_t block_size = 128;
    constexpr size_t warp_size = 32;
    init_array_pointer_kernel <<<block_size, warp_size>>> (X_p, X, outer_size, inner_size);
}

template<>
__host__
void initialize_pointers_device( double *** matrix_list_batched_pp){
}
/*
    template<typename T>
__host__
void initialize_pointers_device( T *** matrix_list_batched_pp, T *** input_batched_pp, T *** output_batched_pp,
        T *** workspace_batched_pp, size_t batch_count, size_t matrix_count, size_t dimensions,
        size_t size_input, size_t matrix_size, size_t matrix_stride, size_t grid_level)
{
    // TODO: should bne changed in template (device == true) function.
    T ** matrix_list_batched_p;
    T ** input_batched_p;
    T ** output_batched_p;
    T ** workspace_batched_p;
    T * input_batched;
    T * output_batched;
    T * workspace_batched;
    cudaMalloc(&matrix_list_batched_p, sizeof(T *) * batch_count * matrix_count);
    cudaMalloc(&input_batched_p, sizeof(T*) * batch_count);
    cudaMalloc(&input_batched, sizeof(T) * batch_count * size_input);
    cudaMalloc(&output_batched_p, sizeof(T*) * batch_count);
    cudaMalloc(&output_batched, sizeof(T) * batch_count * size_input);
    cudaMalloc(&workspace_batched_p, sizeof(T*) * batch_count);
    cudaMalloc(&workspace_batched, sizeof(T) * batch_count * size_input);
    if(NULL == matrix_list_batched_p
            || NULL == input_batched_p
            || NULL == output_batched_p
            || NULL == workspace_batched_p)
    {
        //utils::display_debug(matrix_size, size_input, matrix_stride, dimensions, grid_level, batch_count);
        cudaFree(input_batched_p);
        cudaFree(output_batched_p);
        cudaFree(workspace_batched_p);
        cudaFree(matrix_list_batched_p);
        std::cerr << "Dynamic allocation failed." << std::endl;
        return;
    }
    // Initialization of 2dimensions array
    random_init_flatarray_device(matrix_list_batched_p, batch_count * matrix_count * matrix_size * matrix_size);
    init_array_pointer_device(matrix_list_batched_p, matrix_list_batched_p, batch_count * matrix_count, matrix_size * matrix_size);
    // Initializing 1dimension arrays
    random_init_flatarray_device(input_batched, batch_count * size_input); // tensor matrix_size ^ matrix_number
    value_init_flatarray_device(output_batched, batch_count * size_input, 0.); // tensor matrix_size ^ matrix_number
    value_init_flatarray_device(workspace_batched, batch_count * size_input, 0.); // tensor matrix_size ^ matrix_number
    init_array_pointer_device(input_batched_p, workspace_batched, batch_count, size_input);
    init_array_pointer_device(output_batched_p, workspace_batched, batch_count, size_input);
    init_array_pointer_device(workspace_batched_p, workspace_batched, batch_count, size_input);

    *matrix_list_batched_pp = matrix_list_batched_p;
    *input_batched_pp       = input_batched_p;
    *output_batched_pp      = output_batched_p;
    *workspace_batched_pp   = workspace_batched_p;
}*/

#include "kronmult.cuh"
#include <device_launch_parameters.h>
#include <type_traits>
#include <cublas_v2.h>

// TODO
//  - write doc for functions
//  - put linear algebra in .cuh file

/*
 * computes number^power for integers
 * does not care about performances
 * does not use std::pow as it does an implicit float conversion that could lead to rounding errors for large
 * numbers
 */
__host__ int pow_int(int const number, int const power)
{
    if (power == 0) return 1;
    return number * pow_int(number, power - 1);
}

template<typename T>
__host__ cublasStatus_t multiply_transpose_batched(cublasHandle_t& handle,
                                          T* input_batched[], const int nb_col_input_batched,
                                          T* matrix_batched[], const int matrix_size_batched, const int matrix_stride_batched,
                                          T* output_batched[], int nb_batch);

template<>
__host__ cublasStatus_t multiply_transpose_batched<double>(cublasHandle_t& handle,
                                                  double* input_batched[], const int nb_col_input_batched,
                                                  double* matrix_batched[], const int matrix_size_batched, const int matrix_stride_batched,
                                                  double* output_batched[], int nb_batch)
{
    cublasOperation_t should_transpose_input_batched = CUBLAS_OP_T;
    cublasOperation_t should_transpose_matrix_batched = CUBLAS_OP_T;
    double weight_product = 1.;
    double weight_output = 0.;
    // https://docs.nvidia.com/cuda/cublas/#cublas-lt-t-gt-gemmbatched
    cublasStatus_t errorCode = cublasDgemmBatched(handle, should_transpose_input_batched, should_transpose_matrix_batched,
                                                  nb_col_input_batched, matrix_size_batched, matrix_size_batched,
                                                  &weight_product,
                                                  input_batched, matrix_size_batched,
                                                  matrix_batched, matrix_stride_batched,
                                                  &weight_output, output_batched, nb_col_input_batched, nb_batch);
    return errorCode;
}

template<>
__host__ cublasStatus_t multiply_transpose_batched<float>(cublasHandle_t& handle,
                                                 float* input_batched[], const int nb_col_input_batched,
                                                 float* matrix_batched[], const int matrix_size_batched, const int matrix_stride_batched,
                                                 float* output_batched[], int nb_batch)
{
    cublasOperation_t should_transpose_input_batched = CUBLAS_OP_T;
    cublasOperation_t should_transpose_matrix_batched = CUBLAS_OP_T;
    float weight_product = 1.;
    float weight_output = 0.;
    // https://docs.nvidia.com/cuda/cublas/#cublas-lt-t-gt-gemmbatched
    cublasStatus_t errorCode = cublasSgemmBatched(handle, should_transpose_input_batched, should_transpose_matrix_batched,
                                                  nb_col_input_batched, matrix_size_batched, matrix_size_batched,
                                                  &weight_product,
                                                  input_batched, matrix_size_batched,
                                                  matrix_batched, matrix_stride_batched,
                                                  &weight_output, output_batched, nb_col_input_batched, nb_batch);
    return errorCode;
}

/*
 * reduce in a threadsafe way
 */
template<typename T>
__global__ void cuda_atomic_reduction(T* input_batched[], T* output_batched[], const int size_input)
{
    // gets the inputs for the algorithm
    // each block corresponds to a batch element
    const int batchId = blockIdx.x;
    T* input = input_batched[batchId];
    T* output = output_batched[batchId];

    // each thread t manage the input i such that i%t==0
    for(int i = threadIdx.x; i < size_input; i+=blockDim.x)
    {
        atomicAdd(&output[i], input[i]);
    }
}


/*
 * Calls the cuda kernel with proper thread parameters.
 * This function expects its inputs to already be on the device (GPU).
 *
 */
template<typename T>
__host__ cudaError cuda_kronmult_batched(const int matrix_count, const int matrix_size, T const * const matrix_list_batched[], const int matrix_stride,
                                         T* input_batched[], T* output_batched[], T* workspace_batched[], const int nb_batch)
{
    // gets handle on cublas
    cublasHandle_t handle;
    cublasStatus_t stat = cublasCreate(&handle);
    if (stat != CUBLAS_STATUS_SUCCESS) return cudaErrorUnknown;

    // used to store batch elements
    T** matrix_batched;
    cudaError errorCode = cudaMalloc((void**)&matrix_batched, nb_batch*sizeof(T*));
    if(errorCode != cudaSuccess) return errorCode;

    // algorithm 993
    // numbers of elements in the input vector
    int size_input = pow_int(matrix_size, matrix_count);
    // how many column should `input` have for the multiplications to be legal
    const int nb_col_input = size_input / matrix_size;

    // iterates on the matrices from the last to the one just before first
    // puts the result in input_batched
    for(int m = matrix_count-1; m >= 0; m--)
    {
        // memcopy with stride
        // https://stackoverflow.com/a/13536437/6422174
        // extracts the m^th matrices into matrix_batched
        errorCode = cudaMemcpy2D(matrix_batched, sizeof(T*),
                                 &matrix_list_batched[m], matrix_count*sizeof(T*),
                                 sizeof(T*), // stride
                                 nb_batch, // number of elments to move
                                 cudaMemcpyDeviceToDevice);
        if(errorCode != cudaSuccess) return errorCode;

        multiply_transpose_batched<T>(handle, input_batched, nb_col_input,
                                      matrix_batched, matrix_size, matrix_stride,
                                      workspace_batched, nb_batch);

        std::swap(input_batched, workspace_batched);
    }

    // reduction
    // each block will be a batch element
    // each thread will be a subset of the lines in input
    int deviceId;
    cudaGetDevice(&deviceId);
    int threadsPerBlock;
    cudaDeviceGetAttribute(&threadsPerBlock, cudaDevAttrMaxThreadsPerBlock, deviceId);
    if(size_input < threadsPerBlock) threadsPerBlock = size_input;
    // paralelize on batch elements
    cuda_atomic_reduction<<<nb_batch, threadsPerBlock>>>(input_batched, output_batched, size_input);
    // waits for kernel to succeed
    errorCode = cudaDeviceSynchronize();
    if(errorCode != cudaSuccess) return errorCode;

    // frees memory and returns error code
    errorCode = cudaFree(matrix_batched);
    if(errorCode != cudaSuccess) return errorCode;
    cublasDestroy(handle);
    return errorCode;
}

/*
 * double specialization of kronmult_batched
 */
template<>
__host__ cudaError kronmult_batched<double>(int const matrix_count, int const matrix_size,
                                            double const *const matrix_list_batched[],
                                            int const matrix_stride, double *input_batched[],
                                            double *output_batched[], double *workspace_batched[],
                                            int const nb_batch)
{
    return cuda_kronmult_batched(matrix_count, matrix_size, matrix_list_batched, matrix_stride, input_batched,
                                 output_batched, workspace_batched, nb_batch);
}

/*
 * float specialization of kronmult_batched
 */
template<>
__host__ cudaError kronmult_batched<float>(int const matrix_count, int const matrix_size,
                                           float const *const matrix_list_batched[], int const matrix_stride,
                                           float *input_batched[], float *output_batched[],
                                           float *workspace_batched[], int const nb_batch)
{
    return cuda_kronmult_batched(matrix_count, matrix_size, matrix_list_batched, matrix_stride, input_batched,
                                 output_batched, workspace_batched, nb_batch);
}

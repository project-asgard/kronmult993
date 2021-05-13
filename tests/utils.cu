#include "utils.hpp"

 void init_array_pointer(T ** X_p, T* X, 
                size_t outer_size, size_t inner_size)
{
    int threadId = blockIdx.x * blockDim.x + threadIdx.x;
    if(threadId < outer_size)
        X_p[threadId] = &(X + threadId*inner_size);
}
    int threadId = blockIdx.x * blockDim.x + threadIdx.x;
template<typename  T>
__device__ void init_array_pointer(T ** X_p, T* X, 
                size_t outer_size, size_t inner_size);
        {
            // CPU code
            for (size_t outer_p_index= 0;  outer_p_index < outer_size; outer_p_index++)
            {
                X_p[outer_p_index] = &(X + outer_p_index*inner_size);
            }
            constexpr int block_size = 128;
            constexpr int warp_size = 128;
            assert(block_size * warp_size > outer_size);
            // GPU code
            init_array_pointer<<<block_size, warp_size>>>(X_p, X, outer_size, inner_size);
        }

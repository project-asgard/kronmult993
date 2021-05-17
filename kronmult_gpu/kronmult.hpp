//#pragma once
#include <stdexcept>
#include <cuda_runtime.h>
//#include <cuda.h>

namespace algo_993
{
    template <typename P>
        cudaError kronmult_batched(const int matrix_count, const int matrix_size,
                P const * const matrix_list_batched[], const int matrix_stride,
                P* input_batched[], P* output_batched[], P* workspace_batched[],
                const int nb_batch);
}

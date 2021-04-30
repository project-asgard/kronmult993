#pragma once

/*
 * operations extracted from the old version's tests
 */

static inline void host2gpu(void *dest, void *src, size_t nbytes)
{
    #ifdef USE_GPU
        cudaError_t istat = cudaMemcpy(dest, src, nbytes, cudaMemcpyHostToDevice);
        assert(istat == cudaSuccess);
    #else
        memcpy(dest, src, nbytes);
    #endif
}

static inline void gpu2host(void *dest, void *src, size_t nbytes)
{
    #ifdef USE_GPU
        cudaError_t istat = cudaMemcpy(dest, src, nbytes, cudaMemcpyDeviceToHost);
        assert(istat == cudaSuccess);
    #else
        memcpy(dest, src, nbytes);
    #endif
}

static inline void *myalloc(size_t const nbytes)
{
    void *devPtr = nullptr;
    #ifdef USE_GPU
        cudaError_t istat = cudaMalloc(&devPtr, nbytes);
        assert(istat == cudaSuccess);
    #else
        devPtr = malloc(nbytes);
    #endif
    assert(devPtr != nullptr);
    return (devPtr);
}

static inline void myfree(void *devPtr)
{
    #ifdef USE_GPU
        cudaError_t istat = cudaFree(devPtr);
        assert(istat == cudaSuccess);
    #else
        free(devPtr);
    #endif
}
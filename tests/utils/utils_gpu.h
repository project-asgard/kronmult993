#pragma once
#include <vector>
#include <stdexcept>
#include <cuda.h>

/*
 * throws an exception if an errorCode is meet
 */
void checkCudaErrorCode(const cudaError errorCode, const std::string& functionName)
{
    if(errorCode != cudaSuccess)
    {
        throw std::runtime_error(functionName + ": " + cudaGetErrorString(errorCode));
    }
}

/*
 * allocates an array of the given size on the device
 * returns a pointer to the array
 * similar to `new T[size]`
 */
template<typename T> T* cudaNew(size_t size)
{
    T* pointer;
    const cudaError errorCode = cudaMalloc((void**)&pointer, sizeof(T) * size);
    checkCudaErrorCode(errorCode, "cudaMalloc");
    return pointer;
}

/*
 * wraps an array on device with a proper constructor and destructor
 */
template<typename T>
class DeviceArray
{
  public:
    // lets you access the device pointer directly
    T* rawPointer;

    // creates an array of the given size on device
    explicit DeviceArray(const size_t size)
    {
        rawPointer = cudaNew<T>(size);
    }

    // releases the memory
    ~DeviceArray()
    {
        const cudaError errorCode = cudaFree(rawPointer);
        checkCudaErrorCode(errorCode, "cudaFree (~DeviceArray)");
    }
};

/*
 * wraps a batch of arrays on device with a proper constructor and destructor
 */
template<typename T>
class DeviceArrayBatch
{
  public:
    // lets you access the device pointer directly
    T** rawPointer;
    // host vector full of pointers to device arrays
    std::vector<T*> batchRawPointers;

    // creates an array of `nb_arrays` arrays of size `array_sizes` on device
    DeviceArrayBatch(const size_t array_sizes, const size_t nb_arrays): batchRawPointers(nb_arrays)
    {
        // allocates all the batch elements on device
        for(unsigned int i=0; i<nb_arrays; i++)
        {
            batchRawPointers[i] = cudaNew<T>(array_sizes);
        }
        // copy the pointers to batch elements on device
        rawPointer = cudaNew<T*>(nb_arrays);
        const cudaError errorCode = cudaMemcpy(rawPointer, batchRawPointers.data(), nb_arrays * sizeof(T*), cudaMemcpyHostToDevice);
        checkCudaErrorCode(errorCode, "cudaMemcpy (DeviceArrayBatch)");
    }

    // releases the memory
    ~DeviceArrayBatch()
    {
        // frees the batch elements
        for(T* ptr: batchRawPointers)
        {
            const cudaError errorCode = cudaFree(ptr);
            checkCudaErrorCode(errorCode, "cudaFree (~DeviceArrayBatch[])");
        }
        // free the array of batch elements
        const cudaError errorCode = cudaFree(rawPointer);
        checkCudaErrorCode(errorCode, "cudaFree (~DeviceArrayBatch)");
    }
};

/*
 * wraps a batch of arrays on device with a proper constructor and destructor
 * make sure that the batch onctains only a given number of distinct elements
 */
template<typename T>
class DeviceArrayBatch_withRepetition
{
  public:
    // lets you access the device pointer directly
    T** rawPointer;
    // host vector full of pointers to device arrays
    std::vector<T*> batchRawPointers;

    // creates an array of `nb_arrays` arrays of size `array_sizes` on device
    // contains only `nb_arrays_distinct` distinct elemnts
    DeviceArrayBatch_withRepetition(const size_t array_sizes, const size_t nb_arrays, const size_t nb_arrays_distinct=5): batchRawPointers(nb_arrays_distinct)
    {
        // allocates all the batch elements on device
        for(unsigned int i=0; i<nb_arrays_distinct; i++)
        {
            batchRawPointers[i] = cudaNew<T>(array_sizes);
        }
        // allocates blocks of identical batch elements
        std::vector<T*> batchRawPointers_withRepetition(nb_arrays);
        for(unsigned int i=0; i<nb_arrays; i++)
        {
            const int ptr_index = (i * nb_arrays_distinct) / nb_arrays;
            batchRawPointers_withRepetition[i] = batchRawPointers[ptr_index];
        }
        // copy the pointers to batch elements on device
        rawPointer = cudaNew<T*>(nb_arrays);
        const cudaError errorCode = cudaMemcpy(rawPointer, batchRawPointers_withRepetition.data(), nb_arrays * sizeof(T*), cudaMemcpyHostToDevice);
        checkCudaErrorCode(errorCode, "cudaMemcpy (DeviceArrayBatch_withRepetition)");
    }

    // releases the memory
    ~DeviceArrayBatch_withRepetition()
    {
        // frees the batch elements
        for(T* ptr: batchRawPointers)
        {
            const cudaError errorCode = cudaFree(ptr);
            checkCudaErrorCode(errorCode, "cudaFree (~DeviceArrayBatch_withRepetition[])");
        }
        // free the array of batch elements
        const cudaError errorCode = cudaFree(rawPointer);
        checkCudaErrorCode(errorCode, "cudaFree (~DeviceArrayBatch_withRepetition)");
    }
};
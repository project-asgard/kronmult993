#pragma once
#include <vector>
#include <stdexcept>
#include <cuda_runtime.h>
#include <cuda.h>
/*
 * computes number^power for integers
 * does not care about performances
 * does not use std::pow as it does an implicit float conversion that could lead to rounding errors for high
 * numbers
 */
int pow_int(const int number, const int power)
{
    if(power == 0) return 1;
    return number * pow_int(number, power-1);
}


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
 *
 * use `cudaMallocManaged` to be able to touche memory on host and allocate more than available on GPU
 */
template<typename T> T* cudaNew(size_t size)
{
    T* pointer;
    const cudaError errorCode = cudaMallocManaged((void**)&pointer, sizeof(T) * size);
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
    size_t nb_arrays;

    // creates an array of `nb_arrays` arrays of size `array_sizes` on device
    DeviceArrayBatch(const size_t array_sizes, const size_t nb_arrays_arg): nb_arrays(nb_arrays_arg)
    {
        rawPointer = cudaNew<T*>(nb_arrays);
        for(unsigned int i=0; i<nb_arrays; i++)
        {
            rawPointer[i] = cudaNew<T>(array_sizes);
        }
    }

    // releases the memory
    ~DeviceArrayBatch()
    {
        // frees the batch elements
        for(unsigned int i=0; i<nb_arrays; i++)
        {
            const cudaError errorCode = cudaFree(rawPointer[i]);
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
    size_t nb_arrays_distinct;

    // creates an array of `nb_arrays` arrays of size `array_sizes` on device
    // contains only `nb_arrays_distinct` distinct elemnts
    DeviceArrayBatch_withRepetition(const size_t array_sizes, const size_t nb_arrays, const size_t nb_arrays_distinct_arg=5): nb_arrays_distinct(nb_arrays_distinct_arg)
    {
        rawPointer = cudaNew<T*>(nb_arrays);
        for(unsigned int i=0; i<nb_arrays_distinct; i++)
        {
            rawPointer[i] = cudaNew<T>(array_sizes);
        }
        // allocates blocks of identical batch elements
        for(unsigned int i=nb_arrays_distinct; i<nb_arrays; i++)
        {
            const int ptr_index = (i * nb_arrays_distinct) / nb_arrays;
            rawPointer[i] = rawPointer[ptr_index];
        }
    }

    // releases the memory
    ~DeviceArrayBatch_withRepetition()
    {
        // frees the batch elements
        for(unsigned int i=0; i<nb_arrays_distinct; i++)
        {
            const cudaError errorCode = cudaFree(rawPointer[i]);
            checkCudaErrorCode(errorCode, "cudaFree (~DeviceArrayBatch[])");
        }
        // free the array of batch elements
        const cudaError errorCode = cudaFree(rawPointer);
        checkCudaErrorCode(errorCode, "cudaFree (~DeviceArrayBatch)");
    }
};

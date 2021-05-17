#pragma once
#include <vector>

/*
 * computes number^power for integers
 * does not care about performances
 * does not use std::pow as it does an implicit float conversion that could lead to rounding errors for high numbers
 */
int pow_int_utils(const int number, const int power)
{
    if(power == 0) return 1;
    return number * pow_int_utils(number, power-1);
}

/*
 * wraps a batch of raw arrays with a proper constructor and destructor
 */
template<typename T>
class ArrayBatch
{
  public:
    // lets you access the device pointer directly
    T** rawPointer;
    // host vector full of pointers to device arrays
    std::vector<T*> batchRawPointers;

    // creates an array of `nb_arrays` arrays of size `array_sizes`
    ArrayBatch(const size_t array_sizes, const size_t nb_arrays): batchRawPointers(nb_arrays)
    {
        // allocates all the batch elements
        #pragma omp parallel for
        for(unsigned int i=0; i<nb_arrays; i++)
        {
            batchRawPointers[i] = new T[array_sizes];
        }
        // pointer to the batch elements
        rawPointer = batchRawPointers.data();
    }

    // releases the memory
    ~ArrayBatch()
    {
        for(auto ptr: batchRawPointers)
        {
            delete[] ptr;
        }
    }
};

/*
 * wraps a batch of arrays on device with a proper constructor and destructor
 * make sure that the batch contains only a given number of distinct elements
 */
template<typename T>
class ArrayBatch_withRepetition
{
  public:
    // lets you access the device pointer directly
    T** rawPointer;
    // host vector full of pointers
    std::vector<T*> batchRawPointers;
    std::vector<T*> batchRawPointers_withRepetition;

    // creates an array of `nb_arrays` arrays of size `array_sizes`
    // contains only `nb_arrays_distinct` distinct elements
    ArrayBatch_withRepetition(const size_t array_sizes, const size_t nb_arrays, const size_t nb_arrays_distinct=5): batchRawPointers(nb_arrays_distinct), batchRawPointers_withRepetition(nb_arrays)
    {
        // allocates all the batch elements on device
        for(unsigned int i=0; i<nb_arrays_distinct; i++)
        {
            batchRawPointers[i] = new T[array_sizes];
        }
        // allocates blocks of identical batch elements
        #pragma omp parallel for
        for(unsigned int i=0; i<nb_arrays; i++)
        {
            const int ptr_index = (i * nb_arrays_distinct) / nb_arrays;
            batchRawPointers_withRepetition[i] = batchRawPointers[ptr_index];
        }
        // copy the pointers to batch elements on device
        rawPointer = batchRawPointers_withRepetition.data();
    }

    // releases the memory
    ~ArrayBatch_withRepetition()
    {
        for(auto ptr: batchRawPointers)
        {
            delete[] ptr;
        }
    }
};

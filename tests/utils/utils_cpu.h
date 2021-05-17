#pragma once
#include <vector>
#include "data_generation.h"

/*
 * wraps a batch of arrays on device with a proper constructor and destructor
 */
template<typename T>
class ArrayBatch
{
  public:
    // lets you access the device pointer directly
    T** rawPointer;
    size_t nb_arrays;

    // creates an array of `nb_arrays` arrays of size `array_sizes` on device
    ArrayBatch(const size_t array_sizes, const size_t nb_arrays_arg, const bool should_initialize_data=false): nb_arrays(nb_arrays_arg)
    {
        // random number generator for the data generation
        std::random_device rd{};
        std::default_random_engine rng{rd()};
        // allocating the arrays
        rawPointer = new T*[nb_arrays];
        for(unsigned int i=0; i<nb_arrays; i++)
        {
            rawPointer[i] = new T[array_sizes];
            if(should_initialize_data) fillArray(rawPointer[i], nb_arrays, rng);
        }
    }

    // releases the memory
    ~ArrayBatch()
    {
        // frees the batch elements
        for(unsigned int i=0; i<nb_arrays; i++)
        {
            delete[] rawPointer[i];
        }
        // free the array of batch elements
        delete[] rawPointer;
    }
};

/*
 * wraps a batch of arrays on device with a proper constructor and destructor
 * make sure that the batch onctains only a given number of distinct elements
 */
template<typename T>
class ArrayBatch_withRepetition
{
  public:
    // lets you access the device pointer directly
    T** rawPointer;
    size_t nb_arrays_distinct;

    // creates an array of `nb_arrays` arrays of size `array_sizes` on device
    // contains only `nb_arrays_distinct` distinct elemnts
    ArrayBatch_withRepetition(const size_t array_sizes, const size_t nb_arrays, const size_t nb_arrays_distinct_arg=5, const bool should_initialize_data=false): nb_arrays_distinct(nb_arrays_distinct_arg)
    {
        // random number generator for the data generation
        std::random_device rd{};
        std::default_random_engine rng{rd()};
        // allocating the arrays
        rawPointer = new T*[nb_arrays];
        for(unsigned int i=0; i<nb_arrays_distinct; i++)
        {
            rawPointer[i] = new T[array_sizes];
            if(should_initialize_data) fillArray(rawPointer[i], nb_arrays, rng);
        }
        // allocates blocks of identical batch elements
        for(unsigned int i=nb_arrays_distinct; i<nb_arrays; i++)
        {
            const int ptr_index = (i * nb_arrays_distinct) / nb_arrays;
            rawPointer[i] = rawPointer[ptr_index];
        }
    }

    // releases the memory
    ~ArrayBatch_withRepetition()
    {
        // frees the batch elements
        for(unsigned int i=0; i<nb_arrays_distinct; i++)
        {
            delete[] rawPointer[i];
        }
        // free the array of batch elements
        delete[] rawPointer;
    }
};
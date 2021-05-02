#pragma once
#include <iostream>
#include <omp.h>

/*
 * puts the sum of the intermediate outputs in the first output
 * inspired by https://stackoverflow.com/a/18722879/6422174
 */
template<typename T>
void reduction_slice_rec(T* intermediate_output_batched[], int first_batch, int last_batch, int nb_elements)
{
    // output of the first batch, final output
    T* output = intermediate_output_batched[first_batch];

    if((omp_get_num_threads() < 2) or (last_batch - first_batch < 2))
    {
        // no more threads available or its a no-op
        // puts all the batches in the first one sequentially
        for(int b = first_batch+1; b < last_batch; b++)
        {
            T* intermediate_output = intermediate_output_batched[b];
            #pragma omp simd
            for(int i = 0; i < nb_elements; i++)
            {
                output[i] += intermediate_output[i];
            }
        }
    }
    else
    {
        // cuts the space in two and runs the algorithm recursively
        const int mid_batch = (first_batch + last_batch) / 2;
        #pragma omp task
        {
            reduction_slice_rec(intermediate_output_batched, first_batch, mid_batch, nb_elements);
        }
        #pragma omp task
        {
            reduction_slice_rec(intermediate_output_batched, mid_batch, last_batch, nb_elements);
        }
        #pragma omp taskwait

        // puts the outputs gathered in mid_batch into first_batch
        T* intermediate_output = intermediate_output_batched[mid_batch];
        #pragma omp simd
        for(int i = 0; i < nb_elements; i++)
        {
            output[i] += intermediate_output[i];
        }
    }
}

/*
 * reduces from first_index to last_index excluded
 */
template<typename T>
void reduction_slice(T* intermediate_output_batched[], T output[], int first_batch, int last_batch, int nb_elements)
{
    #pragma omp parallel
    {
        // sums the batches into intermediate_output_batched[first_batch]
        #pragma omp single
        {
            reduction_slice_rec(intermediate_output_batched, first_batch, last_batch, nb_elements);
        }

        // puts the outputs gathered in first_batch into output
        T* intermediate_output = intermediate_output_batched[first_batch];
        #pragma omp for
        for(int i = 0; i < nb_elements; i++)
        {
            output[i] += intermediate_output[i];
        }
    }
}

/*
 * cuts the batch into slices with the same output index
 * and reduce on them, one slice at a time
 */
template<typename T>
void reduction(int nb_batch, T* intermediate_output_batched[], T* output_batched[], int nb_elements)
{
    int first_batch = 0;
    T* current_output = output_batched[0];

    // identify slices with identical output arrays and reduce them one after the other
    for(int b = 0; b < nb_batch; b++)
    {
        // if the output is different, we are starting a new batch
        if(output_batched[b] != current_output)
        {
            // reduce previous slice
            reduction_slice(intermediate_output_batched, current_output, first_batch, b, nb_elements);
            // starts new slice
            first_batch = b;
            current_output = output_batched[b];
        }
    }

    // reduce last slice
    reduction_slice(intermediate_output_batched, current_output, first_batch, nb_batch, nb_elements);
}

/*
 * simplest version of the reduction, used to validate the other implementation
 */
template<typename T>
void reduction_naive(int nb_batch, T* intermediate_output_batched[], T* output_batched[], int nb_elements)
{
    for(int b = 0; b < nb_batch; b++)
    {
        T* output = output_batched[b];
        T* intermediate_output = intermediate_output_batched[b];

        #pragma omp parallel for
        for(int i = 0; i < nb_elements; i++)
        {
            output[i] += intermediate_output[i];
        }
    }
}

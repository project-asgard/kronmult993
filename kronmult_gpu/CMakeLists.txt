cmake_minimum_required(VERSION 3.8)
project(kronmult_gpu CUDA)

set(CMAKE_CUDA_STANDARD 14)

set(SOURCES
        kronmult.cu
        kronmult.cuh
        linear_algebra.cuh)

add_library(kronmult_gpu ${SOURCES})

set_target_properties(kronmult_gpu PROPERTIES CUDA_SEPARABLE_COMPILATION ON)

# TODO can we make the lib INTERFACE?
cmake_minimum_required(VERSION 3.8)
project(kronmult_gpu CUDA)

set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -arch=sm_70 --std=c++17")

add_library(kronmult_gpu
            kronmult.cu
            kronmult.cuh)

set_target_properties(kronmult_gpu PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
target_include_directories(kronmult_gpu PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
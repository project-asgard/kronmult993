cmake_minimum_required(VERSION 3.8 FATAL_ERROR)
project(test LANGUAGES CXX CUDA)

add_library(utils_gpu STATIC
    utils.cu
    #utils.cpp
    utils.hpp
    )

target_include_directories(utils_gpu PRIVATE ${CUDA_ROOT}/include ../kronmult/ .)

target_compile_features(utils_gpu PUBLIC cxx_std_11)
set_target_properties(utils_gpu
    PROPERTIES CUDA_SEPARABLE_COMPILATION ON)

add_executable(test_utils test_utils.cu)
target_include_directories(test_utils PRIVATE ${CUDA_ROOT}/include ../kronmult/ .)

set_property(TARGET test_utils
    PROPERTY CUDA_SEPARABLE_COMPILATION ON)

target_link_libraries(test_utils PRIVATE utils_gpu)

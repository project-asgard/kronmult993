cmake_minimum_required(VERSION 3.18)
# checks whether CUDA is available
find_package(CUDA)
project(kronmult LANGUAGES CXX)

#----------------------------------------------------------------------------------------
# CUDA

# add CUDA Language
add_subdirectory(kronmult_gpu)

#----------------------------------------------------------------------------------------
# OpenMP

# adds OpenMP flags and lib
add_subdirectory(kronmult_omp)

# adds tests
include(CTest)
enable_testing()
add_subdirectory(tests)

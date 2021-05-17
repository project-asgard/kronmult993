cmake_minimum_required(VERSION 3.8 FATAL_ERROR)
cmake_policy(SET CMP0074 NEW)
set(CMAKE_CXX_STANDARD 17)
if (USE_GPU)
    project(kronmult LANGUAGES CUDA CXX)
    add_subdirectory(kronmult_gpu)
else()
    project(kronmult LANGUAGES CXX)
endif()

# add kronmult implementations
add_subdirectory(kronmult_omp)
add_subdirectory(kronorigin)

# adds tests
include(CTest)
enable_testing()
add_subdirectory(tests)

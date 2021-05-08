cmake_minimum_required(VERSION 3.8 FATAL_ERROR)
cmake_policy(SET CMP0074 NEW)
set(CMAKE_CXX_STANDARD 17)

# TODO
# clean up this CMakeList
# we should probably use INTERFACE as the code is header-only
# https://cmake.org/cmake/help/latest/command/add_library.html#interface-libraries

if (USE_GPU)
    project(kronmult LANGUAGES CXX CUDA)
    find_package(CUDA 9.0 REQUIRED)
    add_compile_definitions(USE_GPU)
    set(CMAKE_CUDA_STANDARD 14)
    set(CMAKE_CUDA_STANDARD_REQUIRED ON)
else ()
    project(kronmult LANGUAGES CXX)
endif ()

include(CTest)
enable_testing ()
add_subdirectory(kronmult)
add_subdirectory(tests)

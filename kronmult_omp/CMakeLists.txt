cmake_minimum_required(VERSION 3.8)
project(kronmult_omp)

# C++ standard
set(CMAKE_CXX_STANDARD 17)

find_package(OpenMP)
if (OpenMP_CXX_FOUND)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
    set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${OpenMP_EXE_LINKER_FLAGS}")
else ()
    message(WARNING "Using kronmult_omp without OpenMP support.")
endif ()

# declare a header-only (interface) library
add_library(kronmult_omp INTERFACE)
target_include_directories(kronmult_omp
        INTERFACE
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>
        $<INSTALL_INTERFACE:>)

# TODO is there a way to check whether blas and openmp are linked and throw a warning otherwise

cmake_minimum_required(VERSION 3.8)
# checks whether CUDA is available
find_package(CUDA)
if (NOT CUDA_FOUND)
    message(WARNING "Using kronmult_gpu without CUDA.")
    project(kronmult LANGUAGES CXX)
else()
    project(kronmult LANGUAGES CUDA CXX)
    add_subdirectory(kronmult_gpu)
endif ()


#----------------------------------------------------------------------------------------
# OpenMP

# adds OpenMP flags and lib
find_package(OpenMP)
if (OpenMP_CXX_FOUND)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
    set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${OpenMP_EXE_LINKER_FLAGS}")
    add_subdirectory(kronmult_omp)
else ()
    message(WARNING "Using kronmult_omp without OpenMP support.")
endif ()

# adds tests
include(CTest)
enable_testing()
add_subdirectory(tests)

cmake_minimum_required(VERSION 3.8)
include(CheckLanguage)
check_language(CUDA)
set(CUDA_TARGET kronmult_gpu)
if(CMAKE_CUDA_COMPILER)
    project(kronmult_gpu CUDA)
    add_subdirectory(${CUDA_TARGET})

    set(CMAKE_CUDA_STANDARD 14)
    set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -arch=sm_70")

    set(SOURCES
            kronmult.cu
            kronmult.cuh)

    add_library(kronmult_gpu ${SOURCES})
    set_target_properties(kronmult_gpu PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
    target_include_directories(kronmult_gpu PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
else()
    message(STATUS "No CUDA support: the target ${CUDA_TARGET} will be omitted")
endif()
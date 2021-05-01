cmake_minimum_required(VERSION 3.8 FATAL_ERROR)
cmake_policy(SET CMP0074 NEW)

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

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Wextra -Wpedantic -O3 -g")
find_package(OpenMP)

if (OpenMP_CXX_FOUND)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
    if (USE_MKL)
        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fopenmp")
    endif ()
    set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${OpenMP_EXE_LINKER_FLAGS}")
else ()
    message("-- WARNING - building kronmult without openMP support ---")
endif ()

include(CTest)

set(KRONSRC
        kronmult/kronmult.hpp
        kronmult/gpu_operations.hpp kronmult/omp_reduction.hpp)

if (USE_GPU)
    set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -arch sm_60 --compiler-options -fPIC --keep-device-functions")
    if (USE_MKL)
        set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -arch sm_60 --compiler-options -fopenmp -fPIC --keep-device-functions")
    endif ()
    set_source_files_properties(${KRONSRC} PROPERTIES LANGUAGE CUDA)
endif ()

add_library(kron SHARED ${KRONSRC})
set_target_properties(kron PROPERTIES LINKER_LANGUAGE CXX)

if (USE_GPU)
    set_target_properties(kron PROPERTIES CUDA_ARCHITECTURES OFF)
endif ()
target_compile_features(kron PUBLIC cxx_std_11)

#set(TESTSRC tests/test_kronmult6_xbatched.cpp)

#if (USE_GPU)
#    set_source_files_properties(${TESTSRC} PROPERTIES LANGUAGE CUDA)
#endif ()

#add_executable(test_kronmult6_xbatched tests/test_kronmult6_xbatched.cpp)
#set_target_properties(test_kronmult6_xbatched PROPERTIES CUDA_ARCHITECTURES OFF)
#target_link_libraries(test_kronmult6_xbatched PRIVATE kron)
cmake_minimum_required(VERSION 3.8 FATAL_ERROR)
cmake_policy(SET CMP0074 NEW)

if (USE_GPU)
    project(kronmult-origin LANGUAGES CXX CUDA)
    find_package(CUDA 9.0 REQUIRED)
    add_compile_definitions(USE_GPU)
    set(CMAKE_CUDA_STANDARD 14)
    set(CMAKE_CUDA_STANDARD_REQUIRED ON)
else ()
    project(kronmult-origin LANGUAGES CXX)
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
        kronmult/kroncommon.hpp
        # kgemm
        kronmult/kgemm/kgemm_nn_batched.cpp
        kronmult/kgemm/kgemm_nn_batched.hpp
        kronmult/kgemm/kgemm_nn.cpp
        kronmult/kgemm/kgemm_nn.hpp
        kronmult/kgemm/kgemm_nt_batched.cpp
        kronmult/kgemm/kgemm_nt_batched.hpp
        kronmult/kgemm/kgemm_nt.cpp
        kronmult/kgemm/kgemm_nt.hpp
        # no batch
        kronmult/no_batch/no_batch.hpp
        kronmult/no_batch/kronmult1.hpp
        kronmult/no_batch/kronmult2.hpp
        kronmult/no_batch/kronmult3.hpp
        kronmult/no_batch/kronmult4.hpp
        kronmult/no_batch/kronmult5.hpp
        kronmult/no_batch/kronmult6.hpp
        # batched
        kronmult/batched/batched.hpp
        kronmult/batched/kronmult1_batched.cpp
        kronmult/batched/kronmult1_batched.hpp
        kronmult/batched/kronmult2_batched.cpp
        kronmult/batched/kronmult2_batched.hpp
        kronmult/batched/kronmult3_batched.cpp
        kronmult/batched/kronmult3_batched.hpp
        kronmult/batched/kronmult4_batched.cpp
        kronmult/batched/kronmult4_batched.hpp
        kronmult/batched/kronmult5_batched.cpp
        kronmult/batched/kronmult5_batched.hpp
        kronmult/batched/kronmult6_batched.cpp
        kronmult/batched/kronmult6_batched.hpp
        # pbatched
        kronmult/pbatched/pbatched.hpp
        kronmult/pbatched/kronmult1_pbatched.cpp
        kronmult/pbatched/kronmult1_pbatched.hpp
        kronmult/pbatched/kronmult2_pbatched.cpp
        kronmult/pbatched/kronmult2_pbatched.hpp
        kronmult/pbatched/kronmult3_pbatched.cpp
        kronmult/pbatched/kronmult3_pbatched.hpp
        kronmult/pbatched/kronmult4_pbatched.cpp
        kronmult/pbatched/kronmult4_pbatched.hpp
        kronmult/pbatched/kronmult5_pbatched.cpp
        kronmult/pbatched/kronmult5_pbatched.hpp
        kronmult/pbatched/kronmult6_pbatched.hpp
        kronmult/pbatched/kronmult6_pbatched.cpp
        # xbatched
        kronmult/xbatched/xbatched.hpp
        kronmult/xbatched/kronmult1_xbatched.cpp
        kronmult/xbatched/kronmult1_xbatched.hpp
        kronmult/xbatched/kronmult2_xbatched.cpp
        kronmult/xbatched/kronmult2_xbatched.hpp
        kronmult/xbatched/kronmult3_xbatched.cpp
        kronmult/xbatched/kronmult3_xbatched.hpp
        kronmult/xbatched/kronmult4_xbatched.cpp
        kronmult/xbatched/kronmult4_xbatched.hpp
        kronmult/xbatched/kronmult5_xbatched.cpp
        kronmult/xbatched/kronmult5_xbatched.hpp
        kronmult/xbatched/kronmult6_xbatched.cpp
        kronmult/xbatched/kronmult6_xbatched.hpp)

if (USE_GPU)
    set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -arch sm_60 --compiler-options -fPIC --keep-device-functions")
    if (USE_MKL)
        set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -arch sm_60 --compiler-options -fopenmp -fPIC --keep-device-functions")
    endif ()
    set_source_files_properties(${KRONSRC} PROPERTIES LANGUAGE CUDA)
endif ()

add_library(kron-origin SHARED ${KRONSRC})
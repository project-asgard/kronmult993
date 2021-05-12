set(CMAKE_CXX_STANDARD 17)

# compilation flags
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -g -O3 -march=native")

#----------------------------------------------------------------------------------------
# OpenMP

# adds OpenMP flags and lib
find_package(OpenMP)
if (OpenMP_CXX_FOUND)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
    set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${OpenMP_EXE_LINKER_FLAGS}")
    message("-- OpenMP detected ---")
else ()
    message(WARNING "Building Kronmult without OpenMP support.")
endif ()

# define KRONMULT_USE_BLAS if BLAS is found
find_package(BLAS)
if (BLAS_FOUND)
    add_compile_definitions(KRONMULT_USE_BLAS)
    set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${BLAS_LINKER_FLAGS}")
else()
    message(WARNING "Using kronmult_omp without BLAS support.")
endif()

# benchmark
add_executable(kronmult_bench kronmult_bench.cpp)
target_link_libraries(kronmult_bench PUBLIC kronmult_omp ${BLAS_LIBRARIES})
add_test(NAME kronmult_bench COMMAND ./kronmult_bench)

#----------------------------------------------------------------------------------------
# CUDA

# checks whether CUDA is available
find_package(CUDA)
if (NOT CUDA_FOUND)
    message(WARNING "Using kronmult_gpu without CUDA.")
endif()

# kronmult_gpu benchmark
add_executable(kronmult_bench_gpu kronmult_bench_gpu.cpp)
target_include_directories(kronmult_bench_gpu PUBLIC ${CUDA_INCLUDE_DIRS}) # TODO needs kronmult_gpu?
target_link_libraries(kronmult_bench_gpu PUBLIC kronmult_gpu ${CUDA_LIBRARIES}) # TODO not needed?
add_test(NAME kronmult_bench_gpu COMMAND ./kronmult_bench_gpu)
#    message("-- BLAS detected ---")
#else()
#    message(WARNING "Building Kronmult without BLAS support.")
#endif()
#
## benchmarks
#set(TEST_FILES kronmult_bench)
#foreach(TEST_EXE_NAME ${TEST_FILES})
#    add_executable(${TEST_EXE_NAME} ${TEST_EXE_NAME}.cpp)
#    #set_target_properties(${TEST_EXE_NAME} PROPERTIES LINKER_LANGUAGE  CXX)
#    target_include_directories(${TEST_EXE_NAME} PUBLIC ./)
#    target_link_libraries(${TEST_EXE_NAME} PUBLIC kron ${BLAS_LIBRARIES})
#    add_test(NAME  ${TEST_EXE_NAME} COMMAND ./${TEST_EXE_NAME})
#endforeach(TEST_EXE_NAME)

set(CMAKE_CXX_STANDARD 17)

# compilation flags
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -g -O3 -march=native")

# adds BLAS lib
# define KRONMULT_USE_BLAS if BLAS is found
find_package(BLAS)
if (BLAS_FOUND)
    add_compile_definitions(KRONMULT_USE_BLAS)
    set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${BLAS_LINKER_FLAGS}")
else ()
    message(WARNING "Using kronmult_omp without BLAS support.")
endif ()

if (OpenMP_CXX_FOUND)
    # test
    add_executable(kronmult_test_cpu kronmult_test.cpp)
    target_link_libraries(kronmult_test_cpu PUBLIC kronmult_omp ${BLAS_LIBRARIES})
    add_test(NAME kronmult_test_cpu COMMAND ./kronmult_test)

    # benchmark
    add_executable(kronmult_bench kronmult_bench.cpp)
    target_link_libraries(kronmult_bench PUBLIC kronmult_omp ${BLAS_LIBRARIES})
    add_test(NAME kronmult_bench COMMAND ./kronmult_bench)
endif()

if (CUDA_FOUND)
    # test
    add_executable(kronmult_test_gpu kronmult_test_gpu.cpp)
    target_include_directories(kronmult_test_gpu PUBLIC ${CUDA_INCLUDE_DIRS})
    target_link_libraries(kronmult_test_gpu PUBLIC kronmult_gpu)
    add_test(NAME kronmult_test_gpu COMMAND ./kronmult_test_gpu)

    # benchmark
    add_executable(kronmult_bench_gpu kronmult_bench_gpu.cpp)
    target_include_directories(kronmult_bench_gpu PUBLIC ${CUDA_INCLUDE_DIRS})
    target_link_libraries(kronmult_bench_gpu PUBLIC kronmult_gpu)
    add_test(NAME kronmult_bench_gpu COMMAND ./kronmult_bench_gpu)
endif()
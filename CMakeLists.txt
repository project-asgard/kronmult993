cmake_minimum_required(VERSION 3.8)

# add kronmult implementations
add_subdirectory(kronmult_omp)
add_subdirectory(kronmult_gpu)

# adds tests
include(CTest)
enable_testing()
add_subdirectory(tests)

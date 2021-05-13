set(CMAKE_CXX_STANDARD 17)

## compilation flags
#set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -g -O3 -march=native")
#
## define OpenMP flags
#find_package(OpenMP)
#if (OpenMP_CXX_FOUND)
#    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
#    set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${OpenMP_EXE_LINKER_FLAGS}")
#    message("-- OpenMP detected ---")
#else ()
#    message(WARNING "Building Kronmult without OpenMP support.")
#endif ()
#
## define KRONMULT_USE_BLAS if BLAS is found
#find_package(BLAS)
#if (BLAS_FOUND)
#    add_compile_definitions(KRONMULT_USE_BLAS)
#    set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${BLAS_LINKER_FLAGS}")
#    message("-- BLAS detected ---")
#else()
#    message(WARNING "Building Kronmult without BLAS support.")
#endif()
#
## benchmarks
#set(TEST_FILES test_kronmult kronmult_bench)
#foreach(TEST_EXE_NAME ${TEST_FILES})
#    add_executable(${TEST_EXE_NAME} ${TEST_EXE_NAME}.cpp)
#    #set_target_properties(${TEST_EXE_NAME} PROPERTIES LINKER_LANGUAGE  CXX)
#    target_include_directories(${TEST_EXE_NAME} PUBLIC ./)
#    target_link_libraries(${TEST_EXE_NAME} PUBLIC kron ${BLAS_LIBRARIES})
#    add_test(NAME  ${TEST_EXE_NAME} COMMAND ./${TEST_EXE_NAME})
#endforeach(TEST_EXE_NAME)

## TODO: If use gpu
set(TESTCUDASRC test_utils.cu)
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -arch sm_60 --ptxas-options=-O3")
set_source_files_properties( ${KRONSRC} PROPERTIES LANGUAGE CUDA )
set_target_properties( kron PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
set_target_properties( test_utils PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
target_link_libraries(${TEST_EXE_NAME} PUBLIC kron blas)
set_source_files_properties( ${TESTCUDASRC} PROPERTIES LANGUAGE CUDA )

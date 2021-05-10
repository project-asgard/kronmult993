set(TEST_FILES-ORIGIN
    test_kgemm_nn_batched
    test_kronmult6_xbatched
    test_kgemm_nt_batched
    test_kronmult6_batched
    test_kronmult6_pbatched)
set(TEST_FILES-993
        test_kronmult)
set(CMAKE_CXX_STANDARD 17)

if(BLAS_FOUND)
    set(USE_BLAS 1)
endif()

foreach(TEST_EXE_NAME ${TEST_FILES-ORIGIN})
    add_executable(${TEST_EXE_NAME} ${TEST_EXE_NAME}.cpp)
    #set_target_properties(${TEST_EXE_NAME} PROPERTIES LINKER_LANGUAGE  CXX)
    target_include_directories(${TEST_EXE_NAME} PUBLIC ./)
    target_link_libraries(${TEST_EXE_NAME} PUBLIC kron-origin)
    if (USE_GPU)
        set_target_properties(kron-origin PROPERTIES CUDA_ARCHITECTURES OFF)
        set_source_files_properties(${TESTSRC} PROPERTIES LANGUAGE CUDA)
    endif ()
    add_test(NAME  ${TEST_EXE_NAME} COMMAND  ./${TEST_EXE_NAME})
endforeach(TEST_EXE_NAME)

foreach(TEST_EXE_NAME ${TEST_FILES-993})
    add_executable(${TEST_EXE_NAME} ${TEST_EXE_NAME}.cpp)
    #set_target_properties(${TEST_EXE_NAME} PROPERTIES LINKER_LANGUAGE  CXX)
    target_include_directories(${TEST_EXE_NAME} PUBLIC ./)
    target_link_libraries(${TEST_EXE_NAME} PUBLIC kron)
    add_test(NAME  ${TEST_EXE_NAME} COMMAND  ./${TEST_EXE_NAME})
endforeach(TEST_EXE_NAME)

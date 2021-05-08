set(TEST_FILES
    test_kronmult)
set(CMAKE_CXX_STANDARD 17)

foreach(TEST_EXE_NAME ${TEST_FILES})
    add_executable(${TEST_EXE_NAME} ${TEST_EXE_NAME}.cpp)
    #set_target_properties(${TEST_EXE_NAME} PROPERTIES LINKER_LANGUAGE  CXX)
    target_include_directories(${TEST_EXE_NAME} PUBLIC ./)
    target_link_libraries(${TEST_EXE_NAME} PUBLIC kron blas)
    add_test(NAME  ${TEST_EXE_NAME} COMMAND  ./${TEST_EXE_NAME})
endforeach(TEST_EXE_NAME)

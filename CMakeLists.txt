cmake_minimum_required(VERSION 3.8 FATAL_ERROR)
cmake_policy(SET CMP0074 NEW)

if(USE_GPU)
	project(src LANGUAGES CXX CUDA)
	find_package(CUDA 9.0 REQUIRED)
	add_compile_definitions(USE_GPU)
	set(CMAKE_CUDA_STANDARD 14)
	set(CMAKE_CUDA_STANDARD_REQUIRED ON)
else()
	project(src LANGUAGES CXX)
endif()

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Wextra -Wpedantic -O3 -g")
find_package(OpenMP)

if(OpenMP_CXX_FOUND)
	set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
	if(USE_MKL)
		set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fopenmp")
	endif()
	set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${OpenMP_EXE_LINKER_FLAGS}")
else()
	message("-- WARNING - building kronmult without openMP support ---")
endif()

include(CTest)

set(KRONSRC
		src/kroncommon.hpp
	# kgemm
		src/kgemm/kgemm_nn_batched.cpp
		src/kgemm/kgemm_nn_batched.hpp
		src/kgemm/kgemm_nn.cpp
		src/kgemm/kgemm_nn.hpp
		src/kgemm/kgemm_nt_batched.cpp
		src/kgemm/kgemm_nt_batched.hpp
		src/kgemm/kgemm_nt.cpp
		src/kgemm/kgemm_nt.hpp
	# no batch
		src/no_batch/kronmult1.hpp
		src/no_batch/kronmult2.hpp
		src/no_batch/kronmult3.hpp
		src/no_batch/kronmult4.hpp
		src/no_batch/kronmult5.hpp
		src/no_batch/kronmult6.hpp
	# batched
		src/batched/kronmult1_batched.cpp
		src/batched/kronmult1_batched.hpp
		src/batched/kronmult2_batched.cpp
		src/batched/kronmult2_batched.hpp
		src/batched/kronmult3_batched.cpp
		src/batched/kronmult3_batched.hpp
		src/batched/kronmult4_batched.cpp
		src/batched/kronmult4_batched.hpp
		src/batched/kronmult5_batched.cpp
		src/batched/kronmult5_batched.hpp
		src/batched/kronmult6_batched.cpp
		src/batched/kronmult6_batched.hpp
	# pbatched
		src/pbatched/kronmult1_pbatched.cpp
		src/pbatched/kronmult1_pbatched.hpp
		src/pbatched/kronmult2_pbatched.cpp
		src/pbatched/kronmult2_pbatched.hpp
		src/pbatched/kronmult3_pbatched.cpp
		src/pbatched/kronmult3_pbatched.hpp
		src/pbatched/kronmult4_pbatched.cpp
		src/pbatched/kronmult4_pbatched.hpp
		src/pbatched/kronmult5_pbatched.cpp
		src/pbatched/kronmult5_pbatched.hpp
		src/pbatched/kronmult6_pbatched.hpp
		src/pbatched/kronmult6_pbatched.cpp
	# xbatched
		src/xbatched/kronmult1_xbatched.cpp
		src/xbatched/kronmult1_xbatched.hpp
		src/xbatched/kronmult2_xbatched.cpp
		src/xbatched/kronmult2_xbatched.hpp
		src/xbatched/kronmult3_xbatched.cpp
		src/xbatched/kronmult3_xbatched.hpp
		src/xbatched/kronmult4_xbatched.cpp
		src/xbatched/kronmult4_xbatched.hpp
		src/xbatched/kronmult5_xbatched.cpp
		src/xbatched/kronmult5_xbatched.hpp
		src/xbatched/kronmult6_xbatched.cpp
		src/xbatched/kronmult6_xbatched.hpp)

if(USE_GPU)
	set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -arch sm_60 --compiler-options -fPIC --keep-device-functions")
	if(USE_MKL)
		set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -arch sm_60 --compiler-options -fopenmp -fPIC --keep-device-functions")
	endif()
	set_source_files_properties( ${KRONSRC} PROPERTIES LANGUAGE CUDA )
endif()

add_library(kron SHARED ${KRONSRC})
if(USE_GPU)
	set_target_properties(kron PROPERTIES CUDA_ARCHITECTURES OFF)
endif()
target_compile_features(kron PUBLIC cxx_std_11)

set(TESTSRC
	tests/test_kronmult6_batched.cpp
	tests/test_kronmult6_pbatched.cpp
	tests/test_kronmult6_xbatched.cpp
	tests/test_kgemm_nn_batched.cpp
	tests/test_kgemm_nt_batched.cpp)

if(USE_GPU)
	set_source_files_properties(${TESTSRC} PROPERTIES LANGUAGE CUDA)
endif()

add_executable(kgemm_nn_test tests/test_kgemm_nn_batched.cpp)
add_executable(kgemm_nt_test tests/test_kgemm_nt_batched.cpp)
add_executable(test_kronmult6_batched tests/test_kronmult6_batched.cpp)
add_executable(test_kronmult6_pbatched tests/test_kronmult6_pbatched.cpp)
add_executable(test_kronmult6_xbatched tests/test_kronmult6_xbatched.cpp)

set_target_properties(kgemm_nn_test PROPERTIES CUDA_ARCHITECTURES OFF)
set_target_properties(kgemm_nt_test PROPERTIES CUDA_ARCHITECTURES OFF)
set_target_properties(test_kronmult6_batched PROPERTIES CUDA_ARCHITECTURES OFF)
set_target_properties(test_kronmult6_pbatched PROPERTIES CUDA_ARCHITECTURES OFF)
set_target_properties(test_kronmult6_xbatched PROPERTIES CUDA_ARCHITECTURES OFF)

target_link_libraries(kgemm_nn_test PRIVATE kron)
target_link_libraries(kgemm_nt_test PRIVATE kron)
target_link_libraries(test_kronmult6_batched PRIVATE kron)
target_link_libraries(test_kronmult6_pbatched PRIVATE kron)
target_link_libraries(test_kronmult6_xbatched PRIVATE kron)
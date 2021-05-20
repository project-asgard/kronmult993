# Tests and benchmarks

This folder contains standalone tests and benchmarks designed to emulate the inputs that are received when `kronmult_batched` is used within [ASGarD](https://github.com/project-asgard/asgard).

They are in double precision by default but you can manualy change the `Number` type at the begining of each file to test other precisions such as `float`.

The `utils` folder contains our naive implementation of `kronmult_batched` as wella s utilities to take care of memory management on CPU and GPU.

## Tests

The CMake target `kronmult_test_cpu` (CPU version, file: `kronmult_test.cpp`) and `kronmult_test_gpu` (GPU version, file: `kronmult_test_gpu`) 
check the maximum relative error when comparing the output of our implementation with a naive implementation.
Due to the inefficiency of the naive implementation, they are run on small test cases only.

You can expect a correct implementation to have a value around `1e-15` while an incorrect implementation would have a value around `1`.

The CPU version should tell you if BLAS was correctly detected 
while the GPU version should display basic information on the GPU you are using.

## Benchmarks

The CMake target `kronmult_bench` (CPU version, file: `kronmult_bench.cpp`) and `kronmult_bench_gpu` (GPU version, file: `kronmult_bench_gpu`) 
time our implementation (and only our implementation, not the memory allocation and transfer) on benchmarks of increasing sizes
(displayed here in term of their corresponding ASGarD problem size):

- `toy`: degree=4 dimension=1 grid_level=2
- `small`: degree=4 dimension=2 grid_level=4
- `medium`: degree=6 dimension=3 grid_level=6
- `large`: degree=8 dimension=6 grid_level=7
- `realistic`: degree=8 dimension=6 grid_level=9

Only the `large` and `realistic` cases should be used for benchmarking purposes as the others are too small to be measured accurately in a single run.

The number of batch element is capped at `2^11` to avoid allocation errors on the GPU.
This results in the `large` and `realistic` being identical.

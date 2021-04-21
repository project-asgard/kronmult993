# Kronmult

This code performs a 6-dimensional batched kronecker product on GPU and CPU (using OpenMP).

It is finetuned for the needs of [ASGarD](https://github.com/project-asgard/asgard) and, in particular, we expect our inputs to be slender rectangular matrices.

## Usage

**TODO**

## Compilation

To compile the code for CPU, run:

```
mkdir build && cd build
cmake ../
make
```

To compile the code for an Nvidia GPU, pass the `-DUSE_GPU=1` flag to CMake:

```
mkdir build && cd build
cmake ../ -DUSE_GPU=1
make
```

## Tests

The binary targets `test_kgemm_nt_batched` and `test_kronmult6_batched` can be used to test `kgemm_nt_batched` and `kronmult6_batched` respectively.


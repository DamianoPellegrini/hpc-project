# Assignment - Sistemi di Calcolo Parallelo - Boruvka's MST algorithm

This project aims to show the parallel implementation of the Boruvka's MST algorithm.

The implementation is repeated 3 times using separate hpc paradigms:

- MPI
- OpenMP
- CUDA

To compile and run the code install the following dependencies and use the following commands:

> [!WARNING]  
> This will only work on MacOS

Install:
```bash
brew install openmpi libomp
```

Compile:
```bash
cmake --preset ninja-local
cmake --build --preset ninja-local
```

Run:
```bash
mpirun -np 2 ./build/mpi/mpi_app
# or
./build/mpi/mpi_app
# and
./build/openmp/openmp_app
```

or use the provided Zed tasks.

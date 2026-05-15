# Assignment - Sistemi di Calcolo Parallelo - Boruvka's MST algorithm

This project aims to show the parallel implementation of the Boruvka's MST algorithm.

There are 3 separate implementations using these hpc backends:

- MPI
- OpenMP
- CUDA

CMake auto-detects which backends can be built on the current machine.

> [!WARNING]
> The current setup instructions are MacOS-oriented.

Install local CPU dependencies:

```bash
brew install openmpi libomp
```

Build everything available on this machine:

```bash
cmake --preset ninja-local
cmake --build --preset ninja-local
```

Run OpenMP:

```bash
./build/openmp/openmp_app
```

Run MPI:

```bash
mpirun -np 2 ./build/mpi/mpi_app
```

Build & Run CUDA, on a CUDA machine:

```bash
cmake --preset ninja-local
cmake --build --preset ninja-local --target cuda_app
./build/cuda/cuda_app
```

Zed tasks already match this configuration.

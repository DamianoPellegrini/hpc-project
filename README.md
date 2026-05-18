# Assignment - Sistemi di Calcolo Parallelo - Boruvka's MST algorithm

This project aims to show the parallel implementation of the Boruvka's MST algorithm.

There are 3 separate implementations using these hpc backends:

- MPI
- OpenMP
- CUDA

CMake is the source of truth for the build. It auto-detects which backends can
be built on the current machine: if OpenMP, MPI, or CUDA are found, the
corresponding target is compiled; otherwise CMake prints a status message and
skips that target.

> [!WARNING]
> The current setup instructions are MacOS-oriented.

Install local CPU dependencies:

```bash
brew install openmpi libomp
```

Build everything available on this machine:

```bash
cmake --preset default
cmake --build --preset default
```

Recreate the local CMake build tree when cache state needs to be discarded:

```bash
cmake --fresh --preset recreate
cmake --build --preset recreate
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
cmake --preset default
cmake --build --preset default --target cuda_app
./build/cuda/cuda_app
```

The top-level Makefile is a compatibility wrapper. On a machine with CMake and
Ninja, `make all`, `make openmp`, `make mpi`, and `make cuda` delegate to the
CMake presets. On the Slurm environment where CMake/Ninja are unavailable, use
the direct fallback rules:

```bash
make USE_CMAKE=OFF openmp CXX=g++
make USE_CMAKE=OFF mpi MPICXX=mpicxx
make USE_CMAKE=OFF cuda CXX=g++ NVCC=nvcc NVCC_CCBIN=g++
```

Zed tasks should use the CMake preset path locally.

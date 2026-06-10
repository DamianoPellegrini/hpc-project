# Assignment - Sistemi di Calcolo Parallelo - Boruvka's MST algorithm

This project aims to show the parallel implementation of the Boruvka's MST algorithm.

There are 4 separate implementations using these backends:

- Sequential (CPU baseline)
- OpenMP
- MPI
- CUDA

CMake is the source of truth for the build. It auto-detects which backends can
be built on the current machine: if OpenMP, MPI, or CUDA are found, the
corresponding target is compiled; otherwise CMake prints a status message and
skips that target. The sequential target always builds.

> [!WARNING]
> The current setup instructions are MacOS-oriented.

## Build

Install local CPU dependencies:

```bash
brew install openmpi libomp
```

Build everything available on this machine:

```bash
rm -rf build
cmake --preset default
cmake --build --preset default
```

Format C++ and CUDA sources with:

```bash
clang-format -i src/*.cpp src/*.cu
```

## Running

Each backend takes the same positional arguments: `<vertices> <edges> <seed>`
(all optional, with defaults baked into each program). They generate a random
connected graph, run Borůvka's algorithm, verify the result against Kruskal,
and print timing breakdowns (`overhead_seconds`, `exec_seconds`,
`total_seconds`, `verification`).

Run sequential:

```bash
./build/sequential_app 32768 196608 886261
```

Run OpenMP:

```bash
./build/openmp_app 32768 196608 886261
```

Run MPI:

```bash
mpirun -np 2 ./build/mpi_app 32768 196608 886261
```

MPI also accepts a single graph file argument instead of `<vertices> <edges> <seed>`:

```bash
mpirun -np 2 ./build/mpi_app path/to/graph.txt
```

## CUDA

On a CUDA-capable machine, `cmake --build --preset default` also builds
`cuda_app`. To build and run it explicitly:

```bash
cmake --build --preset default --target cuda_app
./build/cuda_app 32768 196608 886261
```

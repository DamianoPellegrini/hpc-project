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
rm -rf build
cmake --preset default
cmake --build --preset default
```

Run OpenMP:

```bash
./build/openmp/openmp_app --graph test
```

Run MPI:

```bash
mpirun -np 2 ./build/mpi/mpi_app --graph test
```

Build & Run CUDA, on a CUDA machine:

```bash
cmake --preset default
cmake --build --preset default --target cuda_app
./build/cuda/cuda_app --graph test
```

Run tests:

```bash
ctest --test-dir build --output-on-failure
```

Select input graphs with `--graph`. Available values are `test`, `triangle`,
`square`, `tie`, `dense16`, and `random`. The random graph defaults to 32,768
vertices and 196,608 extra edges:

```bash
./build/openmp/openmp_app --graph random --random-vertices 32768 \
  --random-extra-edges 196608 --random-seed 886261 \
  --random-max-weight 10000 --benchmark
```

Set `--report` to write JSON timing reports:

```bash
./build/openmp/openmp_app --graph tie --report results/openmp_tie.json
```

CUDA host edge memory mode (`pageable|pinned|zero_copy`) is fixed at configure
time via `MST_CUDA_HOST_MEMORY_DEFAULT`; pinned is the default.

For benchmark builds without visualization:

```bash
cmake --preset relwithdebinfo
cmake --build --preset relwithdebinfo
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

Fallback build switches are available as make variables:

```bash
make USE_CMAKE=OFF openmp MST_ENABLE_RENDERING=0
make USE_CMAKE=OFF cuda MST_CUDA_HOST_MEMORY_DEFAULT=zero_copy
```

Format C++ and CUDA sources with:

```bash
make format
```

Run a local benchmark and write a sidecar manifest with the command, git
revision, backend, arguments, and report path:

```bash
scripts/run_benchmark.sh openmp --graph random --random-vertices 32768 \
  --random-extra-edges 196608
```

Submit the Slurm benchmark matrix for all named graphs plus the large random
graph:

```bash
scripts/slurm/submit_all.sh
```

Zed tasks should use the CMake preset path locally.

# Repository Guidelines

## Project Structure & Module Organization

This project implements Boruvka's MST algorithm across OpenMP, MPI, and CUDA backends.
Shared C++20 headers live in `include/mst/`: `core` holds typed graph/edge models and examples, `dsu` contains sequential and parallel DSUs, `boruvka` contains backend contracts and the sequential CPU verifier, `app` handles graph selection, and `reporting`/`visualization` provide output helpers.
Backend entry points are `openmp/main.cpp`, `mpi/main.cpp`, and `cuda/main.cu`; CUDA device helpers live in `cuda/*.cuh`.
Regression tests are in `tests/core_tests.cpp`.
Cluster scripts are in `scripts/slurm/`.
The Typst report lives in `report/`: `main.typ` is the entry point, `_prelude.typ` centralizes package imports, `data.typ` indexes and normalizes JSON reports from `results/`, and `figures.typ` contains chart helpers.
Generated outputs belong in ignored paths: `build/`, `results/`, `job_logs/`, and `docs/local/`.

## Build, Test, and Development Commands

- `cmake --preset default` configures the Debug Ninja build and auto-detects OpenMP, MPI, and CUDA.
- `cmake --build --preset default` builds all available targets.
- `cmake --build --preset default --target core_tests openmp_app mpi_app` builds the local CPU/MPI targets explicitly.
- `ctest --test-dir build --output-on-failure` runs the dependency-free test suite.
- `./build/openmp/openmp_app` runs OpenMP; `mpirun -np 4 ./build/mpi/mpi_app` runs MPI.
- `cmake --build --preset default --target cuda_app && ./build/cuda/cuda_app` runs CUDA on a CUDA-capable machine.
- `typst compile --root . report/main.typ report/main.pdf` builds the UniMiB Typst report and allows `report/data.typ` to read JSON files from `results/`.

Use `MST_GRAPH=test|triangle|square|tie|dense16|random` to select inputs.
For random graphs, set `MST_RANDOM_VERTICES`, `MST_RANDOM_EXTRA_EDGES`, `MST_RANDOM_SEED`, and `MST_RANDOM_MAX_WEIGHT`.
Set `MST_REPORT_PATH=results/openmp.json` to emit JSON reports.
When adding, replacing, or renaming benchmark reports used by the Typst document, update the file list in `report/data.typ` so tables and charts continue to reflect the intended run set.

## Coding Style & Naming Conventions

Use C++20, two-space indentation, no compiler extensions, and lower_snake_case for types, functions, variables, and CMake targets.
Keep public APIs inside `mst::<domain>` namespaces and include project headers as `#include "mst/core/graph.hpp"`.
Prefer strong types and smart constructors (`vertex_id`, `edge_index`, `candidate_key`, `random_connected_graph_config`) over raw primitives.
When adding backend behavior, keep shared semantics in `include/mst` and backend mechanics in the backend directory.

## Testing Guidelines

Add focused tests to `tests/core_tests.cpp`.
Cover typed invariants with `static_assert` where possible.
Every backend result should be checked with `mst::boruvka::verify_against_sequential_cpu`; apps should exit nonzero if verification fails.
Run `ctest --test-dir build --output-on-failure` before review.

## Commit & Pull Request Guidelines

Recent commits use Conventional Commit prefixes such as `feat:`, `refactor:`, and `chore:`.
Use short imperative summaries, for example `feat: add cuda verification report`.
PRs should list affected backends, graph inputs used, build/test commands run, CUDA/MPI availability assumptions, and report artifacts when timings or JSON output change.

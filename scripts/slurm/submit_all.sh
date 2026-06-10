#!/bin/bash

set -euo pipefail

export REPO_DIR="${REPO_DIR:-$HOME/hpc-project}"
export EXPERIMENT_DIR="${EXPERIMENT_DIR:-$HOME/experiments/parallel-mst}"

mkdir -p "$EXPERIMENT_DIR/job_logs" "$EXPERIMENT_DIR/results"

make -C "$REPO_DIR" clean

cd "$EXPERIMENT_DIR"

# Sweep a densità crescente sul grafo casuale generato dal programma stesso
# (m/n da 1 a 384, con n = RANDOM_VERTICES = 32768): supera le soglie
# operative di MPI (m/n >= p log p) e
# OpenMP (m/n >= p, Capitolo 2 del report) e copre il regime m >> n rilevante
# per il modello CUDA. Sovrascrivibile impostando RANDOM_EDGES_LIST
# prima del lancio; i tre script per-backend la leggono già.
export RANDOM_EDGES_LIST="${RANDOM_EDGES_LIST:-32768,65536,131072,196608,393216,786432,1572864,3145728,6291456,12582912}"

sequential_job_id="$(
  sbatch --parsable --export=ALL \
    "$REPO_DIR/scripts/slurm/sequential.sh"
)"
openmp_job_id="$(
  sbatch --parsable --export=ALL \
    "$REPO_DIR/scripts/slurm/openmp.sh"
)"
mpi_job_id="$(
  sbatch --parsable --export=ALL \
    "$REPO_DIR/scripts/slurm/mpi.sh"
)"
cuda_job_id="$(
  sbatch --parsable --export=ALL \
    "$REPO_DIR/scripts/slurm/cuda.sh"
)"

printf '%s\n' \
  "Submitted sequential job: $sequential_job_id" \
  "Submitted OpenMP job: $openmp_job_id" \
  "Submitted MPI job: $mpi_job_id" \
  "Submitted CUDA job: $cuda_job_id"

printf 'Random edges sweep: %s\n' "$RANDOM_EDGES_LIST"

printf '%s\n' \
  "Logs: $EXPERIMENT_DIR/job_logs" \
  "Results: $EXPERIMENT_DIR/results"

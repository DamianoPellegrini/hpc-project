#!/bin/bash

set -euo pipefail

export REPO_DIR="${REPO_DIR:-$HOME/hpc-project}"
export EXPERIMENT_DIR="${EXPERIMENT_DIR:-$HOME/experiments/parallel-mst}"

mkdir -p "$EXPERIMENT_DIR/job_logs" "$EXPERIMENT_DIR/results"

make -C "$REPO_DIR" clean

cd "$EXPERIMENT_DIR"

graph_list="${GRAPHS:-test,triangle,square,tie,dense16,random}"
export GRAPHS="$graph_list"

# Sweep completa a densità crescente sul grafo `random` (m/n da ~2 a ~385): supera le
# soglie operative di MPI (m/n >= p log p) e OpenMP (m/n >= p, Capitolo 2 del report) e
# copre il regime m >> n rilevante per il modello CUDA. Sovrascrivibile impostando
# RANDOM_EXTRA_EDGES_LIST prima del lancio; i tre script per-backend la leggono già.
export RANDOM_EXTRA_EDGES_LIST="${RANDOM_EXTRA_EDGES_LIST:-32768,65536,131072,196608,393216,786432,1572864,3145728,6291456,12582912}"

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
  "Submitted OpenMP graphs=[$graph_list] job: $openmp_job_id" \
  "Submitted MPI graphs=[$graph_list] job: $mpi_job_id" \
  "Submitted CUDA graphs=[$graph_list] job: $cuda_job_id"

printf 'Random extra-edges sweep: %s\n' "$RANDOM_EXTRA_EDGES_LIST"

printf '%s\n' \
  "Logs: $EXPERIMENT_DIR/job_logs" \
  "Results: $EXPERIMENT_DIR/results"

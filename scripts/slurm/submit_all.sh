#!/bin/bash

set -euo pipefail

export REPO_DIR="${REPO_DIR:-$HOME/hpc-project}"
export EXPERIMENT_DIR="${EXPERIMENT_DIR:-$HOME/experiments/parallel-mst}"

mkdir -p "$EXPERIMENT_DIR/job_logs" "$EXPERIMENT_DIR/results"

make -C "$REPO_DIR" clean

cd "$EXPERIMENT_DIR"

graph_list="${MST_GRAPHS:-test,triangle,square,tie,dense16,random}"
export MST_GRAPHS="$graph_list"

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

printf '%s\n' \
  "Logs: $EXPERIMENT_DIR/job_logs" \
  "Results: $EXPERIMENT_DIR/results"

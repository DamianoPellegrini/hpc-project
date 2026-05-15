#!/bin/bash

set -euo pipefail

export REPO_DIR="${REPO_DIR:-$HOME/hpc-project}"
export EXPERIMENT_DIR="${EXPERIMENT_DIR:-$HOME/experiments/parallel-mst}"

mkdir -p "$EXPERIMENT_DIR/job_logs" "$EXPERIMENT_DIR/results"

make -C "$REPO_DIR" clean

cd "$EXPERIMENT_DIR"

openmp_job_id="$(sbatch --parsable "$REPO_DIR/scripts/slurm/openmp.sh")"
mpi_job_id="$(sbatch --parsable "$REPO_DIR/scripts/slurm/mpi.sh")"
cuda_job_id="$(sbatch --parsable "$REPO_DIR/scripts/slurm/cuda.sh")"

printf '%s\n' \
  "Submitted OpenMP job: $openmp_job_id" \
  "Submitted MPI job: $mpi_job_id" \
  "Submitted CUDA job: $cuda_job_id" \
  "Logs: $EXPERIMENT_DIR/job_logs" \
  "Results: $EXPERIMENT_DIR/results"

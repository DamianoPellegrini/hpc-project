#!/bin/bash
#SBATCH --account=d.pellegrini-thesis
#SBATCH --job-name=parallel-mst-mpi
#SBATCH --partition=ulow
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=2G
#SBATCH --time=00:15:00
#SBATCH --output=job_logs/out_%x_%j.log
#SBATCH --mail-type=FAIL
#SBATCH --mail-user=d.pellegrini10@campus.unimib.it

set -euo pipefail

export REPO_DIR="${REPO_DIR:-$HOME/hpc-project}"
export EXPERIMENT_DIR="${EXPERIMENT_DIR:-$HOME/experiments/parallel-mst}"
export RESULTS_DIR="$EXPERIMENT_DIR/results"
export MST_REPORT_PATH="$RESULTS_DIR/mpi_${SLURM_JOB_ID}.json"

mkdir -p "$EXPERIMENT_DIR/job_logs" "$RESULTS_DIR"

pwd
hostname
date

module purge
module load amd/gcc/gcc-12
module load amd/gcc-12.2.1/openmpi-4.1.6

cd "$REPO_DIR"
make mpi MPICXX=mpicxx

mpirun -np "${SLURM_CPUS_PER_TASK}" ./build/mpi/mpi_app

date

#!/bin/bash
#SBATCH --account=d.pellegrini-thesis
#SBATCH --job-name=parallel-mst-cuda
#SBATCH --partition=only-one-gpu
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=2G
#SBATCH --gres=gpu:1
#SBATCH --time=00:15:00
#SBATCH --output=job_logs/out_%x_%j.log
#SBATCH --mail-type=FAIL
#SBATCH --mail-user=d.pellegrini10@campus.unimib.it

set -euo pipefail

export REPO_DIR="${REPO_DIR:-$HOME/hpc-project}"
export EXPERIMENT_DIR="${EXPERIMENT_DIR:-$HOME/experiments/parallel-mst}"
export RESULTS_DIR="$EXPERIMENT_DIR/results"
export MST_REPORT_PATH="$RESULTS_DIR/cuda_${SLURM_JOB_ID}.json"
export TMPDIR="/scratch_local/$USER/${SLURM_JOB_NAME}_${SLURM_JOB_ID}"

mkdir -p "$EXPERIMENT_DIR/job_logs" "$RESULTS_DIR" "$TMPDIR"

pwd
hostname
date

module purge
module load amd/gcc/gcc-12
module load amd/gcc-12.2.1/openmpi-4.1.6
module load amd/nvidia/cuda-12.3.2

cd "$REPO_DIR"
make cuda CXX=g++ NVCC=nvcc NVCC_CCBIN=g++

./build/cuda/cuda_app

rm -rf "$TMPDIR"
date

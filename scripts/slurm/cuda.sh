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
GRAPHS="${GRAPHS:-random}"
RANDOM_VERTICES="${RANDOM_VERTICES:-32768}"
RANDOM_EXTRA_EDGES="${RANDOM_EXTRA_EDGES:-196608}"
RANDOM_SEED="${RANDOM_SEED:-886261}"
RANDOM_MAX_WEIGHT="${RANDOM_MAX_WEIGHT:-10000}"
CUDA_HOST_MEMORY="${CUDA_HOST_MEMORY:-pinned}"
export TMPDIR="/scratch_local/$USER/${SLURM_JOB_NAME}_${SLURM_JOB_ID}"
graph_list="${GRAPHS//,/ }"
read -r -a graphs <<< "$graph_list"
extra_edge_list="${RANDOM_EXTRA_EDGES_LIST:-$RANDOM_EXTRA_EDGES}"
extra_edge_list="${extra_edge_list//,/ }"
read -r -a random_extra_edges_values <<< "$extra_edge_list"

mkdir -p "$EXPERIMENT_DIR/job_logs" "$RESULTS_DIR" "$TMPDIR"

pwd
hostname
date
printf 'graphs=%s vertices=%s extra_edges=%s seed=%s max_weight=%s cuda_host_memory=%s\n' \
  "$GRAPHS" "$RANDOM_VERTICES" "$RANDOM_EXTRA_EDGES" \
  "$RANDOM_SEED" "$RANDOM_MAX_WEIGHT" "$CUDA_HOST_MEMORY"
if [[ -n "${RANDOM_EXTRA_EDGES_LIST:-}" ]]; then
  printf 'RANDOM_EXTRA_EDGES_LIST=%s\n' "$RANDOM_EXTRA_EDGES_LIST"
fi

module purge
module load amd/gcc/gcc-12
module load amd/gcc-12.2.1/openmpi-4.1.6
module load amd/nvidia/cuda-12.3.2

cd "$REPO_DIR"
make USE_CMAKE=OFF cuda CXX=g++ NVCC=nvcc NVCC_CCBIN=g++

run_graph() {
  local graph="$1"
  local extra_edges="$2"
  local report_name="cuda_${graph}_${SLURM_JOB_ID}.json"
  if [[ "$graph" == "random" && -n "${RANDOM_EXTRA_EDGES_LIST:-}" ]]; then
    report_name="cuda_${graph}_v${RANDOM_VERTICES}_e${extra_edges}_${SLURM_JOB_ID}.json"
  fi
  local report_path="$RESULTS_DIR/$report_name"
  local args=(--graph "$graph" --report "$report_path" --benchmark
              --cuda-host-memory "$CUDA_HOST_MEMORY")
  if [[ "$graph" == "random" ]]; then
    args+=(
      --random-vertices "$RANDOM_VERTICES"
      --random-extra-edges "$extra_edges"
      --random-seed "$RANDOM_SEED"
      --random-max-weight "$RANDOM_MAX_WEIGHT"
    )
  fi
  printf 'Running CUDA graph=%s report=%s\n' "$graph" "$report_path"
  ./build/cuda/cuda_app "${args[@]}"
}

for graph in "${graphs[@]}"; do
  if [[ "$graph" == "random" ]]; then
    for extra_edges in "${random_extra_edges_values[@]}"; do
      run_graph "$graph" "$extra_edges"
    done
  else
    run_graph "$graph" "$RANDOM_EXTRA_EDGES"
  fi
done

rm -rf "$TMPDIR"
date

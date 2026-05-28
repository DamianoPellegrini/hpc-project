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
export MST_GRAPH="${MST_GRAPH:-random}"
export MST_GRAPHS="${MST_GRAPHS:-$MST_GRAPH}"
export MST_RANDOM_VERTICES="${MST_RANDOM_VERTICES:-32768}"
export MST_RANDOM_EXTRA_EDGES="${MST_RANDOM_EXTRA_EDGES:-196608}"
export MST_RANDOM_SEED="${MST_RANDOM_SEED:-886261}"
export MST_RANDOM_MAX_WEIGHT="${MST_RANDOM_MAX_WEIGHT:-10000}"
export TMPDIR="/scratch_local/$USER/${SLURM_JOB_NAME}_${SLURM_JOB_ID}"
graph_list="${MST_GRAPHS//,/ }"
read -r -a graphs <<< "$graph_list"
extra_edge_list="${MST_RANDOM_EXTRA_EDGES_LIST:-$MST_RANDOM_EXTRA_EDGES}"
extra_edge_list="${extra_edge_list//,/ }"
read -r -a random_extra_edges_values <<< "$extra_edge_list"

mkdir -p "$EXPERIMENT_DIR/job_logs" "$RESULTS_DIR" "$TMPDIR"

pwd
hostname
date
printf 'MST_GRAPHS=%s vertices=%s extra_edges=%s seed=%s max_weight=%s\n' \
  "$MST_GRAPHS" "$MST_RANDOM_VERTICES" "$MST_RANDOM_EXTRA_EDGES" \
  "$MST_RANDOM_SEED" "$MST_RANDOM_MAX_WEIGHT"
if [[ -n "${MST_RANDOM_EXTRA_EDGES_LIST:-}" ]]; then
  printf 'MST_RANDOM_EXTRA_EDGES_LIST=%s\n' "$MST_RANDOM_EXTRA_EDGES_LIST"
fi

module purge
module load amd/gcc/gcc-12
module load amd/gcc-12.2.1/openmpi-4.1.6
module load amd/nvidia/cuda-12.3.2

cd "$REPO_DIR"
make USE_CMAKE=OFF cuda CXX=g++ NVCC=nvcc NVCC_CCBIN=g++

run_graph() {
  local graph="$1"
  export MST_GRAPH="$graph"
  local report_name="cuda_${MST_GRAPH}_${SLURM_JOB_ID}.json"
  if [[ "$MST_GRAPH" == "random" && -n "${MST_RANDOM_EXTRA_EDGES_LIST:-}" ]]; then
    report_name="cuda_${MST_GRAPH}_v${MST_RANDOM_VERTICES}_e${MST_RANDOM_EXTRA_EDGES}_${SLURM_JOB_ID}.json"
  fi
  export MST_REPORT_PATH="$RESULTS_DIR/$report_name"
  printf 'Running CUDA graph=%s report=%s\n' "$MST_GRAPH" "$MST_REPORT_PATH"
  ./build/cuda/cuda_app
}

for graph in "${graphs[@]}"; do
  if [[ "$graph" == "random" ]]; then
    for extra_edges in "${random_extra_edges_values[@]}"; do
      export MST_RANDOM_EXTRA_EDGES="$extra_edges"
      run_graph "$graph"
    done
  else
    run_graph "$graph"
  fi
done

rm -rf "$TMPDIR"
date

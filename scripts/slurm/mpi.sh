#!/bin/bash
#SBATCH --account=d.pellegrini-thesis
#SBATCH --job-name=parallel-mst-mpi
#SBATCH --partition=ulow
#SBATCH --ntasks=2
#SBATCH --cpus-per-task=1
#SBATCH --mem=2G
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
graph_list="${GRAPHS//,/ }"
read -r -a graphs <<< "$graph_list"

mkdir -p "$EXPERIMENT_DIR/job_logs" "$RESULTS_DIR"

pwd
hostname
date
printf 'graphs=%s vertices=%s extra_edges=%s seed=%s max_weight=%s\n' \
  "$GRAPHS" "$RANDOM_VERTICES" "$RANDOM_EXTRA_EDGES" \
  "$RANDOM_SEED" "$RANDOM_MAX_WEIGHT"

module purge
module load amd/gcc/gcc-12
module load amd/gcc-12.2.1/openmpi-4.1.6

cd "$REPO_DIR"
make USE_CMAKE=OFF mpi MPICXX=mpicxx

for graph in "${graphs[@]}"; do
  resource_suffix="np${SLURM_NTASKS:-2}"
  if [[ "$graph" == "random" ]]; then
    report_name="mpi_${graph}_v${RANDOM_VERTICES}_e${RANDOM_EXTRA_EDGES}_s${RANDOM_SEED}_w${RANDOM_MAX_WEIGHT}_${resource_suffix}_${SLURM_JOB_ID}.json"
  else
    report_name="mpi_${graph}_${resource_suffix}_${SLURM_JOB_ID}.json"
  fi
  report_path="$RESULTS_DIR/$report_name"
  args=(--graph "$graph" --report "$report_path" --benchmark)
  if [[ "$graph" == "random" ]]; then
    args+=(
      --random-vertices "$RANDOM_VERTICES"
      --random-extra-edges "$RANDOM_EXTRA_EDGES"
      --random-seed "$RANDOM_SEED"
      --random-max-weight "$RANDOM_MAX_WEIGHT"
    )
  fi
  printf 'Running MPI graph=%s report=%s\n' "$graph" "$report_path"
  mpirun -np "${SLURM_NTASKS}" ./build/mpi/mpi_app "${args[@]}"
done

date

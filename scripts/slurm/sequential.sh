#!/bin/bash
#SBATCH --account=d.pellegrini-thesis
#SBATCH --job-name=parallel-mst-sequential
#SBATCH --partition=ulow
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=00:30:00
#SBATCH --output=job_logs/out_%x_%j.log
#SBATCH --mail-type=FAIL
#SBATCH --mail-user=d.pellegrini10@campus.unimib.it

set -euo pipefail

export REPO_DIR="${REPO_DIR:-$HOME/hpc-project}"
export EXPERIMENT_DIR="${EXPERIMENT_DIR:-$HOME/experiments/parallel-mst}"
export RESULTS_DIR="$EXPERIMENT_DIR/results"
RANDOM_VERTICES="${RANDOM_VERTICES:-32768}"
RANDOM_SEED="${RANDOM_SEED:-886261}"
edges_list="${RANDOM_EDGES_LIST:-32768,65536,131072,196608,393216,786432,1572864,3145728,6291456,12582912}"
read -r -a edges_values <<< "${edges_list//,/ }"

mkdir -p "$EXPERIMENT_DIR/job_logs" "$RESULTS_DIR"

pwd
hostname
date
printf 'vertices=%s seed=%s edges_list=%s\n' \
  "$RANDOM_VERTICES" "$RANDOM_SEED" "$edges_list"

module purge
module load amd/gcc/gcc-12

cd "$REPO_DIR"
make sequential CXX=g++

# Borůvka seriale di riferimento: nessun parallelismo, resources=1.
resources=1
csv_path="$RESULTS_DIR/sequential_${SLURM_JOB_ID}.csv"
printf 'backend,vertices,edges,density,seed,resources,overhead_seconds,exec_seconds,total_seconds,verified\n' \
  > "$csv_path"

# `edges` è il secondo argomento posizionale del programma: il numero totale
# di archi del grafo generato (src/sequential.cpp genera esattamente questo
# numero di archi, a parità di seed identico agli altri backend), quindi
# density = edges/vertices è la densità esatta.
for edges in "${edges_values[@]}"; do
  printf 'Running sequential vertices=%s edges=%s seed=%s\n' \
    "$RANDOM_VERTICES" "$edges" "$RANDOM_SEED"
  out="$(./build/sequential_app "$RANDOM_VERTICES" "$edges" "$RANDOM_SEED")"
  printf '%s\n' "$out"

  overhead="$(grep -o 'overhead_seconds=[0-9.]*' <<<"$out" | cut -d= -f2)"
  exec_t="$(grep -o 'exec_seconds=[0-9.]*'      <<<"$out" | cut -d= -f2)"
  total="$(grep -o 'total_seconds=[0-9.]*'      <<<"$out" | cut -d= -f2)"
  verified="$(grep -o 'verification=[A-Z]*'      <<<"$out" | cut -d= -f2)"
  density="$(awk -v e="$edges" -v v="$RANDOM_VERTICES" 'BEGIN{printf "%.4f", e/v}')"
  printf '%s,%s,%s,%s,%s,%s,%s,%s,%s,%s\n' \
    sequential "$RANDOM_VERTICES" "$edges" "$density" "$RANDOM_SEED" "$resources" \
    "$overhead" "$exec_t" "$total" "$verified" >> "$csv_path"
done

printf 'CSV written to %s\n' "$csv_path"
date

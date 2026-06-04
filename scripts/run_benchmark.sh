#!/bin/bash

set -euo pipefail

REPO_DIR="${REPO_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
BUILD_DIR="${BUILD_DIR:-$REPO_DIR/build}"
RESULTS_DIR="${RESULTS_DIR:-$REPO_DIR/results}"
MPI_PROCS="${MPI_PROCS:-2}"

usage() {
  printf '%s\n' \
    'Usage: scripts/run_benchmark.sh <openmp|mpi|cuda> [app options]' \
    '' \
    'Examples:' \
    '  scripts/run_benchmark.sh openmp --graph tie' \
    '  scripts/run_benchmark.sh mpi --graph random --random-vertices 32768 --random-extra-edges 196608'
}

json_escape() {
  local value="$1"
  value="${value//\\/\\\\}"
  value="${value//\"/\\\"}"
  value="${value//$'\n'/\\n}"
  printf '%s' "$value"
}

has_option() {
  local name="$1"
  shift
  for argument in "$@"; do
    if [[ "$argument" == "$name" || "$argument" == "$name="* ]]; then
      return 0
    fi
  done
  return 1
}

value_after_option() {
  local name="$1"
  shift
  local previous_matches=0
  for argument in "$@"; do
    if [[ "$previous_matches" == 1 ]]; then
      printf '%s' "$argument"
      return 0
    fi
    if [[ "$argument" == "$name="* ]]; then
      printf '%s' "${argument#*=}"
      return 0
    fi
    if [[ "$argument" == "$name" ]]; then
      previous_matches=1
    fi
  done
  return 1
}

value_after_option_or_default() {
  local name="$1"
  local default_value="$2"
  shift 2
  value_after_option "$name" "$@" || printf '%s' "$default_value"
}

if [[ $# -lt 1 ]]; then
  usage
  exit 1
fi

backend="$1"
shift
args=("$@")

case "$backend" in
openmp)
  executable="$BUILD_DIR/openmp/openmp_app"
  command=("$executable")
  ;;
mpi)
  executable="$BUILD_DIR/mpi/mpi_app"
  command=(mpirun -np "$MPI_PROCS" "$executable")
  ;;
cuda)
  executable="$BUILD_DIR/cuda/cuda_app"
  command=("$executable")
  ;;
*)
  usage
  exit 1
  ;;
esac

mkdir -p "$RESULTS_DIR"

if ! has_option "--report" "${args[@]}"; then
  timestamp="$(date -u +%Y%m%dT%H%M%SZ)"
  graph_name="$(value_after_option_or_default "--graph" "test" "${args[@]}")"
  random_vertices="$(value_after_option_or_default "--random-vertices" "32768" "${args[@]}")"
  random_extra_edges="$(value_after_option_or_default "--random-extra-edges" "196608" "${args[@]}")"
  random_seed="$(value_after_option_or_default "--random-seed" "886261" "${args[@]}")"
  random_max_weight="$(value_after_option_or_default "--random-max-weight" "10000" "${args[@]}")"
  case "$backend" in
  openmp)
    resource_suffix="t${OMP_NUM_THREADS:-default}"
    ;;
  mpi)
    resource_suffix="np${MPI_PROCS}"
    ;;
  cuda)
    cuda_host_memory="$(value_after_option_or_default "--cuda-host-memory" "default" "${args[@]}")"
    resource_suffix="hm${cuda_host_memory}"
    ;;
  esac
  if [[ "$graph_name" == "random" ]]; then
    report_path="$RESULTS_DIR/${backend}_${graph_name}_v${random_vertices}_e${random_extra_edges}_s${random_seed}_w${random_max_weight}_${resource_suffix}_${timestamp}.json"
  else
    report_path="$RESULTS_DIR/${backend}_${graph_name}_${resource_suffix}_${timestamp}.json"
  fi
  args+=(--report "$report_path")
else
  report_path="$(value_after_option "--report" "${args[@]}")"
fi

if ! has_option "--benchmark" "${args[@]}" &&
   ! has_option "--render" "${args[@]}" &&
   ! has_option "--no-render" "${args[@]}"; then
  args+=(--benchmark)
fi

git_revision="$(git -C "$REPO_DIR" rev-parse --short HEAD 2>/dev/null || printf 'unknown')"
manifest_path="${report_path%.json}_manifest.json"
command+=("${args[@]}")
command_string="$(printf '%q ' "${command[@]}")"
argument_string="$(printf '%q ' "${args[@]}")"

set +e
"${command[@]}"
status="$?"
set -e

cat > "$manifest_path" <<EOF
{
  "backend": "$(json_escape "$backend")",
  "git_revision": "$(json_escape "$git_revision")",
  "command": "$(json_escape "$command_string")",
  "arguments": "$(json_escape "$argument_string")",
  "report_path": "$(json_escape "$report_path")",
  "exit_status": $status
}
EOF

printf 'Report: %s\nManifest: %s\n' "$report_path" "$manifest_path"
exit "$status"

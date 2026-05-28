#!/bin/bash

set -euo pipefail

usage() {
  cat <<'USAGE'
Usage: scripts/sync_results_from_server.sh [options]

Sync benchmark JSON results from the remote server into this local checkout.

Options:
  -n, --dry-run                  Show what would be copied without copying
      --server HOST              SSH host alias (default: uni-server)
      --remote-results-dir PATH  Remote results directory
                                  (default: ~/experiments/parallel-mst/results)
      --local-results-dir PATH   Local destination (default: ./results)
  -h, --help                     Show this help

Environment overrides:
  SYNC_SERVER                    Same as --server
  REMOTE_RESULTS_DIR             Same as --remote-results-dir
  LOCAL_RESULTS_DIR              Same as --local-results-dir
USAGE
}

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd "$script_dir/.." && pwd)"

dry_run=0
server="${SYNC_SERVER:-uni-server}"
remote_results_dir="${REMOTE_RESULTS_DIR:-~/experiments/parallel-mst/results}"
local_results_dir="${LOCAL_RESULTS_DIR:-$repo_root/results}"

while [[ $# -gt 0 ]]; do
  case "$1" in
    -n|--dry-run)
      dry_run=1
      shift
      ;;
    --server)
      server="${2:?missing value for --server}"
      shift 2
      ;;
    --remote-results-dir)
      remote_results_dir="${2:?missing value for --remote-results-dir}"
      shift 2
      ;;
    --local-results-dir)
      local_results_dir="${2:?missing value for --local-results-dir}"
      shift 2
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      printf 'Unknown option: %s\n\n' "$1" >&2
      usage >&2
      exit 2
      ;;
  esac
done

rsync_args=(
  -av
  --prune-empty-dirs
  --include='*/'
  --include='*.json'
  --exclude='*'
)

if [[ "$dry_run" -eq 1 ]]; then
  rsync_args+=(-n --itemize-changes)
  printf 'DRY RUN: no files will be copied.\n'
else
  printf 'SYNC: copying JSON results from %s:%s\n' "$server" "$remote_results_dir"
  mkdir -p "$local_results_dir"
fi

printf 'Source:      %s:%s/\n' "$server" "${remote_results_dir%/}"
printf 'Destination: %s/\n' "${local_results_dir%/}"
rsync "${rsync_args[@]}" \
  "${server}:${remote_results_dir%/}/" \
  "${local_results_dir%/}/"

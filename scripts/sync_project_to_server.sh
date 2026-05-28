#!/bin/bash

set -euo pipefail

usage() {
  cat <<'USAGE'
Usage: scripts/sync_project_to_server.sh [options]

Sync project files from this local checkout to the remote server.

Options:
  -n, --dry-run              Show what would be copied without copying
      --server HOST          SSH host alias (default: uni-server)
      --local-dir PATH       Local source (default: repository root)
      --remote-dir PATH      Remote destination (default: ~/hpc-project)
  -h, --help                 Show this help

Environment overrides:
  SYNC_SERVER                Same as --server
  REMOTE_PROJECT_DIR         Same as --remote-dir
  LOCAL_PROJECT_DIR          Same as --local-dir
USAGE
}

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd "$script_dir/.." && pwd)"

dry_run=0
server="${SYNC_SERVER:-uni-server}"
remote_dir="${REMOTE_PROJECT_DIR:-~/hpc-project}"
local_dir="${LOCAL_PROJECT_DIR:-$repo_root}"

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
    --remote-dir)
      remote_dir="${2:?missing value for --remote-dir}"
      shift 2
      ;;
    --local-dir)
      local_dir="${2:?missing value for --local-dir}"
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
  --filter=':- .gitignore'
  --exclude='.git/'
  --exclude='AGENTS.md'
  --exclude='agents.md'
  --exclude='Presentazione nv-link/'
  --exclude='BBBBB'
  --exclude='docs/'
)

if [[ "$dry_run" -eq 1 ]]; then
  rsync_args+=(-n --itemize-changes)
  printf 'DRY RUN: no files will be copied.\n'
else
  printf 'SYNC: copying local project to %s:%s\n' "$server" "$remote_dir"
fi

printf 'Source:      %s/\n' "${local_dir%/}"
printf 'Destination: %s:%s/\n' "$server" "${remote_dir%/}"
rsync "${rsync_args[@]}" "${local_dir%/}/" "${server}:${remote_dir%/}/"

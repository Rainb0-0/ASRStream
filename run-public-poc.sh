#!/usr/bin/env bash
set -euo pipefail

project_directory=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
config_path=${1:-"$project_directory/config.public.example.toml"}
if [[ $# -gt 0 ]]; then
  shift
fi

exec env POC_ENTRYPOINT=asr-public-poc "$project_directory/run-poc.sh" "$config_path" "$@"

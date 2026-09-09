#!/usr/bin/env bash
set -euo pipefail

# Bootstrap and launch the local IPTV player proof of concept.
project_directory=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
config_path=${1:-"$project_directory/config.example.toml"}
if [[ $# -gt 0 ]]; then
  shift
fi
if [[ ! -f "$config_path" ]]; then
  echo "POC configuration not found: $config_path" >&2
  exit 1
fi
config_path=$(cd -- "$(dirname -- "$config_path")" && pwd)/$(basename -- "$config_path")

python_binary=${PYTHON_BIN:-python3.12}
venv_directory="$project_directory/.venv"
poc_entrypoint=${POC_ENTRYPOINT:-asr-poc}

if ! command -v "$python_binary" >/dev/null 2>&1; then
  echo "Python 3.12 is required. Install it or set PYTHON_BIN to its path." >&2
  exit 1
fi
if ! "$python_binary" -c 'import sys; raise SystemExit(sys.version_info[:2] != (3, 12))'; then
  echo "PYTHON_BIN must be Python 3.12." >&2
  exit 1
fi

if [[ ! -x "$venv_directory/bin/python" ]]; then
  "$python_binary" -m venv "$venv_directory"
fi
if ! "$venv_directory/bin/python" -c 'import sys; raise SystemExit(sys.version_info[:2] != (3, 12))'; then
  echo "Existing .venv is not Python 3.12; recreate it with Python 3.12." >&2
  exit 1
fi

cd "$project_directory"

"$venv_directory/bin/python" -m pip install --no-build-isolation "$project_directory"

# CTranslate2 discovers these shared libraries at runtime when the POC uses
# the CUDA defaults in config.example.toml. CPU configurations skip them.
if grep -Eq '^[[:space:]]*device[[:space:]]*=[[:space:]]*"cuda"' "$config_path" && \
   [[ ! -d "$venv_directory/lib/python3.12/site-packages/nvidia/cublas/lib" ]]; then
  "$venv_directory/bin/python" -m pip install \
    'nvidia-cublas-cu12>=12,<13' \
    'nvidia-cudnn-cu12>=9,<10' \
    'nvidia-cuda-nvrtc-cu12>=12,<13'
fi

exec "$venv_directory/bin/$poc_entrypoint" --config "$config_path" "$@"

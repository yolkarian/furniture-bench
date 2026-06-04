#!/usr/bin/env bash
set -euo pipefail

cd /workspace

export VIRTUAL_ENV="${UV_PROJECT_ENVIRONMENT:-/workspace/.venv}"
export PATH="${VIRTUAL_ENV}/bin:${PATH}"

if [[ -f pyproject.toml ]]; then
    if [[ -f uv.lock ]]; then
        uv sync --locked
    else
        uv sync
    fi
fi

if [[ $# -eq 0 ]]; then
    exec bash
fi

exec "$@"

#!/usr/bin/env bash

set -euo pipefail

# Install the repository into the active container environment on startup.
/opt/conda/envs/${VENV_NAME}/bin/pip install -e /furniture-bench

# Isaac Gym is still optional. When it is mounted into the container we install
# it as an editable dependency before dropping into the shell.
if [[ -d "/isaacgym" ]]; then
  /opt/conda/envs/${VENV_NAME}/bin/pip install -e /isaacgym/python
fi

exec /bin/bash "$@"

#!/usr/bin/env bash

set -euo pipefail

# Keep this helper as a stable entry point for existing workflows.
# The offline IQL stack was removed, so there are no extra model packages to
# install beyond the main project itself.
python -m pip install -e .

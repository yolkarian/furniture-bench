#!/usr/bin/env bash

set -euo pipefail

# Install the refactored project and the offline-learning extras that remain supported.
pip install -e .
pip install -r implicit_q_learning/requirements.txt

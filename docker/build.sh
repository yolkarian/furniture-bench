#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

IMAGE_NAME="${1:-furniture-bench-sapien}"
IMAGE_TAG="${2:-latest}"

echo "Building ${IMAGE_NAME}:${IMAGE_TAG} from docker/sapien.Dockerfile"
docker build \
    -f "${SCRIPT_DIR}/sapien.Dockerfile" \
    -t "${IMAGE_NAME}:${IMAGE_TAG}" \
    "${REPO_ROOT}"

#!/usr/bin/env bash
set -euo pipefail

IMAGE_NAME="furniture-bench-sapien"
IMAGE_TAG="latest"
NO_CACHE=()

usage() {
    cat <<EOF
Usage: $(basename "$0") [OPTIONS]

Build the furniture-bench SAPIEN image with the uv-managed Python project.

Options:
  -n, --name NAME      Image name (default: ${IMAGE_NAME})
  -t, --tag TAG        Image tag  (default: ${IMAGE_TAG})
      --no-cache       Build without Docker layer cache
  -h, --help           Show this help message and exit
EOF
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        -n|--name)
            IMAGE_NAME="$2"
            shift 2
            ;;
        -t|--tag)
            IMAGE_TAG="$2"
            shift 2
            ;;
        --no-cache)
            NO_CACHE=(--no-cache)
            shift
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        -*)
            echo "Error: unknown option '$1'" >&2
            usage >&2
            exit 1
            ;;
        *)
            echo "Error: unexpected argument '$1'" >&2
            usage >&2
            exit 1
            ;;
    esac
done

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

echo "Building ${IMAGE_NAME}:${IMAGE_TAG} from docker/sapien.Dockerfile"
docker build \
    "${NO_CACHE[@]+"${NO_CACHE[@]}"}" \
    -f "${SCRIPT_DIR}/sapien.Dockerfile" \
    -t "${IMAGE_NAME}:${IMAGE_TAG}" \
    "${REPO_ROOT}"

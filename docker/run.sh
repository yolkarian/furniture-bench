#!/usr/bin/env bash
set -euo pipefail

IMAGE_NAME="furniture-bench-sapien"
IMAGE_TAG="latest"
GPU_FLAG="--gpus all --runtime=nvidia"
EXTRA_VOLUMES=()
DEV_MOUNT=false
DETACHED=false
DETACH_CMD=""

usage() {
    cat <<EOF
Usage: $(basename "$0") [OPTIONS] [MOUNT_DIR ...]

Launch an interactive shell inside the furniture-bench SAPIEN container.

The container is started with GPU access (--gpus all) by default, X11
forwarding for SAPIEN rendering, and is automatically removed on exit.

Positional arguments are host directories to mount into /workspace/:
  $(basename "$0") /data/trajectories /data/checkpoints
  mounts as /workspace/trajectories and /workspace/checkpoints

Options:
  -n, --name NAME      Image name  (default: ${IMAGE_NAME})
  -t, --tag TAG        Image tag   (default: ${IMAGE_TAG})
  -v, --volume SRC:DST Bind-mount with explicit container path (repeatable)
      --gpu DEVICES    Limit visible GPUs (e.g. 0 or 0,1). Default: all
      --cpu            Run without GPU (omit --gpus flag)
      --dev            Mount host project source over /root/furniture-bench
                       for live editing inside the container
  -d, --detach CMD     Run CMD in the background (detached mode).
                       The container keeps running after the script exits.
                       Use 'docker logs <id>' to follow output.
  -h, --help           Show this help message and exit

Examples:
  $(basename "$0")                              # interactive shell, all GPUs
  $(basename "$0") --gpu 0                      # single GPU
  $(basename "$0") --gpu 0,1                    # two GPUs
  $(basename "$0") --cpu                        # CPU only
  $(basename "$0") --dev                        # live-edit host source
  $(basename "$0") /data/trajectories           # mount into /workspace/trajectories
  $(basename "$0") -v /data:/mnt/data           # explicit mount path

  # Mount a training project and run a script in the background
  $(basename "$0") --gpu 0 -d "python /workspace/my_project/train.py" ~/my_project

  # Detached training with dev source and extra data
  $(basename "$0") --dev --gpu 0,1 \\
      -d "python /workspace/my_project/train.py --epochs 100" \\
      ~/my_project /data/datasets
EOF
}

POSITIONAL=()
while [[ $# -gt 0 ]]; do
    case "$1" in
        -n|--name)    IMAGE_NAME="$2"; shift 2 ;;
        -t|--tag)     IMAGE_TAG="$2";  shift 2 ;;
        -v|--volume)  EXTRA_VOLUMES+=("-v" "$2"); shift 2 ;;
        --gpu)        GPU_FLAG="--gpus device=$2 --runtime=nvidia"; shift 2 ;;
        --cpu)        GPU_FLAG=""; shift ;;
        --dev)        DEV_MOUNT=true; shift ;;
        -d|--detach)  DETACHED=true; DETACH_CMD="$2"; shift 2 ;;
        -h|--help)    usage; exit 0 ;;
        -*)           echo "Error: unknown option '$1'" >&2; usage >&2; exit 1 ;;
        *)            POSITIONAL+=("$1"); shift ;;
    esac
done

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

# --dev: bind-mount host project for live editing
SOURCE_MOUNT=()
if [[ "${DEV_MOUNT}" == true ]]; then
    echo "==> Dev mode: mounting ${PROJECT_ROOT} -> /root/furniture-bench"
    SOURCE_MOUNT=(-v "${PROJECT_ROOT}:/root/furniture-bench")
fi

# Positional args: mount each dir into /workspace/<basename>
WORKSPACE_MOUNTS=()
for dir in "${POSITIONAL[@]+"${POSITIONAL[@]}"}"; do
    dir="$(realpath "${dir}")"
    if [[ ! -d "${dir}" ]]; then
        echo "Error: '${dir}' is not a directory" >&2
        exit 1
    fi
    basename="$(basename "${dir}")"
    echo "==> Mounting ${dir} -> /workspace/${basename}"
    WORKSPACE_MOUNTS+=(-v "${dir}:/workspace/${basename}")
done

# X11 forwarding for SAPIEN rendering (interactive only)
X11_FLAGS=()
if [[ "${DETACHED}" == false ]] && [[ -n "${DISPLAY:-}" ]]; then
    X11_FLAGS=(
        -e "DISPLAY=${DISPLAY}"
        -v /tmp/.X11-unix:/tmp/.X11-unix
    )
fi

ENV_FLAGS=()
if [[ -z "${GPU_FLAG}" ]]; then
    ENV_FLAGS=(-e NVIDIA_VISIBLE_DEVICES=void)
fi

if [[ "${DETACHED}" == true ]]; then
    echo "==> Starting ${IMAGE_NAME}:${IMAGE_TAG} (detached)"
    echo "==> Command: ${DETACH_CMD}"

    CID=$(docker run --rm -d \
        ${GPU_FLAG} \
        "${SOURCE_MOUNT[@]+"${SOURCE_MOUNT[@]}"}" \
        "${WORKSPACE_MOUNTS[@]+"${WORKSPACE_MOUNTS[@]}"}" \
        "${ENV_FLAGS[@]+"${ENV_FLAGS[@]}"}" \
        "${EXTRA_VOLUMES[@]+"${EXTRA_VOLUMES[@]}"}" \
        "${IMAGE_NAME}:${IMAGE_TAG}" \
        bash -lc "${DETACH_CMD}")

    echo "==> Container: ${CID:0:12}"
    echo "==> Follow logs:  docker logs -f ${CID:0:12}"
    echo "==> Stop:         docker stop ${CID:0:12}"
else
    echo "==> Starting ${IMAGE_NAME}:${IMAGE_TAG}"

    docker run --rm -it \
        ${GPU_FLAG} \
        "${SOURCE_MOUNT[@]+"${SOURCE_MOUNT[@]}"}" \
        "${WORKSPACE_MOUNTS[@]+"${WORKSPACE_MOUNTS[@]}"}" \
        "${X11_FLAGS[@]+"${X11_FLAGS[@]}"}" \
        "${ENV_FLAGS[@]+"${ENV_FLAGS[@]}"}" \
        "${EXTRA_VOLUMES[@]+"${EXTRA_VOLUMES[@]}"}" \
        "${IMAGE_NAME}:${IMAGE_TAG}" \
        bash
fi

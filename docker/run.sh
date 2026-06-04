#!/usr/bin/env bash
set -euo pipefail

IMAGE_NAME="furniture-bench-sapien"
IMAGE_TAG="latest"
VENV_VOLUME="furniture-bench-sapien-venv"
GPU_ARGS=(--gpus all)
EXTRA_VOLUMES=()
DETACHED=false
DETACH_CMD=()
COMMAND=()

usage() {
    cat <<EOF
Usage: $(basename "$0") [OPTIONS] [MOUNT_DIR ...] [-- COMMAND [ARG ...]]

Run the furniture-bench SAPIEN container. The repository is mounted at
/workspace, /workspace/.venv is backed by a Docker volume, and the container
entrypoint runs uv sync --locked before launching the command.

Positional MOUNT_DIR arguments are host directories mounted into
/workspace/<basename>.

Options:
  -n, --name NAME        Image name (default: ${IMAGE_NAME})
  -t, --tag TAG          Image tag  (default: ${IMAGE_TAG})
  -v, --volume SRC:DST   Bind-mount with explicit container path (repeatable)
      --venv-volume NAME Docker volume for /workspace/.venv
                          (default: ${VENV_VOLUME})
      --gpu DEVICES      Use all GPUs or selected GPUs (all, 0, 0,1). Default: all
      --cpu              Run without GPU access
  -d, --detach CMD       Run CMD in the background
  -h, --help             Show this help message and exit

Examples:
  $(basename "$0")                                  # interactive shell, all GPUs
  $(basename "$0") --gpu 0                          # interactive shell, GPU 0
  $(basename "$0") --cpu -- python --version        # CPU command smoke test
  $(basename "$0") /data/trajectories               # mount data directory
  $(basename "$0") --gpu 0 -d "python train.py"     # detached command
EOF
}

POSITIONAL=()
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
        -v|--volume)
            EXTRA_VOLUMES+=("-v" "$2")
            shift 2
            ;;
        --venv-volume)
            VENV_VOLUME="$2"
            shift 2
            ;;
        --gpu)
            if [[ "$2" == "all" ]]; then
                GPU_ARGS=(--gpus all)
            else
                GPU_ARGS=(--gpus "device=$2")
            fi
            shift 2
            ;;
        --cpu)
            GPU_ARGS=()
            shift
            ;;
        -d|--detach)
            DETACHED=true
            DETACH_CMD=(bash -lc "$2")
            shift 2
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        --)
            shift
            COMMAND=("$@")
            break
            ;;
        -*)
            echo "Error: unknown option '$1'" >&2
            usage >&2
            exit 1
            ;;
        *)
            POSITIONAL+=("$1")
            shift
            ;;
    esac
done

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

WORKSPACE_MOUNTS=(-v "${PROJECT_ROOT}:/workspace" -v "${VENV_VOLUME}:/workspace/.venv")
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

X11_FLAGS=()
if [[ "${DETACHED}" == false ]] && [[ -n "${DISPLAY:-}" ]]; then
    X11_FLAGS=(
        -e "DISPLAY=${DISPLAY}"
        -v /tmp/.X11-unix:/tmp/.X11-unix
    )
fi

ENV_FLAGS=()
if [[ ${#GPU_ARGS[@]} -eq 0 ]]; then
    ENV_FLAGS=(-e NVIDIA_VISIBLE_DEVICES=void)
fi

RUN_COMMAND=("${COMMAND[@]+"${COMMAND[@]}"}")
RUN_MODE=(--rm)
if [[ ${#RUN_COMMAND[@]} -eq 0 && -t 0 && -t 1 ]]; then
    RUN_MODE=(--rm -it)
fi
if [[ "${DETACHED}" == true ]]; then
    RUN_MODE=(--rm -d)
    if [[ ${#DETACH_CMD[@]} -gt 0 ]]; then
        RUN_COMMAND=("${DETACH_CMD[@]}")
    fi
    if [[ ${#RUN_COMMAND[@]} -eq 0 ]]; then
        echo "Error: detached mode requires a command" >&2
        exit 1
    fi
fi

echo "==> Starting ${IMAGE_NAME}:${IMAGE_TAG}"
echo "==> Mounting ${PROJECT_ROOT} -> /workspace"
echo "==> Using ${VENV_VOLUME} -> /workspace/.venv"

if [[ "${DETACHED}" == true ]]; then
    CID=$(docker run \
        "${RUN_MODE[@]}" \
        "${GPU_ARGS[@]+"${GPU_ARGS[@]}"}" \
        "${WORKSPACE_MOUNTS[@]}" \
        "${ENV_FLAGS[@]+"${ENV_FLAGS[@]}"}" \
        "${EXTRA_VOLUMES[@]+"${EXTRA_VOLUMES[@]}"}" \
        "${IMAGE_NAME}:${IMAGE_TAG}" \
        "${RUN_COMMAND[@]}")

    echo "==> Container: ${CID:0:12}"
    echo "==> Follow logs:  docker logs -f ${CID:0:12}"
    echo "==> Stop:         docker stop ${CID:0:12}"
else
    docker run \
        "${RUN_MODE[@]}" \
        "${GPU_ARGS[@]+"${GPU_ARGS[@]}"}" \
        "${WORKSPACE_MOUNTS[@]}" \
        "${X11_FLAGS[@]+"${X11_FLAGS[@]}"}" \
        "${ENV_FLAGS[@]+"${ENV_FLAGS[@]}"}" \
        "${EXTRA_VOLUMES[@]+"${EXTRA_VOLUMES[@]}"}" \
        "${IMAGE_NAME}:${IMAGE_TAG}" \
        "${RUN_COMMAND[@]+"${RUN_COMMAND[@]}"}"
fi

#!/usr/bin/env bash

set -euo pipefail

# Expected CLI usage:
#   bash scripts/launch_client.sh --gpu --built
#   bash scripts/launch_client.sh --cpu --pulled
#   bash scripts/launch_client.sh --sim-gpu --built
mode="${1:-}"
source_kind="${2:-}"

case "${mode}" in
  --gpu|--cpu|--sim-gpu)
    ;;
  *)
    echo "Unknown option: ${mode}"
    exit 1
    ;;
esac

if [[ -n "${CLIENT_DOCKER:-}" ]]; then
  echo "Using ${CLIENT_DOCKER}"
else
  case "${source_kind}" in
    --built)
      case "${mode}" in
        --gpu|--sim-gpu) CLIENT_DOCKER="client-gpu" ;;
        --cpu) CLIENT_DOCKER="client" ;;
      esac
      ;;
    --pulled)
      case "${mode}" in
        --gpu|--sim-gpu) CLIENT_DOCKER="furniturebench/client-gpu:latest" ;;
        --cpu) CLIENT_DOCKER="furniturebench/client:latest" ;;
      esac
      ;;
    "")
      echo "No second argument provided"
      exit 1
      ;;
    *)
      echo "Unknown second argument: ${source_kind}"
      exit 1
      ;;
  esac
fi

if [[ -z "${FURNITURE_BENCH:-}" ]]; then
  echo "FURNITURE_BENCH is not set"
  exit 1
fi

echo "Environment Variables"
echo "---------------------"
echo "CLIENT_DOCKER: ${CLIENT_DOCKER}"
echo "FURNITURE_BENCH: ${FURNITURE_BENCH}"
echo "HOST_DATA_MOUNT: ${HOST_DATA_MOUNT:-}"
echo "CONTAINER_DATA_MOUNT: ${CONTAINER_DATA_MOUNT:-}"
echo "ISAAC_GYM_PATH: ${ISAAC_GYM_PATH:-}"

# Allow Docker to connect to the local X server for interactive rendering.
xhost +

common_args=(
  --network host
  -it
  --privileged
  -v "${FURNITURE_BENCH}:/furniture-bench"
  --rm
  --ipc=host
  -e "DISPLAY=${DISPLAY:-}"
  -v /tmp/.X11-unix:/tmp/.X11-unix
  --env=QT_X11_NO_MITSHM=1
)

if [[ -n "${HOST_DATA_MOUNT:-}" && -n "${CONTAINER_DATA_MOUNT:-}" ]]; then
  common_args+=( -v "${HOST_DATA_MOUNT}:${CONTAINER_DATA_MOUNT}" )
fi

case "${mode}" in
  --gpu)
    docker run "${common_args[@]}" --gpus=all "${CLIENT_DOCKER}"
    ;;
  --cpu)
    docker run "${common_args[@]}" "${CLIENT_DOCKER}"
    ;;
  --sim-gpu)
    if [[ -z "${ISAAC_GYM_PATH:-}" ]]; then
      echo "ISAAC_GYM_PATH is not set"
      exit 1
    fi
    docker run "${common_args[@]}" --gpus=all \
      -v "${ISAAC_GYM_PATH}:/isaacgym" \
      "${CLIENT_DOCKER}"
    ;;
esac

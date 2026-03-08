#!/usr/bin/env bash

set -euo pipefail

# Respect an explicitly provided image first so existing automation keeps
# working without any argument changes.
if [[ -n "${SERVER_DOCKER:-}" ]]; then
  echo "Using ${SERVER_DOCKER}"
else
  case "${1:-}" in
    --built)
      SERVER_DOCKER="server"
      ;;
    --pulled)
      SERVER_DOCKER="furniturebench/server:latest"
      ;;
    "")
      echo "No first argument provided"
      exit 1
      ;;
    *)
      echo "Unknown first argument: ${1}"
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
echo "SERVER_DOCKER: ${SERVER_DOCKER}"
echo "FURNITURE_BENCH: ${FURNITURE_BENCH}"

# Launch the server container with the repository mounted in-place.
docker run -it --rm --network=host --privileged \
  -v "${FURNITURE_BENCH}:/furniture-bench" \
  "${SERVER_DOCKER}" \
  /bin/bash

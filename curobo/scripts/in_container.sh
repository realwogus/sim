#!/usr/bin/env bash
set -Eeuo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd -- "$SCRIPT_DIR/.." && pwd)"
WORKSPACE_DIR="$(cd -- "$PROJECT_DIR/.." && pwd)"
IMAGE="${CUROBO_IMAGE:-piper-curobo-thor:latest}"

if [[ "$#" -eq 0 ]]; then
  set -- bash
fi

TTY_ARGS=()
if [[ -t 0 && -t 1 ]]; then
  TTY_ARGS=(-it)
fi

exec docker run --rm "${TTY_ARGS[@]}" \
  --runtime nvidia \
  --gpus all \
  --ipc host \
  --network host \
  -e CUDA_VISIBLE_DEVICES=0 \
  -v "$WORKSPACE_DIR:/workspace" \
  -w /workspace/curobo \
  "$IMAGE" "$@"

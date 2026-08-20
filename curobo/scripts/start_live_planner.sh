#!/usr/bin/env bash
set -Eeuo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd -- "$SCRIPT_DIR/.." && pwd)"
WORKSPACE_DIR="$(cd -- "$PROJECT_DIR/.." && pwd)"
IMAGE="${CUROBO_IMAGE:-piper-curobo-thor:latest}"
CONTAINER_NAME="${CUROBO_LIVE_CONTAINER:-piper-curobo-live}"

exec docker run --rm \
  --name "$CONTAINER_NAME" \
  --runtime nvidia \
  --gpus all \
  --ipc host \
  --network host \
  -e CUDA_VISIBLE_DEVICES=0 \
  -v "$WORKSPACE_DIR:/workspace" \
  -w /workspace/curobo \
  "$IMAGE" python -u scripts/live_planner_server.py "$@"

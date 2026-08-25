#!/usr/bin/env bash
set -Eeuo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd -- "$SCRIPT_DIR/.." && pwd)"
WORKSPACE_DIR="$(cd -- "$PROJECT_DIR/.." && pwd)"
MUJOCO_PYTHON="$WORKSPACE_DIR/mujoco/.venv/bin/python"
IMAGE="${CUROBO_IMAGE:-piper-curobo-thor:latest}"
CONTAINER_NAME="piper-curobo-rail"
PORT="${CUROBO_RAIL_PORT:-5564}"
OWNED_CONTAINER=0

cleanup() {
  if [[ "$OWNED_CONTAINER" == "1" ]]; then
    docker stop --timeout 3 "$CONTAINER_NAME" >/dev/null 2>&1 || true
    docker rm "$CONTAINER_NAME" >/dev/null 2>&1 || true
  fi
}
trap cleanup EXIT INT TERM

if ! docker inspect --format '{{.State.Running}}' "$CONTAINER_NAME" 2>/dev/null \
  | grep -qx true; then
  docker rm "$CONTAINER_NAME" >/dev/null 2>&1 || true
  echo "Starting independent rail-PiPER 7-DOF planner..."
  docker run --rm \
    --name "$CONTAINER_NAME" \
    --runtime nvidia \
    --gpus all \
    --ipc host \
    --network host \
    -e CUDA_VISIBLE_DEVICES=0 \
    -v "$WORKSPACE_DIR:/workspace" \
    -w /workspace/curobo \
    "$IMAGE" python -u scripts/rail_planner_server.py --port "$PORT" &
  OWNED_CONTAINER=1
fi

echo "Waiting for rail planner on 127.0.0.1:$PORT..."
for _ in $(seq 1 180); do
  if "$MUJOCO_PYTHON" -c \
    "import socket; s=socket.create_connection(('127.0.0.1',$PORT),0.2); s.close()" \
    2>/dev/null; then
    "$MUJOCO_PYTHON" "$PROJECT_DIR/bridge/live_rail_endpoint_gui.py" \
      --port "$PORT" "$@"
    exit $?
  fi
  sleep 1
done

echo "ERROR: rail planner did not become ready within 180 seconds" >&2
exit 1

#!/usr/bin/env bash
set -Eeuo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd -- "$SCRIPT_DIR/.." && pwd)"
MUJOCO_PYTHON="$PROJECT_DIR/../mujoco/.venv/bin/python"
CONTAINER_NAME="${CUROBO_LIVE_CONTAINER:-piper-curobo-live}"
SERVER_PID=""

cleanup() {
  if [[ -n "$SERVER_PID" ]]; then
    docker stop --timeout 3 "$CONTAINER_NAME" >/dev/null 2>&1 || true
    wait "$SERVER_PID" 2>/dev/null || true
    docker rm "$CONTAINER_NAME" >/dev/null 2>&1 || true
  fi
}
trap cleanup EXIT INT TERM

if ! docker inspect --format '{{.State.Running}}' "$CONTAINER_NAME" 2>/dev/null | grep -qx true; then
  # A killed GUI can leave a stopped/dead named container behind.
  docker rm "$CONTAINER_NAME" >/dev/null 2>&1 || true
  echo "Starting persistent cuRobo planner (first warmup can take about a minute)..."
  "$SCRIPT_DIR/start_live_planner.sh" &
  SERVER_PID=$!
fi

echo "Waiting for cuRobo planner on 127.0.0.1:5561..."
for _ in $(seq 1 180); do
  if "$MUJOCO_PYTHON" -c 'import socket; s=socket.create_connection(("127.0.0.1",5561),0.2); s.close()' 2>/dev/null; then
    echo "Planner is ready. Opening MuJoCo GUI..."
    "$MUJOCO_PYTHON" "$PROJECT_DIR/bridge/live_red_gui.py" "$@"
    exit $?
  fi
  sleep 1
done

echo "ERROR: cuRobo planner did not become ready within 180 seconds" >&2
exit 1

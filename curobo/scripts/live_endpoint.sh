#!/usr/bin/env bash
set -Eeuo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd -- "$SCRIPT_DIR/.." && pwd)"
MUJOCO_PYTHON="$PROJECT_DIR/../mujoco/.venv/bin/python"
DUAL_PLANNER="${CUROBO_DUAL_PLANNER:-0}"
COMPARE_PLANNERS="${CUROBO_COMPARE_PLANNERS:-0}"
SERVER_PIDS=()
OWNED_CONTAINERS=()

cleanup() {
  local container_name server_pid
  for container_name in "${OWNED_CONTAINERS[@]}"; do
    docker stop --timeout 3 "$container_name" >/dev/null 2>&1 || true
  done
  for server_pid in "${SERVER_PIDS[@]}"; do
    wait "$server_pid" 2>/dev/null || true
  done
  for container_name in "${OWNED_CONTAINERS[@]}"; do
    docker rm "$container_name" >/dev/null 2>&1 || true
  done
}
trap cleanup EXIT INT TERM

start_planner() {
  local container_name="$1"
  local port="$2"
  local planner_label="$3"
  shift 3

  if docker inspect --format '{{.State.Running}}' "$container_name" 2>/dev/null \
    | grep -qx true; then
    return
  fi

  docker rm "$container_name" >/dev/null 2>&1 || true
  echo "Starting $planner_label cuRobo planner (first warmup can take about a minute)..."
  CUROBO_LIVE_CONTAINER="$container_name" \
    "$SCRIPT_DIR/start_live_planner.sh" \
      --mode endpoint --port "$port" "$@" &
  SERVER_PIDS+=("$!")
  OWNED_CONTAINERS+=("$container_name")
}

wait_for_planner() {
  local port="$1"
  local planner_label="$2"
  echo "Waiting for $planner_label planner on 127.0.0.1:$port..."
  for _ in $(seq 1 180); do
    if "$MUJOCO_PYTHON" -c \
      "import socket; s=socket.create_connection(('127.0.0.1',$port),0.2); s.close()" \
      2>/dev/null; then
      return
    fi
    sleep 1
  done
  echo "ERROR: $planner_label planner did not become ready within 180 seconds" >&2
  exit 1
}

if [[ "$COMPARE_PLANNERS" == "1" ]]; then
  # Warm up serially so CUDA graph capture does not compete for GPU memory.
  start_planner "piper-curobo-endpoint" 5562 "sequential 6+6-DOF"
  wait_for_planner 5562 "sequential 6+6-DOF"
  start_planner "piper-curobo-endpoint-dual" 5563 "joint 12-DOF" --dual
  wait_for_planner 5563 "joint 12-DOF"
  echo "Both planners are ready. Opening comparison GUI..."
  "$MUJOCO_PYTHON" "$PROJECT_DIR/bridge/live_endpoint_gui.py" \
    --port 5563 --joint-dual --single-port 5562 "$@"
elif [[ "$DUAL_PLANNER" == "1" ]]; then
  start_planner "piper-curobo-endpoint-dual" 5563 "joint 12-DOF" --dual
  wait_for_planner 5563 "joint 12-DOF"
  echo "Planner is ready. Opening endpoint GUI..."
  "$MUJOCO_PYTHON" "$PROJECT_DIR/bridge/live_endpoint_gui.py" \
    --port 5563 --joint-dual "$@"
else
  start_planner "piper-curobo-endpoint" 5562 "position-only 6-DOF"
  wait_for_planner 5562 "position-only 6-DOF"
  echo "Planner is ready. Opening endpoint GUI..."
  "$MUJOCO_PYTHON" "$PROJECT_DIR/bridge/live_endpoint_gui.py" \
    --port 5562 "$@"
fi

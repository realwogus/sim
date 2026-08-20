#!/usr/bin/env bash
set -Eeuo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd -- "$SCRIPT_DIR/.." && pwd)"
MUJOCO_PYTHON="$PROJECT_DIR/../mujoco/.venv/bin/python"

if [[ ! -x "$MUJOCO_PYTHON" ]]; then
  echo "ERROR: MuJoCo virtual environment not found: $MUJOCO_PYTHON" >&2
  exit 1
fi

exec "$MUJOCO_PYTHON" "$PROJECT_DIR/bridge/play_in_mujoco.py" "$@"

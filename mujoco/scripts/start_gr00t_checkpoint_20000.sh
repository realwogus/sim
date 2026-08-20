#!/usr/bin/env bash
set -Eeuo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd -- "$SCRIPT_DIR/.." && pwd)"
WORKSPACE_DIR="$(cd -- "$PROJECT_DIR/.." && pwd)"

exec bash "$SCRIPT_DIR/start_gr00t_server.sh" \
  "$WORKSPACE_DIR/gr00t_outputs/checkpoint-20000"

#!/usr/bin/env bash
set -Eeuo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
SIM_DIR="$(cd -- "$SCRIPT_DIR/.." && pwd)"

export CUROBO_DUAL_PLANNER=1
export CUROBO_COMPARE_PLANNERS="${CUROBO_COMPARE_PLANNERS:-1}"
exec "$SIM_DIR/curobo/scripts/live_endpoint.sh" "$@"

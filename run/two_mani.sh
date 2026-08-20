#!/usr/bin/env bash
set -Eeuo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
SIM_DIR="$(cd -- "$SCRIPT_DIR/.." && pwd)"

exec "$SIM_DIR/curobo/scripts/live_endpoint.sh" "$@"

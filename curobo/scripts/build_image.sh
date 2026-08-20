#!/usr/bin/env bash
set -Eeuo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd -- "$SCRIPT_DIR/.." && pwd)"
IMAGE="${CUROBO_IMAGE:-piper-curobo-thor:latest}"

exec docker build \
  --tag "$IMAGE" \
  --file "$PROJECT_DIR/docker/Dockerfile.thor" \
  "$PROJECT_DIR"

#!/usr/bin/env bash
set -Eeuo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd -- "$SCRIPT_DIR/.." && pwd)"

exec "$SCRIPT_DIR/in_container.sh" python -m curobo.examples.getting_started.build_robot_model \
  --urdf /workspace/curobo/robots/piper/piper_arm.urdf \
  --asset-path /workspace/curobo/robots/piper \
  --tool-frames gripper_center \
  --output /workspace/curobo/robots/piper/piper.yml \
  --clip-link arm_base z 0.0 \
  --compute-metrics \
  --seed 42

#!/usr/bin/env bash
set -Eeuo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd -- "$SCRIPT_DIR/.." && pwd)"
OUTPUT="$PROJECT_DIR/outputs/red_approach_trajectory.json"
CONTAINER_OUTPUT="/workspace/curobo/outputs/red_approach_trajectory.json"
CONTAINER_WORLD="/workspace/curobo/worlds/piper_red_gate.yml"
SCENE="$PROJECT_DIR/../mujoco/models/scenes/piper_red_gate.xml"

# Red block in MuJoCo world: center z=0.236, half-height=0.036.
# PiPER base world z=0.20 and approach clearance=0.03:
# target base z = 0.236 + 0.036 + 0.03 - 0.20 = 0.102 m.
TARGET=(0.42 0.08 0.102)
# Downward red-block approach orientation (wxyz), validated for this gate.
GOAL_QUATERNION=(0.53240967 -0.53241307 -0.46533632 -0.46533674)
HOLD=10
GRIPPER_OPENING=""
HEADLESS=false

while [[ "$#" -gt 0 ]]; do
  case "$1" in
    --hold|--gripper-opening)
      if [[ "$#" -lt 2 ]]; then
        echo "ERROR: $1 requires a value" >&2
        exit 2
      fi
      if [[ "$1" == "--hold" ]]; then
        HOLD="$2"
      else
        GRIPPER_OPENING="$2"
      fi
      shift 2
      ;;
    --headless)
      HEADLESS=true
      shift
      ;;
    *)
      echo "Usage: bash scripts/go_to_red.sh [--hold SECONDS] [--gripper-opening METERS] [--headless]" >&2
      exit 2
      ;;
  esac
done

PLAY_ARGS=(--hold "$HOLD" --report-target red_block)
if [[ -n "$GRIPPER_OPENING" ]]; then
  PLAY_ARGS+=(--gripper-opening "$GRIPPER_OPENING")
fi
if [[ "$HEADLESS" == true ]]; then
  PLAY_ARGS+=(--headless)
fi

echo "Red approach target (PiPER base frame): ${TARGET[*]} m"
echo "[1/2] Planning a collision-free trajectory with cuRobo..."
"$SCRIPT_DIR/plan.sh" \
  --world "$CONTAINER_WORLD" \
  --start 0 0 0 0 0 0 \
  --goal-position "${TARGET[@]}" \
  --goal-quaternion "${GOAL_QUATERNION[@]}" \
  --output "$CONTAINER_OUTPUT"

echo "[2/2] Playing the red-object approach in MuJoCo..."
exec "$SCRIPT_DIR/play_mujoco.sh" "$OUTPUT" --scene "$SCENE" "${PLAY_ARGS[@]}"

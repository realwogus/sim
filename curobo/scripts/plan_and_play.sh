#!/usr/bin/env bash
set -Eeuo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"

usage() {
  echo "Usage: bash scripts/plan_and_play.sh [X Y Z] [--hold SECONDS] [--gripper-opening METERS] [--headless]" >&2
  echo "       bash scripts/plan_and_play.sh --default [--hold SECONDS]" >&2
}

PLAN_ARGS=()
PLAY_ARGS=()

if [[ "$#" -ge 1 && "$1" == "--default" ]]; then
  shift
elif [[ "$#" -ge 3 && "$1" != --* && "$2" != --* && "$3" != --* ]]; then
  PLAN_ARGS=(--goal-position "$1" "$2" "$3")
  shift 3
elif [[ "$#" -gt 0 && "$1" != --hold && "$1" != --gripper-opening && "$1" != --headless ]]; then
  usage
  exit 2
fi

while [[ "$#" -gt 0 ]]; do
  case "$1" in
    --hold|--gripper-opening)
      if [[ "$#" -lt 2 ]]; then
        usage
        exit 2
      fi
      PLAY_ARGS+=("$1" "$2")
      shift 2
      ;;
    --headless)
      PLAY_ARGS+=("$1")
      shift
      ;;
    *)
      usage
      exit 2
      ;;
  esac
done

echo "[1/2] Planning with cuRobo..."
"$SCRIPT_DIR/plan.sh" "${PLAN_ARGS[@]}"

echo "[2/2] Playing the trajectory in MuJoCo..."
exec "$SCRIPT_DIR/play_mujoco.sh" "${PLAY_ARGS[@]}"

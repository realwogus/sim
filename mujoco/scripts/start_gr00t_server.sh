#!/usr/bin/env bash
set -Eeuo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd -- "$SCRIPT_DIR/.." && pwd)"
WORKSPACE_DIR="$(cd -- "$PROJECT_DIR/.." && pwd)"
GR00T_REPO="${GR00T_REPO:-$WORKSPACE_DIR/Isaac-GR00T}"
DEFAULT_CHECKPOINT="$WORKSPACE_DIR/gr00t_outputs/main_step2000/checkpoint-2000"
CHECKPOINT="${1:-${GR00T_CHECKPOINT:-$DEFAULT_CHECKPOINT}}"
IMAGE="${GR00T_IMAGE:-gr00t-thor:latest}"
PERSISTENT_HOME="${GR00T_PERSISTENT_HOME:-/home/airlab/.keys/home}"

if [[ "$#" -gt 1 ]]; then
  echo "Usage: bash scripts/start_gr00t_server.sh [CHECKPOINT_DIR]" >&2
  exit 2
fi

# Interpret paths such as gr00t_outputs/checkpoint-20000 relative to the shared
# workspace, independent of the directory from which this script is called.
if [[ "$CHECKPOINT" != /* && -d "$WORKSPACE_DIR/$CHECKPOINT" ]]; then
  CHECKPOINT="$WORKSPACE_DIR/$CHECKPOINT"
fi
if [[ -d "$CHECKPOINT" ]]; then
  CHECKPOINT="$(cd -- "$CHECKPOINT" && pwd)"
fi

if [[ ! -d "$GR00T_REPO/gr00t" ]]; then
  echo "ERROR: Isaac-GR00T repository not found: $GR00T_REPO" >&2
  exit 1
fi
if [[ ! -f "$CHECKPOINT/model.safetensors.index.json" ]]; then
  echo "ERROR: GR00T checkpoint not found: $CHECKPOINT" >&2
  exit 1
fi
if [[ ! -d "$PERSISTENT_HOME" ]]; then
  echo "ERROR: persistent Hugging Face home not found: $PERSISTENT_HOME" >&2
  exit 1
fi

echo "GR00T checkpoint: $CHECKPOINT"
echo "Policy endpoint:  tcp://127.0.0.1:5555"

exec docker run --rm \
  --name piper-gr00t-server \
  --runtime nvidia \
  --gpus all \
  --ipc host \
  --network host \
  -e PYTHONPATH=/workspace/Isaac-GR00T \
  -e CUDA_VISIBLE_DEVICES=0 \
  -v "$PERSISTENT_HOME:/root" \
  -v "$GR00T_REPO:/workspace/Isaac-GR00T:ro" \
  -v "$CHECKPOINT:/workspace/checkpoint:ro" \
  -w /workspace/Isaac-GR00T \
  --entrypoint bash \
  "$IMAGE" -lc '
    source /opt/gr00t-venv/bin/activate
    source scripts/activate_thor.sh
    exec python -m gr00t.eval.run_gr00t_server \
      --model-path /workspace/checkpoint \
      --embodiment-tag new_embodiment \
      --device cuda \
      --host 0.0.0.0 \
      --port 5555
  '

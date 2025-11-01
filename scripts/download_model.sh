#!/usr/bin/env bash
set -euo pipefail

# Simple helper to download a small Qwen model from Hugging Face.
#
# Usage:
#   scripts/download_model.sh <model-id> <target-dir>
#
# Examples:
#   scripts/download_model.sh qwen3-0.6b models/qwen3_0_6b
#   scripts/download_model.sh qwen2.5-0.5b-instruct models/qwen2_5_0_5b
#
# Notes:
# - Public models work without auth. For gated/private models, export HF_TOKEN first.
# - Requires either `hf` CLI (huggingface_hub) or Python with huggingface_hub module.

MODEL_KEY="${1:-}"
TARGET_DIR="${2:-}"

if [[ -z "$MODEL_KEY" || -z "$TARGET_DIR" ]]; then
  echo "Usage: $0 <model-id> <target-dir>" >&2
  exit 2
fi

case "${MODEL_KEY,,}" in
  qwen3-0.6b)
    HF_ID="Qwen/Qwen3-0.6B" ;;
  qwen2.5-0.5b-instruct|qwen2_5-0_5b-instruct)
    HF_ID="Qwen/Qwen2.5-0.5B-Instruct" ;;
  qwen2-0.5b-instruct)
    HF_ID="Qwen/Qwen2-0.5B-Instruct" ;;
  *)
    # Assume a full Hugging Face repo id was passed
    HF_ID="$MODEL_KEY" ;;
esac

mkdir -p "$TARGET_DIR"

have_hf_cli=0
if command -v hf >/dev/null 2>&1; then
  have_hf_cli=1
elif [[ -x ".venv/bin/hf" ]]; then
  have_hf_cli=2
fi

echo "Downloading $HF_ID to $TARGET_DIR ..."

if [[ $have_hf_cli -ne 0 ]]; then
  HF_BIN="hf"
  [[ $have_hf_cli -eq 2 ]] && HF_BIN=".venv/bin/hf"
  $HF_BIN download "$HF_ID" \
    --include "*.json" --include "*.safetensors" \
    --local-dir "$TARGET_DIR" \
    --max-workers 8
  echo "Done. Files in $TARGET_DIR"
  exit 0
fi

# Fallback: use Python huggingface_hub
if ! command -v python3 >/dev/null 2>&1; then
  echo "Python3 not found. Please install huggingface_hub or the hf CLI." >&2
  exit 1
fi

PY_SCRIPT=""'
import os
import sys
from huggingface_hub import snapshot_download

if len(sys.argv) < 3:
    print("usage: script.py <repo_id> <target_dir>")
    sys.exit(2)
repo_id = sys.argv[1]
target_dir = sys.argv[2]

snapshot_download(
    repo_id=repo_id,
    allow_patterns=["*.json", "*.safetensors"],
    local_dir=target_dir,
)
print("Downloaded to", target_dir)
""'

echo "Installing huggingface_hub into local venv ..."
python3 -m venv .venv >/dev/null 2>&1 || true
. .venv/bin/activate
python -m pip -q install --upgrade pip huggingface_hub
python - <<PY
$PY_SCRIPT
PY

echo "Done. Files in $TARGET_DIR"


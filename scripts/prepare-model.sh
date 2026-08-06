#!/usr/bin/env bash
# Download Qwen2.5-Coder-7B-Instruct (safetensors) into the coordinator model dir.
# No Fly / cloud deploy — local disk only.
set -euo pipefail

MODEL_ID="${HYVERK_HF_MODEL:-Qwen/Qwen2.5-Coder-7B-Instruct}"
DEST="${HYVERK_MODEL_DIR:-$HOME/.hyverk/model}"

mkdir -p "$DEST"
export HYVERK_HF_MODEL="$MODEL_ID"
export HYVERK_MODEL_DIR="$DEST"

if ! python3 -c "import huggingface_hub" 2>/dev/null; then
  echo "Installing huggingface_hub into the current Python..."
  python3 -m pip install -q "huggingface_hub"
fi

echo "Downloading $MODEL_ID → $DEST"
python3 - <<'PY'
from huggingface_hub import snapshot_download
import os

snapshot_download(
    repo_id=os.environ["HYVERK_HF_MODEL"],
    local_dir=os.environ["HYVERK_MODEL_DIR"],
    allow_patterns=[
        "config.json",
        "tokenizer.json",
        "tokenizer_config.json",
        "special_tokens_map.json",
        "generation_config.json",
        "*.safetensors",
        "model.safetensors.index.json",
    ],
)
print("ok")
PY

for f in config.json tokenizer.json model.safetensors.index.json; do
  if [[ ! -f "$DEST/$f" ]]; then
    echo "ERROR: missing $DEST/$f after download" >&2
    exit 1
  fi
done

echo
echo "Model ready at: $DEST"
echo "Start coordinator with:"
echo "  export HYVERK_MODEL_DIR=\"$DEST\""
echo "  cargo run -p hyverk-coordinator --release"

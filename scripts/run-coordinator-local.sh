#!/usr/bin/env bash
# Run the Hyverk coordinator against a local model directory (no Fly).
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

export HYVERK_MODEL_DIR="${HYVERK_MODEL_DIR:-$HOME/.hyverk/model}"
export HYVERK_DATA_DIR="${HYVERK_DATA_DIR:-$HOME/.hyverk}"
# Optional: export HYVERK_API_KEY=... to require Bearer auth on inference endpoints.

if [[ ! -f "$HYVERK_MODEL_DIR/config.json" ]]; then
  echo "No model at $HYVERK_MODEL_DIR"
  echo "Run: bash scripts/prepare-model.sh"
  exit 1
fi

if [[ ! -f config.toml ]]; then
  cat > config.toml <<'EOF'
mode = "coordinator"

[node]
name = "local-coordinator"
coordinator_url = "http://127.0.0.1:17000"
models_dir = "~/.hyverk/models"
max_concurrent_tasks = 1
poll_interval_ms = 1000
hardware_info = ""

[coordinator]
grpc_port = 17001
http_port = 17000
bind_addr = "0.0.0.0"
heartbeat_timeout_secs = 30
EOF
  echo "Wrote ./config.toml for local coordinator"
fi

echo "Model:  $HYVERK_MODEL_DIR"
echo "Data:   $HYVERK_DATA_DIR"
echo "HTTP:   http://0.0.0.0:17000"
echo "Health: curl -s http://127.0.0.1:17000/health"

export HYVERK_CONFIG=./config.toml
exec cargo run -p hyverk-coordinator --release

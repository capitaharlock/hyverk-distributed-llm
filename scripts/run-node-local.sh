#!/usr/bin/env bash
# Start a hyverk-node pointing at a local (or LAN) coordinator.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

COORD_URL="${1:-${HYVERK_COORDINATOR:-http://127.0.0.1:17000}}"
HW="${HYVERK_HARDWARE_INFO:-Apple Silicon, Metal GPU}"

mkdir -p "$HOME/.hyverk"
cat > "$HOME/.hyverk/config.node.toml" <<EOF
mode = "node"

[node]
name = "ignored-stable-id-used"
coordinator_url = "$COORD_URL"
models_dir = "~/.hyverk/models"
max_concurrent_tasks = 2
poll_interval_ms = 1000
hardware_info = "$HW"

[coordinator]
grpc_port = 17001
http_port = 17000
bind_addr = "127.0.0.1"
heartbeat_timeout_secs = 30
EOF

echo "Coordinator: $COORD_URL"
echo "hardware_info: $HW"
echo "Identity: ~/.hyverk/node_id (auto)"
echo
echo "On each Mac, use a distinct machine (stable node_id is per-host)."
echo "Wait until: curl -s $COORD_URL/api/v1/cluster/status | jq .status"
echo "Then probe:  curl -s -X POST $COORD_URL/api/v1/ws-inference -H 'Content-Type: application/json' -d '{\"prompt\":\"hi\",\"max_tokens\":32}'"

export HYVERK_CONFIG="$HOME/.hyverk/config.node.toml"
exec cargo run -p hyverk-node --release

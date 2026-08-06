#!/usr/bin/env bash
# Lightweight local smoke test — no HF download, no GPU nodes.
# Stands up the coordinator against a tiny fake model tree, checks health,
# model config, optional API-key gate, then shuts down.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
source "$HOME/.cargo/env" 2>/dev/null || true

TMP="${TMPDIR:-/tmp}/hyverk-smoke-$$"
MODEL="$TMP/model"
DATA="$TMP/data"
CFG="$TMP/config.toml"
mkdir -p "$MODEL" "$DATA"

# Minimal Qwen2-shaped metadata so coordinator_model_available() is true.
cat > "$MODEL/config.json" <<'EOF'
{
  "model_type": "qwen2",
  "num_hidden_layers": 28,
  "hidden_size": 3584,
  "vocab_size": 151936
}
EOF

cat > "$MODEL/model.safetensors.index.json" <<'EOF'
{
  "metadata": {"total_size": 1},
  "weight_map": {
    "model.embed_tokens.weight": "model-00001-of-00001.safetensors",
    "model.layers.0.self_attn.q_proj.weight": "model-00001-of-00001.safetensors",
    "model.norm.weight": "model-00001-of-00001.safetensors",
    "lm_head.weight": "model-00001-of-00001.safetensors"
  }
}
EOF

# Tiny placeholder shard + tokenizer (not used for real forward).
printf 'fake' > "$MODEL/model-00001-of-00001.safetensors"
printf '{"version":"1.0","truncation":null,"padding":null,"added_tokens":[],"normalizer":null,"pre_tokenizer":null,"post_processor":null,"decoder":null,"model":{"type":"BPE","dropout":null,"unk_token":null,"continuing_subword_prefix":null,"end_of_word_suffix":null,"fuse_unk":false,"byte_fallback":false,"vocab":{},"merges":[]}}' > "$MODEL/tokenizer.json"

cat > "$CFG" <<EOF
mode = "coordinator"

[node]
name = "smoke"
coordinator_url = "http://127.0.0.1:17000"
models_dir = "$TMP/models"
max_concurrent_tasks = 1
poll_interval_ms = 1000
hardware_info = ""

[coordinator]
grpc_port = 17001
http_port = 17000
bind_addr = "127.0.0.1"
heartbeat_timeout_secs = 30
EOF

export HYVERK_MODEL_DIR="$MODEL"
export HYVERK_DATA_DIR="$DATA"
export HYVERK_CONFIG="$CFG"
export HYVERK_API_KEY="smoke-test-key"

# Free ports if a previous smoke left something behind
for p in 17000 17001; do
  if lsof -tiTCP:$p -sTCP:LISTEN >/dev/null 2>&1; then
    echo "Port $p busy — aborting smoke (won't kill foreign processes)"
    exit 1
  fi
done

echo "== building coordinator =="
cargo build -p hyverk-coordinator --quiet

echo "== starting coordinator =="
./target/debug/hyverk-coordinator &
PID=$!
cleanup() {
  kill "$PID" 2>/dev/null || true
  wait "$PID" 2>/dev/null || true
  rm -rf "$TMP"
}
trap cleanup EXIT

for i in $(seq 1 40); do
  if curl -sf http://127.0.0.1:17000/health >/dev/null; then
    break
  fi
  sleep 0.25
done

echo "== /health =="
curl -sf http://127.0.0.1:17000/health
echo

echo "== /api/v1/model/config (expect available:true) =="
CFG_JSON=$(curl -sf http://127.0.0.1:17000/api/v1/model/config)
echo "$CFG_JSON" | python3 -c 'import sys,json; d=json.load(sys.stdin); assert d.get("available") is True, d; print("available=true ok")'

echo "== /api/v1/cluster/status (expect no_nodes/forming) =="
curl -sf http://127.0.0.1:17000/api/v1/cluster/status | python3 -c 'import sys,json; d=json.load(sys.stdin); print("status=", d.get("status"))'

echo "== ws-inference without key (expect 401) =="
CODE=$(curl -s -o /dev/null -w '%{http_code}' -X POST http://127.0.0.1:17000/api/v1/ws-inference \
  -H 'Content-Type: application/json' -d '{"prompt":"hi","max_tokens":8}')
test "$CODE" = "401" && echo "401 ok" || { echo "expected 401 got $CODE"; exit 1; }

echo "== ws-inference with key, no nodes (expect JSON error, not 401) =="
RESP=$(curl -sf -X POST http://127.0.0.1:17000/api/v1/ws-inference \
  -H 'Content-Type: application/json' \
  -H "Authorization: Bearer $HYVERK_API_KEY" \
  -d '{"prompt":"hi","max_tokens":8}')
echo "$RESP" | python3 -c 'import sys,json; d=json.load(sys.stdin); assert "error" in d, d; print("error=", d["error"][:80])'

echo
echo "SMOKE OK"

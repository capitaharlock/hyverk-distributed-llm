#!/usr/bin/env bash
# Join MeshKore via cluster invite and write .meshkore.local (gitignored).
# Channel/hub come from .meshkore/public/cluster.yaml.
# Invite URL is NOT in the public repo:
#   MESHKORE_INVITE='https://hub.../agents/invites/<nonce>/join' bash scripts/meshkore-join.sh
# or put "invite" in gitignored .meshkore.local before joining.
#
# Corporate networks often block hub.meshkore.com — use the relay for the join POST:
#   MESHKORE_HUB_URL=https://meshkore-relay.fly.dev bash scripts/meshkore-join.sh
#
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

_HOST="$(hostname -s 2>/dev/null | tr '[:upper:]' '[:lower:]' | tr -cd 'a-z0-9-' | cut -c1-20)"
AGENT_ID="${MESHKORE_AGENT_ID:-hyverk-contributor-${_HOST:-dev}}"
HUB_URL="${MESHKORE_HUB_URL:-https://meshkore-relay.fly.dev}"
HUB_URL="${HUB_URL%/}"

OUT="$(mktemp)"
trap 'rm -f "$OUT"' EXIT

eval "$(python3 - <<'PY'
import json, os, re, sys
from pathlib import Path

try:
    import yaml  # type: ignore
except ImportError:
    yaml = None

def load_cluster():
    p = Path(".meshkore/public/cluster.yaml")
    if not p.is_file():
        sys.stderr.write("ERROR: missing .meshkore/public/cluster.yaml\n")
        sys.exit(1)
    text = p.read_text()
    if yaml is not None:
        return yaml.safe_load(text)
    # Minimal fallback parser for the fields we need (no PyYAML required).
    data = {"legacy_hub": {}, "bootstrap": {}}
    cur = None
    for line in text.splitlines():
        if not line.strip() or line.lstrip().startswith("#"):
            continue
        if re.match(r"^[a-z_]+:\s*$", line):
            cur = line.split(":", 1)[0].strip()
            data.setdefault(cur, {})
            continue
        if cur in ("legacy_hub", "bootstrap") and ":" in line and line.startswith("  "):
            k, v = line.split(":", 1)
            data[cur][k.strip()] = v.strip().strip('"').strip("'")
    return data

m = load_cluster()
local = {}
lp = Path(".meshkore.local")
if lp.is_file():
    try:
        local = json.loads(lp.read_text())
    except json.JSONDecodeError:
        local = {}

invite = (os.environ.get("MESHKORE_INVITE") or "").strip()
if not invite:
    invite = (local.get("invite") or "").strip()

mo = re.search(r"/invites/([0-9a-f]+)/join", invite)
if not mo:
    sys.stderr.write(
        "ERROR: no invite URL. Set MESHKORE_INVITE or put \"invite\" in .meshkore.local\n"
        "  (invite URLs are not stored in the public repo)\n"
    )
    sys.exit(1)
nonce = mo.group(1)
legacy = m.get("legacy_hub") or {}
bootstrap = m.get("bootstrap") or {}
canonical = (bootstrap.get("hub") or "https://hub.meshkore.com").rstrip("/")
channel = legacy.get("channel_id") or ""

def sq(s: str) -> str:
    return "'" + s.replace("'", "'\"'\"'") + "'"
print(f"export INVITE_NONCE={sq(nonce)}")
print(f"export CANONICAL_HUB={sq(canonical)}")
print(f"export CHANNEL_ID={sq(channel)}")
PY
)"


echo "join_hub=$HUB_URL (POST)"
echo "canonical_hub=$CANONICAL_HUB"
echo "channel_id=$CHANNEL_ID"
echo "agent_id=$AGENT_ID"
echo "invite_nonce=$INVITE_NONCE"

curl -sS -m 60 \
  -H "Content-Type: application/json" \
  -H "Accept: application/json" \
  -X POST "${HUB_URL}/agents/invites/${INVITE_NONCE}/join" \
  -d "{\"agent_id\":\"${AGENT_ID}\",\"capabilities\":[\"coding\",\"debugging\",\"rust\",\"code-review\"]}" \
  -o "$OUT"

if grep -qi '<!doctype html' "$OUT"; then
  echo "ERROR: Got HTML instead of JSON (firewall / wrong host). Try MESHKORE_HUB_URL=https://meshkore-relay.fly.dev" >&2
  exit 1
fi

python3 - "$OUT" "$ROOT" "$CHANNEL_ID" "$HUB_URL" "$CANONICAL_HUB" "$AGENT_ID" <<'PY'
import json, sys
from pathlib import Path

out_path, root, channel_default, hub_url, canonical, agent_id_cli = sys.argv[1:7]
root = Path(root)
raw = Path(out_path).read_text()
try:
    data = json.loads(raw)
except json.JSONDecodeError as e:
    print("ERROR: not JSON:", e, file=sys.stderr)
    print(raw[:800], file=sys.stderr)
    sys.exit(1)

if isinstance(data, dict) and data.get("status") == "pending":
    print("Invite pending approval; ask cluster owner to approve.", data, file=sys.stderr)
    sys.exit(2)

token = data.get("token") or data.get("access_token") or data.get("jwt")
api_key = data.get("api_key") or data.get("apiKey")
agent_id = data.get("agent_id") or data.get("agentId") or agent_id_cli
channel_id = data.get("channel_id") or data.get("channelId") or channel_default

if not token or not api_key or not agent_id:
    print("ERROR: response missing token/api_key/agent_id:", data, file=sys.stderr)
    sys.exit(1)

local_path = root / ".meshkore.local"
blob = {}
if local_path.is_file():
    try:
        blob = json.loads(local_path.read_text())
    except json.JSONDecodeError:
        blob = {}
blob.update({
    "meshkore_version": 1,
    "hub_url": hub_url,
    "canonical_hub_url": canonical,
    "channel_id": channel_id,
    "agent_id": agent_id,
    "api_key": api_key,
    "token": token,
})
local_path.write_text(json.dumps(blob, indent=2) + "\n")
local_path.chmod(0o600)
print("Wrote", local_path)
PY

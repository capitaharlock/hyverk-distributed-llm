#!/usr/bin/env bash
# Join MeshKore (dev mesh) via cluster invite and write credentials (gitignored).
# Reads **`.mechcore` first**, else **`.meshkore`** — same JSON schema (`meshkore_version` 1).
# Writes **`.mechcore.local`** if spec is `.mechcore`, else **`.meshkore.local`**.
#
# Corporate networks often block hub.meshkore.com — use the relay for the join POST:
#   MESHKORE_HUB_URL=https://meshkore-relay.fly.dev bash scripts/meshkore-join.sh
#
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

if [[ -f "$ROOT/.mechcore" ]]; then CLUSTER_SPEC="$ROOT/.mechcore"
elif [[ -f "$ROOT/.meshkore" ]]; then CLUSTER_SPEC="$ROOT/.meshkore"
else
  echo "ERROR: missing $ROOT/.mechcore or $ROOT/.meshkore (cluster invite + channel_id)" >&2
  exit 1
fi
if [[ "$(basename "$CLUSTER_SPEC")" == ".mechcore" ]]; then CRED_OUT="$ROOT/.mechcore.local"
else CRED_OUT="$ROOT/.meshkore.local"
fi

AGENT_ID="${MESHKORE_AGENT_ID:-hyverk-contrib-$(openssl rand -hex 4)}"
HUB_URL="${MESHKORE_HUB_URL:-https://meshkore-relay.fly.dev}"
HUB_URL="${HUB_URL%/}"

OUT="$(mktemp)"
trap 'rm -f "$OUT"' EXIT

export CLUSTER_SPEC
eval "$(python3 <<'PY'
import json, re, sys, os
from pathlib import Path
spec = Path(os.environ["CLUSTER_SPEC"])
m = json.loads(spec.read_text())
invite = (m.get("cluster") or {}).get("invite") or ""
mo = re.search(r"/invites/([0-9a-f]+)/join", invite)
if not mo:
    sys.stderr.write(f"ERROR: cluster.invite in {spec.name} has no /invites/<nonce>/join\n")
    sys.exit(1)
nonce = mo.group(1)
canonical = (m.get("hub") or {}).get("url", "https://hub.meshkore.com").rstrip("/")
channel = (m.get("cluster") or {}).get("channel_id") or ""
# shell-safe single-quoted strings
def sq(s: str) -> str:
    return "'" + s.replace("'", "'\"'\"'") + "'"
print(f"export INVITE_NONCE={sq(nonce)}")
print(f"export CANONICAL_HUB={sq(canonical)}")
print(f"export CHANNEL_ID={sq(channel)}")
PY
)"


echo "cluster_spec=$(basename "$CLUSTER_SPEC")"
echo "credentials_out=$(basename "$CRED_OUT")"
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

export CRED_OUT
python3 - "$OUT" "$ROOT" "$CHANNEL_ID" "$HUB_URL" "$CANONICAL_HUB" "$AGENT_ID" <<'PY'
import json, os, sys
from pathlib import Path

out_path, root, channel_default, hub_url, canonical, agent_id_cli = sys.argv[1:7]
root = Path(root)
cred_out = Path(os.environ["CRED_OUT"])
spec = Path(os.environ["CLUSTER_SPEC"])
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

meshv = 1
try:
    meshv = json.loads(spec.read_text()).get("meshkore_version", 1)
except OSError:
    pass

blob = {
    "meshkore_version": meshv,
    "hub_url": hub_url,
    "canonical_hub_url": canonical,
    "channel_id": channel_id,
    "agent_id": agent_id,
    "api_key": api_key,
    "token": token,
}
cred_out.write_text(json.dumps(blob, indent=2) + "\n")
cred_out.chmod(0o600)
print("Wrote", cred_out)
PY

"""Read .meshkore.local — handles both flat and nested formats. Auto-refreshes expired token."""
from __future__ import annotations
import json, ssl, sys, urllib.error, urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
LOCAL = ROOT / ".meshkore.local"


def _http(url: str, *, method: str = "GET", data: bytes | None = None,
          headers: dict | None = None) -> dict:
    try:
        import certifi
        ctx = ssl.create_default_context(cafile=certifi.where())
    except ImportError:
        ctx = ssl.create_default_context()
    req = urllib.request.Request(url, data=data, method=method,
                                 headers=headers or {})
    with urllib.request.urlopen(req, context=ctx, timeout=20) as r:
        return json.loads(r.read())


def load(auto_refresh: bool = True) -> dict:
    """Return normalised flat dict: hub_url, token, api_key, agent_id, channel_id."""
    if not LOCAL.exists():
        print("Missing .meshkore.local — run scripts/meshkore-join.sh", file=sys.stderr)
        sys.exit(1)
    raw = json.loads(LOCAL.read_text())

    # Nested format: {hub, identity:{agent_id,api_key,token}, cluster:{channel_id}}
    if "identity" in raw:
        flat = {
            "hub_url":    str(raw.get("hub") or "").rstrip("/"),
            "token":      raw["identity"].get("token", ""),
            "api_key":    raw["identity"].get("api_key", ""),
            "agent_id":   raw["identity"].get("agent_id", "unknown"),
            "channel_id": raw.get("cluster", {}).get("channel_id", ""),
        }
    else:
        # Already flat
        flat = {
            "hub_url":    str(raw.get("hub_url") or "").rstrip("/"),
            "token":      raw.get("token", ""),
            "api_key":    raw.get("api_key", ""),
            "agent_id":   raw.get("agent_id", "unknown"),
            "channel_id": raw.get("channel_id", ""),
        }

    if auto_refresh and flat["api_key"]:
        flat["token"] = _ensure_valid_token(flat)

    return flat


def _ensure_valid_token(flat: dict) -> str:
    """Verify token works; refresh via api_key if it returns 401."""
    hub, token, api_key, agent_id = (
        flat["hub_url"], flat["token"], flat["api_key"], flat["agent_id"])
    try:
        _http(f"{hub}/agents/messages",
              headers={"Authorization": f"Bearer {token}"})
        return token  # still valid
    except urllib.error.HTTPError as e:
        if e.code != 401:
            raise
    # Refresh
    try:
        resp = _http(f"{hub}/agents/token", method="POST",
                     data=json.dumps({"api_key": api_key, "agent_id": agent_id}).encode(),
                     headers={"Content-Type": "application/json"})
        new_token = resp["token"]
    except Exception as exc:
        print(f"Token refresh failed: {exc}", file=sys.stderr)
        return token  # return old, caller will get 401

    # Persist new token
    raw = json.loads(LOCAL.read_text())
    if "identity" in raw:
        raw["identity"]["token"] = new_token
    else:
        raw["token"] = new_token
    LOCAL.write_text(json.dumps(raw, indent=2))
    return new_token

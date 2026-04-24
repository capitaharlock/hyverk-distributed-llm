#!/usr/bin/env python3
"""POST one broadcast to the cluster channel (MeshKore). Reads .meshkore.local."""
from __future__ import annotations

import argparse
import json
import ssl
import sys
import urllib.error
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
LOCAL = ROOT / ".meshkore.local"


def main() -> int:
    p = argparse.ArgumentParser(description="Send payload to MeshKore channel")
    p.add_argument(
        "--stdin",
        action="store_true",
        help="Read JSON payload object from stdin (merged under payload)",
    )
    p.add_argument(
        "extra_json",
        nargs="?",
        default="",
        help='Optional JSON object merged into payload, e.g. \'{"type":"task.start","task_id":"x"}\'',
    )
    args = p.parse_args()

    if not LOCAL.exists():
        print("Missing .meshkore.local — run scripts/meshkore-join.sh", file=sys.stderr)
        return 1

    raw = json.loads(LOCAL.read_text())
    hub = str(raw.get("hub_url") or "").rstrip("/")
    token = raw.get("token") or ""
    channel_id = raw.get("channel_id") or ""
    agent_id = raw.get("agent_id") or "unknown"
    if not hub or not token or not channel_id:
        print(".meshkore.local missing hub_url, token, or channel_id", file=sys.stderr)
        return 1

    body: dict = {"payload": {"from": agent_id}}
    if args.stdin:
        inner = json.load(sys.stdin)
        if not isinstance(inner, dict):
            print("stdin JSON must be an object", file=sys.stderr)
            return 1
        body["payload"].update(inner)
    if args.extra_json.strip():
        body["payload"].update(json.loads(args.extra_json))

    try:
        import certifi

        ctx = ssl.create_default_context(cafile=certifi.where())
    except ImportError:
        ctx = ssl.create_default_context()

    data = json.dumps(body).encode()
    url = f"{hub}/agents/channels/{channel_id}/send"
    req = urllib.request.Request(
        url,
        data=data,
        method="POST",
        headers={
            "Authorization": f"Bearer {token}",
            "Content-Type": "application/json",
            "Accept": "application/json",
        },
    )
    try:
        with urllib.request.urlopen(req, context=ctx, timeout=45) as resp:
            out = resp.read().decode()
    except urllib.error.HTTPError as e:
        print(e.read().decode()[:2000], file=sys.stderr)
        return 1
    except urllib.error.URLError as e:
        print(e, file=sys.stderr)
        return 1
    print(out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

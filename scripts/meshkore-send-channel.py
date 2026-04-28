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

sys.path.insert(0, str(Path(__file__).parent))
from _meshkore_local import load


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

    c = load(auto_refresh=True)
    hub = c["hub_url"]
    token = c["token"]
    channel_id = c["channel_id"]
    agent_id = c["agent_id"]
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

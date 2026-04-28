#!/usr/bin/env python3
"""Print MeshKore inbox (DMs) for hyverk-lead. Auto-refreshes token if expired."""
from __future__ import annotations
import json, ssl, sys, urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from _meshkore_local import load, _http


def main() -> None:
    c = load(auto_refresh=True)
    hub, token = c["hub_url"], c["token"]

    try:
        import certifi
        ctx = ssl.create_default_context(cafile=certifi.where())
    except ImportError:
        ctx = ssl.create_default_context()

    req = urllib.request.Request(
        f"{hub}/agents/messages",
        headers={"Authorization": f"Bearer {token}"},
    )
    with urllib.request.urlopen(req, context=ctx, timeout=20) as r:
        data = json.loads(r.read())

    msgs = data.get("messages", [])
    if not msgs:
        print("(inbox empty)")
        return
    for m in msgs:
        print(json.dumps(m, indent=2))


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Print latest MeshKore inbox JSON (primary + optional teammate) for hyverk-cluster agents."""
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
LOCAL = ROOT / ".meshkore.local"


def fetch(hub: str, token: str) -> dict:
    r = subprocess.run(
        [
            "curl",
            "-sS",
            "-m",
            "25",
            "-H",
            f"Authorization: Bearer {token}",
            f"{hub.rstrip('/')}/agents/messages",
        ],
        capture_output=True,
        text=True,
    )
    if r.returncode != 0:
        return {"error": r.stderr}
    return json.loads(r.stdout)


def main() -> None:
    if not LOCAL.exists():
        print("Missing .meshkore.local", file=sys.stderr)
        sys.exit(1)
    c = json.loads(LOCAL.read_text())
    hub = c["hub_url"]
    for label, tok in (
        ("primary", c["token"]),
        ("teammate", (c.get("teammate") or {}).get("token")),
    ):
        if not tok:
            continue
        print(f"--- {label} ({c.get('agent_id', '?')}) ---")
        data = fetch(hub, tok)
        print(json.dumps(data, indent=2))
        msgs = data.get("messages") if isinstance(data, dict) else None
        if isinstance(msgs, list) and len(msgs) == 0 and not data.get("error"):
            print(
                "(empty) Hub returned no queued messages — they may have been consumed by a "
                "previous poll, or leader DMs went to another agent_id. "
                "Compare your agent_id above with whom hyverk-lead addresses; "
                "run meshkore-listener.py so nothing is missed between polls.",
                file=sys.stderr,
            )


if __name__ == "__main__":
    main()

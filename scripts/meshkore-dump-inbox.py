#!/usr/bin/env python3
"""Print latest MeshKore inbox JSON (primary + optional teammate) for hyverk-cluster agents."""
import json
import subprocess
import sys
from pathlib import Path

_SCRIPT_DIR = Path(__file__).resolve().parent
if str(_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPT_DIR))
from meshkore_resolve import credentials_path

ROOT = Path(__file__).resolve().parent.parent


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
    local = credentials_path(ROOT)
    if local is None or not local.exists():
        print("Missing .mechcore.local or .meshkore.local — run scripts/meshkore-join.sh", file=sys.stderr)
        sys.exit(1)
    c = json.loads(local.read_text())
    hub = c["hub_url"]
    for label, tok in (
        ("primary", c["token"]),
        ("teammate", (c.get("teammate") or {}).get("token")),
    ):
        if not tok:
            continue
        print(f"--- {label} ({c.get('agent_id', '?')}) ---")
        print(json.dumps(fetch(hub, tok), indent=2))


if __name__ == "__main__":
    main()

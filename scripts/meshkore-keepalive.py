#!/usr/bin/env python3
"""Poll MeshKore messages so agents stay in poll/online mode.

Prefer **meshkore-listener.py** for production: it also persists inbox lines to
``.meshkore-incoming.jsonl`` for Cursor hooks. This script is a minimal keep-alive only.
"""
import json
import subprocess
import sys
import time
from pathlib import Path

_SCRIPT_DIR = Path(__file__).resolve().parent
if str(_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPT_DIR))
from meshkore_resolve import credentials_path

ROOT = Path(__file__).resolve().parent.parent
INTERVAL = 25


def main():
    local = credentials_path(ROOT)
    if local is None or not local.exists():
        raise SystemExit("Missing .mechcore.local or .meshkore.local — run scripts/meshkore-join.sh")
    c = json.loads(local.read_text())
    hub = c["hub_url"].rstrip("/")
    tokens = [c["token"]]
    tm = c.get("teammate")
    if isinstance(tm, dict) and tm.get("token"):
        tokens.append(tm["token"])
    while True:
        for t in tokens:
            subprocess.run(
                [
                    "curl",
                    "-sS",
                    "-m",
                    "20",
                    "-H",
                    f"Authorization: Bearer {t}",
                    f"{hub}/agents/messages",
                ],
                capture_output=True,
            )
        time.sleep(INTERVAL)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
afterFileEdit: broadcast a compact task.progress line to the MeshKore cluster channel
so the other agent sees edits without the human relaying them.

Fails open (prints {}) on any error or non-code paths.
"""
from __future__ import annotations

import json
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent


def _credentials_path(r: Path) -> Path | None:
    a, b = r / ".mechcore.local", r / ".meshkore.local"
    if a.exists():
        return a
    if b.exists():
        return b
    return None


def _cluster_spec_path(r: Path) -> Path | None:
    a, b = r / ".mechcore", r / ".meshkore"
    if a.exists():
        return a
    if b.exists():
        return b
    return None


LOCAL = _credentials_path(ROOT)
MESH = _cluster_spec_path(ROOT)
DEBOUNCE_SEC = 20.0
DEBOUNCE = ROOT / ".meshkore-broadcast-debounce.json"

SKIP_SUBSTR = (
    "/.git/",
    "/target/",
    "node_modules",
    ".meshkore.local",
    ".mechcore.local",
    ".meshkore-incoming",
    ".meshkore-listener",
    ".meshkore-stop",
    ".meshkore-broadcast",
    "/.cursor/hooks/",
)

ALLOW_EXT = frozenset(
    {
        ".rs",
        ".py",
        ".toml",
        ".proto",
        ".md",
        ".yaml",
        ".yml",
        ".json",
        ".html",
        ".js",
        ".ts",
        ".tsx",
    }
)


def _debounce_ok(rel: str) -> bool:
    now = time.time()
    data: dict[str, float] = {}
    if DEBOUNCE.exists():
        try:
            data = json.loads(DEBOUNCE.read_text())
        except json.JSONDecodeError:
            data = {}
    last = data.get(rel, 0.0)
    if now - last < DEBOUNCE_SEC:
        return False
    data[rel] = now
    # prune old keys
    cutoff = now - 3600
    data = {k: v for k, v in data.items() if v > cutoff}
    DEBOUNCE.write_text(json.dumps(data) + "\n")
    DEBOUNCE.chmod(0o600)
    return True


def main() -> None:
    try:
        hook = json.load(sys.stdin)
    except json.JSONDecodeError:
        print("{}")
        return

    fp = hook.get("file_path") or ""
    if not fp:
        print("{}")
        return

    p = Path(fp)
    try:
        rel = str(p.resolve().relative_to(ROOT.resolve()))
    except ValueError:
        print("{}")
        return

    if any(s in fp for s in SKIP_SUBSTR):
        print("{}")
        return

    if p.suffix.lower() not in ALLOW_EXT:
        print("{}")
        return

    if not _debounce_ok(rel):
        print("{}")
        return

    if LOCAL is None or MESH is None or not LOCAL.exists() or not MESH.exists():
        print("{}")
        return

    try:
        cred = json.loads(LOCAL.read_text())
        mesh = json.loads(MESH.read_text())
        hub = str(cred.get("hub_url", "")).rstrip("/")
        tok = cred["token"]
        cid = mesh["cluster"]["channel_id"]
        aid = cred.get("agent_id", "unknown-agent")
    except (KeyError, json.JSONDecodeError, OSError):
        print("{}")
        return

    body = {
        "payload": {
            "type": "task.progress",
            "from": aid,
            "text": f"Cursor file edit: {rel}",
            "refs": rel,
        }
    }
    try:
        subprocess.run(
            [
                "curl",
                "-sS",
                "-m",
                "15",
                "-X",
                "POST",
                "-H",
                f"Authorization: Bearer {tok}",
                "-H",
                "Content-Type: application/json",
                "-d",
                json.dumps(body),
                f"{hub}/agents/channels/{cid}/send",
            ],
            capture_output=True,
            text=True,
            check=False,
        )
    except OSError:
        pass
    print("{}")


if __name__ == "__main__":
    main()

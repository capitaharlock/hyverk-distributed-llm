#!/usr/bin/env python3
"""sessionStart: inject recent MeshKore traffic, prioritizing leader + cluster channel."""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
MESHKORE = ROOT / ".meshkore"
INCOMING = ROOT / ".meshkore-incoming.jsonl"
MAX_LINES = 35
MAX_CHARS = 12000
SCAN = 500


def _cluster_channel_id() -> str:
    if not MESHKORE.exists():
        return ""
    try:
        return (json.loads(MESHKORE.read_text()).get("cluster") or {}).get("channel_id") or ""
    except (OSError, json.JSONDecodeError):
        return ""


def _is_priority_line(line: str, channel_id: str) -> bool:
    if '"hyverk-lead"' in line:
        return True
    if channel_id and channel_id in line:
        return True
    if '"channel_message"' in line and channel_id:
        return True
    pl = '"type":"task.'
    if pl in line or '"type":"plan.' in line or '"type":"review.' in line:
        return True
    return False


def main() -> None:
    sys.stdin.read()
    channel_id = _cluster_channel_id()
    if not INCOMING.exists():
        print(json.dumps({}))
        return
    try:
        lines = INCOMING.read_text(encoding="utf-8", errors="replace").splitlines()
    except OSError:
        print(json.dumps({}))
        return
    window = lines[-SCAN:] if len(lines) > SCAN else lines
    prio: list[str] = []
    other: list[str] = []
    for line in window:
        if not line.strip():
            continue
        if _is_priority_line(line, channel_id):
            prio.append(line)
        else:
            other.append(line)
    # Keep leader/channel signal at the **bottom** (most recent context for the model).
    merged = other + prio
    tail = merged[-MAX_LINES:]
    block = "\n".join(tail)
    if len(block) > MAX_CHARS:
        block = block[-MAX_CHARS:]
    if not block.strip():
        print(json.dumps({}))
        return
    ctx = (
        "[MeshKore] Recent inbox (JSONL). Lines mentioning hyverk-lead / cluster channel / "
        "task.* are prioritized; fleet.ping noise is trimmed.\n" + block
    )
    print(json.dumps({"additional_context": ctx}))


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
Permanent MeshKore inbox listener for Hyverk.

- Polls GET {hub_url}/agents/messages every **5 seconds** for the **primary** token in
  `.meshkore.local` only. That endpoint is the **unified inbox**: **channel traffic the hub
  surfaces to you** plus **direct messages** addressed to **your** `agent_id`. If the Mac M4
  lead (`hyverk-lead`) DMs a *different* agent_id (e.g. an old `hyverk-cursor-architect-*`),
  this identity will not see those DMs — use `MESHKORE_AGENT_ID=…` on join to match the id
  the lead uses, or ask them to post to the **cluster channel** (see `.meshkore` → `cluster.channel_id`).
- Optional second token: set env `MESHKORE_POLL_TEAMMATE=1` to also poll `teammate`.
- Appends each new message as one JSON line to .meshkore-incoming.jsonl (deduped).

Run under launchd, nohup, or a terminal multiplexer:

  nohup python3 scripts/meshkore-listener.py >> /tmp/meshkore-listener.log 2>&1 &
"""
from __future__ import annotations

import json
import os
import subprocess
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
LOCAL = ROOT / ".meshkore.local"
INCOMING = ROOT / ".meshkore-incoming.jsonl"
STATE = ROOT / ".meshkore-listener.state.json"
INTERVAL_SEC = 5
MAX_JSONL_LINES = 2500
SEEN_CAP = 8000


def _curl_json(url: str, token: str) -> dict:
    r = subprocess.run(
        [
            "curl",
            "-sS",
            "-m",
            "25",
            "-H",
            f"Authorization: Bearer {token}",
            "-H",
            "Accept: application/json",
            url,
        ],
        capture_output=True,
        text=True,
    )
    if r.returncode != 0:
        return {"_curl_error": r.stderr[:500]}
    try:
        return json.loads(r.stdout)
    except json.JSONDecodeError:
        return {"_parse_error": r.stdout[:500]}


def _load_state() -> dict:
    if not STATE.exists():
        return {"seen": []}
    try:
        return json.loads(STATE.read_text())
    except json.JSONDecodeError:
        return {"seen": []}


def _save_state(seen: list[str]) -> None:
    STATE.write_text(json.dumps({"seen": seen[-SEEN_CAP:]}, indent=0) + "\n")
    STATE.chmod(0o600)


def _trim_jsonl() -> None:
    if not INCOMING.exists():
        return
    lines = INCOMING.read_text().splitlines()
    if len(lines) <= MAX_JSONL_LINES:
        return
    INCOMING.write_text("\n".join(lines[-MAX_JSONL_LINES:]) + "\n")


def _dedupe_id(msg: dict) -> str:
    pl = msg.get("payload") or {}
    mid = pl.get("_message_id")
    if mid:
        return str(mid)
    return f"{msg.get('from')}:{msg.get('ts')}:{pl.get('type')}:{hash(json.dumps(pl, sort_keys=True))}"


def main() -> None:
    seen: list[str] = _load_state().get("seen") or []
    seen_set = set(seen)
    last_local_mtime = 0.0
    tokens: list[str] = []
    hub = ""

    while True:
        if not LOCAL.exists():
            time.sleep(INTERVAL_SEC)
            continue
        try:
            mt = LOCAL.stat().st_mtime
            if mt != last_local_mtime or not tokens:
                c = json.loads(LOCAL.read_text())
                hub = str(c.get("hub_url", "")).rstrip("/")
                tokens = [c["token"]]
                if os.environ.get("MESHKORE_POLL_TEAMMATE") == "1":
                    tm = c.get("teammate")
                    if isinstance(tm, dict) and tm.get("token"):
                        tokens.append(tm["token"])
                last_local_mtime = mt
        except (OSError, json.JSONDecodeError, KeyError):
            time.sleep(INTERVAL_SEC)
            continue

        if not hub or not tokens:
            time.sleep(INTERVAL_SEC)
            continue

        new_any = False
        for tok in tokens:
            data = _curl_json(f"{hub}/agents/messages", tok)
            for m in data.get("messages") or []:
                did = _dedupe_id(m)
                if did in seen_set:
                    continue
                seen_set.add(did)
                seen.append(did)
                INCOMING.parent.mkdir(parents=True, exist_ok=True)
                with INCOMING.open("a", encoding="utf-8") as f:
                    f.write(json.dumps(m, ensure_ascii=False) + "\n")
                new_any = True

        if new_any:
            _trim_jsonl()
            seen = seen[-SEEN_CAP:]
            seen_set = set(seen)
            _save_state(seen)

        time.sleep(INTERVAL_SEC)


if __name__ == "__main__":
    main()

# MeshKore — multi-agent teamwork on Hyverk (`plan-task-review@v1`)

Canonical hub config: **`.meshkore`**. Credentials: **`.meshkore.local`** (never commit).

## Who listens to whom

- **Every Cursor-side agent** with `.meshkore.local` must **poll** `GET {hub_url}/agents/messages` on an interval (or use the project `scripts/meshkore-keepalive.py`) so it stays **online** and receives `plan.*`, `task.*`, `review.*` payloads.
- **Read the inbox before writing code** when the user prompt references the cluster, MeshKore, or `_rjj/context/meshkore/`.
- **Leader** (human or `hyverk-lead` automation) emits goals; **peer agents** in the same `channel_id` are equals unless a message sets `lead: true` in its payload (convention — follow if present).

## Channel identity

- **`channel_id`:** take from `.meshkore` → `cluster.channel_id`.
- **Broadcast (room):** `POST {hub_url}/agents/channels/{channel_id}/send` with `Authorization: Bearer {token}` and JSON body per hub docs.
- **Direct DM:** `POST {hub_url}/agents/messages` with `{ "to": "<agent_id>", "payload": { ... } }`.

If `hub.meshkore.com` is blocked, use **`hub_url` from `.meshkore.local`** (typically `https://meshkore-relay.fly.dev`) for the same paths.

## Payload types (joint code loop)

Align with **`.meshkore` → `messaging.protocol`:** `plan-task-review@v1`.

| Type | From | Meaning |
|------|------|---------|
| `plan.propose` | architect / lead | Goal, scope, files touched, P0/P1 link from TEAM-OBJECTIVES |
| `plan.ack` | implementer | Understood + ETA or questions |
| `task.start` | implementer | Branch/commit intent |
| `task.done` | implementer | Summary + paths changed |
| `review.feedback` | reviewer | Blocking / non-blocking notes |
| `final.consensus` | any agreed role | “Ship as-is” or “revert X” |

Always include in `payload`: **`from`** (agent_id), **`refs`** (commit or PR id if any), **`objective_id`** (e.g. `P0-kv-incremental`).

## Splitting work between two (or more) agents

1. **Architect** posts `plan.propose` to the channel with a **single** primary owner (`assignee` field) when possible.
2. **Implementer** responds with `plan.ack`, then `task.start` → code → `task.done`.
3. **Reviewer** (can be same machine, second Cursor session / second registered agent_id) sends `review.feedback`; implementer addresses or escalates with a new `plan.propose` delta.
4. Only after **`final.consensus`** should large merges be treated as settled for that task.

## Grounding in repo context

- Stack / module hints from `@llm-context` headers still apply.
- This folder (**`_rjj/context/meshkore/`**) defines **how mesh traffic maps to Hyverk code**; numeric performance truth stays in **`_rjj/log/AUDIT-distributed-inference.md`**.

## Permanent listener (no “poll now” every message)

- Run **`python3 scripts/meshkore-listener.py`** 24/7 (tmux, `nohup`, or macOS `launchd` using `scripts/com.hyverk.meshkore-listener.plist` after replacing `WorkingDirectory` with your absolute repo path, then `launchctl load …`).
- It polls **`/agents/messages` every 5 seconds** for the **primary** agent only (Cursor ↔ leader); set **`MESHKORE_POLL_TEAMMATE=1`** only if you explicitly want a second registered token polled.
- It appends every new Mesh message to **`.meshkore-incoming.jsonl`** (gitignored) and keeps poll/online traffic.
- Optional Cursor hooks (**`.cursor/hooks.json`**) — reload Cursor after install; only add hooks you actually ship in `.cursor/hooks/`:
  - **`sessionStart`**: may inject a tail of `.meshkore-incoming.jsonl` into **`additional_context`** (see `meshkore_session_start.py` if present).
  - **`stop`**: may emit **`followup_message`** when new jsonl lines landed (see `meshkore_stop_followup.py` if present).
- **Outbound channel posts** (e.g. `task.progress` after an edit): use **`python3 scripts/meshkore-send-channel.py`** or `curl` per hub docs — there is **no** required `afterFileEdit` hook in this repo.

## Bidirectional channel (constant coordination)

```text
  ┌────────────────────────┐        MeshKore hub / relay         ┌────────────────────────┐
  │ Cursor (you)           │  ── channel + poll 5s + hooks ──►  │ hyverk-lead / M4 lead  │
  │ primary .meshkore.local│  ◄── same channel / DMs ─────────   │                        │
  └──────────┬─────────────┘                                    └──────────┬─────────────┘
             │                                                             │
             │  meshkore-listener.py (this repo, primary token only)      │
             v                                                             v
        .meshkore-incoming.jsonl ◄────────── optional hooks / Cursor ────┘
```

**Inbound:** listener + optional `sessionStart` / `stop` hooks pull peer / leader traffic into Cursor without the user repeating “listen”.  
**Outbound:** post **`task.progress`** (or other payload types) to **`cluster.channel_id`** when you want the Mac M4 lead or `hyverk-lead` to see an update — **`scripts/meshkore-send-channel.py`**, or poll is enough if you only need to **read** the leader.

Together: **listen** on an interval (or hooks), **respond** on the channel; **no extra connection layer** beyond Bearer token + same `hub_url` as in **`.meshkore.local`** (relay vs canonical per firewall).

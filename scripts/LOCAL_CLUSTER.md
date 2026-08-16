# Local multi-Mac cluster

Prepare code and machines so you can bring the stack up on a LAN.

## Quick smoke (one machine, no model download)

```bash
bash scripts/smoke-coordinator-local.sh
```

Checks health, fake-model availability, API-key gate, and clean shutdown.

## Prereqs (each Mac)

- Rust (`rustup`), CMake, Python 3 + `pip install -r inference/requirements.txt`
- Same git revision of this repo
- LAN reachability: nodes dial out to the coordinator WebSocket; Macs need no public inbound port

## 1. Model on the coordinator host

Pick one Mac (or a Linux box) as coordinator:

```bash
bash scripts/prepare-model.sh
# → ~/.hyverk/model  (override with HYVERK_MODEL_DIR)
```

## 2. Start coordinator

```bash
bash scripts/run-coordinator-local.sh
# HTTP :17000  WS /ws  model from HYVERK_MODEL_DIR
```

Optional public-ish LAN hardening:

```bash
export HYVERK_API_KEY='long-random-string'
# Clients must send: Authorization: Bearer …  (ws-inference / inference only)
```

## 3. Start GPU nodes (2–N Macs)

On each Mac:

```bash
# Point at the coordinator LAN IP
bash scripts/run-node-local.sh http://192.168.x.y:17000
```

Ensure `hardware_info` mentions `Metal` (or `CUDA` on Windows) so the node joins the GPU pool.

## 4. Wait for operational

```bash
curl -s http://COORD:17000/api/v1/cluster/status
curl -s http://COORD:17000/api/v1/clusters   # active / pending / draining generations
```

`status` should become `operational` after every GPU node downloads its layer cache and starts `:18100`.

## 5. Inference

```bash
curl -s -X POST http://COORD:17000/api/v1/ws-inference \
  -H 'Content-Type: application/json' \
  -H "Authorization: Bearer $HYVERK_API_KEY" \
  -d '{"prompt":"Write a Rust hello world","max_tokens":64}'
```

## Dynamic join / leave

- Extra Macs can start `hyverk-node` later; the coordinator forms a **pending** generation, keeps serving the **active** one, then swaps when all new assignments report ready.
- A disconnect mid-hop fails that request immediately (no 120s silent hang).
- More nodes lengthen the layer chain (more hops). Throughput gains need multiple short chains later; correctness comes first.

## Notes

- Default model: Qwen2.5-Coder-7B-Instruct (28 layers), matching the current hard-coded split.
- Layer packs are still HF safetensor shards (possible overlap across nodes). Per-layer bundles remain a follow-up.
- Do not commit `config.toml`, `.meshkore.local`, or API keys.

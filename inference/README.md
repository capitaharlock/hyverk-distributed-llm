# inference/ — node-side worker for Qwen2 distributed pipeline

The Rust `ws_worker` on each contributor machine launches `node_forward.py`
as a subprocess. `serve` mode is the hot path: load the assigned layer
shard once, hold it in GPU/CPU memory, answer HTTP requests from the Rust
worker for every step of a distributed inference chain.

This directory is all CPU / MPS / CUDA Python. The coordinator (Rust, in
`server/coordinator/`) drives the chain and never touches this module
directly — it speaks to the Rust worker over WebSocket, which then POSTs
to `http://127.0.0.1:{port}` here.

Context: `.meshkore/docs/architecture/distributed-inference.md`.

## Modes

```
python3 node_forward.py --mode {download|serve|embed|forward|generate} \
    --model-dir ~/.hyverk/models \
    --layer-start N --layer-end M \
    [--port 18100]
```

| mode       | purpose                                                                  |
|------------|--------------------------------------------------------------------------|
| `download` | Pull this node's safetensors shard(s) from the coordinator.              |
| `serve`    | Persistent HTTP server. This is the **hot path** for distributed infer. |
| `embed`    | One-shot: token ids → hidden states (first node, CLI fallback).          |
| `forward`  | One-shot: hidden states → hidden states (middle node, CLI fallback).     |
| `generate` | One-shot: hidden states → token id via norm + lm_head (last node, CLI).  |

Only `serve` is called during production inference. The one-shot modes
are kept for offline smoke-tests and the legacy file-IO chain.

## Environment flags

All flags are opt-in. Unset = current default behaviour.

| Variable                | Effect                                                                                           | Added in |
|-------------------------|--------------------------------------------------------------------------------------------------|----------|
| `HYVERK_DEBUG_NAN=1`    | Enable the per-layer `isnan/isinf` scan in `run_forward_legacy` and `run_forward_kv`. Off by default because it costs one GPU sync + full tensor reduction per layer per step. The end-of-pipe logits NaN cleanup is always on (cheap and catastrophic if skipped). | DINF-002 |
| `HYVERK_COMPILE=1`      | Wrap each `Qwen2DecoderLayer` in `torch.compile(mode=…, fullgraph=False, dynamic=True)`. CUDA-only; logs and skips on MPS/CPU. First forward pass pays the compile cost. | DINF-005 |
| `HYVERK_COMPILE_MODE`   | Inductor mode — default `reduce-overhead` (good for small-batch decode). Can be `max-autotune` for prefill-heavy loads. Ignored unless `HYVERK_COMPILE=1`. | DINF-005 |
| `HYVERK_FLASH_ATTN=1`   | Set `config._attn_implementation="flash_attention_2"` on layer construction. Requires the `flash-attn` package and a CUDA device; falls back to SDPA silently otherwise. CPU path is unaffected (pinned to eager + int8 dynamic quant). | DINF-004 |
| `HYVERK_TEMPERATURE`    | Default sampling temperature for the generate path. Header `X-Temperature` wins if present. `0.0` (default) = argmax / greedy. | DINF-009 |
| `HYVERK_TOP_P`          | Default nucleus-sampling top-p. `1.0` (default) = no filter. Overridden by `X-Top-P` header. | DINF-009 |
| `HYVERK_TOP_K`          | Default top-k filter for sampling. `0` (default) = no filter. Overridden by `X-Top-K` header. | DINF-009 |
| `HYVERK_KV_MAX_ENTRIES` | Max concurrent request_ids to keep KV cache for. Default `8`. On overflow we evict the LRU entry and log the drop to stderr. | DINF-010 |
| `HYVERK_KV_IDLE_TIMEOUT_S` | Drop KV entries not touched in N seconds (coordinator crashed / aborted). Default `300`. `0` disables the reaper. | DINF-012 |
| `HYVERK_KV_REAPER_INTERVAL_S` | Background sweep frequency. Default `30`. Min `5`. Ignored if idle-timeout disabled. | DINF-012 |

## HTTP contract (serve mode)

### `GET /health`

Lock-free. Returns readiness + snapshot counters; answers in ~2 ms even
while a POST thread is inside `run_forward` holding `model_lock`.

```json
{
  "status": "ready",
  "device": "cuda",
  "layers": "0-10",
  "port": 18100,
  "kv_incremental": true,
  "active_requests": 2,
  "max_cache_entries": 8,
  "uptime_s": 142,
  "kv_entries": [
    {"request_id": "abc-123", "kv_tokens": 512},
    {"request_id": "abc-124", "kv_tokens": 17}
  ]
}
```

### `POST /`

Two dispatch flavours on the same URL — the `Content-Type` picks the path.

#### JSON body (control / embed)

```
Content-Type: application/json
```

```jsonc
// prefill — embed a fresh prompt and run through this node's layers
{"mode": "embed", "request_id": "abc-123", "token_ids": [1, 2, 3, ...]}

// incremental decode step — embed ONE new token id, prepend to KV cache
{"mode": "embed_step", "request_id": "abc-123", "token_id": 42}

// drop the KV entry for a finished request
{"mode": "clear_cache", "request_id": "abc-123"}
```

`embed` and `embed_step` respond with `Content-Type: application/octet-stream`
and the same binary protocol as `forward` (`X-Shape` header, fp16 bytes).

#### Binary body (forward / generate)

```
Content-Type: application/octet-stream
X-Mode:       forward | generate
X-Request-Id: <uuid>            ; identifies the KV cache entry
X-Shape:      [1, <seq>, <h>]   ; JSON-encoded shape
X-Temperature: 0.7              ; optional, sampling
X-Top-P:       0.9              ; optional, nucleus filter
X-Top-K:       50               ; optional, top-k filter
```

As of DINF-011 (commit `ea42d3f`), the Rust coordinator + ws_worker
actually populate these headers from the original request parameters
(`CoordinatorMessage::InferenceStart/Forward/Continue` carry
`temperature` / `top_p` / `top_k`; the node keeps them in a per-`request_id`
map and attaches them when `mode == generate`). So the end-to-end
sampling pipeline is: coordinator → ws_worker → Python headers →
`_sample_next_token`. Operators can still use the HYVERK_* env defaults
as a global floor, but per-request overrides now win all the way through.

Body: raw fp16 bytes for the hidden-state tensor in shape `X-Shape`.

Response for `X-Mode: forward`:

```
HTTP 200
Content-Type:   application/octet-stream
X-Shape:        [1, <seq>, <h>]
X-Elapsed-Ms:   <ms>
<fp16 bytes>
```

Response for `X-Mode: generate` (last-node only, when lm_head is loaded):

```json
{"token_id": 4242, "elapsed_ms": 12}
```

Sampling defaults preserve the legacy argmax behaviour (`temperature=0`).
See DINF-006 for the nucleus sampler implementation.

## Benchmarking

`inference/bench_node_forward.py` runs a tiny synthetic Qwen2 through
prefill + decode on whatever device is available. No checkpoint needed.

```bash
python3 inference/bench_node_forward.py \
    --device auto --num-layers 4 --prompt-len 64 --steps 64

# Measure the CUDA compile win:
HYVERK_COMPILE=1 python3 inference/bench_node_forward.py --device cuda

# Measure FlashAttention-2:
HYVERK_FLASH_ATTN=1 python3 inference/bench_node_forward.py --device cuda
```

Output is a JSON line with `prefill_ms`, `decode_p50_ms`, `decode_p95_ms`,
`decode_mean_ms`, and `tokens_per_second`. Compare within the same device
class — not across CPU vs GPU.

## Tests

```bash
python3 inference/test_node_forward.py
```

Covers:

- Prefill + decode shape invariants; no-mask path on `seq_len=1` with KV.
- `lm_head` fp32 cache numerically equivalent to the legacy per-step
  `lm_head.float()` cast.
- Incremental decode (prefill + 1 step) matches a single full forward
  within fp32 tolerance → proves DINF-002's mask-skip is mathematically
  sound.
- Sampler: argmax at `T=0`, `top_k=1`, and `top_p→0`; variety at `T=1`;
  degenerate `-inf` input handled.
- All four env-gate flags parse correctly.
- `/health` responds lock-free while a POST holds `model_lock`.

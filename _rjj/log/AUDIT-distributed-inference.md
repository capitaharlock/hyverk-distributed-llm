# Hyverk — Distributed inference audit

**Artifact:** engineering audit (not a product “audio” audit).  
**Scope:** GPU-oriented paths (Metal / MPS, CUDA, Windows “envy-class” discrete GPUs), coordinator ↔ WebSocket nodes ↔ local Python layer servers.  
**Date:** 2026-04-15  

---

## Verdict (short)

| Domain | Dominant limit today | Notes |
|--------|----------------------|--------|
| **Throughput / tokens·s⁻¹** | **Software** | Coordinator-driven decode **recomputes the full prefix** every new token (no cross-node KV reuse). Complexity behaves like **O(n²)** in output length across the chain. This dwarfs micro-optimizations until fixed. |
| **Latency per hop** | **Mixed** | WebSocket + JSON control plane + **f16 activation tensors** over the network are real physical costs; still, payload sizes grow with sequence length because of the recompute strategy. |
| **Single-node compute** | **Physical + library** | Per-GPU **memory** caps shard size; **memory bandwidth** caps matmul-heavy layers. Stack uses **PyTorch + HuggingFace `Qwen2DecoderLayer`** with **SDPA** on GPU (not FlashAttention-2 / Triton kernels / TensorRT-LLM). |
| **Orchestration “SoTA”** | **Software / architecture** | Design is **manual pipeline parallelism** (layer ranges on different machines), not **NCCL tensor parallel**, not **vLLM/SGLang**-class serving, not **Ray**-style elastic scheduling. |

**Bottom line:** Hyverk is **not yet applying state-of-the-art LLM serving stacks** end-to-end. It is a **custom research/distribution** layout with a **correctable algorithmic gap** (incremental decode + KV) and several **incremental engineering** upgrades possible after that.

---

## Architecture recap (what actually runs)

1. **Coordinator** (`server/coordinator`): builds a **chain** of GPU WebSocket nodes; drives **InferenceStart** / **InferenceForward**; merges **TokenGenerated**; exposes HTTP APIs (e.g. `/api/v1/ws-inference`).
2. **Client node** (`client/node` `ws_worker.rs`): receives coordinator messages; for each forward step calls **`http://127.0.0.1:18100`** on the **same machine** (Python `inference/node_forward.py` “serve” mode).
3. **Python worker** (`inference/node_forward.py`): loads a **slice** of Qwen2 weights; runs embed / layer forward / optional `lm_head` on **MPS, CUDA, or CPU** (CPU uses dynamic int8 quant).

Cross-machine traffic is **activations** (and control messages), not a fused NCCL collective inside one process.

---

## Physical limits (what hardware will always cap)

- **VRAM:** Each node holds only its **layer shard + embeddings/norm/lm_head** as designed. If the shard does not fit, you must quantize, shrink the model, or change sharding — not fixable in software without trade-offs.
- **Memory bandwidth:** FP16/BF16 decode is often **bandwidth-bound** on attention and large MLP projections. Apple Silicon **unified memory** helps avoid PCIe copies; discrete **PCIe GPUs** pay host↔device transfer costs when the stack is not careful.
- **Network:** Each hop ships **O(seq_len × hidden × 2 bytes)** for f16 hidden states in the current “full sequence” decode pattern. **1 GbE vs 10 GbE vs Wi‑Fi** changes wall time quickly. **RDMA / GPUDirect** is not in play here.
- **Single-threaded Python HTTP (stdlib):** `http.server.HTTPServer` is **process-serial** for a given worker in practice; under concurrent clients, this becomes a **software-imposed serialization** that feels like a hardware queue but is not.
- **Metal vs CUDA ecosystem:** **CUDA** has the richest **FlashAttention / fused MLP / custom epilogue** ecosystem; **MPS** improves steadily but is not automatically at parity with the fastest CUDA serving paths.

These limits define the **ceiling** once the software path is sane (KV, minimal payloads, efficient kernels).

---

## Software limits (what Hyverk controls and should iterate)

| Issue | Where | Impact |
|-------|--------|--------|
| **Full-sequence recompute each token** | `ws_handler.rs` `handle_generated_token` → `InferenceStart` with `token_ids` = prompt + **all** generated tokens | **Primary throughput killer**; quadratic work vs length. |
| **KV cache unused in serve path** | `node_forward.py`: `kv_cache` dict exists; `run_forward` uses **`use_cache=False`** and rebuilds **full S×S causal mask** each call | Wasted memory bandwidth + compute; blocks true incremental decode. |
| **Control-plane chatter** | WebSocket JSON + separate binary frames per step | Latency floor for **many small steps**; acceptable if each step were **one real decode**, not a full rerun. |
| **No continuous batching / no multi-request scheduling** | Whole design is **one request linear chain** | SoTA servers batch independent sequences to saturate GPUs. |
| **Sampling vs temperature** | Distributed path uses **argmax** in Python for next token; temperature in messages may be **ignored** for quality knobs | Not a speed bug, but a **behavior consistency** issue. |
| **Weight load path** | `safetensors` → CPU → `.to(device)` per tensor | Startup cost; can be optimized (mmap, direct GPU load) but is not decode-hot. |

---

## State of the art (reference) vs Hyverk

| Technique | Typical stack | Hyverk today |
|-----------|----------------|--------------|
| **Paged / chunked KV** (limit fragmentation) | vLLM, SGLang | Not used |
| **Prefill vs decode separation** | Most serving systems | **Not explicit**; prefill is repeated every token |
| **FlashAttention-2 / Triton** | Fast CUDA attention | **SDPA** on GPU (better than eager; not FA2-class on CUDA) |
| **Tensor parallel (intra-node)** | Megatron, DeepSpeed, NCCL | **Not used** — one GPU per process slice |
| **Pipeline parallel (inter-node)** | Megatron PP, GPipe concepts | **Yes**, manually via coordinator chain |
| **Speculative decoding** | Draft model + verify | Not used |
| **torch.compile / Inductor** | PyTorch 2+ | Not used in `node_forward.py` hot path |
| **Dedicated inference server** | Triton, TensorRT-LLM, llama.cpp server | **stdlib HTTPServer** |

Hyverk’s **pipeline-over-the-network** idea is **legitimate** for heterogeneous volunteers, but **without incremental KV and without a modern server runtime**, it will not sit on the **Pareto frontier** of tokens·s⁻¹ vs quality.

---

## Recommendations (priority order)

1. **P0 — Incremental decode + per-node KV** tied to `request_id` (coordinator sends **one new token** after prefill, or equivalent activation-only step). Aligns with standard **prefill / decode** APIs.
2. **P0 — Stop allocating full causal mask every step** when using KV (mask only the **new row** / use cache-aware APIs).
3. **P1 — Replace stdlib HTTP with an async server** (e.g. **uvicorn + Starlette/FastAPI**) or **gRPC** for layer RPC; add **health + concurrency** controls.
4. **P1 — CUDA path:** evaluate **FlashAttention-2** (where supported) behind a config flag; keep MPS on SDPA until validated.
5. **P2 — Orchestration:** if the product grows, consider **Ray Serve** / **Kubernetes + job queue** for placement, back-pressure, and autoscaling instead of ad-hoc WS routing alone.
6. **P2 — Quality:** wire **temperature / top‑p** consistently or document “greedy only” for distributed mode.

---

## Task cards (import / copy to tracker)

Use these as Kanban items (title + acceptance sketch).

### Card DINF-001 — Incremental KV decode (P0) — **implemented 2026-04-15**

- **Category:** Algorithm / coordinator / Python  
- **Acceptance:** After first prefill, **no** `InferenceStart` with full `token_ids` per token; per-request KV lives on each node; tokens/s scales **~linearly** with output length in the mid-range (measure vs baseline).  
- **Shipped:** `InferenceContinue` / `InferenceEnd`, `embed_step`, `DynamicCache` path in `node_forward.py` (requires recent `transformers`).  

### Card DINF-002 — Causal mask + cache API cleanup (P0)

- **Category:** Python inference  
- **Acceptance:** `run_forward` uses **`use_cache=True`** on decode path; no full **S×S** mask allocation when `seq_len == 1` and cache present; NaN regression tests pass.  

### Card DINF-003 — Async inference HTTP (P1)

- **Category:** Python infra  
- **Acceptance:** Local port 18100 served by async stack; concurrent health check + single decode under load without head-of-line blocking from stdlib server.  

### Card DINF-004 — FlashAttention / kernel flag (P1, CUDA)

- **Category:** Performance / optional dep  
- **Acceptance:** Config toggles FA2 when `device.type == "cuda"` and package present; falls back to SDPA; benchmark doc in `_rjj/log/`.  

### Card DINF-005 — Sampling parity (P2)

- **Category:** Correctness  
- **Acceptance:** `temperature` / `top_p` applied in distributed generate path or explicitly documented as greedy-only.  

### Card DINF-006 — Network payload metrics (P2)

- **Category:** Observability  
- **Acceptance:** Coordinator logs **bytes/hop/token** and **p50/p95** hop latency; dashboard or structured logs.  

### Card DINF-007 — Weight load optimization (P3)

- **Category:** Cold start  
- **Acceptance:** Document or implement faster shard load (mmap / direct GPU) with measured startup delta.  

---

## Machine-specific notes (Metal / Windows NVIDIA)

- **Apple Metal (MPS):** Good for **single-process** inference; verify SDPA paths on your OS + torch combo; watch for **fp16 overflow** in long contexts (code already guards some logits).  
- **Windows + NVIDIA:** Prefer **CUDA** build of PyTorch; this is where **FlashAttention / TensorRT-LLM** class wins appear **if** you adopt those libraries.  
- **Multi-GPU on one box:** Hyverk’s current **split is across processes/machines**, not **NCCL all-reduce** tensor parallel; **intra-node** multi-GPU would be a **new** topology (bigger change, higher SoTA alignment).

---

## Closing

**Physical limits** matter (VRAM, NIC, memory BW), but today’s **Hyverk bottleneck is overwhelmingly software**: **re-decoding the entire sequence every token** and **not using KV/incremental attention** in the distributed path. Addressing that moves the project from a **clever prototype chain** toward something competitive with **modern serving** assumptions, without abandoning the **volunteer / multi-machine** story.

---

*Maintainers: keep this file under `_rjj/log/` as the canonical audit; link it from internal docs if you add a `_rjj/README.md` later.*

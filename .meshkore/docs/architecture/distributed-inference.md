---
title: "Distributed inference"
category: architecture
updated: 2026-08-16
owner: hyverk-lead
status: active
---

# Distributed inference

- **Base model:** Qwen2.5-Coder-7B-Instruct (28 layers).
- **Serving:** layer ranges assigned per node; coordinator builds an active generation of slots.
- **Formats:** safetensors for training shards; GGUF Q4_K_M for llama.cpp paths where used.
- **Local defaults:** coordinator `http://127.0.0.1:17000`; optional `HYVERK_API_KEY` on inference POSTs.
- **Rebalance:** generation-aware routing with drain so in-flight requests finish before retiring a generation.

Code entry points: `server/coordinator/src/serving_clusters.rs`, `ws_handler.rs`, `inference/node_forward.py`.


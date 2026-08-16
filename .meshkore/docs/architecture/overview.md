---
title: "Architecture overview"
category: architecture
updated: 2026-08-16
owner: hyverk-lead
status: active
---

# Architecture overview

```
Contributors (Mac/Windows/Linux)     Coordinator (your host)
┌──────────────┐                     ┌──────────────────┐
│  GPU Client  │◄──── WebSocket ────►│  Orchestrator    │
│  LoRA Train  │     + HTTP API      │  Task Router     │
│  Inference   │                     │  Dashboard       │
└──────────────┘                     │  Dataset Store   │
┌──────────────┐                     │  Training Mgmt   │
│  CPU Client  │◄──── WebSocket ────►│                  │
│  Synthesis   │                     └──────────────────┘
│  Training    │
└──────────────┘
```

**Coordinator** (`server/coordinator`) — manages nodes, shard assignment, inference routing, model HTTP.

**Clients** (`client/`, `hyverk-node`) — connect over WebSocket; Metal / CUDA / CPU according to hardware.

**Inference runtime** (`inference/`) — Python layer-parallel forward for distributed generation.


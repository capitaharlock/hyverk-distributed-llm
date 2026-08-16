# Hyverk

Distributed network for training and serving open-source coding models. Contributors donate compute (CPU/GPU) to collectively fine-tune and serve a shared LLM.

This repository is a **demo / source** tree. There is no required hosted coordinator: run locally, or deploy the stack on any host you choose.

## Architecture

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

**Coordinator** — Hub you run yourself (LAN or cloud). Manages nodes, training shards, inference routing, and the dashboard.

**Clients** — Run on contributor machines. Connect to the coordinator and contribute compute for training, synthesis, or inference.

## Model

- **Base**: Qwen2.5-Coder-7B-Instruct (7.6B params, 28 layers)
- **Fine-tuning**: Layer-sharded LoRA (rank 16, ~14M trainable params)
- **Inference**: llama.cpp — Metal (Mac) / CUDA (NVIDIA) / CPU fallback
- **Format**: GGUF Q4_K_M (4.4GB) for inference, safetensors for training

## Quick Start

### macOS / Linux

```bash
git clone https://github.com/capitaharlock/hyverk-distributed-llm.git
cd hyverk-distributed-llm
bash deploy/setup.sh
```

### Windows

```powershell
git clone https://github.com/capitaharlock/hyverk-distributed-llm.git
cd hyverk-distributed-llm
powershell -ExecutionPolicy Bypass -File deploy\setup.ps1
```

### Requirements

| | macOS | Windows | Linux |
|---|---|---|---|
| Rust | ✓ | ✓ + MSVC Build Tools | ✓ |
| CMake | `brew install cmake` | `winget install cmake` | `apt install cmake` |
| GPU | Metal (automatic) | CUDA Toolkit | CUDA Toolkit |

### Run (local cluster)

```bash
bash scripts/prepare-model.sh          # once, on the coordinator host
bash scripts/run-coordinator-local.sh  # HTTP :17000
# on each GPU Mac:
bash scripts/run-node-local.sh http://<coordinator-lan-ip>:17000
```

See `scripts/LOCAL_CLUSTER.md`. Optional: `HYVERK_API_KEY` gates `POST /api/v1/ws-inference`.

### Run (legacy single binary)

```bash
./target/release/hyverk --config ~/.hyverk/config.toml --mode node
```

### Hyverk-node (WS client for distributed inference / training)

```bash
cargo build --release -p hyverk-node
HYVERK_CONFIG=./config.toml ./target/release/hyverk-node
```

Use a **gitignored** `config.toml` with `coordinator_url`, `node.name`, and `hardware_info` (see **Config** below).

## MeshKore Standard

This repo adopts the [MeshKore Standard](https://meshkore.com/standard) for agentic collaboration:

- Public cluster identity: `.meshkore/public/cluster.yaml`
- Docs / tasks: `.meshkore/docs/`, `.meshkore/modules/`
- Secrets: `.meshkore/credentials/` and `.meshkore.local` (**never commit**)

```bash
MESHKORE_INVITE='https://hub.meshkore.com/agents/invites/<nonce>/join' \
  MESHKORE_HUB_URL=https://meshkore-relay.fly.dev \
  bash scripts/meshkore-join.sh
python3 scripts/meshkore-dump-inbox.py
```

Ask a lead for an invite URL. Spec for agents: https://api.meshkore.com/v1/standard.md

## Project Structure

```
.meshkore/         MeshKore Standard (docs, modules, public cluster.yaml)
server/
  coordinator/     Orchestrator: HTTP API, WebSocket, gRPC, dashboard
  rag/             Knowledge base (BM25 + SQLite)
  sandbox/         Code verification (compile check)

client/
  node/            Connects to coordinator, trains, infers
  inference/       llama.cpp wrapper (Metal / CUDA / CPU)
  cli/             Unified CLI entry point
  desktop/         Electron desktop UI

shared/
  core/            Config, errors
  proto/           gRPC protobuf definitions
  comms/           WebSocket protocol messages
  training/        LoRA fine-tuning (candle)
  synthesis/       Data generation via LLM APIs

inference/         Python: layer-parallel forward pass (runtime)
training/          Python: LoRA training + adapter merge (runtime)
proto/             .proto source files
scripts/           Local cluster + MeshKore helpers
```

## API

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/health` | GET | Health check |
| `/api/v1/nodes` | GET | Connected nodes |
| `/api/v1/cluster/status` | GET | Inference cluster state |
| `/api/v1/inference` | POST | Submit inference task |
| `/api/v1/inference/{id}` | GET | Poll task result |
| `/api/v1/ws-inference` | POST | Distributed inference via WebSocket chain |
| `/api/v1/dataset/stats` | GET | Training dataset statistics |
| `/api/v1/layer-training/rounds` | GET | Training round status |

## Config

```toml
mode = "node"

[node]
name = "my-machine"
coordinator_url = "http://127.0.0.1:17000"
models_dir = "~/.hyverk/models"
max_concurrent_tasks = 2
hardware_info = "Apple M4 Max, Metal GPU"

[synthesis]
enabled = true

[[synthesis.providers]]
name = "groq"
api_key = "gsk_..."
model = "llama-3.3-70b-versatile"
```

## License

MIT

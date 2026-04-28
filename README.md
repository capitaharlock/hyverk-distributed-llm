# Hyverk

Distributed network for training and serving open-source coding models. Contributors donate compute (CPU/GPU) to collectively fine-tune and serve a shared LLM.

## Architecture

```
Contributors (Mac/Windows/Linux)     Coordinator (Fly.io)
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

**Coordinator** — Central hub on Fly.io. Manages nodes, distributes training shards, routes inference, serves the dashboard.

**Clients** — Run on contributor machines. Connect to the coordinator, contribute compute for training, synthesis, or inference depending on hardware.

## Model

- **Base**: Qwen2.5-Coder-7B-Instruct (7.6B params, 28 layers)
- **Fine-tuning**: Layer-sharded LoRA (rank 16, ~14M trainable params)
- **Inference**: llama.cpp — Metal (Mac) / CUDA (NVIDIA) / CPU fallback
- **Format**: GGUF Q4_K_M (4.4GB) for inference, safetensors for training

## Quick Start

### macOS / Linux

```bash
git clone https://github.com/capitaharlock/hyverk.com.git
cd hyverk.com
bash deploy/setup.sh
```

### Windows

```powershell
git clone https://github.com/capitaharlock/hyverk.com.git
cd hyverk.com
powershell -ExecutionPolicy Bypass -File deploy\setup.ps1
```

### Requirements

| | macOS | Windows | Linux |
|---|---|---|---|
| Rust | ✓ | ✓ + MSVC Build Tools | ✓ |
| CMake | `brew install cmake` | `winget install cmake` | `apt install cmake` |
| GPU | Metal (automatic) | CUDA Toolkit | CUDA Toolkit |

### Run

```bash
./target/release/hyverk --config ~/.hyverk/config.toml --mode node
```

Dashboard: https://hyverk-coordinator.fly.dev

### Hyverk-node (WS client for distributed inference / training)

The **`hyverk-node`** binary connects to the coordinator over **WebSocket** and participates in GPU layer work when `hardware_info` advertises Metal/CUDA. Python helpers (`inference/node_forward.py`, etc.) need **`pip install -r inference/requirements.txt`** (includes `certifi` for HTTPS downloads from the coordinator).

```bash
cargo build --release -p hyverk-node
HYVERK_CONFIG=./config.toml ./target/release/hyverk-node
```

Use a **gitignored** `config.toml` with `coordinator_url`, `node.name`, and `hardware_info` (see **Config** below). The coordinator must expose model shards under `/api/v1/model/*` when this node should download weights.

### MeshKore (dev mesh — Mac M4 lead + `hyverk-lead` messages)

This is **separate** from the Hyverk Fly coordinator: same repo, **`.meshkore`** (invite + `channel_id`) + **`.meshkore.local`** (Bearer token, never commit). Join once, then poll inbox or run the listener.

```bash
MESHKORE_HUB_URL=https://meshkore-relay.fly.dev bash scripts/meshkore-join.sh
# If the lead DMs a fixed agent_id, match it:
# MESHKORE_AGENT_ID=hyverk-cursor-architect-435d95 bash scripts/meshkore-join.sh
python3 scripts/meshkore-dump-inbox.py
# optional: python3 scripts/meshkore-listener.py
```

Cached hub onboarding (join / poll / channel / §A3 token refresh): **`_rjj/context/meshkore/AGENT-DOCS.relay.md`**. Cluster viewer link is in **`.meshkore`** → `cluster.viewer`.

## Project Structure

```
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
coordinator_url = "https://hyverk-coordinator.fly.dev"
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

## Encrypted Files

Everything under `_rjj/` is encrypted in Git via [git-crypt](https://github.com/AGWA/git-crypt). Without the team key, those paths are opaque binary blobs in history and in clones.

**GitHub visibility:** git-crypt protects **confidentiality** (who can read the *contents*). It does **not** make files visible to “only one GitHub user.” Anyone who can **clone** the repository can still **fetch** the ciphertext for `_rjj/`. To restrict who can access the repo at all, use a **private** GitHub repository and grant access only to the accounts that should clone it (e.g. your user, or a small collaborator list). Combine **private repo** + **git-crypt** if you want both access control and defense-in-depth for `_rjj/`.

**To decrypt (authorized contributors only):**

```bash
# Install git-crypt
brew install git-crypt          # macOS
scoop install git-crypt         # Windows
sudo apt install git-crypt      # Linux

# Unlock after cloning
git-crypt unlock /path/to/hyverk-git-crypt-key
```

The key file is distributed privately to authorized team members.

## License

MIT

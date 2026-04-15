// WebSocket handler for the coordinator.
// Accepts persistent connections from clients.
// Routes inference requests through node chains.

use axum::extract::ws::{Message, WebSocket, WebSocketUpgrade};
use axum::extract::State;
use axum::response::IntoResponse;
use futures_util::{SinkExt, StreamExt};
use hyverk_comms::messages::{ClientMessage, CoordinatorMessage};
use std::collections::HashMap;
use std::sync::Arc;
use tokio::sync::{mpsc, RwLock};
use tracing::{error, info, warn};

/// Connected WebSocket node
pub struct WsNode {
    pub node_id: String,
    pub node_name: String,
    pub has_gpu: bool,
    pub ram_mb: u64,
    pub state: String, // "connecting", "downloading", "ready", "processing", "error"
    pub state_detail: String,
    pub client_version: String,
    pub os: String,
    pub tx: mpsc::UnboundedSender<Message>,
}

/// Shared state for WebSocket connections
pub struct WsState {
    pub nodes: RwLock<HashMap<String, WsNode>>,
    /// Pending inference results: request_id → (hidden_states, next_step)
    pub pending_forwards: RwLock<HashMap<String, PendingForward>>,
}

pub struct PendingForward {
    pub hidden_data: Vec<u8>,
    pub chain: Vec<ChainStep>,
    pub current_step: usize,
    pub token_ids: Vec<u32>,
    pub generated: Vec<u32>,
    pub max_tokens: usize,
    pub temperature: f32,
    pub result_tx: Option<tokio::sync::oneshot::Sender<InferenceResult>>,
}

#[derive(Clone)]
pub struct ChainStep {
    pub node_id: String,
    pub layer_start: usize,
    pub layer_end: usize,
    pub is_last: bool,
}

pub struct InferenceResult {
    pub text: String,
    pub tokens: usize,
    pub elapsed_secs: f64,
    pub cluster: Vec<serde_json::Value>,
    pub generated_ids: Vec<u32>,
}

impl WsState {
    pub fn new() -> Self {
        Self {
            nodes: RwLock::new(HashMap::new()),
            pending_forwards: RwLock::new(HashMap::new()),
        }
    }

    pub async fn node_count(&self) -> usize {
        self.nodes.read().await.len()
    }

    pub async fn send_to_node(&self, node_id: &str, msg: CoordinatorMessage) -> bool {
        let nodes = self.nodes.read().await;
        if let Some(node) = nodes.get(node_id) {
            let json = serde_json::to_string(&msg).unwrap_or_default();
            node.tx.send(Message::Text(json.into())).is_ok()
        } else {
            false
        }
    }

    /// Send binary data (hidden states) to a node
    pub async fn send_binary_to_node(&self, node_id: &str, data: Vec<u8>) -> bool {
        let nodes = self.nodes.read().await;
        if let Some(node) = nodes.get(node_id) {
            node.tx.send(Message::Binary(data.into())).is_ok()
        } else {
            false
        }
    }
}

/// Dynamically assign layers to GPU nodes for inference.
/// Only GPU nodes participate in inference clusters.
/// Layers split evenly among available GPU nodes, sorted by name for determinism.
pub fn assign_gpu_layers(gpu_nodes: &[&WsNode]) -> Vec<(String, usize, usize)> {
    if gpu_nodes.is_empty() { return vec![]; }
    let total_layers: usize = 28;
    let n = gpu_nodes.len();
    let per_node = total_layers / n;
    let remainder = total_layers % n;
    let mut assignments = Vec::new();
    let mut start = 0;
    for (i, node) in gpu_nodes.iter().enumerate() {
        let extra = if i < remainder { 1 } else { 0 };
        let end = start + per_node + extra;
        assignments.push((node.node_id.clone(), start, end));
        start = end;
    }
    assignments
}

/// Build inference chain from currently connected GPU nodes that are "ready".
/// CPU nodes are excluded from inference — they do training/synthesis only.
pub async fn build_inference_chain(ws_state: &WsState) -> Vec<ChainStep> {
    let nodes = ws_state.nodes.read().await;
    let mut gpu_nodes: Vec<&WsNode> = nodes.values()
        .filter(|node| node.state == "ready" && node.has_gpu)
        .collect();
    gpu_nodes.sort_by_key(|n| &n.node_name);
    let assignments = assign_gpu_layers(&gpu_nodes);
    assignments.iter().map(|(node_id, start, end)| {
        ChainStep {
            node_id: node_id.clone(),
            layer_start: *start,
            layer_end: *end,
            is_last: *end >= 28,
        }
    }).collect()
}

/// Check if the inference cluster is complete (all layers 0-28 covered by ready GPU nodes).
pub async fn cluster_status(ws_state: &WsState) -> ClusterStatus {
    let nodes = ws_state.nodes.read().await;
    let mut node_states: Vec<NodeInferenceState> = Vec::new();

    // GPU nodes get dynamic layer assignments; CPU nodes show 0-0 (no inference)
    let mut gpu_nodes: Vec<&WsNode> = nodes.values().filter(|n| n.has_gpu).collect();
    gpu_nodes.sort_by_key(|n| &n.node_name);
    let gpu_assignments = assign_gpu_layers(&gpu_nodes);
    let gpu_map: std::collections::HashMap<String, (usize, usize)> = gpu_assignments
        .iter().map(|(id, s, e)| (id.clone(), (*s, *e))).collect();

    for node in nodes.values() {
        let (start, end) = gpu_map.get(&node.node_id).copied().unwrap_or((0, 0));
        node_states.push(NodeInferenceState {
            node_name: node.node_name.clone(),
            state: node.state.clone(),
            detail: node.state_detail.clone(),
            layer_start: start,
            layer_end: end,
            client_version: node.client_version.clone(),
            os: node.os.clone(),
        });
    }
    node_states.sort_by_key(|n| n.layer_start);

    let ready_gpu: Vec<&NodeInferenceState> = node_states.iter()
        .filter(|n| n.state == "ready" && n.layer_end > 0)
        .collect();
    let all_gpu_ready = !ready_gpu.is_empty() && gpu_nodes.iter().all(|n| n.state == "ready");

    // Check full layer coverage from GPU nodes only
    let mut covered = false;
    if !ready_gpu.is_empty() {
        let starts_at_zero = ready_gpu.first().map_or(false, |n| n.layer_start == 0);
        let ends_at_28 = ready_gpu.last().map_or(false, |n| n.layer_end >= 28);
        let mut no_gaps = true;
        for i in 1..ready_gpu.len() {
            if ready_gpu[i].layer_start > ready_gpu[i - 1].layer_end {
                no_gaps = false;
                break;
            }
        }
        covered = starts_at_zero && ends_at_28 && no_gaps;
    }

    let status = if covered && all_gpu_ready {
        "operational".to_string()
    } else if node_states.is_empty() {
        "no_nodes".to_string()
    } else {
        "forming".to_string()
    };

    ClusterStatus {
        status,
        nodes: node_states,
    }
}

#[derive(Debug, Clone, serde::Serialize)]
pub struct ClusterStatus {
    pub status: String,
    pub nodes: Vec<NodeInferenceState>,
}

#[derive(Debug, Clone, serde::Serialize)]
pub struct NodeInferenceState {
    pub node_name: String,
    pub state: String,
    pub detail: String,
    pub layer_start: usize,
    pub layer_end: usize,
    pub client_version: String,
    pub os: String,
}

/// Axum handler for WebSocket upgrade
pub async fn ws_upgrade(
    ws: WebSocketUpgrade,
    State(state): State<Arc<crate::http_api::AppState>>,
) -> impl IntoResponse {
    let ws_state = state.ws_state.clone();
    ws.on_upgrade(move |socket| handle_ws(socket, ws_state))
}

async fn handle_ws(socket: WebSocket, state: Arc<WsState>) {
    let (mut sink, mut stream) = socket.split();
    let (tx, mut rx) = mpsc::unbounded_channel::<Message>();

    let node_id = uuid::Uuid::new_v4().to_string();
    let node_id_clone = node_id.clone();

    // Spawn task to forward messages from channel to WebSocket
    let send_task = tokio::spawn(async move {
        while let Some(msg) = rx.recv().await {
            if sink.send(msg).await.is_err() {
                break;
            }
        }
    });

    // Receive messages from client
    while let Some(Ok(msg)) = stream.next().await {
        match msg {
            Message::Text(text) => match serde_json::from_str::<ClientMessage>(&text) {
                Ok(client_msg) => {
                    handle_client_message(&state, &node_id, client_msg, &tx).await;
                }
                Err(e) => warn!(node = %node_id, "Bad message: {e}"),
            },
            Message::Binary(data) => {
                // Binary = hidden states from inference forward
                handle_binary_data(&state, &node_id, data.to_vec()).await;
            }
            Message::Ping(d) => {
                let _ = tx.send(Message::Pong(d));
            }
            Message::Close(_) => break,
            _ => {}
        }
    }

    // Cleanup
    state.nodes.write().await.remove(&node_id_clone);
    info!(node_id = %node_id_clone, "WebSocket client disconnected");
    send_task.abort();
}

async fn handle_client_message(
    state: &Arc<WsState>,
    node_id: &str,
    msg: ClientMessage,
    tx: &mpsc::UnboundedSender<Message>,
) {
    match msg {
        ClientMessage::Register {
            node_name,
            hardware_info,
            models,
            ram_mb,
            has_gpu,
            layer_range,
            client_version,
            os,
        } => {
            let node = WsNode {
                node_id: node_id.to_string(),
                node_name: node_name.clone(),
                has_gpu,
                ram_mb,
                state: "connecting".to_string(),
                state_detail: String::new(),
                client_version: client_version.clone(),
                os: os.clone(),
                tx: tx.clone(),
            };
            // Remove stale entries with same name (prevents duplicates on reconnect)
            {
                let mut nodes = state.nodes.write().await;
                let stale: Vec<String> = nodes.iter()
                    .filter(|(id, n)| n.node_name == node_name && id.as_str() != node_id)
                    .map(|(id, _)| id.clone())
                    .collect();
                for id in stale { nodes.remove(&id); }
                nodes.insert(node_id.to_string(), node);
            }
            let layers_info = layer_range
                .map(|(s, e)| format!("{}-{}", s, e))
                .unwrap_or_else(|| "none".to_string());
            info!(node_id, name = %node_name, gpu = has_gpu, ram = ram_mb, layers = %layers_info, "WS node registered");

            // GPU node registered: recalculate and broadcast layer assignments to all GPU nodes
            if has_gpu {
                let nodes = state.nodes.read().await;
                let mut gpu_nodes: Vec<&WsNode> = nodes.values().filter(|n| n.has_gpu).collect();
                gpu_nodes.sort_by_key(|n| &n.node_name);
                let assignments = assign_gpu_layers(&gpu_nodes);
                for (nid, start, end) in &assignments {
                    if let Some(n) = nodes.get(nid.as_str()) {
                        let msg = CoordinatorMessage::LayerAssignment { layer_start: *start, layer_end: *end };
                        let json = serde_json::to_string(&msg).unwrap_or_default();
                        let _ = n.tx.send(Message::Text(json.into()));
                        info!(node_id = %nid, name = %n.node_name, layers = format!("{start}-{end}"), "Sent layer assignment");
                    }
                }
            }
        }
        ClientMessage::StateUpdate {
            state: node_state,
            detail,
        } => {
            let mut nodes = state.nodes.write().await;
            if let Some(node) = nodes.get_mut(node_id) {
                info!(node_id, name = %node.node_name, state = %node_state, detail = %detail, "Node state update");
                node.state = node_state;
                node.state_detail = detail;
            }
        }
        ClientMessage::Heartbeat {
            active_tasks,
            current_role,
            ..
        } => {
            // Update node status
        }
        ClientMessage::ForwardResult {
            request_id,
            hidden_states: _,
            shape: _,
        } => {
            // Text-only signal: actual hidden states arrive as binary frame.
            // Do NOT route here — handle_binary_data does the routing.
            info!(request_id = %request_id, "ForwardResult text received (awaiting binary)");
        }
        ClientMessage::TokenGenerated {
            request_id,
            token_id,
            is_eos,
        } => {
            info!(request_id = %request_id, token_id, eos = is_eos, "Token generated");
            handle_generated_token(state, &request_id, token_id, is_eos).await;
        }
        ClientMessage::Pong => {}
        _ => {}
    }
}

async fn handle_binary_data(state: &Arc<WsState>, node_id: &str, data: Vec<u8>) {
    // Binary frames contain: request_id (first 36 bytes as UUID string) + hidden states
    if data.len() < 36 {
        warn!("Binary data too short");
        return;
    }
    let request_id = String::from_utf8_lossy(&data[..36]).to_string();
    let hidden_states = data[36..].to_vec();
    info!(request_id = %request_id, size = hidden_states.len(), "Binary hidden states from {node_id}");

    route_forward_result(state, &request_id, hidden_states, vec![]).await;
}

async fn route_forward_result(
    state: &Arc<WsState>,
    request_id: &str,
    hidden_states: Vec<u8>,
    _shape: Vec<usize>,
) {
    let mut forwards = state.pending_forwards.write().await;
    if let Some(pending) = forwards.get_mut(request_id) {
        // Do not clone activations into `hidden_data` — it was unused and doubled RAM + memcpy
        // per hop (activations are already large: seq_len × hidden × 2 bytes).
        pending.hidden_data.clear();
        pending.current_step += 1;

        if pending.current_step < pending.chain.len() {
            // Send to next node in chain
            let next = &pending.chain[pending.current_step];
            let msg = CoordinatorMessage::InferenceForward {
                request_id: request_id.to_string(),
                hidden_states_ref: String::new(), // binary sent separately
                layer_start: next.layer_start,
                layer_end: next.layer_end,
                is_last: next.is_last,
            };
            state.send_to_node(&next.node_id, msg).await;
            // Send binary hidden states (36-byte request id prefix, padded — matches node client)
            let mut payload = vec![0u8; 36];
            let rid = request_id.as_bytes();
            let n = rid.len().min(36);
            payload[..n].copy_from_slice(&rid[..n]);
            payload.extend_from_slice(&hidden_states);
            state.send_binary_to_node(&next.node_id, payload).await;
        }
        // If last node, the TokenGenerated message handles completion
    }
}

/// Tell each GPU node in `chain` to drop KV for this request (local `node_forward` clear_cache).
pub async fn broadcast_inference_end(state: &Arc<WsState>, chain: &[ChainStep], request_id: &str) {
    let rid = request_id.to_string();
    let mut seen = std::collections::HashSet::new();
    for step in chain {
        if seen.insert(step.node_id.clone()) {
            let _ = state
                .send_to_node(
                    &step.node_id,
                    CoordinatorMessage::InferenceEnd {
                        request_id: rid.clone(),
                    },
                )
                .await;
        }
    }
}

async fn handle_generated_token(
    state: &Arc<WsState>,
    request_id: &str,
    token_id: u32,
    is_eos: bool,
) {
    let mut forwards = state.pending_forwards.write().await;
    if let Some(pending) = forwards.get_mut(request_id) {
        pending.generated.push(token_id);

        if is_eos || pending.generated.len() >= pending.max_tokens {
            // Snapshot then remove before notifying nodes (avoids holding the lock across sends).
            let chain = pending.chain.clone();
            let cluster_info: Vec<serde_json::Value> = pending
                .chain
                .iter()
                .map(|s| {
                    serde_json::json!({
                        "node": s.node_id,
                        "layers": format!("{}-{}", s.layer_start, s.layer_end),
                        "position": if s.is_last { "last" } else { "first/middle" },
                    })
                })
                .collect();
            let generated_ids = pending.generated.clone();
            let tokens = pending.generated.len();
            let result_tx = pending.result_tx.take();
            forwards.remove(request_id);
            drop(forwards);

            if let Some(tx) = result_tx {
                let _ = tx.send(InferenceResult {
                    text: String::new(),
                    tokens,
                    elapsed_secs: 0.0,
                    cluster: cluster_info,
                    generated_ids,
                });
            }
            broadcast_inference_end(state, &chain, request_id).await;
        } else {
            // Incremental decode: first node embeds only the last generated token (KV on workers).
            pending.current_step = 0;
            let first = &pending.chain[0];
            let msg = CoordinatorMessage::InferenceContinue {
                request_id: request_id.to_string(),
                new_token_id: token_id,
                layer_start: first.layer_start,
                layer_end: first.layer_end,
                max_tokens: pending.max_tokens,
                temperature: pending.temperature,
            };
            state.send_to_node(&first.node_id, msg).await;
        }
    }
}

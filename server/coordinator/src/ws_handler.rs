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
    /// Cluster generation manager — tracks infra states and rebalancing
    pub cluster_mgr: crate::serving_clusters::ClusterManager,
    /// Per-node reliability stats — feeds the Fase 2/3 scheduler
    pub node_stats: crate::node_stats::NodeStatsRegistry,
}

pub struct PendingForward {
    pub hidden_data: Vec<u8>,
    pub chain: Vec<ChainStep>,
    pub current_step: usize,
    pub token_ids: Vec<u32>,
    pub generated: Vec<u32>,
    pub max_tokens: usize,
    pub temperature: f32,
    pub top_p: Option<f32>,
    pub top_k: Option<u32>,
    pub result_tx: Option<tokio::sync::oneshot::Sender<InferenceResult>>,
    /// Active-generation in_flight counter; decremented when the request ends.
    pub in_flight_guard: Option<std::sync::Arc<std::sync::atomic::AtomicU32>>,
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
            cluster_mgr: crate::serving_clusters::ClusterManager::new(),
            node_stats: crate::node_stats::NodeStatsRegistry::new(),
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

/// Compute new layer split across all ready GPU nodes and send LayerAssignment
/// to every node that needs to change (or confirm if same range).
/// Existing nodes keep serving their old ranges until they report ready on the new gen.
pub async fn trigger_rebalance(state: &Arc<WsState>) {
    let nodes = state.nodes.read().await;
    let mut gpu_nodes: Vec<(&str, &str)> = nodes
        .values()
        .filter(|n| n.has_gpu)
        .map(|n| (n.node_id.as_str(), n.node_name.as_str()))
        .collect();
    gpu_nodes.sort_by_key(|(_, name)| *name);
    let ready: Vec<(String, String)> = gpu_nodes
        .iter()
        .map(|(id, name)| (id.to_string(), name.to_string()))
        .collect();
    drop(nodes);

    let assignments = state.cluster_mgr.rebalance(&ready).await;
    let nodes = state.nodes.read().await;
    for (node_id, start, end, skip_dl, gen) in &assignments {
        if let Some(node) = nodes.get(node_id.as_str()) {
            let msg = CoordinatorMessage::LayerAssignment {
                layer_start: *start,
                layer_end: *end,
                skip_download: *skip_dl,
                generation: *gen,
            };
            let json = serde_json::to_string(&msg).unwrap_or_default();
            let _ = node.tx.send(Message::Text(json.into()));
            info!(
                node_id = %node_id,
                name = %node.node_name,
                layers = format!("{start}-{end}"),
                skip_download = skip_dl,
                gen,
                "Sent layer assignment (rebalance)"
            );
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

/// Build inference chain from the **active generation** only.
///
/// Does not recompute ranges from the live ready set — that race used to route
/// pending (new) ranges to nodes still serving the previous generation.
/// Returns empty if the active cluster is incomplete or a slot's node is gone.
pub async fn build_inference_chain(ws_state: &WsState) -> Vec<ChainStep> {
    let slots = ws_state.cluster_mgr.active_slots().await;
    if slots.is_empty() {
        return vec![];
    }

    let nodes = ws_state.nodes.read().await;
    let mut chain = Vec::with_capacity(slots.len());
    for slot in &slots {
        let Some(node) = nodes.get(&slot.node_id) else {
            warn!(
                node_id = %slot.node_id,
                "Active slot missing from WS map — refusing chain"
            );
            return vec![];
        };
        if node.state != "ready" || !node.has_gpu {
            warn!(
                node_id = %slot.node_id,
                state = %node.state,
                "Active slot node not ready — refusing chain"
            );
            return vec![];
        }
        chain.push(ChainStep {
            node_id: slot.node_id.clone(),
            layer_start: slot.layer_start,
            layer_end: slot.layer_end,
            is_last: slot.layer_end >= 28,
        });
    }

    if !ws_state.cluster_mgr.is_operational().await {
        return vec![];
    }
    chain
}

/// Check if the inference cluster is complete.
/// Prefers active-generation assignments; falls back to live node state for UI.
pub async fn cluster_status(ws_state: &WsState) -> ClusterStatus {
    let nodes = ws_state.nodes.read().await;
    let active_slots = ws_state.cluster_mgr.active_slots().await;
    let infra = ws_state.cluster_mgr.snapshot().await;

    // Prefer active gen ranges for display; else show pending / live recompute for operators.
    let mut gpu_map: std::collections::HashMap<String, (usize, usize)> = active_slots
        .iter()
        .map(|s| (s.node_id.clone(), (s.layer_start, s.layer_end)))
        .collect();

    if gpu_map.is_empty() {
        if let Some(pending) = &infra.pending {
            for s in &pending.nodes {
                gpu_map.insert(s.node_id.clone(), (s.layer_start, s.layer_end));
            }
        } else {
            let mut gpu_nodes: Vec<&WsNode> = nodes.values().filter(|n| n.has_gpu).collect();
            gpu_nodes.sort_by_key(|n| &n.node_name);
            for (id, s, e) in assign_gpu_layers(&gpu_nodes) {
                gpu_map.insert(id, (s, e));
            }
        }
    }

    let mut node_states: Vec<NodeInferenceState> = Vec::new();
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

    let operational = ws_state.cluster_mgr.is_operational().await
        && !active_slots.is_empty()
        && active_slots.iter().all(|s| {
            nodes
                .get(&s.node_id)
                .map(|n| n.state == "ready" && n.has_gpu)
                .unwrap_or(false)
        });

    let status = if operational {
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

    // Cleanup — remove node and rebalance remaining GPU nodes
    let (was_gpu, node_name_dc) = {
        let mut nodes = state.nodes.write().await;
        let info = nodes.get(&node_id_clone).map(|n| (n.has_gpu, n.node_name.clone()));
        nodes.remove(&node_id_clone);
        info.unwrap_or((false, String::new()))
    };
    info!(node_id = %node_id_clone, name = %node_name_dc, "WebSocket client disconnected");

    // Immediately fail any in-flight request whose current hop was targeting this node.
    // Without this, those requests would silently hang until the 120s HTTP timeout.
    {
        let mut forwards = state.pending_forwards.write().await;
        let to_fail: Vec<String> = forwards
            .iter()
            .filter(|(_, p)| {
                p.current_step < p.chain.len()
                    && p.chain[p.current_step].node_id == node_id_clone
            })
            .map(|(id, _)| id.clone())
            .collect();
        for req_id in to_fail {
            if let Some(pending) = forwards.remove(&req_id) {
                warn!(
                    request_id = %req_id,
                    node_id = %node_id_clone,
                    "Failing request — node disconnected mid-hop"
                );
                release_in_flight(&pending, &state.cluster_mgr);
                drop(pending.result_tx);
            }
        }
    }

    if !node_name_dc.is_empty() {
        state.node_stats.on_disconnect(&node_name_dc).await;
    }
    if was_gpu && crate::http_api::coordinator_model_available().await {
        trigger_rebalance(&state).await;
    }
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
            state.node_stats.on_connect(&node_name).await;

            // GPU node registered: trigger rebalance across all connected GPU nodes.
            // Skip if coordinator has no model — nodes will idle until model is populated.
            if has_gpu {
                if !crate::http_api::coordinator_model_available().await {
                    info!(
                        node_id = %node_id,
                        name = %node_name,
                        "Skipping layer assignment: coordinator has no model (set HYVERK_MODEL_DIR)"
                    );
                } else {
                    trigger_rebalance(state).await;
                }
            }
        }
        ClientMessage::StateUpdate {
            state: node_state,
            detail,
        } => {
            {
                let mut nodes = state.nodes.write().await;
                if let Some(node) = nodes.get_mut(node_id) {
                    info!(node_id, name = %node.node_name, state = %node_state, detail = %detail, "Node state update");
                    node.state = node_state.clone();
                    node.state_detail = detail;
                }
            }
            match node_state.as_str() {
                "ready" => {
                    state.cluster_mgr.node_ready(node_id).await;
                }
                "reinitializing" => {
                    state.cluster_mgr.node_reinitializing(node_id).await;
                }
                _ => {}
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

/// Drop in_flight accounting for a finished/failed request.
pub fn release_in_flight(pending: &PendingForward, cluster_mgr: &crate::serving_clusters::ClusterManager) {
    if let Some(ref guard) = pending.in_flight_guard {
        crate::serving_clusters::ClusterManager::request_end(guard);
        // Fire-and-forget retirement check — cheap.
        let mgr = cluster_mgr.clone();
        tokio::spawn(async move {
            mgr.maybe_retire_draining().await;
        });
    }
}

/// Remove a pending request and drop result_tx, causing the HTTP handler to see
/// a closed channel (Ok(Err)) and return a 503 immediately — no silent 600s hang.
async fn fail_pending_request(state: &Arc<WsState>, request_id: &str, failed_node: &str) {
    let mut forwards = state.pending_forwards.write().await;
    if let Some(pending) = forwards.remove(request_id) {
        warn!(
            request_id = %request_id,
            node_id = %failed_node,
            "Failing in-flight request — node unreachable"
        );
        release_in_flight(&pending, &state.cluster_mgr);
        drop(pending.result_tx); // triggers Ok(Err) in http_api result_rx
    }
}

async fn route_forward_result(
    state: &Arc<WsState>,
    request_id: &str,
    hidden_states: Vec<u8>,
    _shape: Vec<usize>,
) {
    // Extract next-step params under the lock, then drop before sending.
    // Keeps the critical section small and avoids holding pending_forwards write
    // across network sends (which can block).
    let send_params = {
        let mut forwards = state.pending_forwards.write().await;
        let Some(pending) = forwards.get_mut(request_id) else { return };
        // Do not clone activations into `hidden_data` — it was unused and doubled RAM + memcpy
        // per hop (activations are already large: seq_len × hidden × 2 bytes).
        pending.hidden_data.clear();
        pending.current_step += 1;

        if pending.current_step < pending.chain.len() {
            let next = &pending.chain[pending.current_step];
            let (temperature, top_p, top_k) = if next.is_last {
                (Some(pending.temperature), pending.top_p, pending.top_k)
            } else {
                (None, None, None)
            };
            Some((
                next.node_id.clone(),
                next.layer_start,
                next.layer_end,
                next.is_last,
                temperature,
                top_p,
                top_k,
            ))
        } else {
            None // last node — TokenGenerated handles completion
        }
    }; // write lock released here

    let Some((next_node_id, layer_start, layer_end, is_last, temperature, top_p, top_k)) =
        send_params
    else {
        return;
    };

    let msg = CoordinatorMessage::InferenceForward {
        request_id: request_id.to_string(),
        hidden_states_ref: String::new(), // binary sent separately
        layer_start,
        layer_end,
        is_last,
        temperature,
        top_p,
        top_k,
    };

    if !state.send_to_node(&next_node_id, msg).await {
        warn!(
            request_id = %request_id,
            node_id = %next_node_id,
            "Node gone before JSON forward — failing request"
        );
        fail_pending_request(state, request_id, &next_node_id).await;
        return;
    }

    // Send binary hidden states (36-byte request_id prefix, padded — matches node client)
    let mut payload = vec![0u8; 36];
    let rid = request_id.as_bytes();
    let n = rid.len().min(36);
    payload[..n].copy_from_slice(&rid[..n]);
    payload.extend_from_slice(&hidden_states);

    if !state.send_binary_to_node(&next_node_id, payload).await {
        warn!(
            request_id = %request_id,
            node_id = %next_node_id,
            "Node gone during binary send — failing request"
        );
        fail_pending_request(state, request_id, &next_node_id).await;
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
            let in_flight_guard = pending.in_flight_guard.clone();
            forwards.remove(request_id);
            drop(forwards);

            if let Some(ref guard) = in_flight_guard {
                crate::serving_clusters::ClusterManager::request_end(guard);
                let mgr = state.cluster_mgr.clone();
                tokio::spawn(async move {
                    mgr.maybe_retire_draining().await;
                });
            }

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
            let first_node_id = first.node_id.clone();
            let msg = CoordinatorMessage::InferenceContinue {
                request_id: request_id.to_string(),
                new_token_id: token_id,
                layer_start: first.layer_start,
                layer_end: first.layer_end,
                max_tokens: pending.max_tokens,
                temperature: pending.temperature,
                top_p: pending.top_p,
                top_k: pending.top_k,
            };
            drop(forwards);
            if !state.send_to_node(&first_node_id, msg).await {
                warn!(
                    request_id = %request_id,
                    node_id = %first_node_id,
                    "First node gone during decode — failing request"
                );
                fail_pending_request(state, request_id, &first_node_id).await;
            }
            return;
        }
    }
}

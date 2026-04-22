// WebSocket server — runs on the coordinator.
// Accepts connections from clients. Sends work, receives results.
// Routes hidden states between nodes in an inference chain.

use crate::messages::{ClientMessage, CoordinatorMessage};
use std::collections::HashMap;
use std::sync::Arc;
use tokio::sync::{RwLock, mpsc};
use tracing::{info, warn};

/// A connected client
pub struct ConnectedNode {
    pub node_id: String,
    pub node_name: String,
    pub has_gpu: bool,
    pub ram_mb: u64,
    pub current_role: String,
    /// Channel to send messages TO this client
    pub tx: mpsc::UnboundedSender<CoordinatorMessage>,
}

/// Manages all WebSocket connections
pub struct WsServer {
    nodes: Arc<RwLock<HashMap<String, ConnectedNode>>>,
}

impl WsServer {
    pub fn new() -> Self {
        Self {
            nodes: Arc::new(RwLock::new(HashMap::new())),
        }
    }

    pub async fn register_node(&self, node_id: String, node: ConnectedNode) {
        info!(node_id = %node_id, name = %node.node_name, "WebSocket node registered");
        self.nodes.write().await.insert(node_id, node);
    }

    pub async fn remove_node(&self, node_id: &str) {
        self.nodes.write().await.remove(node_id);
        info!(node_id, "WebSocket node disconnected");
    }

    /// Send a message to a specific node
    pub async fn send_to(&self, node_id: &str, msg: CoordinatorMessage) -> bool {
        let nodes = self.nodes.read().await;
        if let Some(node) = nodes.get(node_id) {
            node.tx.send(msg).is_ok()
        } else {
            warn!(node_id, "Node not found for message delivery");
            false
        }
    }

    /// Send hidden states through an inference chain
    /// chain: ordered list of node_ids, each processes its layers
    pub async fn run_inference_chain(
        &self,
        chain: &[(String, usize, usize)], // (node_id, layer_start, layer_end)
        request_id: &str,
        token_ids: Vec<u32>,
        max_tokens: usize,
        temperature: f32,
    ) {
        if chain.is_empty() { return; }

        // Send InferenceStart to first node
        let (first_id, l_start, l_end) = &chain[0];
        let is_last = chain.len() == 1;
        self.send_to(first_id, CoordinatorMessage::InferenceStart {
            request_id: request_id.to_string(),
            token_ids,
            layer_start: *l_start,
            layer_end: *l_end,
            max_tokens,
            temperature: Some(temperature),
            top_p: None,
            top_k: None,
        }).await;

        // The chain continues when each node sends ForwardResult back,
        // and the coordinator forwards to the next node.
        // This is handled in the message loop (coordinator's WS handler).
    }

    pub async fn list_nodes(&self) -> Vec<(String, String, bool, u64)> {
        let nodes = self.nodes.read().await;
        nodes.values().map(|n| (
            n.node_id.clone(), n.node_name.clone(), n.has_gpu, n.ram_mb
        )).collect()
    }

    pub async fn node_count(&self) -> usize {
        self.nodes.read().await.len()
    }
}

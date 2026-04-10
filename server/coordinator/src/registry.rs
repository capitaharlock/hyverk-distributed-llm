// @llm-context: _rjj/stack.md
// @llm-depends: grpc_server.rs, router.rs

use hyverk_proto::NodeCapabilities;
use std::collections::HashMap;
use std::sync::Arc;
use tokio::sync::RwLock;
use tokio::time::Instant;
use tracing::{info, warn};
use uuid::Uuid;

#[derive(Clone, Debug)]
pub struct NodeInfo {
    pub node_id: String,
    pub node_name: String,
    pub capabilities: NodeCapabilities,
    pub active_tasks: u32,
    pub last_heartbeat: Instant,
    pub registered_at: Instant,
}

#[derive(Clone)]
pub struct NodeRegistry {
    nodes: Arc<RwLock<HashMap<String, NodeInfo>>>,
}

impl NodeRegistry {
    pub fn new() -> Self {
        Self {
            nodes: Arc::new(RwLock::new(HashMap::new())),
        }
    }

    pub async fn register(&self, node_name: String, capabilities: NodeCapabilities) -> String {
        let node_id = Uuid::new_v4().to_string();
        let now = Instant::now();
        let info = NodeInfo {
            node_id: node_id.clone(),
            node_name: node_name.clone(),
            capabilities,
            active_tasks: 0,
            last_heartbeat: now,
            registered_at: now,
        };
        // Remove any existing node with the same name (prevents duplicates on reconnect)
        let mut nodes = self.nodes.write().await;
        let old_ids: Vec<String> = nodes.iter()
            .filter(|(_, n)| n.node_name == node_name)
            .map(|(id, _)| id.clone())
            .collect();
        for old_id in &old_ids {
            nodes.remove(old_id);
            info!(old_node_id = %old_id, name = %node_name, "Removed stale registration");
        }
        nodes.insert(node_id.clone(), info);
        info!(node_id = %node_id, name = %node_name, "Node registered");
        node_id
    }

    pub async fn deregister(&self, node_id: &str) {
        if self.nodes.write().await.remove(node_id).is_some() {
            info!(node_id = %node_id, "Node deregistered");
        } else {
            warn!(node_id = %node_id, "Attempted to deregister unknown node");
        }
    }

    pub async fn heartbeat(&self, node_id: &str, active_tasks: u32) -> bool {
        let mut nodes = self.nodes.write().await;
        if let Some(node) = nodes.get_mut(node_id) {
            node.last_heartbeat = Instant::now();
            node.active_tasks = active_tasks;
            true
        } else {
            false
        }
    }

    pub async fn get_node(&self, node_id: &str) -> Option<NodeInfo> {
        self.nodes.read().await.get(node_id).cloned()
    }

    pub async fn list_nodes(&self) -> Vec<NodeInfo> {
        self.nodes.read().await.values().cloned().collect()
    }

    pub async fn remove_stale_nodes(&self, timeout: std::time::Duration) {
        let mut nodes = self.nodes.write().await;
        let before = nodes.len();
        nodes.retain(|_, info| info.last_heartbeat.elapsed() < timeout);
        let removed = before - nodes.len();
        if removed > 0 {
            warn!(removed, "Removed stale nodes");
        }
    }
}

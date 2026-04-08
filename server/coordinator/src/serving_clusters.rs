// Serving Clusters — Groups nodes into inference pipelines.
//
// A cluster = N nodes that together hold all model layers.
// Each node serves a range of layers. Requests flow through the chain:
//   Node A (layers 0-9) → Node B (layers 10-19) → Node C (layers 20-27) → token
//
// Coordinator forms clusters from available nodes, preferring low-latency groups.

use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::Arc;
use tokio::sync::RwLock;

/// A serving cluster — a group of nodes that can serve the full model together
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ServingCluster {
    pub cluster_id: String,
    pub nodes: Vec<ClusterNode>,
    pub total_layers: usize,
    pub status: ClusterStatus,
    pub region: String,
    pub created_at: u64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ClusterNode {
    pub node_id: String,
    pub node_name: String,
    pub layer_start: usize,
    pub layer_end: usize,
    pub serve_port: u16,
    pub position: NodePosition,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum NodePosition { First, Middle, Last }

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum ClusterStatus { Forming, Ready, Serving, Degraded }

#[derive(Clone)]
pub struct ClusterManager {
    inner: Arc<RwLock<Inner>>,
}

struct Inner {
    clusters: Vec<ServingCluster>,
    /// node_id → cluster_id mapping
    node_assignments: HashMap<String, String>,
}

impl ClusterManager {
    pub fn new() -> Self {
        Self {
            inner: Arc::new(RwLock::new(Inner {
                clusters: Vec::new(),
                node_assignments: HashMap::new(),
            })),
        }
    }

    /// Form a new cluster from available nodes
    pub async fn form_cluster(
        &self,
        nodes: Vec<(String, String, usize)>, // (node_id, node_name, available_ram_mb)
        total_layers: usize,
        region: &str,
    ) -> Option<String> {
        if nodes.is_empty() { return None; }

        let cluster_id = uuid::Uuid::new_v4().to_string();
        let layers_per_node = total_layers / nodes.len();
        let mut cluster_nodes = Vec::new();
        let mut layer = 0;

        for (i, (node_id, node_name, _ram)) in nodes.iter().enumerate() {
            let end = if i == nodes.len() - 1 { total_layers } else { layer + layers_per_node };
            let position = if i == 0 { NodePosition::First }
                else if i == nodes.len() - 1 { NodePosition::Last }
                else { NodePosition::Middle };

            cluster_nodes.push(ClusterNode {
                node_id: node_id.clone(),
                node_name: node_name.clone(),
                layer_start: layer,
                layer_end: end,
                serve_port: 18000 + i as u16,
                position,
            });
            layer = end;
        }

        let now = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default()
            .as_secs();

        let cluster = ServingCluster {
            cluster_id: cluster_id.clone(),
            nodes: cluster_nodes,
            total_layers,
            status: ClusterStatus::Forming,
            region: region.to_string(),
            created_at: now,
        };

        let mut inner = self.inner.write().await;
        for node in &cluster.nodes {
            inner.node_assignments.insert(node.node_id.clone(), cluster_id.clone());
        }
        inner.clusters.push(cluster);

        tracing::info!(cluster_id = %cluster_id, region, nodes = nodes.len(), "Cluster formed");
        Some(cluster_id)
    }

    /// Get a ready cluster for serving an inference request
    pub async fn get_serving_cluster(&self) -> Option<ServingCluster> {
        let inner = self.inner.read().await;
        inner.clusters.iter()
            .find(|c| c.status == ClusterStatus::Ready || c.status == ClusterStatus::Serving)
            .cloned()
    }

    pub async fn list_clusters(&self) -> Vec<ServingCluster> {
        self.inner.read().await.clusters.clone()
    }

    pub async fn get_node_assignment(&self, node_id: &str) -> Option<(String, ClusterNode)> {
        let inner = self.inner.read().await;
        if let Some(cluster_id) = inner.node_assignments.get(node_id) {
            if let Some(cluster) = inner.clusters.iter().find(|c| c.cluster_id == *cluster_id) {
                if let Some(node) = cluster.nodes.iter().find(|n| n.node_id == node_id) {
                    return Some((cluster_id.clone(), node.clone()));
                }
            }
        }
        None
    }
}

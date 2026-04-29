// Cluster generation lifecycle — parallel infra management.
//
// States per generation:
//   Forming  — nodes receiving new layer assignments, not yet serving
//   Active   — serving inference requests (in_flight tracked)
//   Draining — new generation took over; wait for in_flight → 0 then retire
//
// Rebalance strategy:
//   All nodes own the full model (via local symlink), so reassignment is purely
//   logical — no file download, just restart the Python inference server with
//   the new layer range. Coordinator sends LayerAssignment{skip_download:true}.
//
// Infrastructure states visible to operators:
//   Available     — node connected, not yet in any cluster (just joined)
//   InCluster     — node actively serving its assigned range
//   Reinitializing — node received new range, restarting inference server
//
// This module is independent of training and synthesis.

use serde::Serialize;
use std::collections::HashMap;
use std::sync::{
    atomic::{AtomicU32, Ordering},
    Arc,
};
use tokio::sync::RwLock;
use tracing::{info, warn};

const TOTAL_LAYERS: usize = 28;

// ── Public types ─────────────────────────────────────────────────────────────

#[derive(Debug, Clone, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum InfraState {
    Available,
    InCluster,
    Reinitializing,
}

#[derive(Debug, Clone, Serialize)]
pub struct NodeSlot {
    pub node_id: String,
    pub node_name: String,
    pub layer_start: usize,
    pub layer_end: usize,
    pub infra_state: InfraState,
}

#[derive(Debug, Serialize)]
pub struct ClusterSnapshot {
    pub generation: u64,
    pub state: &'static str,
    pub in_flight: u32,
    pub nodes: Vec<NodeSlot>,
}

#[derive(Debug, Serialize)]
pub struct InfraSnapshot {
    pub active: Option<ClusterSnapshot>,
    pub pending: Option<ClusterSnapshot>,
}

// ── Internal types ────────────────────────────────────────────────────────────

struct Generation {
    gen: u64,
    slots: Vec<NodeSlot>,
    in_flight: Arc<AtomicU32>,
    state: GenState,
}

#[derive(PartialEq)]
enum GenState {
    Forming,
    Active,
    Draining,
}

// ── ClusterManager ────────────────────────────────────────────────────────────

#[derive(Clone)]
pub struct ClusterManager {
    inner: Arc<RwLock<Inner>>,
}

struct Inner {
    next_gen: u64,
    active: Option<Generation>,
    pending: Option<Generation>,
}

impl ClusterManager {
    pub fn new() -> Self {
        Self {
            inner: Arc::new(RwLock::new(Inner {
                next_gen: 1,
                active: None,
                pending: None,
            })),
        }
    }

    /// Compute new layer assignments for `ready_nodes` and update cluster state.
    ///
    /// Returns `(node_id, layer_start, layer_end, skip_download, generation)` for every
    /// node that needs a LayerAssignment message.  Nodes that keep the same range and
    /// are already InCluster get `skip_download=true` so they confirm ready without
    /// restarting.
    pub async fn rebalance(
        &self,
        ready_nodes: &[(String, String)], // (node_id, node_name) sorted for determinism
    ) -> Vec<(String, usize, usize, bool, u64)> {
        if ready_nodes.is_empty() {
            return vec![];
        }

        let mut inner = self.inner.write().await;
        let gen = inner.next_gen;
        inner.next_gen += 1;

        // Compute even split across N nodes
        let n = ready_nodes.len();
        let per_node = TOTAL_LAYERS / n;
        let remainder = TOTAL_LAYERS % n;

        let mut new_slots: Vec<NodeSlot> = Vec::with_capacity(n);
        let mut start = 0;
        for (i, (node_id, node_name)) in ready_nodes.iter().enumerate() {
            let extra = if i < remainder { 1 } else { 0 };
            let end = start + per_node + extra;
            new_slots.push(NodeSlot {
                node_id: node_id.clone(),
                node_name: node_name.clone(),
                layer_start: start,
                layer_end: end,
                infra_state: InfraState::Reinitializing,
            });
            start = end;
        }

        // Determine which nodes keep their range vs need to change.
        // Build lookup from the CURRENT active gen (if any).
        let current_assignments: HashMap<String, (usize, usize)> = inner
            .active
            .as_ref()
            .map(|g| {
                g.slots
                    .iter()
                    .map(|s| (s.node_id.clone(), (s.layer_start, s.layer_end)))
                    .collect()
            })
            .unwrap_or_default();

        let mut assignments_to_send = Vec::new();
        for slot in &new_slots {
            let current_range = current_assignments.get(&slot.node_id);
            let same_range = current_range == Some(&(slot.layer_start, slot.layer_end));
            // skip_download=true: files already present (all nodes have full model via symlink)
            // For same range AND already active: node can just confirm ready quickly.
            assignments_to_send.push((
                slot.node_id.clone(),
                slot.layer_start,
                slot.layer_end,
                true, // always skip download — files present via symlink
                gen,
            ));
            if same_range {
                info!(
                    node = %slot.node_name,
                    layers = format!("{}-{}", slot.layer_start, slot.layer_end),
                    gen,
                    "Node keeps same range — quick ready"
                );
            }
        }

        // Drop any previous pending generation — superseded
        if inner.pending.is_some() {
            warn!(gen, "Superseding previous pending generation");
        }

        inner.pending = Some(Generation {
            gen,
            slots: new_slots,
            in_flight: Arc::new(AtomicU32::new(0)),
            state: GenState::Forming,
        });

        info!(
            gen,
            nodes = n,
            "Cluster rebalance scheduled — pending generation forming"
        );

        assignments_to_send
    }

    /// Called when a node reports StateUpdate{ready} after receiving a LayerAssignment.
    /// Marks the node as InCluster in whichever pending generation it belongs to.
    /// If all pending nodes are now ready, promotes pending → Active.
    /// Returns the old generation's in_flight counter (for drain watching) if a promotion happened.
    pub async fn node_ready(&self, node_id: &str) -> Option<Arc<AtomicU32>> {
        let mut inner = self.inner.write().await;

        let node_in_pending = inner
            .pending
            .as_ref()
            .map_or(false, |g| g.slots.iter().any(|s| s.node_id == node_id));

        if !node_in_pending {
            return None;
        }

        let pending = inner.pending.as_mut().unwrap();
        for slot in &mut pending.slots {
            if slot.node_id == node_id {
                slot.infra_state = InfraState::InCluster;
                break;
            }
        }

        let all_ready = pending
            .slots
            .iter()
            .all(|s| s.infra_state == InfraState::InCluster);

        if !all_ready {
            return None;
        }

        // Promote pending → Active
        let new_gen = inner.pending.take().unwrap();
        let promoted_gen = new_gen.gen;
        let old_gen = inner.active.replace(new_gen);
        inner.active.as_mut().unwrap().state = GenState::Active;

        info!(gen = promoted_gen, "All nodes ready — generation promoted to Active");

        if let Some(old) = old_gen {
            let counter = old.in_flight.clone();
            drop(old); // if in_flight already 0, this retires the gen immediately
            return Some(counter);
        }
        None
    }

    /// Call before routing an inference request. Returns a guard counter to decrement when done.
    /// Returns None if no active cluster.
    pub async fn request_start(&self) -> Option<Arc<AtomicU32>> {
        let inner = self.inner.read().await;
        inner.active.as_ref().map(|g| {
            g.in_flight.fetch_add(1, Ordering::Relaxed);
            g.in_flight.clone()
        })
    }

    /// Call when a request finishes (success or error).
    pub fn request_end(counter: &Arc<AtomicU32>) {
        counter.fetch_sub(1, Ordering::Relaxed);
    }

    /// Returns true if there is an active cluster with all layers covered (0-28).
    pub async fn is_operational(&self) -> bool {
        let inner = self.inner.read().await;
        inner.active.as_ref().map_or(false, |g| {
            g.state == GenState::Active
                && !g.slots.is_empty()
                && g.slots.iter().all(|s| s.infra_state == InfraState::InCluster)
        })
    }

    /// Snapshot for the HTTP API / dashboard.
    pub async fn snapshot(&self) -> InfraSnapshot {
        let inner = self.inner.read().await;
        InfraSnapshot {
            active: inner.active.as_ref().map(|g| ClusterSnapshot {
                generation: g.gen,
                state: match g.state {
                    GenState::Forming => "forming",
                    GenState::Active => "active",
                    GenState::Draining => "draining",
                },
                in_flight: g.in_flight.load(Ordering::Relaxed),
                nodes: g.slots.clone(),
            }),
            pending: inner.pending.as_ref().map(|g| ClusterSnapshot {
                generation: g.gen,
                state: "forming",
                in_flight: 0,
                nodes: g.slots.clone(),
            }),
        }
    }

    /// Update infra state for a node (e.g., when it sends StateUpdate{reinitializing}).
    pub async fn node_reinitializing(&self, node_id: &str) {
        let mut inner = self.inner.write().await;
        // Check pending first, then active (can't hold two &mut borrows at once)
        let found = inner.pending.as_mut().map_or(false, |g| {
            g.slots.iter_mut().any(|s| {
                if s.node_id == node_id {
                    s.infra_state = InfraState::Reinitializing;
                    true
                } else {
                    false
                }
            })
        });
        if !found {
            if let Some(g) = inner.active.as_mut() {
                for slot in &mut g.slots {
                    if slot.node_id == node_id {
                        slot.infra_state = InfraState::Reinitializing;
                        break;
                    }
                }
            }
        }
    }
}

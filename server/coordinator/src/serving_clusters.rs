// Cluster generation lifecycle — parallel infra management.
//
// States per generation:
//   Forming  — nodes receiving new layer assignments, not yet serving
//   Active   — serving inference requests (in_flight tracked)
//   Draining — new generation took over; wait for in_flight → 0 then retire
//
// Rebalance strategy:
//   New ranges download (or reuse local cache). Same-range InCluster nodes get
//   skip_download=true and confirm ready without restarting the Python server.
//
// Infrastructure states visible to operators:
//   Available      — node connected, not yet in any cluster (just joined)
//   InCluster      — node actively serving its assigned range
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
    pub draining: Option<ClusterSnapshot>,
}

/// Immutable layer assignment for one hop in an active generation.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ActiveSlot {
    pub node_id: String,
    pub node_name: String,
    pub layer_start: usize,
    pub layer_end: usize,
    pub generation: u64,
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
    draining: Option<Generation>,
}

impl ClusterManager {
    pub fn new() -> Self {
        Self {
            inner: Arc::new(RwLock::new(Inner {
                next_gen: 1,
                active: None,
                pending: None,
                draining: None,
            })),
        }
    }

    /// Compute new layer assignments for `gpu_nodes` and update cluster state.
    ///
    /// Returns `(node_id, layer_start, layer_end, skip_download, generation)` for every
    /// node that needs a LayerAssignment message.
    ///
    /// `skip_download` is true only when the node already holds the same range in the
    /// active generation and is InCluster (quick ready confirm, no restart).
    pub async fn rebalance(
        &self,
        gpu_nodes: &[(String, String)], // (node_id, node_name) sorted for determinism
    ) -> Vec<(String, usize, usize, bool, u64)> {
        if gpu_nodes.is_empty() {
            let mut inner = self.inner.write().await;
            // No GPU nodes left — clear pending/active (drain in-flight via existing requests).
            if let Some(old) = inner.active.take() {
                if old.in_flight.load(Ordering::Relaxed) > 0 {
                    let mut draining = old;
                    draining.state = GenState::Draining;
                    inner.draining = Some(draining);
                }
            }
            inner.pending = None;
            return vec![];
        }

        let mut inner = self.inner.write().await;
        let gen = inner.next_gen;
        inner.next_gen += 1;

        let n = gpu_nodes.len();
        let per_node = TOTAL_LAYERS / n;
        let remainder = TOTAL_LAYERS % n;

        let mut new_slots: Vec<NodeSlot> = Vec::with_capacity(n);
        let mut start = 0;
        for (i, (node_id, node_name)) in gpu_nodes.iter().enumerate() {
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

        let current_assignments: HashMap<String, (usize, usize, InfraState)> = inner
            .active
            .as_ref()
            .map(|g| {
                g.slots
                    .iter()
                    .map(|s| {
                        (
                            s.node_id.clone(),
                            (s.layer_start, s.layer_end, s.infra_state.clone()),
                        )
                    })
                    .collect()
            })
            .unwrap_or_default();

        let mut assignments_to_send = Vec::new();
        for slot in &mut new_slots {
            let (same_range, was_in_cluster) =
                match current_assignments.get(&slot.node_id) {
                    Some(&(ls, le, ref st)) => {
                        (ls == slot.layer_start && le == slot.layer_end, *st == InfraState::InCluster)
                    }
                    None => (false, false),
                };
            // Quick path only when the active gen already serves this exact range.
            let skip_download = same_range && was_in_cluster;
            if skip_download {
                slot.infra_state = InfraState::InCluster;
                info!(
                    node = %slot.node_name,
                    layers = format!("{}-{}", slot.layer_start, slot.layer_end),
                    gen,
                    "Node keeps same range — quick ready"
                );
            }
            assignments_to_send.push((
                slot.node_id.clone(),
                slot.layer_start,
                slot.layer_end,
                skip_download,
                gen,
            ));
        }

        if inner.pending.is_some() {
            warn!(gen, "Superseding previous pending generation");
        }

        // If every slot is already InCluster (all same-range), promote immediately.
        let all_ready = new_slots
            .iter()
            .all(|s| s.infra_state == InfraState::InCluster);

        if all_ready {
            let new_gen = Generation {
                gen,
                slots: new_slots,
                in_flight: Arc::new(AtomicU32::new(0)),
                state: GenState::Active,
            };
            if let Some(old) = inner.active.replace(new_gen) {
                retire_or_drain(&mut inner.draining, old);
            }
            inner.pending = None;
            info!(gen, nodes = n, "Cluster rebalance — same topology, generation active");
        } else {
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
        }

        assignments_to_send
    }

    /// Called when a node reports StateUpdate{ready} after receiving a LayerAssignment.
    /// Marks the node as InCluster in the pending generation. When all pending slots are
    /// ready, promotes pending → Active and moves the previous Active into Draining.
    pub async fn node_ready(&self, node_id: &str) -> bool {
        let mut inner = self.inner.write().await;

        let node_in_pending = inner
            .pending
            .as_ref()
            .map_or(false, |g| g.slots.iter().any(|s| s.node_id == node_id));

        if !node_in_pending {
            return false;
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
            return false;
        }

        let new_gen = inner.pending.take().unwrap();
        let promoted_gen = new_gen.gen;
        let mut promoted = new_gen;
        promoted.state = GenState::Active;

        if let Some(old) = inner.active.replace(promoted) {
            retire_or_drain(&mut inner.draining, old);
        }

        info!(gen = promoted_gen, "All nodes ready — generation promoted to Active");
        true
    }

    /// Slots from the Active generation, ordered by layer_start.
    /// Empty when no active cluster exists.
    pub async fn active_slots(&self) -> Vec<ActiveSlot> {
        let inner = self.inner.read().await;
        let Some(g) = inner.active.as_ref() else {
            return vec![];
        };
        if g.state != GenState::Active {
            return vec![];
        }
        let mut slots: Vec<ActiveSlot> = g
            .slots
            .iter()
            .filter(|s| s.infra_state == InfraState::InCluster)
            .map(|s| ActiveSlot {
                node_id: s.node_id.clone(),
                node_name: s.node_name.clone(),
                layer_start: s.layer_start,
                layer_end: s.layer_end,
                generation: g.gen,
            })
            .collect();
        slots.sort_by_key(|s| s.layer_start);
        slots
    }

    /// Call before routing an inference request. Returns a guard counter to decrement when done.
    /// Returns None if no active cluster.
    pub async fn request_start(&self) -> Option<Arc<AtomicU32>> {
        let inner = self.inner.read().await;
        let g = inner.active.as_ref()?;
        if g.state != GenState::Active {
            return None;
        }
        g.in_flight.fetch_add(1, Ordering::Relaxed);
        Some(g.in_flight.clone())
    }

    /// Call when a request finishes (success or error).
    pub fn request_end(counter: &Arc<AtomicU32>) {
        let prev = counter.fetch_sub(1, Ordering::Relaxed);
        if prev == 0 {
            // Underflow protection — should not happen.
            counter.fetch_add(1, Ordering::Relaxed);
            warn!("request_end called with in_flight already 0");
        }
    }

    /// Drop draining generation once its in-flight counter hits zero.
    pub async fn maybe_retire_draining(&self) {
        let mut inner = self.inner.write().await;
        if let Some(g) = inner.draining.as_ref() {
            if g.in_flight.load(Ordering::Relaxed) == 0 {
                let gen = g.gen;
                inner.draining = None;
                info!(gen, "Draining generation retired");
            }
        }
    }

    /// Returns true if there is an active cluster with InCluster slots covering 0..28.
    pub async fn is_operational(&self) -> bool {
        let slots = self.active_slots().await;
        if slots.is_empty() {
            return false;
        }
        let starts = slots.first().map(|s| s.layer_start) == Some(0);
        let ends = slots.last().map(|s| s.layer_end) >= Some(TOTAL_LAYERS);
        let mut cursor = 0usize;
        for s in &slots {
            if s.layer_start > cursor {
                return false;
            }
            cursor = cursor.max(s.layer_end);
        }
        starts && ends && cursor >= TOTAL_LAYERS
    }

    /// Snapshot for the HTTP API / dashboard.
    pub async fn snapshot(&self) -> InfraSnapshot {
        let inner = self.inner.read().await;
        InfraSnapshot {
            active: inner.active.as_ref().map(snapshot_gen),
            pending: inner.pending.as_ref().map(snapshot_gen),
            draining: inner.draining.as_ref().map(snapshot_gen),
        }
    }

    /// Update infra state for a node (e.g., when it sends StateUpdate{reinitializing}).
    pub async fn node_reinitializing(&self, node_id: &str) {
        let mut inner = self.inner.write().await;
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

fn snapshot_gen(g: &Generation) -> ClusterSnapshot {
    ClusterSnapshot {
        generation: g.gen,
        state: match g.state {
            GenState::Forming => "forming",
            GenState::Active => "active",
            GenState::Draining => "draining",
        },
        in_flight: g.in_flight.load(Ordering::Relaxed),
        nodes: g.slots.clone(),
    }
}

fn retire_or_drain(draining_slot: &mut Option<Generation>, mut old: Generation) {
    if old.in_flight.load(Ordering::Relaxed) == 0 {
        info!(gen = old.gen, "Previous generation retired (no in-flight)");
        return;
    }
    old.state = GenState::Draining;
    info!(
        gen = old.gen,
        in_flight = old.in_flight.load(Ordering::Relaxed),
        "Previous generation draining"
    );
    *draining_slot = Some(old);
}

#[cfg(test)]
mod tests {
    use super::*;

    fn nodes(names: &[&str]) -> Vec<(String, String)> {
        names
            .iter()
            .enumerate()
            .map(|(i, n)| (format!("id-{i}"), (*n).to_string()))
            .collect()
    }

    #[tokio::test]
    async fn rebalance_three_nodes_even_split() {
        let mgr = ClusterManager::new();
        let n = nodes(&["a", "b", "c"]);
        let assigns = mgr.rebalance(&n).await;
        assert_eq!(assigns.len(), 3);
        // 28/3 → 10,9,9 or 10,10,8 depending on remainder: 9+9+10
        let ranges: Vec<(usize, usize)> = assigns
            .iter()
            .map(|(_, s, e, _, _)| (*s, *e))
            .collect();
        assert_eq!(ranges[0], (0, 10));
        assert_eq!(ranges[1], (10, 19));
        assert_eq!(ranges[2], (19, 28));
        // First join: never skip download
        assert!(assigns.iter().all(|(_, _, _, skip, _)| !*skip));
        assert!(!mgr.is_operational().await);
    }

    #[tokio::test]
    async fn promote_then_route_uses_active_slots() {
        let mgr = ClusterManager::new();
        let n = nodes(&["a", "b", "c"]);
        let assigns = mgr.rebalance(&n).await;
        for (id, _, _, _, _) in &assigns {
            mgr.node_ready(id).await;
        }
        assert!(mgr.is_operational().await);
        let slots = mgr.active_slots().await;
        assert_eq!(slots.len(), 3);
        assert_eq!(slots[0].layer_start, 0);
        assert_eq!(slots[2].layer_end, 28);
    }

    #[tokio::test]
    async fn join_fourth_keeps_old_active_until_pending_ready() {
        let mgr = ClusterManager::new();
        let three = nodes(&["a", "b", "c"]);
        let assigns = mgr.rebalance(&three).await;
        for (id, _, _, _, _) in &assigns {
            mgr.node_ready(id).await;
        }
        assert!(mgr.is_operational().await);
        let old_slots = mgr.active_slots().await;
        assert_eq!(old_slots.len(), 3);

        let four = nodes(&["a", "b", "c", "d"]);
        let assigns2 = mgr.rebalance(&four).await;
        assert_eq!(assigns2.len(), 4);
        // During pending form, active still has 3-node topology
        let mid = mgr.active_slots().await;
        assert_eq!(mid.len(), 3);
        assert_eq!(mid[0].layer_end, old_slots[0].layer_end);

        // Complete pending
        for (id, _, _, _, _) in &assigns2 {
            mgr.node_ready(id).await;
        }
        let new_slots = mgr.active_slots().await;
        assert_eq!(new_slots.len(), 4);
        assert_eq!(new_slots[0].layer_start, 0);
        assert_eq!(new_slots[3].layer_end, 28);
    }

    #[tokio::test]
    async fn same_topology_skip_download_and_stays_active() {
        let mgr = ClusterManager::new();
        let n = nodes(&["a", "b"]);
        let a1 = mgr.rebalance(&n).await;
        for (id, _, _, _, _) in &a1 {
            mgr.node_ready(id).await;
        }
        assert!(mgr.is_operational().await);

        let a2 = mgr.rebalance(&n).await;
        assert!(a2.iter().all(|(_, _, _, skip, _)| *skip));
        assert!(mgr.is_operational().await);
        let snap = mgr.snapshot().await;
        assert!(snap.pending.is_none());
        assert!(snap.active.is_some());
    }

    #[tokio::test]
    async fn request_start_end_tracks_in_flight() {
        let mgr = ClusterManager::new();
        let n = nodes(&["a"]);
        let a = mgr.rebalance(&n).await;
        mgr.node_ready(&a[0].0).await;
        let g = mgr.request_start().await.expect("active");
        let snap = mgr.snapshot().await;
        assert_eq!(snap.active.unwrap().in_flight, 1);
        ClusterManager::request_end(&g);
        let snap = mgr.snapshot().await;
        assert_eq!(snap.active.unwrap().in_flight, 0);
    }

    #[tokio::test]
    async fn drain_keeps_counter_until_zero() {
        let mgr = ClusterManager::new();
        let n = nodes(&["a", "b"]);
        let a = mgr.rebalance(&n).await;
        for (id, _, _, _, _) in &a {
            mgr.node_ready(id).await;
        }
        let guard = mgr.request_start().await.unwrap();

        // Force new topology so old gen drains with in_flight=1
        let three = nodes(&["a", "b", "c"]);
        let a2 = mgr.rebalance(&three).await;
        for (id, _, _, _, _) in &a2 {
            mgr.node_ready(id).await;
        }
        let snap = mgr.snapshot().await;
        assert!(snap.draining.is_some());
        assert_eq!(snap.draining.unwrap().in_flight, 1);

        ClusterManager::request_end(&guard);
        mgr.maybe_retire_draining().await;
        let snap = mgr.snapshot().await;
        assert!(snap.draining.is_none());
    }
}

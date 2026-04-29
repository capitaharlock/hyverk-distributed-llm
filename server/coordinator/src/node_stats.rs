// Node reliability statistics — Fase 1 del cluster scheduler.
//
// Tracks per-node connection history so the scheduler can score nodes
// by reliability before assigning them to an active cluster.
//
// Reliability score formula (0.0–1.0):
//   score = uptime_fraction × smoothing_factor
//   where uptime_fraction = total_uptime_secs / total_observed_secs
//   and   smoothing_factor = 1 / (1 + disconnect_count_24h × 0.1)
//
// A node with score < 0.5 should be demoted to backup or excluded.
// A node with score > 0.9 for 24h+ is a candidate for the active chain.
//
// This module is read by serving_clusters.rs (Fase 2) and scheduler.rs (Fase 3).
// It does NOT make scheduling decisions — only records facts.

use std::collections::HashMap;
use std::sync::Arc;
use std::time::{Duration, Instant};
use tokio::sync::RwLock;

const DISCONNECT_PENALTY_WINDOW_SECS: u64 = 86_400; // 24h rolling window

#[derive(Debug, Clone)]
pub struct NodeStats {
    pub node_name: String,
    /// When this node first connected (ever)
    pub first_seen: Instant,
    /// When this connection session started
    pub connected_at: Instant,
    /// Accumulated uptime across all sessions (seconds)
    pub total_uptime_secs: u64,
    /// Total observed time since first_seen (seconds) — uptime + downtime
    pub total_observed_secs: u64,
    /// Disconnect events with timestamp (rolling 24h kept)
    pub disconnect_events: Vec<Instant>,
    /// Exponential moving average of forward-hop latency (ms), updated on each hop
    pub avg_latency_ms: f64,
    /// Number of successful inference hops
    pub hops_ok: u64,
    /// Number of failed inference hops (timeout / error)
    pub hops_failed: u64,
}

impl NodeStats {
    fn new(node_name: String) -> Self {
        let now = Instant::now();
        Self {
            node_name,
            first_seen: now,
            connected_at: now,
            total_uptime_secs: 0,
            total_observed_secs: 0,
            disconnect_events: Vec::new(),
            avg_latency_ms: 0.0,
            hops_ok: 0,
            hops_failed: 0,
        }
    }

    /// Reliability score in [0.0, 1.0]. Higher = more trustworthy.
    pub fn reliability(&self) -> f64 {
        let total = self.total_observed_secs.max(1) as f64;
        let uptime = (self.total_uptime_secs + self.current_session_secs()) as f64;
        let uptime_frac = (uptime / total).min(1.0);

        let recent_disconnects = self.disconnect_count_24h() as f64;
        let smoothing = 1.0 / (1.0 + recent_disconnects * 0.15);

        let hop_success = if self.hops_ok + self.hops_failed > 0 {
            self.hops_ok as f64 / (self.hops_ok + self.hops_failed) as f64
        } else {
            1.0 // no data → assume ok
        };

        (uptime_frac * smoothing * hop_success).clamp(0.0, 1.0)
    }

    fn current_session_secs(&self) -> u64 {
        self.connected_at.elapsed().as_secs()
    }

    fn disconnect_count_24h(&self) -> usize {
        let cutoff = Instant::now()
            .checked_sub(Duration::from_secs(DISCONNECT_PENALTY_WINDOW_SECS))
            .unwrap_or(Instant::now());
        self.disconnect_events
            .iter()
            .filter(|&&t| t > cutoff)
            .count()
    }

    fn record_disconnect(&mut self) {
        let now = Instant::now();
        // Accumulate session uptime
        self.total_uptime_secs += self.current_session_secs();
        self.total_observed_secs = self.first_seen.elapsed().as_secs();
        self.disconnect_events.push(now);
        // Prune events older than 48h to bound memory
        let cutoff = now
            .checked_sub(Duration::from_secs(DISCONNECT_PENALTY_WINDOW_SECS * 2))
            .unwrap_or(now);
        self.disconnect_events.retain(|&t| t > cutoff);
    }

    fn record_reconnect(&mut self) {
        self.connected_at = Instant::now();
        self.total_observed_secs = self.first_seen.elapsed().as_secs();
    }

    /// Update latency EMA: α=0.1 (slow-moving average, resistant to spikes)
    fn record_hop_latency(&mut self, latency_ms: f64, success: bool) {
        const ALPHA: f64 = 0.1;
        if self.avg_latency_ms == 0.0 {
            self.avg_latency_ms = latency_ms;
        } else {
            self.avg_latency_ms = ALPHA * latency_ms + (1.0 - ALPHA) * self.avg_latency_ms;
        }
        if success {
            self.hops_ok += 1;
        } else {
            self.hops_failed += 1;
        }
    }
}

// ── NodeStatsRegistry ─────────────────────────────────────────────────────────

#[derive(Clone)]
pub struct NodeStatsRegistry {
    stats: Arc<RwLock<HashMap<String, NodeStats>>>,
}

impl NodeStatsRegistry {
    pub fn new() -> Self {
        Self {
            stats: Arc::new(RwLock::new(HashMap::new())),
        }
    }

    /// Call when a node connects (or reconnects).
    pub async fn on_connect(&self, node_name: &str) {
        let mut map = self.stats.write().await;
        if let Some(s) = map.get_mut(node_name) {
            s.record_reconnect();
        } else {
            map.insert(node_name.to_string(), NodeStats::new(node_name.to_string()));
        }
    }

    /// Call when a node disconnects.
    pub async fn on_disconnect(&self, node_name: &str) {
        let mut map = self.stats.write().await;
        if let Some(s) = map.get_mut(node_name) {
            s.record_disconnect();
        }
    }

    /// Call after each inference hop completes or times out.
    pub async fn record_hop(&self, node_name: &str, latency_ms: f64, success: bool) {
        let mut map = self.stats.write().await;
        if let Some(s) = map.get_mut(node_name) {
            s.record_hop_latency(latency_ms, success);
        }
    }

    /// Get reliability score for a node (0.0–1.0). Returns 0.5 if unknown.
    pub async fn reliability(&self, node_name: &str) -> f64 {
        self.stats
            .read()
            .await
            .get(node_name)
            .map_or(0.5, |s| s.reliability())
    }

    /// Get all stats sorted by reliability descending.
    /// Used by Fase 2 scheduler to rank nodes for cluster assignment.
    pub async fn ranked(&self) -> Vec<(String, f64, f64)> {
        let map = self.stats.read().await;
        let mut v: Vec<(String, f64, f64)> = map
            .values()
            .map(|s| (s.node_name.clone(), s.reliability(), s.avg_latency_ms))
            .collect();
        v.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        v
    }

    /// Snapshot for the HTTP API / dashboard.
    pub async fn snapshot(&self) -> Vec<serde_json::Value> {
        let map = self.stats.read().await;
        let mut v: Vec<_> = map.values().collect();
        v.sort_by(|a, b| b.reliability().partial_cmp(&a.reliability()).unwrap_or(std::cmp::Ordering::Equal));
        v.iter().map(|s| serde_json::json!({
            "node_name": s.node_name,
            "reliability": (s.reliability() * 1000.0).round() / 1000.0,
            "uptime_secs": s.total_uptime_secs + s.current_session_secs(),
            "disconnect_count_24h": s.disconnect_count_24h(),
            "avg_latency_ms": (s.avg_latency_ms * 10.0).round() / 10.0,
            "hops_ok": s.hops_ok,
            "hops_failed": s.hops_failed,
        })).collect()
    }
}

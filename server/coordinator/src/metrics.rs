// Phase 2.5: Network Operations Dashboard — metrics aggregation.
// Reads from registry, dataset_store, training_store and returns a single snapshot.
// Called by GET /api/v1/metrics every ~5s from the Electron dashboard.

use crate::http_api::AppState;
use crate::training_store::JobStatus;
use serde::Serialize;
use std::collections::HashMap;
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::{SystemTime, UNIX_EPOCH};

/// Global atomic counters for live network activity.
/// Incremented from gRPC handlers, HTTP handlers, etc.
pub struct LiveCounters {
    pub grpc_requests: AtomicU64,
    pub http_requests: AtomicU64,
    pub heartbeats: AtomicU64,
    pub bytes_in: AtomicU64,
    pub bytes_out: AtomicU64,
    pub inference_requests: AtomicU64,
    pub inference_completed: AtomicU64,
    pub tensor_ops: AtomicU64,
}

impl LiveCounters {
    pub fn new() -> Self {
        Self {
            grpc_requests: AtomicU64::new(0),
            http_requests: AtomicU64::new(0),
            heartbeats: AtomicU64::new(0),
            bytes_in: AtomicU64::new(0),
            bytes_out: AtomicU64::new(0),
            inference_requests: AtomicU64::new(0),
            inference_completed: AtomicU64::new(0),
            tensor_ops: AtomicU64::new(0),
        }
    }
}

#[derive(Serialize)]
pub struct MetricsSnapshot {
    pub nodes: NodeMetrics,
    pub dataset: DatasetMetrics,
    pub training: TrainingMetrics,
    pub network: NetworkMetrics,
    pub uptime_secs: u64,
    pub timestamp_secs: u64,
}

#[derive(Serialize)]
pub struct NetworkMetrics {
    pub grpc_requests: u64,
    pub http_requests: u64,
    pub heartbeats: u64,
    pub bytes_in: u64,
    pub bytes_out: u64,
    pub inference_requests: u64,
    pub inference_completed: u64,
    pub tensor_ops: u64,
}

#[derive(Serialize)]
pub struct NodeMetrics {
    /// Currently registered (all stale nodes are reaped by the background reaper)
    pub total: usize,
    pub active_tasks_total: u32,
    /// Deduplicated list of all available model names across nodes
    pub models: Vec<String>,
}

#[derive(Serialize)]
pub struct DatasetMetrics {
    pub total_items: usize,
    /// Rough estimate: avg ~500 bytes per (instruction, response) pair
    pub size_bytes_est: u64,
    pub execution_verified: usize,
    /// 0.0–100.0
    pub verified_pct: f32,
    pub refined: usize,
    pub deduplicated: usize,
    pub by_category: HashMap<String, usize>,
    pub by_provider: HashMap<String, usize>,
    /// Top-5 contributing nodes [(node_id, count)]
    pub top_nodes: Vec<(String, usize)>,
}

#[derive(Serialize)]
pub struct TrainingMetrics {
    pub total_jobs: usize,
    pub active_jobs: usize,
    pub shards_total: usize,
    pub shards_done: usize,
    pub shards_in_flight: usize,
    pub shards_available: usize,
    /// Average training loss across all submitted adapters; -1.0 if no data
    pub avg_loss: f32,
}

pub async fn compute_metrics(state: &AppState) -> MetricsSnapshot {
    // ── Nodes ──────────────────────────────────────────────────────────────────
    let nodes = state.registry.list_nodes().await;
    let active_tasks_total: u32 = nodes.iter().map(|n| n.active_tasks).sum();
    let mut models_set = std::collections::HashSet::new();
    for n in &nodes {
        for m in &n.capabilities.available_models {
            models_set.insert(m.clone());
        }
    }
    let mut models: Vec<String> = models_set.into_iter().collect();
    models.sort();

    // ── Dataset ────────────────────────────────────────────────────────────────
    let stats = state.dataset_store.stats().await;
    let total_items = stats.total_examples;
    let size_bytes_est = total_items as u64 * 500;
    let verified_pct = if total_items > 0 {
        stats.execution_verified as f32 / total_items as f32 * 100.0
    } else {
        0.0
    };
    let mut node_contribs: Vec<(String, usize)> = stats.by_node.into_iter().collect();
    node_contribs.sort_by(|a, b| b.1.cmp(&a.1));
    node_contribs.truncate(5);

    // ── Training ───────────────────────────────────────────────────────────────
    let jobs = state.training_store.list_jobs().await;
    let total_jobs = jobs.len();
    let active_jobs = jobs
        .iter()
        .filter(|j| matches!(j.status, JobStatus::InProgress | JobStatus::Aggregating))
        .count();

    let (mut shards_total, mut shards_done, mut shards_in_flight, mut shards_available) =
        (0usize, 0usize, 0usize, 0usize);
    let mut total_loss = 0.0f32;
    let mut loss_count = 0usize;

    for job in &jobs {
        // Shard counts via job_stats (reads under a single lock)
        if let Some(js) = state.training_store.job_stats(&job.job_id).await {
            shards_total += js["shards_total"].as_u64().unwrap_or(0) as usize;
            shards_done += js["shards_completed"].as_u64().unwrap_or(0) as usize;
            shards_in_flight += js["shards_assigned"].as_u64().unwrap_or(0) as usize;
            shards_available += js["shards_available"].as_u64().unwrap_or(0) as usize;
        }
        // Loss from submitted adapters
        for sub in state.training_store.get_submissions(&job.job_id).await {
            if sub.training_loss.is_finite() && sub.training_loss > 0.0 {
                total_loss += sub.training_loss;
                loss_count += 1;
            }
        }
    }

    let avg_loss = if loss_count > 0 {
        total_loss / loss_count as f32
    } else {
        -1.0
    };

    // ── Uptime ─────────────────────────────────────────────────────────────────
    let now_secs = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs();

    // ── Network counters ─────────────────────────────────────────────────────
    let counters = &state.counters;
    let network = NetworkMetrics {
        grpc_requests: counters.grpc_requests.load(Ordering::Relaxed),
        http_requests: counters.http_requests.load(Ordering::Relaxed),
        heartbeats: counters.heartbeats.load(Ordering::Relaxed),
        bytes_in: counters.bytes_in.load(Ordering::Relaxed),
        bytes_out: counters.bytes_out.load(Ordering::Relaxed),
        inference_requests: counters.inference_requests.load(Ordering::Relaxed),
        inference_completed: counters.inference_completed.load(Ordering::Relaxed),
        tensor_ops: counters.tensor_ops.load(Ordering::Relaxed),
    };

    MetricsSnapshot {
        nodes: NodeMetrics {
            total: nodes.len(),
            active_tasks_total,
            models,
        },
        dataset: DatasetMetrics {
            total_items,
            size_bytes_est,
            execution_verified: stats.execution_verified,
            verified_pct,
            refined: stats.refined,
            deduplicated: stats.deduplicated,
            by_category: stats.by_category,
            by_provider: stats.by_provider,
            top_nodes: node_contribs,
        },
        training: TrainingMetrics {
            total_jobs,
            active_jobs,
            shards_total,
            shards_done,
            shards_in_flight,
            shards_available,
            avg_loss,
        },
        network,
        uptime_secs: now_secs.saturating_sub(state.started_at_secs),
        timestamp_secs: now_secs,
    }
}

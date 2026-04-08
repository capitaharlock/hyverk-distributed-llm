// @llm-depends: http_api.rs, dataset_store.rs
// TrainingStore: manages distributed LoRA fine-tuning jobs.
// Coordinator splits dataset into shards, assigns to nodes, collects adapters, runs FedAvg.

use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::Arc;
use std::time::Duration;
use tokio::sync::RwLock;
use uuid::Uuid;

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum JobStatus {
    Pending,
    InProgress,
    Aggregating,
    Completed,
    Failed,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum ShardStatus {
    Available,
    Assigned { node_id: String },
    Completed { node_id: String },
    Failed { node_id: String, error: String },
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DataShard {
    pub shard_id: String,
    pub job_id: String,
    pub start_line: usize,
    pub end_line: usize,
    pub status: ShardStatus,
    pub assigned_at_secs: Option<u64>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AdapterSubmission {
    pub node_id: String,
    pub shard_id: String,
    /// Serialized safetensors bytes (base64-encoded for JSON transport)
    pub adapter_b64: String,
    pub training_loss: f32,
    pub training_steps: usize,
    pub submitted_at_secs: u64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TrainingJob {
    pub job_id: String,
    pub status: JobStatus,
    pub base_model: String,
    pub lora_rank: u32,
    pub lora_alpha: f64,
    pub num_epochs: u32,
    pub shard_count: usize,
    pub created_at_secs: u64,
    pub completed_at_secs: Option<u64>,
    /// Path to final merged adapter (after FedAvg)
    pub output_adapter_path: Option<String>,
}

#[derive(Clone)]
pub struct TrainingStore {
    inner: Arc<RwLock<Inner>>,
}

struct Inner {
    jobs: HashMap<String, TrainingJob>,
    shards: HashMap<String, Vec<DataShard>>,       // job_id → shards
    shard_data: HashMap<String, String>,            // shard_id → JSONL content
    submissions: HashMap<String, Vec<AdapterSubmission>>, // job_id → submissions
}

impl TrainingStore {
    pub fn new() -> Self {
        Self {
            inner: Arc::new(RwLock::new(Inner {
                jobs: HashMap::new(),
                shards: HashMap::new(),
                shard_data: HashMap::new(),
                submissions: HashMap::new(),
            })),
        }
    }

    /// Create a new training job by splitting a dataset into shards.
    pub async fn create_job(
        &self,
        base_model: String,
        dataset_jsonl: String,
        shard_size: usize,
        lora_rank: u32,
        lora_alpha: f64,
        num_epochs: u32,
    ) -> String {
        let job_id = Uuid::new_v4().to_string();
        let lines: Vec<&str> = dataset_jsonl.lines().filter(|l| !l.trim().is_empty()).collect();
        let now = epoch_secs();

        let mut shards = Vec::new();
        let mut shard_data_map = HashMap::new();

        for (chunk_idx, chunk) in lines.chunks(shard_size).enumerate() {
            let shard_id = Uuid::new_v4().to_string();
            let start = chunk_idx * shard_size;
            let end = start + chunk.len();

            shard_data_map.insert(shard_id.clone(), chunk.join("\n"));
            shards.push(DataShard {
                shard_id,
                job_id: job_id.clone(),
                start_line: start,
                end_line: end,
                status: ShardStatus::Available,
                assigned_at_secs: None,
            });
        }

        let shard_count = shards.len();
        let job = TrainingJob {
            job_id: job_id.clone(),
            status: JobStatus::Pending,
            base_model,
            lora_rank,
            lora_alpha,
            num_epochs,
            shard_count,
            created_at_secs: now,
            completed_at_secs: None,
            output_adapter_path: None,
        };

        let mut inner = self.inner.write().await;
        inner.jobs.insert(job_id.clone(), job);
        inner.shards.insert(job_id.clone(), shards);
        for (k, v) in shard_data_map {
            inner.shard_data.insert(k, v);
        }

        tracing::info!(job_id = %job_id, shards = shard_count, "Training job created");
        job_id
    }

    /// Assign next available shard to a node. Returns (shard_id, jsonl_content).
    pub async fn claim_next_shard(
        &self,
        job_id: &str,
        node_id: &str,
    ) -> Option<(String, String)> {
        let mut inner = self.inner.write().await;
        let shards = inner.shards.get_mut(job_id)?;

        let shard = shards.iter_mut().find(|s| s.status == ShardStatus::Available)?;
        let shard_id = shard.shard_id.clone();

        shard.status = ShardStatus::Assigned { node_id: node_id.to_string() };
        shard.assigned_at_secs = Some(epoch_secs());

        // Update job status to InProgress
        if let Some(job) = inner.jobs.get_mut(job_id) {
            if job.status == JobStatus::Pending {
                job.status = JobStatus::InProgress;
            }
        }

        let content = inner.shard_data.get(&shard_id)?.clone();
        Some((shard_id, content))
    }

    /// Submit a trained LoRA adapter for a completed shard.
    pub async fn submit_adapter(
        &self,
        job_id: &str,
        shard_id: &str,
        node_id: &str,
        adapter_b64: String,
        loss: f32,
        steps: usize,
    ) -> bool {
        let mut inner = self.inner.write().await;

        // Mark shard as completed
        if let Some(shards) = inner.shards.get_mut(job_id) {
            if let Some(shard) = shards.iter_mut().find(|s| s.shard_id == shard_id) {
                shard.status = ShardStatus::Completed { node_id: node_id.to_string() };
            }
        }

        // Record submission
        let submission = AdapterSubmission {
            node_id: node_id.to_string(),
            shard_id: shard_id.to_string(),
            adapter_b64,
            training_loss: loss,
            training_steps: steps,
            submitted_at_secs: epoch_secs(),
        };
        inner.submissions.entry(job_id.to_string()).or_default().push(submission);

        // Check if all shards are done → trigger aggregation
        let all_done = inner.shards.get(job_id).map(|shards| {
            shards.iter().all(|s| matches!(s.status, ShardStatus::Completed { .. } | ShardStatus::Failed { .. }))
        }).unwrap_or(false);

        if all_done {
            if let Some(job) = inner.jobs.get_mut(job_id) {
                job.status = JobStatus::Aggregating;
                tracing::info!(job_id = %job_id, "All shards complete — ready for FedAvg aggregation");
            }
        }

        true
    }

    pub async fn get_job(&self, job_id: &str) -> Option<TrainingJob> {
        self.inner.read().await.jobs.get(job_id).cloned()
    }

    pub async fn list_jobs(&self) -> Vec<TrainingJob> {
        self.inner.read().await.jobs.values().cloned().collect()
    }

    pub async fn get_submissions(&self, job_id: &str) -> Vec<AdapterSubmission> {
        self.inner.read().await.submissions.get(job_id).cloned().unwrap_or_default()
    }

    pub async fn job_stats(&self, job_id: &str) -> Option<serde_json::Value> {
        let inner = self.inner.read().await;
        let job = inner.jobs.get(job_id)?;
        let shards = inner.shards.get(job_id)?;
        let submissions = inner.submissions.get(job_id).map(|v| v.len()).unwrap_or(0);

        let available = shards.iter().filter(|s| s.status == ShardStatus::Available).count();
        let assigned = shards.iter().filter(|s| matches!(s.status, ShardStatus::Assigned { .. })).count();
        let completed = shards.iter().filter(|s| matches!(s.status, ShardStatus::Completed { .. })).count();

        Some(serde_json::json!({
            "job_id": job.job_id,
            "status": format!("{:?}", job.status),
            "shards_total": shards.len(),
            "shards_available": available,
            "shards_assigned": assigned,
            "shards_completed": completed,
            "adapters_submitted": submissions,
        }))
    }

    /// Reassign shards that have been stuck in Assigned for too long (node died)
    pub async fn reassign_stale_shards(&self, timeout: Duration) {
        let now = epoch_secs();
        let mut inner = self.inner.write().await;
        for shards in inner.shards.values_mut() {
            for shard in shards.iter_mut() {
                if let ShardStatus::Assigned { .. } = &shard.status {
                    let assigned_at = shard.assigned_at_secs.unwrap_or(0);
                    if now.saturating_sub(assigned_at) > timeout.as_secs() {
                        tracing::warn!(shard_id = %shard.shard_id, "Reassigning stale shard");
                        shard.status = ShardStatus::Available;
                        shard.assigned_at_secs = None;
                    }
                }
            }
        }
    }
}

fn epoch_secs() -> u64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs()
}

// Layer-Sharded LoRA Training — SQLite-persisted
//
// Rounds, assignments, stats, and adapters survive coordinator restarts.
// In-memory cache for fast reads, SQLite for durability.

use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::Arc;
use tokio::sync::RwLock;
use uuid::Uuid;
use rusqlite::{Connection, params};
use std::sync::Mutex as StdMutex;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TrainingRound {
    pub round_id: String,
    pub version: String,
    pub base_model: String,
    pub total_layers: usize,
    pub layers_per_assignment: usize,
    pub status: RoundStatus,
    pub assignments: Vec<LayerAssignment>,
    pub created_at: u64,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum RoundStatus { Active, Complete, Merging }

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LayerAssignment {
    pub assignment_id: String,
    pub layer_start: usize,
    pub layer_end: usize,
    pub data_shard_start: usize,
    pub data_shard_size: usize,
    pub status: AssignmentStatus,
    pub assigned_to: Option<String>,
    pub assigned_at: Option<u64>,
    pub adapter_received: bool,
    pub training_loss: Option<f32>,
    pub training_steps: Option<usize>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum AssignmentStatus { Available, Assigned, Complete, Failed }

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GoalEstimate {
    pub rounds_remaining: u64,
    pub rounds_per_hour: f64,
    pub eta_hours: f64,
    pub clients_needed_for_7d: u64,
    pub clients_needed_for_30d: u64,
}

#[derive(Clone)]
pub struct LayerTrainingStore {
    inner: Arc<RwLock<Inner>>,
    db: Arc<StdMutex<Connection>>,
}

struct Inner {
    rounds: Vec<TrainingRound>,
    total_rounds_completed: u64,
    first_round_at: Option<u64>,
}

fn now_secs() -> u64 {
    std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).unwrap_or_default().as_secs()
}

const SCHEMA: &str = "
CREATE TABLE IF NOT EXISTS training_rounds (
    round_id TEXT PRIMARY KEY,
    version TEXT NOT NULL,
    data_json TEXT NOT NULL,
    status TEXT NOT NULL,
    created_at INTEGER NOT NULL
);
CREATE TABLE IF NOT EXISTS training_stats (
    key TEXT PRIMARY KEY,
    value TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS adapters (
    assignment_id TEXT PRIMARY KEY,
    adapter_data BLOB NOT NULL,
    created_at INTEGER NOT NULL
);
";

impl LayerTrainingStore {
    pub fn open(db_path: &str) -> Self {
        let conn = Connection::open(db_path).expect("open training DB");
        conn.execute_batch(SCHEMA).expect("create training schema");

        // Load existing rounds from DB
        let mut rounds = Vec::new();
        {
            let mut stmt = conn.prepare("SELECT data_json FROM training_rounds ORDER BY created_at").unwrap();
            let rows: Vec<String> = stmt.query_map([], |r| r.get(0)).unwrap().flatten().collect();
            for json in rows {
                if let Ok(r) = serde_json::from_str::<TrainingRound>(&json) {
                    rounds.push(r);
                }
            }
        }

        // Load stats
        let total_rounds = conn.query_row(
            "SELECT value FROM training_stats WHERE key='total_rounds_completed'",
            [], |r| r.get::<_, String>(0)
        ).ok().and_then(|v| v.parse::<u64>().ok()).unwrap_or(0);

        let first_round_at = conn.query_row(
            "SELECT value FROM training_stats WHERE key='first_round_at'",
            [], |r| r.get::<_, String>(0)
        ).ok().and_then(|v| v.parse::<u64>().ok());

        tracing::info!(
            rounds = rounds.len(),
            completed = total_rounds,
            "Training store loaded from SQLite"
        );

        Self {
            inner: Arc::new(RwLock::new(Inner { rounds, total_rounds_completed: total_rounds, first_round_at })),
            db: Arc::new(StdMutex::new(conn)),
        }
    }

    pub async fn create_round(
        &self, version: &str, base_model: &str, total_layers: usize,
        layers_per_assignment: usize, dataset_size: usize, examples_per_shard: usize,
    ) -> String {
        let round_id = Uuid::new_v4().to_string();
        let now = now_secs();
        let mut assignments = Vec::new();
        let mut layer = 0;
        let mut data_offset = 0;
        while layer < total_layers {
            let end = (layer + layers_per_assignment).min(total_layers);
            let shard_size = examples_per_shard.min(dataset_size.saturating_sub(data_offset));
            assignments.push(LayerAssignment {
                assignment_id: Uuid::new_v4().to_string(),
                layer_start: layer, layer_end: end,
                data_shard_start: data_offset, data_shard_size: shard_size,
                status: AssignmentStatus::Available,
                assigned_to: None, assigned_at: None,
                adapter_received: false, training_loss: None, training_steps: None,
            });
            layer = end;
            data_offset = (data_offset + examples_per_shard) % dataset_size.max(1);
        }
        let round = TrainingRound {
            round_id: round_id.clone(), version: version.to_string(),
            base_model: base_model.to_string(), total_layers, layers_per_assignment,
            status: RoundStatus::Active, assignments, created_at: now,
        };

        // Persist to SQLite
        {
            let db = self.db.lock().unwrap();
            let json = serde_json::to_string(&round).unwrap_or_default();
            db.execute("INSERT OR REPLACE INTO training_rounds (round_id, version, data_json, status, created_at) VALUES (?1,?2,?3,?4,?5)",
                params![round.round_id, round.version, json, "Active", now as i64]).ok();
            // Set first_round_at if not set
            db.execute("INSERT OR IGNORE INTO training_stats (key, value) VALUES ('first_round_at', ?1)",
                params![now.to_string()]).ok();
        }

        tracing::info!(round_id = %round_id, version, assignments = round.assignments.len(), "Training round created (persisted)");
        self.inner.write().await.rounds.push(round);
        if self.inner.read().await.first_round_at.is_none() {
            self.inner.write().await.first_round_at = Some(now);
        }
        round_id
    }

    pub async fn claim_assignment(&self, node_id: &str) -> Option<LayerAssignment> {
        let (result, round_snapshot) = {
            let mut inner = self.inner.write().await;
            let now = now_secs();
            let mut found = None;
            let mut snapshot = None;
            for round in &mut inner.rounds {
                if round.status != RoundStatus::Active { continue; }
                for assignment in &mut round.assignments {
                    if assignment.status == AssignmentStatus::Available {
                        assignment.status = AssignmentStatus::Assigned;
                        assignment.assigned_to = Some(node_id.to_string());
                        assignment.assigned_at = Some(now);
                        found = Some(assignment.clone());
                        break;
                    }
                }
                if found.is_some() { snapshot = Some(round.clone()); break; }
            }
            (found, snapshot)
        };
        if let Some(ref r) = round_snapshot { self.persist_round(r); }
        result
    }

    pub async fn submit_adapter(&self, assignment_id: &str, adapter_bytes: Vec<u8>, loss: f32, steps: usize) -> bool {
        let (found, round_snapshot, total_completed) = {
            let mut inner = self.inner.write().await;
            let mut found = false;
            let mut snapshot = None;
            let mut round_complete = false;

            for i in 0..inner.rounds.len() {
                for j in 0..inner.rounds[i].assignments.len() {
                    if inner.rounds[i].assignments[j].assignment_id == assignment_id {
                        inner.rounds[i].assignments[j].status = AssignmentStatus::Complete;
                        inner.rounds[i].assignments[j].adapter_received = true;
                        inner.rounds[i].assignments[j].training_loss = Some(loss);
                        inner.rounds[i].assignments[j].training_steps = Some(steps);
                        found = true;
                        tracing::info!(assignment_id, loss, steps, "Adapter received");
                        break;
                    }
                }
                if found {
                    let all_done = inner.rounds[i].assignments.iter().all(|a| a.status == AssignmentStatus::Complete);
                    if all_done {
                        inner.rounds[i].status = RoundStatus::Merging;
                        inner.total_rounds_completed += 1;
                        round_complete = true;
                        tracing::info!(total = inner.total_rounds_completed, "Round complete");
                    }
                    snapshot = Some(inner.rounds[i].clone());
                    break;
                }
            }
            (found, snapshot, inner.total_rounds_completed)
        };
        if let Some(ref r) = round_snapshot { self.persist_round(r); }
        if found {
            let db = self.db.lock().unwrap();
            db.execute("INSERT OR REPLACE INTO adapters (assignment_id, adapter_data, created_at) VALUES (?1,?2,?3)",
                params![assignment_id, adapter_bytes, now_secs() as i64]).ok();
            db.execute("INSERT OR REPLACE INTO training_stats (key, value) VALUES ('total_rounds_completed', ?1)",
                params![total_completed.to_string()]).ok();
        }
        found
    }

    pub async fn reassign_stale(&self, timeout_secs: u64) {
        let now = now_secs();
        let snapshots: Vec<TrainingRound> = {
            let mut inner = self.inner.write().await;
            let mut reassigned = 0;
            let mut changed_rounds = Vec::new();
            for round in &mut inner.rounds {
                if round.status != RoundStatus::Active { continue; }
                let mut changed = false;
                for assignment in &mut round.assignments {
                    if assignment.status == AssignmentStatus::Assigned {
                        if let Some(at) = assignment.assigned_at {
                            if now - at > timeout_secs {
                                tracing::warn!(id = %assignment.assignment_id, "Reassigning stale shard");
                                assignment.status = AssignmentStatus::Available;
                                assignment.assigned_to = None;
                                assignment.assigned_at = None;
                                reassigned += 1;
                                changed = true;
                            }
                        }
                    }
                }
                if changed { changed_rounds.push(round.clone()); }
            }
            if reassigned > 0 { tracing::info!(count = reassigned, "Reassigned stale shards"); }
            changed_rounds
        };
        for r in &snapshots { self.persist_round(r); }
    }

    /// Merge adapters for completed rounds and save to disk
    pub async fn merge_completed_rounds(&self, data_dir: &str) {
        let merging_rounds: Vec<TrainingRound> = {
            let inner = self.inner.read().await;
            inner.rounds.iter().filter(|r| r.status == RoundStatus::Merging).cloned().collect()
        };

        for round in merging_rounds {
            let adapter_dir = format!("{}/adapters_{}", data_dir, round.round_id.chars().take(8).collect::<String>());
            std::fs::create_dir_all(&adapter_dir).ok();

            // Extract adapter blobs from SQLite
            let mut extracted = 0;
            {
                let db = self.db.lock().unwrap();
                for assignment in &round.assignments {
                    if !assignment.adapter_received { continue; }
                    let result: Result<Vec<u8>, _> = db.query_row(
                        "SELECT adapter_data FROM adapters WHERE assignment_id = ?1",
                        params![assignment.assignment_id],
                        |r| r.get(0),
                    );
                    if let Ok(data) = result {
                        let path = format!("{}/layers_{}_{}.safetensors", adapter_dir, assignment.layer_start, assignment.layer_end);
                        if std::fs::write(&path, &data).is_ok() {
                            extracted += 1;
                        }
                    }
                }
            }

            if extracted == 0 {
                tracing::warn!(round_id = %round.round_id, "No adapters to merge");
                continue;
            }

            let output_path = format!("{}/hyverk-{}.safetensors", data_dir, round.version);

            // Run merge script
            let result = tokio::process::Command::new("python3")
                .arg("/app/training/merge_adapters.py")
                .arg("--adapter-dir").arg(&adapter_dir)
                .arg("--output").arg(&output_path)
                .output()
                .await;

            match result {
                Ok(out) if out.status.success() => {
                    let stdout = String::from_utf8_lossy(&out.stdout);
                    tracing::info!(
                        round = %round.version,
                        output = %output_path,
                        "Adapters merged successfully: {}", stdout.trim()
                    );
                }
                Ok(out) => {
                    let stderr = String::from_utf8_lossy(&out.stderr);
                    tracing::error!(round = %round.version, "Merge failed: {}", stderr);
                }
                Err(e) => {
                    tracing::warn!(round = %round.version, "Merge script not available: {e}");
                }
            }

            // Cleanup temp adapter dir
            std::fs::remove_dir_all(&adapter_dir).ok();
        }
    }

    pub async fn auto_advance(&self, dataset_size: usize) {
        let (should_create, completed_rounds) = {
            let inner = self.inner.read().await;
            let has_merging = inner.rounds.iter().any(|r| r.status == RoundStatus::Merging);
            let no_active = !inner.rounds.iter().any(|r| r.status == RoundStatus::Active);
            (has_merging && no_active, inner.total_rounds_completed)
        };
        if should_create && dataset_size > 0 {
            let snapshots: Vec<TrainingRound> = {
                let mut inner = self.inner.write().await;
                let mut changed = Vec::new();
                for r in &mut inner.rounds {
                    if r.status == RoundStatus::Merging {
                        r.status = RoundStatus::Complete;
                        changed.push(r.clone());
                    }
                }
                changed
            };
            for r in &snapshots { self.persist_round(r); }
            self.create_round(&format!("v0.1-r{}", completed_rounds + 1), "Qwen2.5-Coder-7B", 28, 2, dataset_size, 500).await;
        }
    }

    pub async fn active_round(&self) -> Option<TrainingRound> {
        let inner = self.inner.read().await;
        inner.rounds.iter().find(|r| r.status == RoundStatus::Active || r.status == RoundStatus::Merging).cloned()
    }

    pub async fn all_rounds(&self) -> Vec<TrainingRound> {
        self.inner.read().await.rounds.clone()
    }

    pub async fn stats(&self) -> (usize, usize, usize, f32) {
        let inner = self.inner.read().await;
        let mut total = 0; let mut completed = 0; let mut total_loss = 0.0f32; let mut loss_count = 0;
        for r in &inner.rounds {
            for a in &r.assignments {
                total += 1;
                if a.status == AssignmentStatus::Complete {
                    completed += 1;
                    if let Some(l) = a.training_loss { total_loss += l; loss_count += 1; }
                }
            }
        }
        let avg = if loss_count > 0 { total_loss / loss_count as f32 } else { -1.0 };
        (inner.rounds.len(), total, completed, avg)
    }

    pub async fn estimate(&self, target_rounds: u64, current_clients: usize) -> GoalEstimate {
        let inner = self.inner.read().await;
        let done = inner.total_rounds_completed;
        let remaining = target_rounds.saturating_sub(done);

        // Calculate rate from actual elapsed time
        let rph = if let Some(first) = inner.first_round_at {
            let elapsed_h = (now_secs() - first) as f64 / 3600.0;
            if elapsed_h > 0.01 { done as f64 / elapsed_h } else { 0.0 }
        } else { 0.0 };

        let eta = if rph > 0.0 { remaining as f64 / rph }
            else if current_clients > 0 { remaining as f64 / (current_clients as f64 * 0.5) }
            else { f64::INFINITY };

        let rph_per_client = if rph > 0.0 && current_clients > 0 { rph / current_clients as f64 } else { 0.5 };
        let c7 = if rph_per_client > 0.0 { (remaining as f64 / (7.0 * 24.0 * rph_per_client)).ceil() as u64 } else { 0 };
        let c30 = if rph_per_client > 0.0 { (remaining as f64 / (30.0 * 24.0 * rph_per_client)).ceil() as u64 } else { 0 };

        GoalEstimate { rounds_remaining: remaining, rounds_per_hour: rph, eta_hours: eta, clients_needed_for_7d: c7, clients_needed_for_30d: c30 }
    }

    fn persist_round(&self, round: &TrainingRound) {
        let db = self.db.lock().unwrap();
        let json = serde_json::to_string(round).unwrap_or_default();
        let status = match round.status { RoundStatus::Active => "Active", RoundStatus::Complete => "Complete", RoundStatus::Merging => "Merging" };
        db.execute("INSERT OR REPLACE INTO training_rounds (round_id, version, data_json, status, created_at) VALUES (?1,?2,?3,?4,?5)",
            params![round.round_id, round.version, json, status, round.created_at as i64]).ok();
    }
}

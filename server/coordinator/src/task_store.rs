// @llm-depends: grpc_server.rs, http_api.rs, router.rs

use hyverk_proto::InferenceTask;
use std::collections::{HashMap, VecDeque};
use std::sync::Arc;
use std::time::Duration;
use tokio::sync::RwLock;
use tokio::time::Instant;
use tracing::{info, warn};
use uuid::Uuid;

const COMPLETED_TASK_TTL: Duration = Duration::from_secs(3600); // 1 hour
const MAX_TOKENS_LIMIT: u32 = 8192;

#[derive(Clone, Debug)]
pub enum TaskStatus {
    Pending,
    Assigned {
        node_id: String,
        assigned_at: Instant,
    },
    Completed {
        response_text: String,
        duration_ms: u64,
    },
    Failed {
        error: String,
    },
}

#[derive(Clone, Debug)]
pub struct TaskInfo {
    pub task_id: String,
    pub model: String,
    pub prompt: String,
    pub temperature: f64,
    pub max_tokens: u32,
    pub status: TaskStatus,
    pub created_at: Instant,
}

#[derive(Clone)]
pub struct TaskStore {
    tasks: Arc<RwLock<HashMap<String, TaskInfo>>>,
    pending_queue: Arc<RwLock<VecDeque<String>>>,
}

impl TaskStore {
    pub fn new() -> Self {
        Self {
            tasks: Arc::new(RwLock::new(HashMap::new())),
            pending_queue: Arc::new(RwLock::new(VecDeque::new())),
        }
    }

    pub async fn create_task(
        &self,
        model: String,
        prompt: String,
        temperature: f64,
        max_tokens: u32,
    ) -> String {
        let max_tokens = max_tokens.min(MAX_TOKENS_LIMIT);
        let task_id = Uuid::new_v4().to_string();
        let info = TaskInfo {
            task_id: task_id.clone(),
            model,
            prompt,
            temperature,
            max_tokens,
            status: TaskStatus::Pending,
            created_at: Instant::now(),
        };
        self.tasks.write().await.insert(task_id.clone(), info);
        self.pending_queue.write().await.push_back(task_id.clone());
        info!(task_id = %task_id, "Task created");
        task_id
    }

    /// Find a pending task matching the node's models. Uses queue for O(1) typical case.
    pub async fn assign_pending_task(
        &self,
        node_id: &str,
        available_models: &[String],
    ) -> Option<InferenceTask> {
        let mut queue = self.pending_queue.write().await;
        let mut tasks = self.tasks.write().await;

        // Scan queue for a task matching this node's models
        let len = queue.len();
        for _ in 0..len {
            let task_id = queue.pop_front()?;
            let task = match tasks.get_mut(&task_id) {
                Some(t) if matches!(t.status, TaskStatus::Pending) => t,
                _ => continue, // task was removed or already assigned
            };

            if available_models.contains(&task.model) {
                task.status = TaskStatus::Assigned {
                    node_id: node_id.to_string(),
                    assigned_at: Instant::now(),
                };
                info!(task_id = %task_id, node_id = %node_id, "Task assigned");
                return Some(InferenceTask {
                    task_id: task.task_id.clone(),
                    model: task.model.clone(),
                    prompt: task.prompt.clone(),
                    temperature: task.temperature,
                    max_tokens: task.max_tokens,
                });
            } else {
                // Not for this node, put back at end of queue
                queue.push_back(task_id);
            }
        }
        None
    }

    pub async fn complete_task(
        &self,
        task_id: &str,
        response_text: String,
        duration_ms: u64,
    ) -> bool {
        let mut tasks = self.tasks.write().await;
        if let Some(task) = tasks.get_mut(task_id) {
            task.status = TaskStatus::Completed {
                response_text,
                duration_ms,
            };
            info!(task_id = %task_id, "Task completed");
            true
        } else {
            warn!(task_id = %task_id, "Attempted to complete unknown task");
            false
        }
    }

    pub async fn fail_task(&self, task_id: &str, error: String) -> bool {
        let mut tasks = self.tasks.write().await;
        if let Some(task) = tasks.get_mut(task_id) {
            task.status = TaskStatus::Failed { error };
            info!(task_id = %task_id, "Task failed");
            true
        } else {
            warn!(task_id = %task_id, "Attempted to fail unknown task");
            false
        }
    }

    pub async fn get_task(&self, task_id: &str) -> Option<TaskInfo> {
        self.tasks.read().await.get(task_id).cloned()
    }

    /// Reassign tasks from dead nodes back to pending.
    pub async fn reassign_stale_tasks(&self, timeout: Duration) {
        let mut tasks = self.tasks.write().await;
        let mut queue = self.pending_queue.write().await;
        for task in tasks.values_mut() {
            if let TaskStatus::Assigned { assigned_at, .. } = &task.status {
                if assigned_at.elapsed() > timeout {
                    info!(task_id = %task.task_id, "Reassigning stale task back to pending");
                    task.status = TaskStatus::Pending;
                    queue.push_back(task.task_id.clone());
                }
            }
        }
    }

    /// Remove completed/failed tasks older than TTL.
    pub async fn cleanup_old_tasks(&self) {
        let mut tasks = self.tasks.write().await;
        let before = tasks.len();
        tasks.retain(|_, task| match &task.status {
            TaskStatus::Completed { .. } | TaskStatus::Failed { .. } => {
                task.created_at.elapsed() < COMPLETED_TASK_TTL
            }
            _ => true,
        });
        let removed = before - tasks.len();
        if removed > 0 {
            info!(removed, "Cleaned up old tasks");
        }
    }
}

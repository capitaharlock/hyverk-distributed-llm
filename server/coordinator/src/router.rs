// @llm-depends: registry.rs, task_store.rs

use crate::registry::NodeRegistry;
use crate::task_store::TaskStore;
use hyverk_proto::InferenceTask;

#[derive(Clone)]
pub struct TaskRouter {
    registry: NodeRegistry,
    task_store: TaskStore,
}

impl TaskRouter {
    pub fn new(registry: NodeRegistry, task_store: TaskStore) -> Self {
        Self {
            registry,
            task_store,
        }
    }

    /// Called when a node polls: find a pending task matching the node's available models.
    pub async fn assign_task_for_node(&self, node_id: &str) -> Option<InferenceTask> {
        let node = self.registry.get_node(node_id).await?;
        let models = node.capabilities.available_models;
        self.task_store.assign_pending_task(node_id, &models).await
    }
}

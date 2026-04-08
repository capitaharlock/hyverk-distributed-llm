// HTTP client to coordinator — replaces gRPC for Fly.io compatibility.
// All communication goes through HTTPS port 443 which Fly.io routes correctly.

use hyverk_core::error::HyverkError;
use std::time::Duration;
use tracing::info;

#[derive(Clone)]
pub struct CoordinatorConnection {
    client: reqwest::Client,
    base_url: String,
    node_id: String,
    node_name: String,
    heartbeat_interval: Duration,
}

impl CoordinatorConnection {
    pub async fn connect(
        coordinator_url: &str,
        node_name: String,
        capabilities: hyverk_proto::NodeCapabilities,
    ) -> Result<Self, HyverkError> {
        let client = reqwest::Client::builder()
            .timeout(Duration::from_secs(30))
            .build()
            .map_err(|e| HyverkError::Transport(format!("HTTP client error: {e}")))?;

        // Derive HTTP base URL from coordinator_url
        // Input might be "http://localhost:17001" (old gRPC) or "https://hyverk-coordinator.fly.dev"
        let base_url = if coordinator_url.contains(":17001") {
            coordinator_url.replace(":17001", ":17000")
        } else if coordinator_url.contains("fly.dev") && !coordinator_url.starts_with("https") {
            format!("https://{}", coordinator_url.trim_start_matches("http://"))
        } else {
            coordinator_url.to_string()
        };

        let resp = client
            .post(format!("{base_url}/api/v1/node/register"))
            .json(&serde_json::json!({
                "node_name": node_name,
                "models": capabilities.available_models,
                "hardware_info": capabilities.hardware_info,
                "max_concurrent_tasks": capabilities.max_concurrent_tasks,
            }))
            .send()
            .await
            .map_err(|e| HyverkError::Transport(format!("Register failed: {e}")))?;

        let body: serde_json::Value = resp.json().await
            .map_err(|e| HyverkError::Transport(format!("Register parse error: {e}")))?;

        let node_id = body["node_id"].as_str()
            .ok_or_else(|| HyverkError::Transport("No node_id in register response".into()))?
            .to_string();

        let interval_secs = body["heartbeat_interval_secs"].as_u64().unwrap_or(20);

        info!(node_id = %node_id, heartbeat_interval = interval_secs, "Registered with coordinator");

        Ok(Self {
            client,
            base_url,
            node_id,
            node_name,
            heartbeat_interval: Duration::from_secs(interval_secs),
        })
    }

    pub fn node_id(&self) -> &str {
        &self.node_id
    }

    pub fn base_url(&self) -> &str {
        &self.base_url
    }

    pub fn heartbeat_interval(&self) -> Duration {
        self.heartbeat_interval
    }

    /// Returns any pending signal from coordinator (role switch, etc.)
    /// Auto-re-registers if the coordinator doesn't recognize this node.
    pub async fn heartbeat(&self, active_tasks: u32) -> Result<Option<serde_json::Value>, HyverkError> {
        let resp = self.client
            .post(format!("{}/api/v1/node/heartbeat", self.base_url))
            .json(&serde_json::json!({
                "node_id": self.node_id,
                "active_tasks": active_tasks,
            }))
            .send()
            .await
            .map_err(|e| HyverkError::Transport(format!("Heartbeat failed: {e}")))?;

        let body: serde_json::Value = resp.json().await
            .map_err(|e| HyverkError::Transport(format!("Heartbeat parse: {e}")))?;

        // If coordinator says "not registered" — re-register automatically
        if body.get("error").and_then(|e| e.as_str()) == Some("Node not registered") {
            tracing::warn!("Coordinator lost our registration — re-registering...");
            let _ = self.re_register().await;
            return Ok(None);
        }

        if let Some(signal) = body.get("signal") {
            if !signal.is_null() {
                return Ok(Some(signal.clone()));
            }
        }
        Ok(None)
    }

    async fn re_register(&self) -> Result<(), HyverkError> {
        let resp = self.client
            .post(format!("{}/api/v1/node/register", self.base_url))
            .json(&serde_json::json!({
                "node_name": self.node_name,
                "models": [],
                "hardware_info": "",
                "max_concurrent_tasks": 2,
            }))
            .send()
            .await
            .map_err(|e| HyverkError::Transport(format!("Re-register failed: {e}")))?;
        let body: serde_json::Value = resp.json().await.unwrap_or_default();
        if let Some(id) = body.get("node_id").and_then(|v| v.as_str()) {
            tracing::info!(node_id = id, "Re-registered with coordinator");
        }
        Ok(())
    }

    pub async fn poll_task(&self) -> Result<Option<hyverk_proto::InferenceTask>, HyverkError> {
        let resp = self.client
            .post(format!("{}/api/v1/node/poll", self.base_url))
            .json(&serde_json::json!({"node_id": self.node_id}))
            .send()
            .await
            .map_err(|e| HyverkError::Transport(format!("Poll failed: {e}")))?;

        let body: serde_json::Value = resp.json().await
            .map_err(|e| HyverkError::Transport(format!("Poll parse: {e}")))?;

        if body["has_task"].as_bool() == Some(true) {
            let task = body.get("task").cloned().unwrap_or_default();
            Ok(Some(hyverk_proto::InferenceTask {
                task_id: task["task_id"].as_str().unwrap_or("").to_string(),
                model: task["model"].as_str().unwrap_or("").to_string(),
                prompt: task["prompt"].as_str().unwrap_or("").to_string(),
                temperature: task["temperature"].as_f64().unwrap_or(0.7),
                max_tokens: task["max_tokens"].as_u64().unwrap_or(256) as u32,
            }))
        } else {
            Ok(None)
        }
    }

    pub async fn submit_result(&self, result: hyverk_proto::SubmitResultRequest) -> Result<(), HyverkError> {
        self.client
            .post(format!("{}/api/v1/node/submit", self.base_url))
            .json(&serde_json::json!({
                "task_id": result.task_id,
                "success": result.success,
                "response_text": result.response_text,
                "error_message": result.error_message,
                "duration_ms": result.duration_ms,
            }))
            .send()
            .await
            .map_err(|e| HyverkError::Transport(format!("Submit failed: {e}")))?;
        Ok(())
    }

    pub async fn deregister(&self) -> Result<(), HyverkError> {
        let _ = self.client
            .post(format!("{}/api/v1/node/deregister", self.base_url))
            .json(&serde_json::json!({"node_id": self.node_id}))
            .send()
            .await;
        info!("Deregistered from coordinator");
        Ok(())
    }
}

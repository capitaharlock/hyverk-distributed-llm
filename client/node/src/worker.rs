// Inference worker — routes tasks through the distributed Python cluster.
// The cluster (serve_layers.py) must be running on localhost:18000.
// It splits the model across multiple processes/nodes for distributed inference.

use hyverk_proto::{InferenceTask, SubmitResultRequest};
use std::sync::OnceLock;
use std::time::{Duration, Instant};
use tracing::{error, info};

fn distributed_cluster_client() -> &'static reqwest::Client {
    static CLIENT: OnceLock<reqwest::Client> = OnceLock::new();
    CLIENT.get_or_init(|| {
        reqwest::Client::builder()
            .pool_max_idle_per_host(8)
            .timeout(Duration::from_secs(300))
            .build()
            .expect("reqwest client for distributed cluster")
    })
}

pub async fn execute_task(
    _engine: &std::sync::Arc<hyverk_inference::engine::InferenceEngine>,
    task: InferenceTask,
    node_id: &str,
) -> SubmitResultRequest {
    info!(task_id = %task.task_id, "Executing via distributed cluster");
    let start = Instant::now();

    // Call the local distributed cluster
    let cluster_url = "http://localhost:18000/generate";
    let prompt = format!(
        "<|im_start|>system\nYou are Hyverk, an expert coding assistant.<|im_end|>\n<|im_start|>user\n{}<|im_end|>\n<|im_start|>assistant\n",
        task.prompt
    );

    let client = distributed_cluster_client();

    match client
        .post(cluster_url)
        .json(&serde_json::json!({
            "prompt": prompt,
            "max_tokens": task.max_tokens,
            "temperature": task.temperature,
        }))
        .send()
        .await
    {
        Ok(resp) => {
            match resp.json::<serde_json::Value>().await {
                Ok(data) => {
                    let duration_ms = start.elapsed().as_millis() as u64;
                    let text = data["text"].as_str().unwrap_or("").to_string();
                    let tokens = data["tokens"].as_u64().unwrap_or(0);
                    let cluster_info = data.get("cluster")
                        .map(|c| serde_json::to_string(c).unwrap_or_default())
                        .unwrap_or_default();

                    info!(
                        task_id = %task.task_id,
                        tokens,
                        duration_ms,
                        distributed = true,
                        "Distributed inference complete"
                    );

                    // Append cluster metadata to response
                    let response_with_meta = format!(
                        "{}\n\n<!-- CLUSTER_META: {} -->",
                        text, cluster_info
                    );

                    SubmitResultRequest {
                        node_id: node_id.to_string(),
                        task_id: task.task_id,
                        success: true,
                        response_text: response_with_meta,
                        error_message: String::new(),
                        duration_ms,
                    }
                }
                Err(e) => {
                    let duration_ms = start.elapsed().as_millis() as u64;
                    error!(task_id = %task.task_id, "Cluster response parse error: {e}");
                    SubmitResultRequest {
                        node_id: node_id.to_string(),
                        task_id: task.task_id,
                        success: false,
                        response_text: String::new(),
                        error_message: format!("Cluster parse error: {e}"),
                        duration_ms,
                    }
                }
            }
        }
        Err(e) => {
            let duration_ms = start.elapsed().as_millis() as u64;
            error!(task_id = %task.task_id, "Cluster unreachable: {e}");
            SubmitResultRequest {
                node_id: node_id.to_string(),
                task_id: task.task_id,
                success: false,
                response_text: String::new(),
                error_message: format!("Cluster offline: {e}. Run: make inference"),
                duration_ms,
            }
        }
    }
}

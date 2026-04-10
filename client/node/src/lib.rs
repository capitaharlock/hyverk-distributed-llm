pub mod coordinator_client;
pub mod inference_worker;
pub mod layer_worker;
pub mod worker;
pub mod ws_worker;

use coordinator_client::CoordinatorConnection;
use hyverk_core::config::NodeConfig;
use hyverk_inference::engine::InferenceEngine;
use hyverk_proto::NodeCapabilities;
use std::sync::atomic::{AtomicU32, Ordering};
use std::sync::Arc;
use std::time::Duration;
use tokio::sync::Semaphore;
use tokio_util::sync::CancellationToken;
use tracing::{error, info, warn};

/// Start the node (register, heartbeat, poll, execute). Returns when cancelled.
pub async fn run_node(
    config: &NodeConfig,
    shutdown: CancellationToken,
) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
    let engine = Arc::new(InferenceEngine::new(config.models_dir.clone())?);
    let available_models = engine.list_models();
    info!(models = ?available_models, "Discovered local models");

    if available_models.is_empty() {
        warn!(
            "No GGUF models found in {:?}. Node will register but cannot serve inference.",
            config.models_dir
        );
    }

    let capabilities = NodeCapabilities {
        available_models: available_models.clone(),
        hardware_info: config.hardware_info.clone(),
        max_concurrent_tasks: config.max_concurrent_tasks,
    };

    let conn = connect_with_retry(&config.coordinator_url, config.name.clone(), capabilities, &shutdown).await?;

    let node_id = conn.node_id().to_string();
    let heartbeat_interval = conn.heartbeat_interval();
    let poll_interval = Duration::from_millis(config.poll_interval_ms);
    let semaphore = Arc::new(Semaphore::new(config.max_concurrent_tasks as usize));
    let active_tasks = Arc::new(AtomicU32::new(0));

    // Heartbeat loop — also receives signals from coordinator
    let conn_hb = conn.clone();
    let active_hb = active_tasks.clone();
    let shutdown_hb = shutdown.clone();
    tokio::spawn(async move {
        loop {
            tokio::select! {
                _ = shutdown_hb.cancelled() => break,
                _ = tokio::time::sleep(heartbeat_interval) => {}
            }
            let count = active_hb.load(Ordering::Relaxed);
            match conn_hb.heartbeat(count).await {
                Ok(Some(signal)) => {
                    let sig_type = signal["signal"].as_str().unwrap_or("");
                    let role = signal["role"].as_str().unwrap_or("");
                    info!(signal = sig_type, role, "Received signal from coordinator");
                    // TODO: implement role switching
                    // For now: log the signal. Full implementation will:
                    // - "switch_role" + "executor" → stop training, start inference server
                    // - "switch_role" + "trainer" → stop inference, resume training
                }
                Ok(None) => {} // no signal
                Err(e) => warn!("Heartbeat failed: {e}"),
            }
        }
    });

    // Layer training worker — uses same HTTP URL as main connection
    let coordinator_http = conn.base_url().to_string();
    let layer_node_id = node_id.clone();
    // Model dir for training: use qwen2.5-7b safetensors, not GGUF models dir
    let model_dir = dirs::home_dir()
        .unwrap_or_default()
        .join(".hyverk")
        .join("qwen2.5-7b")
        .to_string_lossy()
        .to_string();
    let shutdown_layer = shutdown.clone();
    // WebSocket worker — persistent connection for distributed inference
    let ws_coord = conn.base_url().to_string();
    let ws_name = config.name.clone();
    let ws_hw = config.hardware_info.clone();
    let ws_model_dir = model_dir.clone();
    let ws_shutdown = shutdown.clone();
    // Detect GPU from hardware_info string or model availability
    let has_gpu = config.hardware_info.contains("Metal")
        || config.hardware_info.contains("NVIDIA")
        || config.hardware_info.contains("RTX")
        || config.hardware_info.contains("GTX")
        || config.hardware_info.contains("M1")
        || config.hardware_info.contains("M2")
        || config.hardware_info.contains("M3")
        || config.hardware_info.contains("M4")
        || !available_models.is_empty(); // has GGUF = likely has GPU
    tokio::spawn(async move {
        ws_worker::run_ws_worker(
            &ws_coord, &ws_name, &ws_hw,
            has_gpu,
            48000, // ram_mb — TODO: detect dynamically
            &ws_model_dir,
            ws_shutdown,
        ).await;
    });

    tokio::spawn(async move {
        layer_worker::run_layer_worker(&coordinator_http, &layer_node_id, &model_dir, shutdown_layer).await;
    });

    // Poll loop
    info!("Node running. Polling for tasks...");
    loop {
        tokio::select! {
            _ = shutdown.cancelled() => {
                info!("Node shutting down, deregistering...");
                if let Err(e) = conn.deregister().await {
                    warn!("Deregister failed: {e}");
                }
                break;
            }
            _ = tokio::time::sleep(poll_interval) => {}
        }

        let permit = match semaphore.clone().try_acquire_owned() {
            Ok(p) => p,
            Err(_) => continue,
        };

        match conn.poll_task().await {
            Ok(Some(task)) => {
                active_tasks.fetch_add(1, Ordering::Relaxed);
                let engine = engine.clone();
                let node_id = node_id.clone();
                let conn_submit = conn.clone();
                let active_counter = active_tasks.clone();
                tokio::spawn(async move {
                    let result = worker::execute_task(&engine, task, &node_id).await;
                    if let Err(e) = conn_submit.submit_result(result).await {
                        error!("Failed to submit result: {e}");
                    }
                    active_counter.fetch_sub(1, Ordering::Relaxed);
                    drop(permit);
                });
            }
            Ok(None) => {
                drop(permit);
            }
            Err(e) => {
                warn!("Poll failed: {e}");
                drop(permit);
                tokio::time::sleep(Duration::from_secs(2)).await;
            }
        }
    }

    // Wait for in-flight tasks
    let max = config.max_concurrent_tasks as usize;
    let _ = tokio::time::timeout(Duration::from_secs(30), async {
        let _ = semaphore.acquire_many(max as u32).await;
    })
    .await;

    info!("Node stopped.");
    Ok(())
}

async fn connect_with_retry(
    url: &str,
    name: String,
    capabilities: NodeCapabilities,
    shutdown: &CancellationToken,
) -> Result<CoordinatorConnection, Box<dyn std::error::Error + Send + Sync>> {
    let mut attempt = 0u32;
    loop {
        match CoordinatorConnection::connect(url, name.clone(), capabilities.clone()).await {
            Ok(conn) => return Ok(conn),
            Err(e) => {
                attempt += 1;
                let delay = Duration::from_secs((2u64).pow(attempt.min(5)));
                error!(attempt, delay_secs = delay.as_secs(), "Connection failed: {e}, retrying...");
                tokio::select! {
                    _ = shutdown.cancelled() => {
                        return Err(Box::new(e));
                    }
                    _ = tokio::time::sleep(delay) => {}
                }
            }
        }
    }
}

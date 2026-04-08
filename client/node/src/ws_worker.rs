// WebSocket worker — connects to coordinator via persistent WebSocket.
// Receives inference tasks (forward passes) and training assignments.
// Sends results back through the same connection.
// Works behind NAT/firewalls because the client initiates the connection.

use hyverk_comms::messages::{ClientMessage, CoordinatorMessage};
use futures_util::{SinkExt, StreamExt};
use tokio_tungstenite::{connect_async, tungstenite::Message};
use tracing::{info, warn, error};

/// Run the WebSocket worker loop. Reconnects on failure.
pub async fn run_ws_worker(
    coordinator_url: &str,
    node_name: &str,
    hardware_info: &str,
    has_gpu: bool,
    ram_mb: u64,
    model_dir: &str,
    shutdown: tokio_util::sync::CancellationToken,
) {
    // Convert HTTPS URL to WSS
    let ws_url = coordinator_url
        .replace("https://", "wss://")
        .replace("http://", "ws://");
    let ws_url = format!("{}/ws", ws_url.trim_end_matches('/'));

    // Download assigned layers on startup
    // Layer assignment: coordinator tells each node which layers via config
    // For now: auto-detect based on node name
    let (layer_start, layer_end) = if node_name.contains("node-2") || node_name.contains("fly-node-iad-2") {
        (14, 28) // node-2: layers 14-27 + norm + lm_head
    } else {
        (0, 14) // node-1: embed + layers 0-13
    };

    let layer_cache = format!("{}/inference_layers_{layer_start}_{layer_end}", model_dir);
    let coordinator_http = coordinator_url
        .replace("wss://", "https://")
        .replace("ws://", "http://")
        .replace("/ws", "");

    // Track whether layers are ready (persists across reconnections)
    let layers_ready = std::sync::Arc::new(std::sync::atomic::AtomicBool::new(false));
    let download_failed = std::sync::Arc::new(std::sync::atomic::AtomicBool::new(false));

    // Download layers with retry (up to 3 attempts)
    info!(layers = format!("{layer_start}-{layer_end}"), cache = %layer_cache, "Downloading assigned layers...");
    let mut download_ok = false;
    for attempt in 1..=3 {
        match download_layers(&layer_cache, &coordinator_http, layer_start, layer_end).await {
            Ok(()) => {
                info!("Layer weights ready (attempt {attempt})");
                layers_ready.store(true, std::sync::atomic::Ordering::SeqCst);
                download_ok = true;
                break;
            }
            Err(e) => {
                warn!("Layer download failed (attempt {attempt}/3): {e}");
                if attempt < 3 {
                    info!("Retrying download in 10s...");
                    tokio::time::sleep(std::time::Duration::from_secs(10)).await;
                }
            }
        }
    }
    if !download_ok {
        error!("Layer download failed after 3 attempts — inference will not work on this node");
        download_failed.store(true, std::sync::atomic::Ordering::SeqCst);
    }

    loop {
        tokio::select! {
            _ = shutdown.cancelled() => {
                info!("WS worker shutting down");
                break;
            }
            result = connect_and_run(&ws_url, node_name, hardware_info, has_gpu, ram_mb, &layer_cache, &layers_ready, &download_failed, &shutdown) => {
                match result {
                    Ok(()) => info!("WS connection closed cleanly"),
                    Err(e) => warn!("WS connection error: {e}"),
                }
                if shutdown.is_cancelled() { break; }
                info!("Reconnecting in 5s...");
                tokio::time::sleep(std::time::Duration::from_secs(5)).await;
            }
        }
    }
}

async fn connect_and_run(
    ws_url: &str,
    node_name: &str,
    hardware_info: &str,
    has_gpu: bool,
    ram_mb: u64,
    model_dir: &str,
    layers_ready: &std::sync::Arc<std::sync::atomic::AtomicBool>,
    download_failed: &std::sync::Arc<std::sync::atomic::AtomicBool>,
    shutdown: &tokio_util::sync::CancellationToken,
) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
    info!(url = ws_url, "Connecting WebSocket...");

    let (ws, _) = connect_async(ws_url).await?;
    let (mut sink, mut stream) = ws.split();

    // Register
    let reg = ClientMessage::Register {
        node_name: node_name.to_string(),
        hardware_info: hardware_info.to_string(),
        models: vec![],
        ram_mb,
        has_gpu,
    };
    sink.send(Message::Text(serde_json::to_string(&reg)?.into())).await?;

    // Report current state
    let current_state = if layers_ready.load(std::sync::atomic::Ordering::SeqCst) {
        ("ready".to_string(), "Layer weights loaded".to_string())
    } else if download_failed.load(std::sync::atomic::Ordering::SeqCst) {
        ("error".to_string(), "Layer download failed after 3 attempts".to_string())
    } else {
        ("downloading".to_string(), "Downloading model layers".to_string())
    };
    let state_msg = ClientMessage::StateUpdate {
        state: current_state.0,
        detail: current_state.1,
    };
    sink.send(Message::Text(serde_json::to_string(&state_msg)?.into())).await?;
    info!("WebSocket connected and registered");

    let mut ping_interval = tokio::time::interval(std::time::Duration::from_secs(30));

    loop {
        tokio::select! {
            _ = shutdown.cancelled() => break,
            _ = ping_interval.tick() => {
                let pong = serde_json::to_string(&ClientMessage::Pong)?;
                sink.send(Message::Text(pong.into())).await?;
            }
            msg = stream.next() => {
                match msg {
                    Some(Ok(Message::Text(text))) => {
                        match serde_json::from_str::<CoordinatorMessage>(&text) {
                            Ok(coord_msg) => {
                                if let Some(response) = handle_coordinator_message(coord_msg, model_dir).await {
                                    match response {
                                        WsResponse::Text(msg) => {
                                            sink.send(Message::Text(serde_json::to_string(&msg)?.into())).await?;
                                        }
                                        WsResponse::Binary(data) => {
                                            sink.send(Message::Binary(data.into())).await?;
                                        }
                                        WsResponse::TextAndBinary(msg, data) => {
                                            sink.send(Message::Text(serde_json::to_string(&msg)?.into())).await?;
                                            sink.send(Message::Binary(data.into())).await?;
                                        }
                                    }
                                }
                            }
                            Err(e) => warn!("Bad coordinator message: {e}"),
                        }
                    }
                    Some(Ok(Message::Binary(data))) => {
                        // Binary = hidden states for inference forward
                        if let Some(response) = handle_binary_forward(data.to_vec(), model_dir).await {
                            match response {
                                WsResponse::Text(msg) => {
                                    sink.send(Message::Text(serde_json::to_string(&msg)?.into())).await?;
                                }
                                WsResponse::Binary(data) => {
                                    sink.send(Message::Binary(data.into())).await?;
                                }
                                WsResponse::TextAndBinary(msg, data) => {
                                    sink.send(Message::Text(serde_json::to_string(&msg)?.into())).await?;
                                    sink.send(Message::Binary(data.into())).await?;
                                }
                            }
                        }
                    }
                    Some(Ok(Message::Ping(d))) => {
                        sink.send(Message::Pong(d)).await?;
                    }
                    Some(Ok(Message::Close(_))) | None => break,
                    Some(Err(e)) => { error!("WS error: {e}"); break; }
                    _ => {}
                }
            }
        }
    }

    Ok(())
}

enum WsResponse {
    Text(ClientMessage),
    Binary(Vec<u8>),
    TextAndBinary(ClientMessage, Vec<u8>),
}

async fn handle_coordinator_message(
    msg: CoordinatorMessage,
    model_dir: &str,
) -> Option<WsResponse> {
    match msg {
        CoordinatorMessage::InferenceStart { request_id, token_ids, layer_start, layer_end, max_tokens, temperature } => {
            info!(request_id = %request_id, layers = format!("{layer_start}-{layer_end}"), tokens = token_ids.len(), "Inference start: embed + forward");

            let result = run_layer_forward(model_dir, &token_ids, layer_start, layer_end, true).await;

            match result {
                Ok((hidden_data, shape)) => {
                    info!(request_id = %request_id, hidden_size = hidden_data.len(), "Forward complete, sending hidden states");
                    let msg = ClientMessage::ForwardResult {
                        request_id: request_id.clone(),
                        hidden_states: vec![],
                        shape,
                    };
                    let mut payload = vec![0u8; 36];
                    payload[..request_id.len().min(36)].copy_from_slice(&request_id.as_bytes()[..request_id.len().min(36)]);
                    payload.extend_from_slice(&hidden_data);
                    Some(WsResponse::TextAndBinary(msg, payload))
                }
                Err(e) => {
                    error!(request_id = %request_id, "Embed+forward failed: {e}");
                    None
                }
            }
        }
        CoordinatorMessage::InferenceForward { request_id, layer_start, layer_end, is_last, .. } => {
            info!(request_id = %request_id, layers = format!("{layer_start}-{layer_end}"), last = is_last, "Inference forward received");
            // Hidden states arrive as binary frame — handled in handle_binary_forward
            None
        }
        CoordinatorMessage::Ping => {
            Some(WsResponse::Text(ClientMessage::Pong))
        }
        _ => None,
    }
}

async fn handle_binary_forward(
    data: Vec<u8>,
    model_dir: &str,
) -> Option<WsResponse> {
    if data.len() < 36 { return None; }
    let request_id = String::from_utf8_lossy(&data[..36]).trim_end_matches('\0').to_string();
    let hidden_data = data[36..].to_vec();

    info!(request_id = %request_id, size = hidden_data.len(), "Received hidden states for forward pass");

    // Save hidden states to temp file for Python
    let input_file = format!("/tmp/hyverk_ws_in_{}.pt", &request_id[..8]);
    let output_file = format!("/tmp/hyverk_ws_out_{}.pt", &request_id[..8]);
    if std::fs::write(&input_file, &hidden_data).is_err() {
        error!("Failed to write hidden states to temp file");
        return None;
    }

    // Determine our layer range from model_dir path
    // model_dir format: .../inference_layers_10_20
    let (layer_start, layer_end) = parse_layer_range(model_dir);
    let is_last = layer_end >= 28;

    let script = find_script("inference/node_forward.py");
    let python = find_python();
    let mode = if is_last { "generate" } else { "forward" };

    info!(request_id = %request_id, mode, layers = format!("{layer_start}-{layer_end}"), "Running forward pass");

    let mut cmd = tokio::process::Command::new(&python);
    cmd.arg(&script)
        .arg("--mode").arg(mode)
        .arg("--model-dir").arg(model_dir)
        .arg("--layer-start").arg(layer_start.to_string())
        .arg("--layer-end").arg(layer_end.to_string())
        .arg("--input-file").arg(&input_file);

    if !is_last {
        cmd.arg("--output-file").arg(&output_file);
    }

    let out = match tokio::time::timeout(std::time::Duration::from_secs(600), cmd.output()).await {
        Ok(Ok(o)) => o,
        Ok(Err(e)) => { error!("Forward script error: {e}"); return None; }
        Err(_) => { error!("Forward timed out"); return None; }
    };

    let _ = std::fs::remove_file(&input_file);

    if !out.status.success() {
        let stderr = String::from_utf8_lossy(&out.stderr);
        error!("Forward failed: {stderr}");
        return None;
    }

    let stdout = String::from_utf8_lossy(&out.stdout);

    if is_last {
        // Last node: return token ID
        let result: serde_json::Value = serde_json::from_str(stdout.trim()).ok()?;
        let token_id = result["token_id"].as_u64()? as u32;
        let is_eos = token_id == 151643 || token_id == 151644 || token_id == 151645;
        info!(request_id = %request_id, token_id, eos = is_eos, "Token generated");
        Some(WsResponse::Text(ClientMessage::TokenGenerated {
            request_id,
            token_id,
            is_eos,
        }))
    } else {
        // Middle node: return hidden states
        let hidden_out = std::fs::read(&output_file).ok()?;
        let _ = std::fs::remove_file(&output_file);
        let result: serde_json::Value = serde_json::from_str(stdout.trim()).ok()?;
        let shape = result["shape"].as_array()
            .map(|a| a.iter().filter_map(|v| v.as_u64().map(|n| n as usize)).collect())
            .unwrap_or_default();

        info!(request_id = %request_id, hidden_size = hidden_out.len(), "Forward complete");
        let msg = ClientMessage::ForwardResult {
            request_id: request_id.clone(),
            hidden_states: vec![],
            shape,
        };
        let mut payload = vec![0u8; 36];
        payload[..request_id.len().min(36)].copy_from_slice(&request_id.as_bytes()[..request_id.len().min(36)]);
        payload.extend_from_slice(&hidden_out);
        Some(WsResponse::TextAndBinary(msg, payload))
    }
}

fn parse_layer_range(model_dir: &str) -> (usize, usize) {
    // Parse from path like: .../inference_layers_10_20
    let parts: Vec<&str> = model_dir.rsplit('/').next().unwrap_or("").split('_').collect();
    if parts.len() >= 4 {
        let start = parts[parts.len()-2].parse().unwrap_or(0);
        let end = parts[parts.len()-1].parse().unwrap_or(28);
        (start, end)
    } else {
        (0, 28) // fallback: all layers
    }
}

/// Call Python to run forward pass on assigned layers
async fn run_layer_forward(
    model_dir: &str,
    token_ids: &[u32],
    layer_start: usize,
    layer_end: usize,
    is_first: bool,
) -> Result<(Vec<u8>, Vec<usize>), Box<dyn std::error::Error + Send + Sync>> {
    let script = find_script("inference/node_forward.py");
    let python = find_python();

    let output_file = format!("/tmp/hyverk_hidden_{}_{}.pt", layer_start, layer_end);
    let mode = if is_first { "embed" } else { "forward" };

    let mut cmd = tokio::process::Command::new(&python);
    cmd.arg(&script)
        .arg("--mode").arg(mode)
        .arg("--model-dir").arg(model_dir)
        .arg("--layer-start").arg(layer_start.to_string())
        .arg("--layer-end").arg(layer_end.to_string())
        .arg("--output-file").arg(&output_file);

    if is_first {
        let ids_str: String = token_ids.iter().map(|id| id.to_string()).collect::<Vec<_>>().join(",");
        cmd.arg("--token-ids").arg(&ids_str);
    }

    let out = tokio::time::timeout(
        std::time::Duration::from_secs(600),
        cmd.output(),
    ).await
        .map_err(|_| "Forward pass timed out")?
        .map_err(|e| format!("Failed to run forward script: {e}"))?;

    if !out.status.success() {
        let stderr = String::from_utf8_lossy(&out.stderr);
        return Err(format!("Forward failed: {stderr}").into());
    }

    let stdout = String::from_utf8_lossy(&out.stdout);
    let result: serde_json::Value = serde_json::from_str(stdout.trim())
        .map_err(|e| format!("Bad forward result: {e}"))?;

    let hidden_data = std::fs::read(&output_file)
        .map_err(|e| format!("Can't read hidden states: {e}"))?;
    let _ = std::fs::remove_file(&output_file);

    let shape = result["shape"].as_array()
        .map(|a| a.iter().filter_map(|v| v.as_u64().map(|n| n as usize)).collect())
        .unwrap_or_default();

    Ok((hidden_data, shape))
}

/// Download assigned layers from coordinator (called once on startup)
async fn download_layers(
    model_dir: &str,
    coordinator_url: &str,
    layer_start: usize,
    layer_end: usize,
) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
    let script = find_script("inference/node_forward.py");
    let python = find_python();

    let out = tokio::process::Command::new(&python)
        .arg(&script)
        .arg("--mode").arg("download")
        .arg("--model-dir").arg(model_dir)
        .arg("--coordinator").arg(coordinator_url)
        .arg("--layer-start").arg(layer_start.to_string())
        .arg("--layer-end").arg(layer_end.to_string())
        .output()
        .await?;

    let stderr = String::from_utf8_lossy(&out.stderr);
    for line in stderr.lines() { info!(target: "download", "{}", line); }

    if !out.status.success() {
        return Err(format!("Layer download failed: {stderr}").into());
    }

    let stdout = String::from_utf8_lossy(&out.stdout);
    info!("Layer download result: {}", stdout.trim());
    Ok(())
}

fn find_python() -> String {
    for p in &[".venv/bin/python3", "/usr/bin/python3", "python3"] {
        if std::path::Path::new(p).exists() { return p.to_string(); }
    }
    "python3".to_string()
}

fn find_script(name: &str) -> String {
    for prefix in &["", "../", "../../", "/app/"] {
        let p = format!("{}{}", prefix, name);
        if std::path::Path::new(&p).exists() { return p; }
    }
    name.to_string()
}

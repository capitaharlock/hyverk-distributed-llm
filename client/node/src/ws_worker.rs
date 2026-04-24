// WebSocket worker — connects to coordinator via persistent WebSocket.
// Receives inference tasks (forward passes) and training assignments.
// Sends results back through the same connection.
// Works behind NAT/firewalls because the client initiates the connection.

use futures_util::{SinkExt, StreamExt};
use hyverk_comms::messages::{ClientMessage, CoordinatorMessage};
use std::collections::HashMap;
use std::sync::{Arc, Mutex, OnceLock};
use std::time::Duration;
use tokio_tungstenite::{connect_async, tungstenite::Message};
use tracing::{error, info, warn};

/// Single shared client for `127.0.0.1` layer inference (one request per generated token).
/// Creating a new `reqwest::Client` each call drops the connection pool and forces new TCP
/// handshakes to localhost — measurable latency on autoregressive decoding.
/// Per-request sampling forwarded to the local Python `generate` path (X-* headers).
#[derive(Clone, Debug, Default, PartialEq)]
pub struct SamplingParams {
    pub temperature: Option<f32>,
    pub top_p: Option<f32>,
    pub top_k: Option<u32>,
}

pub fn merge_optional_sampling(
    into: &mut SamplingParams,
    temperature: Option<f32>,
    top_p: Option<f32>,
    top_k: Option<u32>,
) {
    if temperature.is_some() {
        into.temperature = temperature;
    }
    if top_p.is_some() {
        into.top_p = top_p;
    }
    if top_k.is_some() {
        into.top_k = top_k;
    }
}

pub fn merge_from_continue(
    into: &mut SamplingParams,
    temperature: f32,
    top_p: Option<f32>,
    top_k: Option<u32>,
) {
    into.temperature = Some(temperature);
    if top_p.is_some() {
        into.top_p = top_p;
    }
    if top_k.is_some() {
        into.top_k = top_k;
    }
}

pub fn apply_sampling_headers(
    mut req: reqwest::RequestBuilder,
    params: &SamplingParams,
) -> reqwest::RequestBuilder {
    if let Some(t) = params.temperature {
        req = req.header("X-Temperature", format!("{t}"));
    }
    if let Some(p) = params.top_p {
        req = req.header("X-Top-P", format!("{p}"));
    }
    if let Some(k) = params.top_k {
        req = req.header("X-Top-K", k.to_string());
    }
    req
}

fn local_inference_http_client() -> &'static reqwest::Client {
    static CLIENT: OnceLock<reqwest::Client> = OnceLock::new();
    CLIENT.get_or_init(|| {
        reqwest::Client::builder()
            .pool_max_idle_per_host(32)
            .tcp_keepalive(Some(Duration::from_secs(120)))
            .timeout(Duration::from_secs(300))
            .build()
            .expect("reqwest client for local inference")
    })
}

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

    let coordinator_http = coordinator_url
        .replace("wss://", "https://")
        .replace("ws://", "http://")
        .replace("/ws", "");

    // Layer assignment comes from coordinator dynamically after registration.
    let assigned_layers = std::sync::Arc::new(tokio::sync::RwLock::new(None::<(usize, usize)>));
    let model_dir = model_dir.to_string();
    let coordinator_http = coordinator_http.to_string();

    // Track whether layers are ready (persists across reconnections)
    let layers_ready = std::sync::Arc::new(std::sync::atomic::AtomicBool::new(false));
    let download_failed = std::sync::Arc::new(std::sync::atomic::AtomicBool::new(false));

    // GPU nodes: wait for LayerAssignment, then download safetensors layers
    // CPU nodes: training/synthesis only, report ready immediately
    if !has_gpu {
        layers_ready.store(true, std::sync::atomic::Ordering::SeqCst);
        info!("CPU node — training/synthesis only, no inference layers needed");
    }

    let sampling_by_request: Arc<Mutex<HashMap<String, SamplingParams>>> =
        Arc::new(Mutex::new(HashMap::new()));

    loop {
        tokio::select! {
            _ = shutdown.cancelled() => {
                info!("WS worker shutting down");
                break;
            }
            result = connect_and_run(&ws_url, node_name, hardware_info, has_gpu, ram_mb, &model_dir, &coordinator_http, &assigned_layers, &layers_ready, &download_failed, &shutdown, &sampling_by_request) => {
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
    coordinator_http: &str,
    assigned_layers: &std::sync::Arc<tokio::sync::RwLock<Option<(usize, usize)>>>,
    layers_ready: &std::sync::Arc<std::sync::atomic::AtomicBool>,
    download_failed: &std::sync::Arc<std::sync::atomic::AtomicBool>,
    shutdown: &tokio_util::sync::CancellationToken,
    sampling_by_request: &Arc<Mutex<HashMap<String, SamplingParams>>>,
) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
    info!(url = ws_url, "Connecting WebSocket...");

    let (ws, _) = connect_async(ws_url).await?;
    let (mut sink, mut stream) = ws.split();

    // Register — coordinator will send LayerAssignment for GPU nodes
    let reg = ClientMessage::Register {
        node_name: node_name.to_string(),
        hardware_info: hardware_info.to_string(),
        models: vec![],
        ram_mb,
        has_gpu,
        layer_range: None, // coordinator assigns dynamically
        client_version: concat!(env!("CARGO_PKG_VERSION"), "-", env!("GIT_HASH")).to_string(),
        os: std::env::consts::OS.to_string(),
    };
    sink.send(Message::Text(serde_json::to_string(&reg)?.into()))
        .await?;

    // Report initial state
    let initial_state = if layers_ready.load(std::sync::atomic::Ordering::SeqCst) {
        (
            "ready".to_string(),
            if has_gpu {
                "GPU node — waiting for layer assignment".to_string()
            } else {
                "CPU node — training/synthesis only".to_string()
            },
        )
    } else {
        (
            "connecting".to_string(),
            "Waiting for layer assignment from coordinator".to_string(),
        )
    };
    let state_msg = ClientMessage::StateUpdate {
        state: initial_state.0,
        detail: initial_state.1,
    };
    sink.send(Message::Text(serde_json::to_string(&state_msg)?.into()))
        .await?;
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
                                // Handle LayerAssignment from coordinator
                                if let CoordinatorMessage::LayerAssignment { layer_start, layer_end } = &coord_msg {
                                    info!(layers = format!("{layer_start}-{layer_end}"), "Received layer assignment from coordinator");
                                    let ls = *layer_start;
                                    let le = *layer_end;
                                    *assigned_layers.write().await = Some((ls, le));

                                    // GPU nodes download safetensors layers for distributed inference

                                    let dl_msg = ClientMessage::StateUpdate {
                                        state: "downloading".to_string(),
                                        detail: format!("Downloading layers {ls}-{le}"),
                                    };
                                    let _ = sink.send(Message::Text(serde_json::to_string(&dl_msg).unwrap_or_default().into())).await;

                                    let cache = format!("{model_dir}/inference_layers_{ls}_{le}");
                                    let mut ok = false;
                                    for attempt in 1..=3 {
                                        match download_layers(&cache, coordinator_http, ls, le).await {
                                            Ok(()) => { info!("Layer weights ready (attempt {attempt})"); layers_ready.store(true, std::sync::atomic::Ordering::SeqCst); ok = true; break; }
                                            Err(e) => { warn!("Layer download failed (attempt {attempt}/3): {e}"); if attempt < 3 { tokio::time::sleep(std::time::Duration::from_secs(10)).await; } }
                                        }
                                    }
                                    let state_msg = if ok {
                                        // Start persistent inference server
                                        info!("Starting inference server for layers {ls}-{le}");
                                        match start_inference_server(&cache, ls, le, 18100).await {
                                            Ok(_child) => {
                                                info!("Inference server running on port 18100");
                                                // Wait a moment for server to be fully ready
                                                tokio::time::sleep(std::time::Duration::from_secs(1)).await;
                                            }
                                            Err(e) => warn!("Failed to start inference server: {e}"),
                                        }
                                        ClientMessage::StateUpdate { state: "ready".to_string(), detail: format!("Layers {ls}-{le} loaded, inference server running") }
                                    } else {
                                        download_failed.store(true, std::sync::atomic::Ordering::SeqCst);
                                        ClientMessage::StateUpdate { state: "error".to_string(), detail: "Layer download failed".to_string() }
                                    };
                                    let _ = sink.send(Message::Text(serde_json::to_string(&state_msg).unwrap_or_default().into())).await;
                                    continue;
                                }

                                let current_model_dir = if let Some((ls, le)) = *assigned_layers.read().await {
                                    format!("{model_dir}/inference_layers_{ls}_{le}")
                                } else { model_dir.to_string() };

                                if let Some(response) = handle_coordinator_message(
                                    coord_msg,
                                    &current_model_dir,
                                    sampling_by_request,
                                ).await {
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
                        let bin_model_dir = if let Some((ls, le)) = *assigned_layers.read().await {
                            format!("{model_dir}/inference_layers_{ls}_{le}")
                        } else { model_dir.to_string() };
                        if let Some(response) = handle_binary_forward(
                            data.to_vec(),
                            &bin_model_dir,
                            sampling_by_request,
                        ).await {
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
    sampling_by_request: &Arc<Mutex<HashMap<String, SamplingParams>>>,
) -> Option<WsResponse> {
    match msg {
        CoordinatorMessage::InferenceStart {
            request_id,
            token_ids,
            layer_start,
            layer_end,
            max_tokens: _,
            temperature,
            top_p,
            top_k,
        } => {
            if let Ok(mut g) = sampling_by_request.lock() {
                let mut p = SamplingParams::default();
                merge_optional_sampling(&mut p, temperature, top_p, top_k);
                g.insert(request_id.clone(), p);
            }
            info!(request_id = %request_id, layers = format!("{layer_start}-{layer_end}"), tokens = token_ids.len(), "Inference start: embed + forward");

            let result =
                run_layer_forward(model_dir, &token_ids, layer_start, layer_end, true, &request_id).await;

            match result {
                Ok((hidden_data, shape)) => {
                    info!(request_id = %request_id, hidden_size = hidden_data.len(), "Forward complete, sending hidden states");
                    let msg = ClientMessage::ForwardResult {
                        request_id: request_id.clone(),
                        hidden_states: vec![],
                        shape,
                    };
                    let mut payload = vec![0u8; 36];
                    payload[..request_id.len().min(36)]
                        .copy_from_slice(&request_id.as_bytes()[..request_id.len().min(36)]);
                    payload.extend_from_slice(&hidden_data);
                    Some(WsResponse::TextAndBinary(msg, payload))
                }
                Err(e) => {
                    error!(request_id = %request_id, "Embed+forward failed: {e}");
                    None
                }
            }
        }
        CoordinatorMessage::InferenceForward {
            request_id,
            layer_start,
            layer_end,
            is_last,
            temperature,
            top_p,
            top_k,
            ..
        } => {
            if is_last {
                if let Ok(mut g) = sampling_by_request.lock() {
                    let e = g.entry(request_id.clone()).or_default();
                    merge_optional_sampling(e, temperature, top_p, top_k);
                }
            }
            info!(request_id = %request_id, layers = format!("{layer_start}-{layer_end}"), last = is_last, "Inference forward received");
            // Hidden states arrive as binary frame — handled in handle_binary_forward
            None
        }
        CoordinatorMessage::InferenceContinue {
            request_id,
            new_token_id,
            layer_start,
            layer_end,
            temperature,
            top_p,
            top_k,
            ..
        } => {
            if let Ok(mut g) = sampling_by_request.lock() {
                let e = g.entry(request_id.clone()).or_default();
                merge_from_continue(e, temperature, top_p, top_k);
            }
            info!(
                request_id = %request_id,
                new_token_id,
                layers = format!("{layer_start}-{layer_end}"),
                "Inference continue: single-token embed + forward"
            );
            match run_embed_step(model_dir, new_token_id, &request_id).await {
                Ok((hidden_data, shape)) => {
                    let msg = ClientMessage::ForwardResult {
                        request_id: request_id.clone(),
                        hidden_states: vec![],
                        shape,
                    };
                    let mut payload = vec![0u8; 36];
                    payload[..request_id.len().min(36)]
                        .copy_from_slice(&request_id.as_bytes()[..request_id.len().min(36)]);
                    payload.extend_from_slice(&hidden_data);
                    Some(WsResponse::TextAndBinary(msg, payload))
                }
                Err(e) => {
                    error!(request_id = %request_id, "embed_step failed: {e}");
                    None
                }
            }
        }
        CoordinatorMessage::InferenceEnd { request_id } => {
            if let Ok(mut g) = sampling_by_request.lock() {
                g.remove(&request_id);
            }
            clear_local_kv_cache(&request_id).await;
            None
        }
        CoordinatorMessage::Ping => Some(WsResponse::Text(ClientMessage::Pong)),
        _ => None,
    }
}

// ---------------------------------------------------------------------------
// TCP inference transport
// Frame (both directions): [4B LE total_payload][4B LE json_len][json][binary]
// ---------------------------------------------------------------------------

struct TcpForwardResponse {
    shape: Vec<usize>,
    next_token: Option<u32>,
    hidden_data: Vec<u8>,
}

async fn forward_via_tcp(
    hidden_data: &[u8],
    mode: &str,
    request_id: &str,
    seq_len: usize,
    hidden_size: usize,
    sampling: &SamplingParams,
) -> Result<TcpForwardResponse, Box<dyn std::error::Error + Send + Sync>> {
    use tokio::io::{AsyncReadExt, AsyncWriteExt};
    use tokio::net::TcpStream;

    let addr = "127.0.0.1:18101";
    let mut stream = tokio::time::timeout(
        Duration::from_millis(200),
        TcpStream::connect(addr),
    )
    .await
    .map_err(|_| format!("TCP connect to {addr} timed out"))?
    .map_err(|e| format!("TCP connect to {addr} failed: {e}"))?;

    // Build JSON metadata header
    let mut meta = serde_json::json!({
        "mode": mode,
        "request_id": request_id,
        "shape": [1u32, seq_len as u32, hidden_size as u32],
    });
    if let Some(t) = sampling.temperature { meta["temperature"] = t.into(); }
    if let Some(p) = sampling.top_p      { meta["top_p"]       = p.into(); }
    if let Some(k) = sampling.top_k      { meta["top_k"]       = k.into(); }
    let json_bytes = serde_json::to_vec(&meta)?;

    // Write request frame
    let total_payload = json_bytes.len() + hidden_data.len();
    let mut frame = Vec::with_capacity(8 + total_payload);
    frame.extend_from_slice(&(total_payload as u32).to_le_bytes());
    frame.extend_from_slice(&(json_bytes.len() as u32).to_le_bytes());
    frame.extend_from_slice(&json_bytes);
    frame.extend_from_slice(hidden_data);
    stream.write_all(&frame).await?;

    // Read response frame header
    let mut hdr = [0u8; 8];
    stream.read_exact(&mut hdr).await?;
    let resp_total  = u32::from_le_bytes(hdr[..4].try_into().unwrap()) as usize;
    let resp_jlen   = u32::from_le_bytes(hdr[4..8].try_into().unwrap()) as usize;
    let resp_binlen = resp_total.saturating_sub(resp_jlen);

    let mut resp_json_buf = vec![0u8; resp_jlen];
    stream.read_exact(&mut resp_json_buf).await?;
    let mut resp_bin = vec![0u8; resp_binlen];
    if resp_binlen > 0 {
        stream.read_exact(&mut resp_bin).await?;
    }

    let resp: serde_json::Value = serde_json::from_slice(&resp_json_buf)?;
    let status = resp["status"].as_u64().unwrap_or(200) as u16;
    if status >= 400 {
        let err = resp["error"].as_str().unwrap_or("unknown error");
        return Err(format!("inference server error {status}: {err}").into());
    }

    let shape = resp["shape"]
        .as_array()
        .map(|a| a.iter().filter_map(|v| v.as_u64().map(|n| n as usize)).collect())
        .unwrap_or_default();
    let next_token = resp["next_token"].as_u64().map(|t| t as u32);

    Ok(TcpForwardResponse { shape, next_token, hidden_data: resp_bin })
}

async fn handle_binary_forward(
    data: Vec<u8>,
    model_dir: &str,
    sampling_by_request: &Arc<Mutex<HashMap<String, SamplingParams>>>,
) -> Option<WsResponse> {
    if data.len() < 36 {
        return None;
    }
    let request_id = String::from_utf8_lossy(&data[..36])
        .trim_end_matches('\0')
        .to_string();
    let hidden_data = data[36..].to_vec();

    info!(request_id = %request_id, size = hidden_data.len(), "Received hidden states for forward pass");

    let (layer_start, layer_end) = parse_layer_range(model_dir);
    let is_last = layer_end >= 28;
    let mode = if is_last { "generate" } else { "forward" };
    let hidden_size = 3584usize; // Qwen2.5-7B hidden size
    let seq_len = hidden_data.len() / (hidden_size * 2); // fp16 = 2 bytes

    let sampling = sampling_by_request
        .lock()
        .ok()
        .and_then(|g| g.get(&request_id).cloned())
        .unwrap_or_default();

    info!(request_id = %request_id, mode, layers = format!("{layer_start}-{layer_end}"), "Running forward pass");

    // Try TCP first (lower framing overhead); fall back to HTTP on connect failure.
    match forward_via_tcp(&hidden_data, mode, &request_id, seq_len, hidden_size, &sampling).await {
        Ok(tcp_resp) => {
            if is_last {
                let token_id = match tcp_resp.next_token {
                    Some(t) => t,
                    None => {
                        error!(request_id = %request_id, "TCP generate: missing next_token in response");
                        return None;
                    }
                };
                let is_eos = token_id == 151643 || token_id == 151644 || token_id == 151645;
                info!(request_id = %request_id, token_id, eos = is_eos, transport = "tcp", "Token generated");
                Some(WsResponse::Text(ClientMessage::TokenGenerated { request_id, token_id, is_eos }))
            } else {
                info!(request_id = %request_id, hidden_size = tcp_resp.hidden_data.len(), transport = "tcp", "Forward complete");
                let msg = ClientMessage::ForwardResult { request_id: request_id.clone(), hidden_states: vec![], shape: tcp_resp.shape };
                let mut payload = vec![0u8; 36];
                payload[..request_id.len().min(36)].copy_from_slice(&request_id.as_bytes()[..request_id.len().min(36)]);
                payload.extend_from_slice(&tcp_resp.hidden_data);
                Some(WsResponse::TextAndBinary(msg, payload))
            }
        }
        Err(tcp_err) => {
            // TCP unavailable (old Python server or port not yet open) — fall back to HTTP.
            warn!(request_id = %request_id, "TCP forward failed ({tcp_err}), falling back to HTTP");
            handle_binary_forward_http(&hidden_data, mode, &request_id, seq_len, hidden_size, is_last, &sampling).await
        }
    }
}

async fn handle_binary_forward_http(
    hidden_data: &[u8],
    mode: &str,
    request_id: &str,
    seq_len: usize,
    hidden_size: usize,
    is_last: bool,
    sampling: &SamplingParams,
) -> Option<WsResponse> {
    let url = "http://127.0.0.1:18100";
    let client = local_inference_http_client();

    let post = client
        .post(url)
        .header("X-Mode", mode)
        .header("X-Shape", format!("[1,{seq_len},{hidden_size}]"))
        .header("X-Request-Id", request_id)
        .header("Content-Type", "application/octet-stream");
    let post = if is_last { apply_sampling_headers(post, sampling) } else { post };

    let resp = match post.body(hidden_data.to_vec()).send().await {
        Ok(r) => r,
        Err(e) => { error!("Inference server unreachable: {e}"); return None; }
    };

    if is_last {
        let result: serde_json::Value = match resp.json().await {
            Ok(r) => r,
            Err(e) => { error!("Bad generate response: {e}"); return None; }
        };
        if let Some(err) = result.get("error") { error!("Generate error: {err}"); return None; }
        let token_id = result["token_id"].as_u64()? as u32;
        let is_eos = token_id == 151643 || token_id == 151644 || token_id == 151645;
        info!(request_id = %request_id, token_id, eos = is_eos, transport = "http", "Token generated");
        Some(WsResponse::Text(ClientMessage::TokenGenerated { request_id: request_id.to_string(), token_id, is_eos }))
    } else {
        let shape_str = resp.headers().get("X-Shape")
            .and_then(|v| v.to_str().ok()).unwrap_or("[]").to_string();
        let shape: Vec<usize> = serde_json::from_str(&shape_str).unwrap_or_default();
        let hidden_out = match resp.bytes().await {
            Ok(b) => b.to_vec(),
            Err(e) => { error!("Bad forward response: {e}"); return None; }
        };
        info!(request_id = %request_id, hidden_size = hidden_out.len(), transport = "http", "Forward complete");
        let msg = ClientMessage::ForwardResult { request_id: request_id.to_string(), hidden_states: vec![], shape };
        let mut payload = vec![0u8; 36];
        payload[..request_id.len().min(36)].copy_from_slice(&request_id.as_bytes()[..request_id.len().min(36)]);
        payload.extend_from_slice(&hidden_out);
        Some(WsResponse::TextAndBinary(msg, payload))
    }
}

fn parse_layer_range(model_dir: &str) -> (usize, usize) {
    // Parse from path like: .../inference_layers_10_20
    let parts: Vec<&str> = model_dir
        .rsplit('/')
        .next()
        .unwrap_or("")
        .split('_')
        .collect();
    if parts.len() >= 4 {
        let start = parts[parts.len() - 2].parse().unwrap_or(0);
        let end = parts[parts.len() - 1].parse().unwrap_or(28);
        (start, end)
    } else {
        (0, 28) // fallback: all layers
    }
}

/// Start the persistent Python inference server (loads model once, serves via HTTP)
async fn start_inference_server(
    model_dir: &str,
    layer_start: usize,
    layer_end: usize,
    port: u16,
) -> Result<tokio::process::Child, Box<dyn std::error::Error + Send + Sync>> {
    let script = find_script("inference/node_forward.py");
    let python = find_python();

    info!(port, layers = format!("{layer_start}-{layer_end}"), "Starting persistent inference server");

    let mut child = tokio::process::Command::new(&python)
        .arg(&script)
        .arg("--mode").arg("serve")
        .arg("--model-dir").arg(model_dir)
        .arg("--layer-start").arg(layer_start.to_string())
        .arg("--layer-end").arg(layer_end.to_string())
        .arg("--port").arg(port.to_string())
        .stdout(std::process::Stdio::piped())
        .stderr(std::process::Stdio::piped())
        .spawn()
        .map_err(|e| format!("Failed to start inference server: {e}"))?;

    // Wait for "ready" on stdout
    let stdout = child.stdout.take().unwrap();
    let mut reader = tokio::io::BufReader::new(stdout);
    let mut line = String::new();
    tokio::time::timeout(
        std::time::Duration::from_secs(120),
        tokio::io::AsyncBufReadExt::read_line(&mut reader, &mut line),
    ).await
        .map_err(|_| "Inference server startup timed out (120s)")?
        .map_err(|e| format!("Failed to read server stdout: {e}"))?;

    info!("Inference server ready: {}", line.trim());
    Ok(child)
}

/// Call the persistent inference server via HTTP
async fn run_layer_forward(
    model_dir: &str,
    token_ids: &[u32],
    layer_start: usize,
    layer_end: usize,
    is_first: bool,
    request_id: &str,
) -> Result<(Vec<u8>, Vec<usize>), Box<dyn std::error::Error + Send + Sync>> {
    let port = 18100u16;
    let url = format!("http://127.0.0.1:{port}");
    let client = local_inference_http_client();

    if is_first {
        // Embed mode: send token IDs, get hidden states back
        let resp = client.post(&url)
            .json(&serde_json::json!({"mode": "embed", "token_ids": token_ids, "request_id": request_id}))
            .send().await
            .map_err(|e| format!("Inference server unreachable: {e}. Is it running?"))?;

        let shape_str = resp.headers().get("X-Shape")
            .and_then(|v| v.to_str().ok())
            .unwrap_or("[]").to_string();
        let shape: Vec<usize> = serde_json::from_str(&shape_str).unwrap_or_default();
        let hidden_data = resp.bytes().await?.to_vec();
        Ok((hidden_data, shape))
    } else {
        // Forward/generate: send hidden states as raw binary
        let input_file = format!("/tmp/hyverk_ws_in_{}.bin", layer_start);
        let hidden_bytes = std::fs::read(&input_file)
            .map_err(|e| format!("Can't read hidden states: {e}"))?;

        let (layer_start, layer_end) = parse_layer_range(model_dir);
        let is_last = layer_end >= 28;
        let mode = if is_last { "generate" } else { "forward" };

        let hidden_size = 3584usize;
        let seq_len = hidden_bytes.len() / (hidden_size * 2);

        let resp = client.post(&url)
            .header("X-Mode", mode)
            .header("X-Shape", format!("[1,{seq_len},{hidden_size}]"))
            .header("X-Request-Id", request_id)
            .header("Content-Type", "application/octet-stream")
            .body(hidden_bytes)
            .send().await
            .map_err(|e| format!("Inference server error: {e}"))?;

        if is_last {
            // Generate mode: response is JSON with token_id
            let result: serde_json::Value = resp.json().await?;
            // Return empty hidden data + token_id encoded in shape
            let token_id = result["token_id"].as_u64().unwrap_or(0) as usize;
            Ok((vec![], vec![token_id]))
        } else {
            let shape_str = resp.headers().get("X-Shape")
                .and_then(|v| v.to_str().ok())
                .unwrap_or("[]").to_string();
            let shape: Vec<usize> = serde_json::from_str(&shape_str).unwrap_or_default();
            let hidden_data = resp.bytes().await?.to_vec();
            Ok((hidden_data, shape))
        }
    }
}

/// Incremental decode: embed one new token using KV cache on the local inference server.
async fn run_embed_step(
    _model_dir: &str,
    new_token_id: u32,
    request_id: &str,
) -> Result<(Vec<u8>, Vec<usize>), Box<dyn std::error::Error + Send + Sync>> {
    let port = 18100u16;
    let url = format!("http://127.0.0.1:{port}");
    let client = local_inference_http_client();
    let resp = client
        .post(&url)
        .json(&serde_json::json!({
            "mode": "embed_step",
            "token_id": new_token_id,
            "request_id": request_id,
        }))
        .send()
        .await
        .map_err(|e| format!("Inference server unreachable: {e}. Is it running?"))?;
    let shape_str = resp
        .headers()
        .get("X-Shape")
        .and_then(|v| v.to_str().ok())
        .unwrap_or("[]")
        .to_string();
    let shape: Vec<usize> = serde_json::from_str(&shape_str).unwrap_or_default();
    let hidden_data = resp.bytes().await?.to_vec();
    Ok((hidden_data, shape))
}

async fn clear_local_kv_cache(request_id: &str) {
    let client = local_inference_http_client();
    let _ = client
        .post("http://127.0.0.1:18100")
        .json(&serde_json::json!({"mode": "clear_cache", "request_id": request_id}))
        .send()
        .await;
}

/// Legacy: Call Python subprocess (used only for download)
async fn run_layer_forward_subprocess(
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
        .arg("--mode")
        .arg(mode)
        .arg("--model-dir")
        .arg(model_dir)
        .arg("--layer-start")
        .arg(layer_start.to_string())
        .arg("--layer-end")
        .arg(layer_end.to_string())
        .arg("--output-file")
        .arg(&output_file);

    if is_first {
        let ids_str: String = token_ids
            .iter()
            .map(|id| id.to_string())
            .collect::<Vec<_>>()
            .join(",");
        cmd.arg("--token-ids").arg(&ids_str);
    }

    let out = tokio::time::timeout(std::time::Duration::from_secs(600), cmd.output())
        .await
        .map_err(|_| "Forward pass timed out")?
        .map_err(|e| format!("Failed to run forward script: {e}"))?;

    if !out.status.success() {
        let stderr = String::from_utf8_lossy(&out.stderr);
        return Err(format!("Forward failed: {stderr}").into());
    }

    let stdout = String::from_utf8_lossy(&out.stdout);
    let result: serde_json::Value =
        serde_json::from_str(stdout.trim()).map_err(|e| format!("Bad forward result: {e}"))?;

    let hidden_data =
        std::fs::read(&output_file).map_err(|e| format!("Can't read hidden states: {e}"))?;
    let _ = std::fs::remove_file(&output_file);

    let shape = result["shape"]
        .as_array()
        .map(|a| {
            a.iter()
                .filter_map(|v| v.as_u64().map(|n| n as usize))
                .collect()
        })
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
    // Skip download if safetensors files already present in the target directory.
    // This lets operators pre-populate ~/.hyverk/qwen2.5-7b/inference_layers_X_Y
    // (or symlink it) without relying on coordinator-side model hosting.
    if let Ok(mut entries) = std::fs::read_dir(model_dir) {
        let has_weights = entries.any(|e| {
            e.ok().and_then(|e| e.file_name().into_string().ok())
                .map(|n| n.ends_with(".safetensors") || n == "config.json")
                .unwrap_or(false)
        });
        if has_weights {
            info!("Layer weights already present at {model_dir}, skipping download");
            return Ok(());
        }
    }

    let script = find_script("inference/node_forward.py");
    let python = find_python();

    let out = tokio::process::Command::new(&python)
        .arg(&script)
        .arg("--mode")
        .arg("download")
        .arg("--model-dir")
        .arg(model_dir)
        .arg("--coordinator")
        .arg(coordinator_url)
        .arg("--layer-start")
        .arg(layer_start.to_string())
        .arg("--layer-end")
        .arg(layer_end.to_string())
        .output()
        .await?;

    let stderr = String::from_utf8_lossy(&out.stderr);
    for line in stderr.lines() {
        info!(target: "download", "{}", line);
    }

    let stdout = String::from_utf8_lossy(&out.stdout);
    if !out.status.success() {
        let detail = if stderr.trim().is_empty() {
            format!(
                "exit={} stdout={}",
                out.status.code().map(|c| c.to_string()).unwrap_or_else(|| "?".into()),
                stdout.trim()
            )
        } else {
            stderr.into_owned()
        };
        return Err(format!("Layer download failed: {detail}").into());
    }

    info!("Layer download result: {}", stdout.trim());
    Ok(())
}

fn find_python() -> String {
    // Windows: check for python.exe, python3.exe
    // Unix: check for python3 in common locations
    let candidates = if cfg!(windows) {
        vec![".venv\\Scripts\\python.exe", "python.exe", "python3.exe"]
    } else {
        vec![".venv/bin/python3", "/usr/bin/python3", "python3"]
    };

    for p in &candidates {
        if std::path::Path::new(p).exists() {
            return p.to_string();
        }
    }

    // Try to find python in PATH using 'which' or 'where'
    if cfg!(windows) {
        if let Ok(output) = std::process::Command::new("where").arg("python").output() {
            if output.status.success() {
                if let Ok(path) = String::from_utf8(output.stdout) {
                    if let Some(first_line) = path.lines().next() {
                        return first_line.trim().to_string();
                    }
                }
            }
        }
        "python.exe".to_string()
    } else {
        "python3".to_string()
    }
}

fn find_script(name: &str) -> String {
    for prefix in &["", "../", "../../", "/app/"] {
        let p = format!("{}{}", prefix, name);
        if std::path::Path::new(&p).exists() {
            return p;
        }
    }
    name.to_string()
}

#[cfg(test)]
mod sampling_cache_tests {
    use super::*;

    #[test]
    fn insert_lookup_cleanup_lifecycle() {
        let m = Arc::new(Mutex::new(HashMap::new()));
        let rid = "req-a".to_string();

        {
            let mut g = m.lock().unwrap();
            let mut p = SamplingParams::default();
            merge_optional_sampling(&mut p, Some(0.8), Some(0.95), Some(40));
            g.insert(rid.clone(), p);
        }
        assert_eq!(
            m.lock().unwrap().get(&rid).cloned(),
            Some(SamplingParams {
                temperature: Some(0.8),
                top_p: Some(0.95),
                top_k: Some(40),
            })
        );

        {
            let mut g = m.lock().unwrap();
            let e = g.entry(rid.clone()).or_default();
            merge_from_continue(e, 0.2, None, Some(50));
        }
        assert_eq!(
            m.lock().unwrap().get(&rid).cloned(),
            Some(SamplingParams {
                temperature: Some(0.2),
                top_p: Some(0.95),
                top_k: Some(50),
            })
        );

        {
            let mut g = m.lock().unwrap();
            g.remove(&rid);
        }
        assert!(m.lock().unwrap().get(&rid).is_none());
    }

    #[test]
    fn merge_optional_preserves_unset() {
        let mut p = SamplingParams {
            temperature: Some(1.0),
            top_p: Some(0.9),
            top_k: None,
        };
        merge_optional_sampling(&mut p, None, Some(0.5), None);
        assert_eq!(
            p,
            SamplingParams {
                temperature: Some(1.0),
                top_p: Some(0.5),
                top_k: None,
            }
        );
    }
}

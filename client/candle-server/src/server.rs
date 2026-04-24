/// HTTP (axum) + TCP server matching node_forward.py protocol exactly.
///
/// HTTP endpoints:
///   GET /health  → JSON status
///   GET /        → same as /health
///   POST /       → inference (X-Mode: forward | generate, X-Shape, X-Request-Id headers)
///
/// TCP framing (port+1):
///   Request:  [4B LE total_payload][4B LE json_header_len][json][binary]
///   Response: [4B LE total_payload][4B LE json_header_len][json][binary]
use anyhow::Result;
use axum::{
    body::Bytes,
    extract::State,
    http::{HeaderMap, StatusCode},
    response::IntoResponse,
    routing::get,
    Json, Router,
};
use serde_json::{json, Value};
use std::{net::SocketAddr, sync::Arc};
use tokio::{
    io::{AsyncReadExt, AsyncWriteExt},
    net::{TcpListener, TcpStream},
    sync::Mutex,
};
use tracing::{info, warn};

use crate::worker::InferenceWorker;

pub type SharedWorker = Arc<Mutex<InferenceWorker>>;

// ── HTTP ──────────────────────────────────────────────────────────────────────

async fn health(State(worker): State<SharedWorker>) -> impl IntoResponse {
    let w = worker.lock().await;
    Json(w.health_json())
}

async fn inference_http(
    State(worker): State<SharedWorker>,
    headers: HeaderMap,
    body: Bytes,
) -> impl IntoResponse {
    let mode = headers
        .get("X-Mode")
        .and_then(|v| v.to_str().ok())
        .unwrap_or("forward")
        .to_string();
    let shape_str = headers
        .get("X-Shape")
        .and_then(|v| v.to_str().ok())
        .unwrap_or("[1,1,3584]")
        .to_string();
    let request_id = headers
        .get("X-Request-Id")
        .and_then(|v| v.to_str().ok())
        .unwrap_or("unknown")
        .to_string();

    // Parse temperature/top_p/top_k from X-Temperature etc.
    let temperature = headers
        .get("X-Temperature")
        .and_then(|v| v.to_str().ok())
        .and_then(|s| s.parse::<f64>().ok());
    let top_p = headers
        .get("X-Top-P")
        .and_then(|v| v.to_str().ok())
        .and_then(|s| s.parse::<f64>().ok());
    let top_k = headers
        .get("X-Top-K")
        .and_then(|v| v.to_str().ok())
        .and_then(|s| s.parse::<usize>().ok());

    let shape = parse_shape(&shape_str);
    let sampling = crate::sampling::SamplingParams { temperature, top_p, top_k };

    let mut w = worker.lock().await;
    match w.run_inference(&mode, &request_id, &shape, body.as_ref(), &sampling) {
        Ok((resp_headers, resp_body)) => {
            let mut response = axum::response::Response::new(axum::body::Body::from(resp_body));
            for (k, v) in resp_headers {
                response.headers_mut().insert(k, v);
            }
            response
        }
        Err(e) => {
            warn!("inference error: {e}");
            let mut r = axum::response::Response::new(axum::body::Body::from(
                format!("{{\"error\":\"{e}\"}}").into_bytes(),
            ));
            *r.status_mut() = StatusCode::INTERNAL_SERVER_ERROR;
            r
        }
    }
}

async fn root_handler(
    state: State<SharedWorker>,
    headers: HeaderMap,
    body: Bytes,
) -> impl IntoResponse {
    if body.is_empty() && !headers.contains_key("X-Mode") {
        health(state).await.into_response()
    } else {
        inference_http(state, headers, body).await.into_response()
    }
}

pub fn build_router(worker: SharedWorker) -> Router {
    Router::new()
        .route("/health", get(health))
        .route("/", axum::routing::any(root_handler))
        .with_state(worker)
}

// ── TCP ───────────────────────────────────────────────────────────────────────

pub async fn run_tcp_server(addr: SocketAddr, worker: SharedWorker) -> Result<()> {
    let listener = TcpListener::bind(addr).await?;
    info!("TCP server listening on {addr}");
    loop {
        let (stream, peer) = listener.accept().await?;
        let w = worker.clone();
        tokio::spawn(async move {
            if let Err(e) = handle_tcp_client(stream, w).await {
                warn!("TCP client {peer} error: {e}");
            }
        });
    }
}

async fn handle_tcp_client(mut stream: TcpStream, worker: SharedWorker) -> Result<()> {
    // Read 8-byte header
    let mut hdr = [0u8; 8];
    stream.read_exact(&mut hdr).await?;
    let total = u32::from_le_bytes(hdr[0..4].try_into().unwrap()) as usize;
    let json_len = u32::from_le_bytes(hdr[4..8].try_into().unwrap()) as usize;
    let bin_len = total.saturating_sub(json_len);

    let mut json_buf = vec![0u8; json_len];
    stream.read_exact(&mut json_buf).await?;
    let mut bin_buf = vec![0u8; bin_len];
    if bin_len > 0 {
        stream.read_exact(&mut bin_buf).await?;
    }

    let meta: Value = serde_json::from_slice(&json_buf)?;
    let mode = meta["mode"].as_str().unwrap_or("forward");
    let request_id = meta["request_id"].as_str().unwrap_or("unknown");
    let shape = if let Some(arr) = meta["shape"].as_array() {
        arr.iter().filter_map(|v| v.as_u64().map(|u| u as usize)).collect::<Vec<_>>()
    } else {
        vec![1, 1, 3584]
    };
    let sampling = crate::sampling::SamplingParams {
        temperature: meta["temperature"].as_f64(),
        top_p: meta["top_p"].as_f64(),
        top_k: meta["top_k"].as_u64().map(|u| u as usize),
    };

    let (resp_headers_map, resp_body) = {
        let mut w = worker.lock().await;
        w.run_inference(mode, request_id, &shape, &bin_buf, &sampling)?
    };

    // Build response JSON from headers
    let mut resp_meta = serde_json::Map::new();
    resp_meta.insert("status".to_string(), json!("ok"));
    for (k, v) in &resp_headers_map {
        if let Ok(s) = v.to_str() {
            resp_meta.insert(k.as_str().to_string(), json!(s));
        }
    }
    let resp_json = serde_json::to_vec(&Value::Object(resp_meta))?;
    let total_resp = resp_json.len() + resp_body.len();

    let mut frame = Vec::with_capacity(8 + total_resp);
    frame.extend_from_slice(&(total_resp as u32).to_le_bytes());
    frame.extend_from_slice(&(resp_json.len() as u32).to_le_bytes());
    frame.extend_from_slice(&resp_json);
    frame.extend_from_slice(&resp_body);
    stream.write_all(&frame).await?;
    Ok(())
}

// ── helpers ───────────────────────────────────────────────────────────────────

fn parse_shape(s: &str) -> Vec<usize> {
    s.trim_matches(|c| c == '[' || c == ']')
        .split(',')
        .filter_map(|p| p.trim().parse::<usize>().ok())
        .collect()
}

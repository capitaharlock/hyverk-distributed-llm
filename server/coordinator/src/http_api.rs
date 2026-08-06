// @llm-context: _rjj/links.md
// @llm-depends: task_store.rs, registry.rs, dataset_store.rs, metrics.rs

use crate::dataset_store::{DatasetExample, DatasetStore};
use crate::registry::NodeRegistry;
use crate::task_store::{TaskStatus, TaskStore};
use crate::training_store::TrainingStore;
use crate::layer_training::LayerTrainingStore;
use crate::metrics::LiveCounters;
use hyverk_rag::{RagConfig, SourceType, store::RagStore};
use axum::extract::{Path, Query, Request, State};
use axum::http::{HeaderMap, StatusCode};
use axum::middleware::{self, Next};
use axum::response::IntoResponse;
use axum::routing::{get, post};
use axum::{Json, Router};
use serde::{Deserialize, Serialize};
use std::sync::Arc;
use std::time::{SystemTime, UNIX_EPOCH};
use uuid::Uuid;

#[derive(Clone)]
pub struct AppState {
    pub task_store: TaskStore,
    pub registry: NodeRegistry,
    pub router: crate::router::TaskRouter,
    pub dataset_store: DatasetStore,
    pub training_store: TrainingStore,
    pub rag_store: Arc<RagStore>,
    pub layer_training: LayerTrainingStore,
    pub cluster_manager: crate::serving_clusters::ClusterManager,
    pub pending_signals: Arc<tokio::sync::RwLock<std::collections::HashMap<String, serde_json::Value>>>,
    pub ws_state: Arc<crate::ws_handler::WsState>,
    pub counters: Arc<LiveCounters>,
    pub started_at_secs: u64,
}

pub fn create_router(state: AppState) -> Router {
    let state = Arc::new(state);

    // Public inference surface — gated when HYVERK_API_KEY is set.
    let inference_api = Router::new()
        .route("/api/v1/ws-inference", post(ws_inference))
        .route("/api/v1/inference", post(create_inference))
        .route("/api/v1/inference/distributed", post(proxy_distributed_inference))
        .route_layer(middleware::from_fn(require_api_key));

    Router::new()
        .route("/", get(dashboard))
        .route("/models", get(models_page))
        .route("/devices", get(devices_page))
        .route("/test", get(test_page))
        .route("/health", get(health))
        .route("/ws", get(crate::ws_handler::ws_upgrade))
        // Model weight serving — clients download assigned layers (LAN / local trust)
        .route("/api/v1/model/shard/{filename}", get(serve_model_shard))
        .route("/api/v1/model/config", get(serve_model_config))
        .route("/api/v1/inference/{task_id}", get(get_inference))
        .route("/api/v1/nodes", get(list_nodes))
        // Dataset (synthesis Phase 2 + verification Phase 3)
        .route("/api/v1/dataset/prompts", get(get_prompts))
        .route("/api/v1/dataset/examples", post(submit_example))
        .route("/api/v1/dataset/stats", get(dataset_stats))
        .route("/api/v1/dataset/export", get(export_dataset))
        .route("/api/v1/dataset/verify", post(verify_code))
        .route("/api/v1/dataset/bulk", post(bulk_import))
        // Training (Phase 4: distributed LoRA)
        .route("/api/v1/training/jobs", post(create_training_job))
        .route("/api/v1/training/jobs", get(list_training_jobs))
        .route("/api/v1/training/jobs/{job_id}", get(get_training_job))
        .route("/api/v1/training/jobs/{job_id}/shards/claim", post(claim_shard))
        .route("/api/v1/training/jobs/{job_id}/shards/{shard_id}/submit", post(submit_adapter))
        // Dashboard metrics (Phase 2.5)
        .route("/api/v1/metrics", get(get_metrics))
        // Node registration + control
        .route("/api/v1/node/register", post(http_register))
        .route("/api/v1/node/heartbeat", post(http_heartbeat))
        .route("/api/v1/node/poll", post(http_poll_task))
        .route("/api/v1/node/submit", post(http_submit_result))
        .route("/api/v1/node/deregister", post(http_deregister))
        .route("/api/v1/node/signal", post(http_node_signal))
        // Serving clusters
        .route("/api/v1/clusters", get(list_clusters))
        // Inference cluster status (node states + readiness)
        .route("/api/v1/cluster/status", get(get_cluster_status))
        // Layer-sharded training
        .route("/api/v1/layer-training/rounds", post(create_training_round))
        .route("/api/v1/layer-training/rounds", get(list_training_rounds))
        .route("/api/v1/layer-training/claim", post(claim_layer_assignment))
        .route("/api/v1/layer-training/submit", post(submit_layer_adapter))
        .route("/api/v1/layer-training/tokenizer", get(serve_tokenizer))
        .route("/api/v1/layer-training/layers/{layer_start}/{layer_end}", get(serve_layer_weights))
        .route("/api/v1/layer-training/shard/{offset}/{size}", get(serve_data_shard))
        // Model info
        .route("/api/v1/model", get(get_model_info))
        // RAG (Phase 2C: real-time knowledge)
        .route("/api/v1/rag/index", post(rag_index))
        .route("/api/v1/rag/search", get(rag_search))
        .route("/api/v1/rag/sources", get(rag_sources))
        .route("/api/v1/rag/context", get(rag_context))
        .merge(inference_api)
        .layer(axum::extract::DefaultBodyLimit::max(256 * 1024 * 1024)) // 256MB
        .with_state(state)
}

/// Optional API key gate. Unset/empty `HYVERK_API_KEY` → open (local/dev).
/// When set, require `Authorization: Bearer <key>` or `X-Api-Key: <key>`.
async fn require_api_key(req: Request, next: Next) -> Result<impl IntoResponse, StatusCode> {
    let expected = match std::env::var("HYVERK_API_KEY") {
        Ok(k) if !k.is_empty() => k,
        _ => return Ok(next.run(req).await),
    };

    let headers = req.headers();
    let provided = headers
        .get(axum::http::header::AUTHORIZATION)
        .and_then(|v| v.to_str().ok())
        .and_then(|v| v.strip_prefix("Bearer ").map(|s| s.to_string()))
        .or_else(|| {
            headers
                .get("x-api-key")
                .and_then(|v| v.to_str().ok())
                .map(|s| s.to_string())
        });

    match provided {
        Some(key) if key == expected => Ok(next.run(req).await),
        _ => Err(StatusCode::UNAUTHORIZED),
    }
}

async fn health() -> &'static str {
    "ok"
}

// ─── HTTP Node Registration (replaces gRPC for Fly.io compatibility) ─────────

async fn http_register(
    State(state): State<Arc<AppState>>,
    Json(req): Json<serde_json::Value>,
) -> Json<serde_json::Value> {
    state.counters.http_requests.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
    let node_name = req["node_name"].as_str().unwrap_or("unknown").to_string();
    let caps = hyverk_proto::NodeCapabilities {
        available_models: req["models"].as_array()
            .map(|a| a.iter().filter_map(|v| v.as_str().map(String::from)).collect())
            .unwrap_or_default(),
        hardware_info: req["hardware_info"].as_str().unwrap_or("").to_string(),
        max_concurrent_tasks: req["max_concurrent_tasks"].as_u64().unwrap_or(1) as u32,
    };
    let node_id = state.registry.register(node_name, caps).await;
    Json(serde_json::json!({
        "node_id": node_id,
        "heartbeat_interval_secs": 20
    }))
}

async fn http_heartbeat(
    State(state): State<Arc<AppState>>,
    Json(req): Json<serde_json::Value>,
) -> impl IntoResponse {
    state.counters.heartbeats.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
    state.counters.bytes_in.fetch_add(64, std::sync::atomic::Ordering::Relaxed);
    let node_id = req["node_id"].as_str().unwrap_or("");
    let active_tasks = req["active_tasks"].as_u64().unwrap_or(0) as u32;
    if state.registry.heartbeat(node_id, active_tasks).await {
        // Check for pending signal
        let signal = state.pending_signals.write().await.remove(node_id);
        match signal {
            Some(sig) => Json(serde_json::json!({"acknowledged": true, "signal": sig})),
            None => Json(serde_json::json!({"acknowledged": true})),
        }
    } else {
        Json(serde_json::json!({"error": "Node not registered"}))
    }
}

async fn http_poll_task(
    State(state): State<Arc<AppState>>,
    Json(req): Json<serde_json::Value>,
) -> Json<serde_json::Value> {
    let node_id = req["node_id"].as_str().unwrap_or("");
    match state.router.assign_task_for_node(node_id).await {
        Some(task) => {
            Json(serde_json::json!({
                "has_task": true,
                "task": {
                    "task_id": task.task_id,
                    "model": task.model,
                    "prompt": task.prompt,
                    "temperature": task.temperature,
                    "max_tokens": task.max_tokens,
                }
            }))
        }
        None => Json(serde_json::json!({"has_task": false}))
    }
}

async fn http_submit_result(
    State(state): State<Arc<AppState>>,
    Json(req): Json<serde_json::Value>,
) -> Json<serde_json::Value> {
    let task_id = req["task_id"].as_str().unwrap_or("");
    let success = req["success"].as_bool().unwrap_or(false);
    if success {
        state.task_store.complete_task(task_id, req["response_text"].as_str().unwrap_or("").to_string(), req["duration_ms"].as_u64().unwrap_or(0)).await;
    } else {
        state.task_store.fail_task(task_id, req["error_message"].as_str().unwrap_or("").to_string()).await;
    }
    Json(serde_json::json!({"acknowledged": true}))
}

async fn http_deregister(
    State(state): State<Arc<AppState>>,
    Json(req): Json<serde_json::Value>,
) -> Json<serde_json::Value> {
    let node_id = req["node_id"].as_str().unwrap_or("");
    state.registry.deregister(node_id).await;
    Json(serde_json::json!({"acknowledged": true}))
}

async fn dashboard() -> impl IntoResponse {
    (
        StatusCode::OK,
        [("content-type", "text/html; charset=utf-8")],
        include_str!("dashboard.html"),
    )
}

async fn models_page() -> impl IntoResponse {
    (
        StatusCode::OK,
        [("content-type", "text/html; charset=utf-8")],
        include_str!("models.html"),
    )
}

async fn devices_page() -> impl IntoResponse {
    (
        StatusCode::OK,
        [("content-type", "text/html; charset=utf-8")],
        include_str!("devices.html"),
    )
}

async fn test_page() -> impl IntoResponse {
    (
        StatusCode::OK,
        [("content-type", "text/html; charset=utf-8")],
        include_str!("test.html"),
    )
}

#[derive(Deserialize)]
struct InferenceRequest {
    model: String,
    prompt: String,
    temperature: Option<f64>,
    max_tokens: Option<u32>,
}

#[derive(Serialize)]
struct InferenceCreated {
    task_id: String,
    status: String,
}

async fn create_inference(
    State(state): State<Arc<AppState>>,
    Json(req): Json<InferenceRequest>,
) -> impl IntoResponse {
    let temperature = req.temperature.unwrap_or(0.7);
    let max_tokens = req.max_tokens.unwrap_or(512);
    let task_id = state
        .task_store
        .create_task(req.model, req.prompt, temperature, max_tokens)
        .await;
    (
        StatusCode::CREATED,
        Json(InferenceCreated {
            task_id,
            status: "pending".to_string(),
        }),
    )
}

#[derive(Serialize)]
struct InferenceStatus {
    task_id: String,
    status: String,
    model: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    response_text: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    error: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    duration_ms: Option<u64>,
}

async fn get_inference(
    State(state): State<Arc<AppState>>,
    Path(task_id): Path<String>,
) -> impl IntoResponse {
    match state.task_store.get_task(&task_id).await {
        Some(task) => {
            let (status, response_text, error, duration_ms) = match &task.status {
                TaskStatus::Pending => ("pending", None, None, None),
                TaskStatus::Assigned { .. } => ("assigned", None, None, None),
                TaskStatus::Completed {
                    response_text,
                    duration_ms,
                } => (
                    "completed",
                    Some(response_text.clone()),
                    None,
                    Some(*duration_ms),
                ),
                TaskStatus::Failed { error } => ("failed", None, Some(error.clone()), None),
            };
            (
                StatusCode::OK,
                Json(InferenceStatus {
                    task_id: task.task_id,
                    status: status.to_string(),
                    model: task.model,
                    response_text,
                    error,
                    duration_ms,
                }),
            )
                .into_response()
        }
        None => (
            StatusCode::NOT_FOUND,
            Json(serde_json::json!({"error": "Task not found"})),
        )
            .into_response(),
    }
}


async fn list_nodes(State(state): State<Arc<AppState>>) -> Json<serde_json::Value> {
    let nodes = state.registry.list_nodes().await;
    let dataset_stats = state.dataset_store.stats().await;
    let now = tokio::time::Instant::now();

    let summaries: Vec<serde_json::Value> = nodes
        .into_iter()
        .map(|n| {
            let uptime_secs = now.duration_since(n.registered_at).as_secs();
            let last_seen_secs = now.duration_since(n.last_heartbeat).as_secs();
            let hw = &n.capabilities.hardware_info;
            let has_gpu = !n.capabilities.available_models.is_empty()
                || hw.contains("M1") || hw.contains("M2") || hw.contains("M3") || hw.contains("M4")
                || hw.contains("NVIDIA") || hw.contains("RTX") || hw.contains("GTX");
            let role = if has_gpu { "trainer" } else { "worker" };

            // Count contributions from this node
            let contributions = dataset_stats.by_node.get(&n.node_id)
                .or_else(|| dataset_stats.by_node.get(&n.node_name))
                .copied()
                .unwrap_or(0);

            serde_json::json!({
                "node_id": n.node_id,
                "node_name": n.node_name,
                "models": n.capabilities.available_models,
                "active_tasks": n.active_tasks,
                "hardware": if hw.is_empty() { "unknown" } else { hw },
                "has_gpu": has_gpu,
                "role": role,
                "uptime_secs": uptime_secs,
                "last_seen_secs": last_seen_secs,
                "status": if last_seen_secs < 60 { "online" } else { "stale" },
                "contributions": contributions,
            })
        })
        .collect();
    Json(serde_json::json!({ "nodes": summaries }))
}

// ─── Dataset endpoints (Phase 2: Synthesis) ──────────────────────────────────

/// Returns a batch of prompts from the bank for synthesis nodes to process
async fn get_prompts(
    State(_state): State<Arc<AppState>>,
    Query(params): Query<std::collections::HashMap<String, String>>,
) -> Json<serde_json::Value> {
    use hyverk_synthesis::prompts;
    let count: usize = params.get("count")
        .and_then(|s| s.parse().ok())
        .unwrap_or(10)
        .min(100);

    // Seed with current time for variety
    let base_seed = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_nanos() as u64;

    let prompts_list: Vec<_> = (0..count)
        .map(|i| {
            let p = prompts::random_prompt(base_seed.wrapping_add(i as u64 * 31337));
            serde_json::json!({
                "category": p.category,
                "instruction": p.instruction
            })
        })
        .collect();

    Json(serde_json::json!({ "prompts": prompts_list }))
}

#[derive(Deserialize)]
struct SubmitExampleRequest {
    example: serde_json::Value,
    node_id: String,
}

/// Receives a generated training example from a synthesis node
async fn submit_example(
    State(state): State<Arc<AppState>>,
    Json(req): Json<SubmitExampleRequest>,
) -> impl IntoResponse {
    let now_secs = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs();

    let example = DatasetExample {
        id: req.example["id"].as_str().unwrap_or(&Uuid::new_v4().to_string()).to_string(),
        instruction: match req.example["instruction"].as_str() {
            Some(s) if !s.is_empty() => s.to_string(),
            _ => return (StatusCode::BAD_REQUEST, "Missing instruction").into_response(),
        },
        response: match req.example["response"].as_str() {
            Some(s) if !s.is_empty() => s.to_string(),
            _ => return (StatusCode::BAD_REQUEST, "Missing response").into_response(),
        },
        category: req.example["category"].as_str().unwrap_or("unknown").to_string(),
        provider: req.example["provider"].as_str().unwrap_or("unknown").to_string(),
        model: req.example["model"].as_str().unwrap_or("").to_string(),
        node_id: req.node_id,
        refined: req.example["refined"].as_bool().unwrap_or(false),
        execution_verified: req.example["execution_verified"].as_bool().unwrap_or(false),
        quality_score: req.example["quality_score"].as_f64().map(|f| f as f32),
        submitted_at_secs: now_secs,
    };

    let accepted = state.dataset_store.add_example(example).await;

    if accepted {
        (StatusCode::CREATED, Json(serde_json::json!({"status": "accepted"}))).into_response()
    } else {
        (StatusCode::OK, Json(serde_json::json!({"status": "duplicate"}))).into_response()
    }
}

/// Returns dataset statistics
async fn dataset_stats(State(state): State<Arc<AppState>>) -> Json<serde_json::Value> {
    let stats = state.dataset_store.stats().await;
    Json(serde_json::to_value(stats).unwrap_or_default())
}

/// On-demand code verification endpoint
/// POST body: { "code": "fn foo() {...}" }
/// Returns: VerificationResult as JSON
async fn verify_code(
    State(_state): State<Arc<AppState>>,
    Json(req): Json<serde_json::Value>,
) -> impl IntoResponse {
    let code = match req["code"].as_str() {
        Some(c) if !c.is_empty() => c.to_string(),
        _ => {
            return (
                StatusCode::BAD_REQUEST,
                Json(serde_json::json!({"error": "Missing 'code' field"})),
            ).into_response();
        }
    };

    let result = hyverk_sandbox::verify_response(&code, None).await;
    (StatusCode::OK, Json(serde_json::to_value(result).unwrap_or_default())).into_response()
}

/// Bulk import: accepts JSONL body (one example per line)
/// Much faster than submitting one at a time.
async fn bulk_import(
    State(state): State<Arc<AppState>>,
    body: String,
) -> impl IntoResponse {
    let now_secs = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs();

    let mut examples = Vec::new();
    for line in body.lines() {
        let line = line.trim();
        if line.is_empty() { continue; }
        if let Ok(raw) = serde_json::from_str::<serde_json::Value>(line) {
            let inst = raw["instruction"].as_str().unwrap_or("").to_string();
            let resp = raw["response"].as_str().unwrap_or("").to_string();
            if inst.is_empty() || resp.is_empty() { continue; }

            let lower = inst.to_lowercase();
            let cat = if lower.contains("debug") || lower.contains("fix") || lower.contains("error") { "debugging" }
                else if lower.contains("test") { "testing" }
                else if lower.contains("refactor") || lower.contains("optimize") { "refactoring" }
                else if lower.contains("algorithm") || lower.contains("sort") { "algorithms" }
                else if lower.contains("api") || lower.contains("endpoint") { "api_design" }
                else if lower.contains("database") || lower.contains("sql") { "database" }
                else if lower.contains("docker") || lower.contains("deploy") { "devops" }
                else { "code_generation" };

            examples.push(DatasetExample {
                id: Uuid::new_v4().to_string(),
                instruction: inst,
                response: resp,
                category: cat.to_string(),
                provider: raw["source"].as_str().or(raw["provider"].as_str()).unwrap_or("huggingface").to_string(),
                model: raw["model"].as_str().unwrap_or("").to_string(),
                node_id: "bulk-import".to_string(),
                refined: false,
                execution_verified: false,
                quality_score: None,
                submitted_at_secs: now_secs,
            });
        }
    }

    let total = examples.len();
    let (accepted, rejected) = state.dataset_store.add_bulk(examples).await;
    (StatusCode::OK, Json(serde_json::json!({
        "total_lines": total,
        "accepted": accepted,
        "rejected": rejected
    })))
}

/// Exports the full dataset as JSONL (one example per line)
async fn export_dataset(State(state): State<Arc<AppState>>) -> impl IntoResponse {
    let jsonl = state.dataset_store.export_jsonl().await;
    let mut headers = HeaderMap::new();
    headers.insert("Content-Type", "application/x-ndjson".parse().unwrap());
    headers.insert(
        "Content-Disposition",
        "attachment; filename=\"hyverk-dataset.jsonl\"".parse().unwrap(),
    );
    (headers, jsonl)
}

// ─── Training endpoints (Phase 4: distributed LoRA) ──────────────────────────

#[derive(Deserialize)]
struct CreateJobRequest {
    base_model: String,
    #[serde(default = "default_shard_size")]
    shard_size: usize,
    #[serde(default = "default_lora_rank")]
    lora_rank: u32,
    #[serde(default = "default_lora_alpha")]
    lora_alpha: f64,
    #[serde(default = "default_num_epochs")]
    num_epochs: u32,
    /// JSONL content (the dataset to split into shards)
    dataset: String,
}
fn default_shard_size() -> usize { 50 }
fn default_lora_rank() -> u32 { 16 }
fn default_lora_alpha() -> f64 { 32.0 }
fn default_num_epochs() -> u32 { 3 }

async fn create_training_job(
    State(state): State<Arc<AppState>>,
    Json(req): Json<CreateJobRequest>,
) -> impl IntoResponse {
    let job_id = state.training_store.create_job(
        req.base_model,
        req.dataset,
        req.shard_size,
        req.lora_rank,
        req.lora_alpha,
        req.num_epochs,
    ).await;

    (StatusCode::CREATED, Json(serde_json::json!({ "job_id": job_id })))
}

async fn list_training_jobs(State(state): State<Arc<AppState>>) -> Json<serde_json::Value> {
    let jobs = state.training_store.list_jobs().await;
    Json(serde_json::json!({ "jobs": jobs }))
}

async fn get_training_job(
    State(state): State<Arc<AppState>>,
    Path(job_id): Path<String>,
) -> impl IntoResponse {
    match state.training_store.job_stats(&job_id).await {
        Some(stats) => (StatusCode::OK, Json(stats)).into_response(),
        None => (StatusCode::NOT_FOUND, Json(serde_json::json!({"error": "Job not found"}))).into_response(),
    }
}

#[derive(Deserialize)]
struct ClaimShardRequest {
    node_id: String,
}

async fn claim_shard(
    State(state): State<Arc<AppState>>,
    Path(job_id): Path<String>,
    Json(req): Json<ClaimShardRequest>,
) -> impl IntoResponse {
    match state.training_store.claim_next_shard(&job_id, &req.node_id).await {
        Some((shard_id, content)) => (
            StatusCode::OK,
            Json(serde_json::json!({
                "shard_id": shard_id,
                "content": content,
                "job_id": job_id
            })),
        ).into_response(),
        None => (
            StatusCode::NO_CONTENT,
            Json(serde_json::json!({"message": "No shards available"})),
        ).into_response(),
    }
}

#[derive(Deserialize)]
struct SubmitAdapterRequest {
    node_id: String,
    /// Base64-encoded safetensors adapter bytes
    adapter_b64: String,
    training_loss: Option<f32>,
    training_steps: Option<usize>,
}

async fn submit_adapter(
    State(state): State<Arc<AppState>>,
    Path((job_id, shard_id)): Path<(String, String)>,
    Json(req): Json<SubmitAdapterRequest>,
) -> impl IntoResponse {
    let accepted = state.training_store.submit_adapter(
        &job_id,
        &shard_id,
        &req.node_id,
        req.adapter_b64,
        req.training_loss.unwrap_or(f32::NAN),
        req.training_steps.unwrap_or(0),
    ).await;

    if accepted {
        (StatusCode::CREATED, Json(serde_json::json!({"status": "accepted"})))
    } else {
        (StatusCode::NOT_FOUND, Json(serde_json::json!({"error": "Shard not found"})))
    }
}

// ─── Metrics endpoint (Phase 2.5: Network Dashboard) ─────────────────────────

async fn get_metrics(State(state): State<Arc<AppState>>) -> Json<serde_json::Value> {
    state.counters.http_requests.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
    let snapshot = crate::metrics::compute_metrics(&state).await;
    let json = serde_json::to_value(&snapshot).unwrap_or_default();
    let bytes = json.to_string().len() as u64;
    state.counters.bytes_out.fetch_add(bytes, std::sync::atomic::Ordering::Relaxed);
    Json(json)
}

// ─── Distributed Inference Proxy ──────────────────────────────────────────────

/// Coordinator proxies inference to the first node of an active cluster.
/// The node handles the distributed pipeline internally.
async fn proxy_distributed_inference(
    State(state): State<Arc<AppState>>,
    Json(req): Json<serde_json::Value>,
) -> impl IntoResponse {
    // CLUSTER_ENTRY_URL takes priority — if set, route directly without checking registered nodes.
    // If not set, require at least one GPU node to be registered.
    let cluster_url = std::env::var("CLUSTER_ENTRY_URL").unwrap_or_default();
    if cluster_url.is_empty() {
        let nodes = state.registry.list_nodes().await;
        let has_gpu = nodes.iter().any(|n| {
            let hw = &n.capabilities.hardware_info;
            hw.contains("M1") || hw.contains("M2") || hw.contains("M3") || hw.contains("M4")
                || hw.contains("NVIDIA") || hw.contains("RTX")
        });
        if !has_gpu {
            return Json(serde_json::json!({
                "error": "No GPU cluster available. Set CLUSTER_ENTRY_URL or register GPU nodes."
            })).into_response();
        }
    }
    let cluster_url = if cluster_url.is_empty() {
        "http://localhost:18200".to_string()
    } else {
        cluster_url
    };

    let client = reqwest::Client::builder()
        .timeout(std::time::Duration::from_secs(300))
        .build()
        .unwrap_or_else(|_| reqwest::Client::new());

    let prompt = req["prompt"].as_str().unwrap_or("");
    let max_tokens = req["max_tokens"].as_u64().unwrap_or(256);
    let temperature = req["temperature"].as_f64().unwrap_or(0.3);

    let full_prompt = format!(
        "<|im_start|>system\nYou are Hyverk, an expert coding assistant.<|im_end|>\n<|im_start|>user\n{}<|im_end|>\n<|im_start|>assistant\n",
        prompt
    );

    match client
        .post(format!("{}/generate", cluster_url))
        .json(&serde_json::json!({
            "prompt": full_prompt,
            "max_tokens": max_tokens,
            "temperature": temperature,
        }))
        .send()
        .await
    {
        Ok(resp) => {
            match resp.json::<serde_json::Value>().await {
                Ok(data) => Json(data).into_response(),
                Err(e) => Json(serde_json::json!({"error": format!("Parse error: {e}")})).into_response(),
            }
        }
        Err(e) => {
            Json(serde_json::json!({"error": format!("Cluster unreachable: {e}. Is inference mode active? Run: make inference")})).into_response()
        }
    }
}

// ─── Model Weight Serving ─────────────────────────────────────────────────

/// Serve a model shard file to clients (streamed — doesn't load into RAM)
async fn serve_model_shard(
    Path(filename): Path<String>,
) -> impl IntoResponse {
    use tokio_util::io::ReaderStream;

    if !filename.ends_with(".safetensors") && !filename.ends_with(".json") {
        return (StatusCode::BAD_REQUEST, "Invalid file type").into_response();
    }
    let safe_name = filename.replace("..", "").replace("/", "");
    let path = crate::model_paths::model_file(&safe_name);

    match tokio::fs::File::open(&path).await {
        Ok(file) => {
            let meta = file.metadata().await.ok();
            let stream = ReaderStream::new(file);
            let body = axum::body::Body::from_stream(stream);

            let mut headers = HeaderMap::new();
            headers.insert("Content-Type", "application/octet-stream".parse().unwrap());
            if let Some(m) = meta { headers.insert("Content-Length", m.len().to_string().parse().unwrap()); }
            (StatusCode::OK, headers, body).into_response()
        }
        Err(_) => (StatusCode::NOT_FOUND, format!("Model file not found: {safe_name}")).into_response(),
    }
}

/// Read and validate the on-disk model state under `HYVERK_MODEL_DIR` (default `/data/model`).
/// Returned tuple: (index_val, config_val, available). When `available` is false
/// the JSON values may still be partial (Null) — callers must not trust them.
pub async fn read_coordinator_model_state() -> (serde_json::Value, serde_json::Value, bool) {
    let index_path = crate::model_paths::model_file("model.safetensors.index.json");
    let config_path = crate::model_paths::model_file("config.json");

    let index_raw = tokio::fs::read_to_string(&index_path).await.unwrap_or_default();
    let config_raw = tokio::fs::read_to_string(&config_path).await.unwrap_or_default();

    let index_val: serde_json::Value =
        serde_json::from_str(index_raw.trim()).unwrap_or(serde_json::Value::Null);
    let config_val: serde_json::Value =
        serde_json::from_str(config_raw.trim()).unwrap_or(serde_json::Value::Null);

    let index_ok = index_val
        .get("weight_map")
        .and_then(|w| w.as_object())
        .is_some_and(|m| !m.is_empty());
    let config_ok = config_val
        .get("num_hidden_layers")
        .and_then(|v| v.as_u64().or_else(|| v.as_i64().map(|i| i as u64)))
        .is_some();

    (index_val, config_val, index_ok && config_ok)
}

/// Cheap availability check used to gate work assignments before the coordinator
/// has a model on disk. Same predicate as `read_coordinator_model_state`.
pub async fn coordinator_model_available() -> bool {
    let (_, _, ok) = read_coordinator_model_state().await;
    ok
}

/// Serve model config (tells client which shards contain which layers)
async fn serve_model_config() -> impl IntoResponse {
    let (index_val, config_val, available) = read_coordinator_model_state().await;

    if !available {
        let dir = crate::model_paths::model_dir();
        return Json(serde_json::json!({
            "available": false,
            "config": null,
            "index": null,
            "coordinator_model_status": "missing_or_invalid",
            "model_dir": dir.display().to_string(),
            "hint": format!(
                "Coordinator reads {{config.json,model.safetensors.index.json,tokenizer.json,*.safetensors}} under {}. Set HYVERK_MODEL_DIR or run scripts/prepare-model.sh. Nodes poll GET /api/v1/model/config before layer download.",
                dir.display()
            ),
        }));
    }

    Json(serde_json::json!({
        "available": true,
        "index": index_val,
        "config": config_val,
    }))
}

// ─── Node Signals & Clusters ──────────────────────────────────────────────────

/// Send a role-switch signal to a node
/// The node will poll this via heartbeat response
async fn http_node_signal(
    State(state): State<Arc<AppState>>,
    Json(req): Json<serde_json::Value>,
) -> Json<serde_json::Value> {
    let node_id = req["node_id"].as_str().unwrap_or("");
    let signal = req["signal"].as_str().unwrap_or("");
    let role = req["role"].as_str().unwrap_or("trainer");
    let urgency = req["urgency"].as_str().unwrap_or("graceful");

    // Store pending signal for the node (delivered via next heartbeat)
    state.pending_signals.write().await.insert(
        node_id.to_string(),
        serde_json::json!({
            "signal": signal,
            "role": role,
            "urgency": urgency,
            "cluster_id": req.get("cluster_id"),
            "layer_start": req.get("layer_start"),
            "layer_end": req.get("layer_end"),
            "next_node": req.get("next_node"),
            "position": req.get("position"),
        }),
    );

    Json(serde_json::json!({"queued": true, "node_id": node_id}))
}

async fn list_clusters(
    State(state): State<Arc<AppState>>,
) -> Json<serde_json::Value> {
    let infra = state.ws_state.cluster_mgr.snapshot().await;
    let reliability = state.ws_state.node_stats.snapshot().await;
    Json(serde_json::json!({
        "infra": infra,
        "node_reliability": reliability,
    }))
}

async fn get_cluster_status(
    State(state): State<Arc<AppState>>,
) -> Json<serde_json::Value> {
    let status = crate::ws_handler::cluster_status(&state.ws_state).await;
    Json(serde_json::to_value(status).unwrap_or_default())
}

// ─── Layer-Sharded Training ───────────────────────────────────────────────────

#[derive(Deserialize)]
struct CreateRoundRequest {
    #[serde(default = "default_version")]
    version: String,
    #[serde(default = "default_layers_per")]
    layers_per_assignment: usize,
    #[serde(default = "default_examples_per")]
    examples_per_shard: usize,
}
fn default_version() -> String { "v0.1".to_string() }
fn default_layers_per() -> usize { 2 }
fn default_examples_per() -> usize { 1000 }

async fn create_training_round(
    State(state): State<Arc<AppState>>,
    Json(req): Json<CreateRoundRequest>,
) -> impl IntoResponse {
    let dataset_size = state.dataset_store.stats().await.total_examples;
    if dataset_size == 0 {
        return (StatusCode::BAD_REQUEST, Json(serde_json::json!({"error": "No training data available"}))).into_response();
    }

    let round_id = state.layer_training.create_round(
        &req.version,
        "Qwen2.5-Coder-7B",
        28, // total layers
        req.layers_per_assignment,
        dataset_size,
        req.examples_per_shard,
    ).await;

    (StatusCode::CREATED, Json(serde_json::json!({
        "round_id": round_id,
        "total_layers": 28,
        "assignments": 28 / req.layers_per_assignment.max(1),
        "dataset_size": dataset_size,
    }))).into_response()
}

async fn list_training_rounds(
    State(state): State<Arc<AppState>>,
) -> Json<serde_json::Value> {
    let rounds = state.layer_training.all_rounds().await;
    Json(serde_json::json!({ "rounds": rounds }))
}

#[derive(Deserialize)]
struct ClaimRequest {
    node_id: String,
}

async fn claim_layer_assignment(
    State(state): State<Arc<AppState>>,
    Json(req): Json<ClaimRequest>,
) -> impl IntoResponse {
    match state.layer_training.claim_assignment(&req.node_id).await {
        Some(assignment) => (StatusCode::OK, Json(serde_json::json!({
            "assignment": assignment,
            "layers_url": format!("/api/v1/layer-training/layers/{}/{}", assignment.layer_start, assignment.layer_end),
            "data_url": format!("/api/v1/layer-training/shard/{}/{}", assignment.data_shard_start, assignment.data_shard_size),
        }))).into_response(),
        None => (StatusCode::NOT_FOUND, Json(serde_json::json!({
            "error": "No assignments available"
        }))).into_response(),
    }
}

#[derive(Deserialize)]
struct SubmitLayerAdapterRequest {
    assignment_id: String,
    loss: f32,
    steps: usize,
    adapter_base64: String,
}

async fn submit_layer_adapter(
    State(state): State<Arc<AppState>>,
    Json(req): Json<SubmitLayerAdapterRequest>,
) -> impl IntoResponse {
    use base64::Engine;
    let adapter_bytes = match base64::engine::general_purpose::STANDARD.decode(&req.adapter_base64) {
        Ok(b) => b,
        Err(_) => return (StatusCode::BAD_REQUEST, Json(serde_json::json!({"error": "Invalid base64"}))).into_response(),
    };

    let ok = state.layer_training.submit_adapter(
        &req.assignment_id,
        adapter_bytes,
        req.loss,
        req.steps,
    ).await;

    if ok {
        (StatusCode::OK, Json(serde_json::json!({"status": "accepted"}))).into_response()
    } else {
        (StatusCode::NOT_FOUND, Json(serde_json::json!({"error": "Assignment not found"}))).into_response()
    }
}

/// Serve the tokenizer.json for nodes that don't have it locally
async fn serve_tokenizer() -> impl IntoResponse {
    // Try common paths
    let paths = [
        "/app/tokenizer.json",
        "/data/tokenizer.json",
        "/root/.hyverk/qwen2.5-7b/tokenizer.json",
    ];
    for p in &paths {
        if let Ok(data) = std::fs::read(p) {
            return (
                StatusCode::OK,
                [("content-type", "application/json")],
                data,
            ).into_response();
        }
    }
    // Return a minimal tokenizer info if file not found
    (StatusCode::NOT_FOUND, "tokenizer not found on coordinator").into_response()
}

/// Serve specific layer weights as safetensors subset
/// Nodes call this to download only the layers they need (~500MB per layer)
async fn serve_layer_weights(
    Path((layer_start, layer_end)): Path<(usize, usize)>,
) -> impl IntoResponse {
    // For now, return info about which files to download
    // Full implementation will extract and serve specific layer tensors
    let layer_names: Vec<String> = (layer_start..layer_end)
        .flat_map(|l| {
            vec![
                format!("model.layers.{l}.self_attn.q_proj.weight"),
                format!("model.layers.{l}.self_attn.k_proj.weight"),
                format!("model.layers.{l}.self_attn.v_proj.weight"),
                format!("model.layers.{l}.self_attn.o_proj.weight"),
                format!("model.layers.{l}.self_attn.q_proj.bias"),
                format!("model.layers.{l}.self_attn.k_proj.bias"),
                format!("model.layers.{l}.self_attn.v_proj.bias"),
                format!("model.layers.{l}.input_layernorm.weight"),
                format!("model.layers.{l}.post_attention_layernorm.weight"),
                format!("model.layers.{l}.mlp.gate_proj.weight"),
                format!("model.layers.{l}.mlp.up_proj.weight"),
                format!("model.layers.{l}.mlp.down_proj.weight"),
            ]
        })
        .collect();

    Json(serde_json::json!({
        "layer_start": layer_start,
        "layer_end": layer_end,
        "tensor_names": layer_names,
        "note": "Download these tensors from the model safetensors files"
    }))
}

/// Serve a data shard (subset of training examples as JSONL)
async fn serve_data_shard(
    State(state): State<Arc<AppState>>,
    Path((offset, size)): Path<(usize, usize)>,
) -> impl IntoResponse {
    let examples = state.dataset_store.list(offset, size).await;
    let jsonl: String = examples.iter()
        .filter_map(|ex| serde_json::to_string(ex).ok())
        .collect::<Vec<_>>()
        .join("\n");

    let mut headers = HeaderMap::new();
    headers.insert("Content-Type", "application/x-ndjson".parse().unwrap());
    (headers, jsonl)
}

// ─── Model info ───────────────────────────────────────────────────────────────

async fn get_model_info(State(state): State<Arc<AppState>>) -> Json<serde_json::Value> {
    let nodes = state.registry.list_nodes().await;
    let gpu_nodes = nodes.iter().filter(|n| {
        let hw = &n.capabilities.hardware_info;
        !n.capabilities.available_models.is_empty()
            || hw.contains("M1") || hw.contains("M2") || hw.contains("M3") || hw.contains("M4")
            || hw.contains("NVIDIA") || hw.contains("RTX") || hw.contains("GTX")
    }).count();
    let cpu_nodes = nodes.len() - gpu_nodes;

    // Real training pipeline status — derived from actual coordinator state
    let jobs = state.training_store.list_jobs().await;
    let active_jobs = jobs.len();
    let dataset_stats = state.dataset_store.stats().await;
    let verified_examples = dataset_stats.execution_verified;
    let total_submitted = dataset_stats.total_examples;

    // Layer training stats
    let (lt_rounds, lt_total, lt_completed, lt_avg_loss) = state.layer_training.stats().await;
    let active_round = state.layer_training.active_round().await;
    // Estimate based on first active goal
    let target_rounds = 100u64; // v0.1 target
    let estimate = state.layer_training.estimate(target_rounds, nodes.len()).await;

    // Pipeline phases — each must complete before the next starts
    // Phase 1: Data Collection — gather instruction/response pairs from contributors
    // Phase 2: Verification — execute code examples in sandbox, confirm they compile/run
    // Phase 3: Training — distribute LoRA shards to GPU nodes, train adapters
    // Phase 4: Evaluation — benchmark the model on HumanEval, MBPP, etc.

    // Dataset source files on disk (real inventory)
    let disk_datasets = vec![
        serde_json::json!({"name": "Magicoder-OSS-Instruct", "file": "magicoder_oss.jsonl", "examples": 75000, "size_mb": 175, "source": "HuggingFace", "status": "downloaded"}),
        serde_json::json!({"name": "CodeFeedback-Filtered", "file": "codefeedback.jsonl", "examples": 157000, "size_mb": 360, "source": "HuggingFace", "status": "downloaded"}),
        serde_json::json!({"name": "Evol-Instruct-Code", "file": "evol_code_80k.jsonl", "examples": 80000, "size_mb": 120, "source": "HuggingFace", "status": "downloaded"}),
        serde_json::json!({"name": "Evol-CodeAlpaca", "file": "evol_codealpaca.jsonl", "examples": 152000, "size_mb": 253, "source": "HuggingFace", "status": "downloaded"}),
        serde_json::json!({"name": "Code-Alpaca-20K", "file": "code_alpaca_122k.jsonl", "examples": 122000, "size_mb": 79, "source": "HuggingFace", "status": "downloaded"}),
    ];
    let total_disk = 586445u64;
    let total_disk_gb = 1.1f64;

    // Determine version
    let version = if active_jobs > 0 {
        "v0.1 (training in progress)"
    } else if total_submitted > 0 {
        "v0 (collecting data)"
    } else {
        "v0 (not yet started)"
    };

    // Collection progress: how much of disk data has been submitted to coordinator
    let collection_pct = if total_disk > 0 {
        (total_submitted as f64 / total_disk as f64 * 100.0).min(100.0)
    } else { 0.0 };

    Json(serde_json::json!({
        "name": "Hyverk",
        "base_model": "Qwen2.5-Coder-7B-Instruct",
        "parameters": "7.6B",
        "description": "A coding assistant fine-tuned by distributed contributors",
        "architecture": {
            "layers": 28,
            "hidden_size": 3584,
            "attention_heads": 28,
            "kv_heads": 4,
            "vocab_size": 152064,
            "context_length": 131072
        },
        "training_method": {
            "name": "LoRA (Low-Rank Adaptation)",
            "lora_rank": 16,
            "trainable_params": "14M",
            "trainable_pct": 0.18,
            "optimizer": "AdamW",
            "precision": "FP16 base + FP32 adapters"
        },
        "datasets_on_disk": {
            "sources": disk_datasets,
            "total_examples": total_disk,
            "total_size_gb": total_disk_gb,
        },
        "pipeline": {
            "phase": if lt_rounds > 0 { "training" } else if total_submitted > 0 { "collection" } else { "not_started" },
            "data_collected": total_submitted,
            "data_verified": verified_examples,
            "rounds_completed": lt_rounds,
            "active_round": active_round,
            "training_avg_loss": lt_avg_loss,
            "version": version,
        },
        "goals": [
            {
                "name": "v0.1",
                "description": "Proof of concept — first working LoRA adapter",
                "target_rounds": 100,
                "rounds_done": lt_rounds,
                "dataset_size": total_submitted,
                "min_clients": 3,
                "est_days": 2,
                "status": if lt_rounds >= 100 { "complete" } else if lt_rounds > 0 { "active" } else { "pending" }
            },
            {
                "name": "v0.2",
                "description": "First usable model — measurable improvement over base",
                "target_rounds": 500,
                "rounds_done": 0,
                "dataset_size": 200000,
                "min_clients": 100,
                "est_days": 7,
                "status": "pending"
            },
            {
                "name": "v0.3",
                "description": "Competitive model — beats base on HumanEval",
                "target_rounds": 2000,
                "rounds_done": 0,
                "dataset_size": 500000,
                "min_clients": 1000,
                "est_days": 30,
                "status": "pending"
            },
            {
                "name": "v1.0",
                "description": "Production quality — top open-source coding model",
                "target_rounds": 10000,
                "rounds_done": 0,
                "dataset_size": 2000000,
                "min_clients": 10000,
                "est_days": 60,
                "status": "pending"
            }
        ],
        "network": {
            "total_nodes": nodes.len(),
            "gpu_nodes": gpu_nodes,
            "cpu_nodes": cpu_nodes
        },
        "estimation": {
            "rounds_completed": lt_rounds,
            "rounds_per_hour": estimate.rounds_per_hour,
            "eta_hours": if estimate.eta_hours.is_infinite() { -1.0 } else { estimate.eta_hours },
            "clients_for_7d": estimate.clients_needed_for_7d,
            "clients_for_30d": estimate.clients_needed_for_30d,
        }
    }))
}

// ─── RAG endpoints (Phase 2C: Real-time Knowledge Layer) ──────────────────────

#[derive(Deserialize)]
struct RagIndexRequest {
    /// "crate:tokio", "dir:./my-project", "url:https://..."
    source: String,
}

async fn rag_index(
    State(state): State<Arc<AppState>>,
    Json(req): Json<RagIndexRequest>,
) -> impl IntoResponse {
    let (source_type, source_ref) = match req.source.split_once(':') {
        Some(("crate", name)) => (SourceType::CrateDocs, name.to_string()),
        Some(("dir", path)) => (SourceType::LocalDir, path.to_string()),
        Some(("url", url)) => (SourceType::Url, url.to_string()),
        _ => {
            return (
                StatusCode::BAD_REQUEST,
                Json(serde_json::json!({"error": "source must be crate:NAME, dir:PATH, or url:URL"})),
            );
        }
    };

    let store = Arc::clone(&state.rag_store);
    let config = RagConfig::default();
    let source_ref_clone = source_ref.clone();
    let result = hyverk_rag::sources::index_source(&store, &config, source_type, &source_ref_clone).await;

    match result {
        Ok(chunks) => (
            StatusCode::OK,
            Json(serde_json::json!({"source": source_ref, "chunks_indexed": chunks})),
        ),
        Err(e) => (
            StatusCode::INTERNAL_SERVER_ERROR,
            Json(serde_json::json!({"error": e.to_string()})),
        ),
    }
}

#[derive(Deserialize)]
struct RagSearchQuery {
    q: String,
    #[serde(default = "default_top_k")]
    k: usize,
}
fn default_top_k() -> usize { 5 }

async fn rag_search(
    State(state): State<Arc<AppState>>,
    Query(params): Query<RagSearchQuery>,
) -> Json<serde_json::Value> {
    match state.rag_store.search(&params.q, params.k) {
        Ok(results) => Json(serde_json::json!({
            "query": params.q,
            "results": results.iter().map(|r| serde_json::json!({
                "title": r.chunk.title,
                "source": r.chunk.source_ref,
                "score": r.score,
                "content": r.chunk.content,
            })).collect::<Vec<_>>()
        })),
        Err(e) => Json(serde_json::json!({"error": e.to_string()})),
    }
}

async fn rag_sources(State(state): State<Arc<AppState>>) -> Json<serde_json::Value> {
    match state.rag_store.list_sources() {
        Ok(sources) => Json(serde_json::json!({
            "total_chunks": state.rag_store.chunk_count(),
            "sources": sources,
        })),
        Err(e) => Json(serde_json::json!({"error": e.to_string()})),
    }
}

async fn rag_context(
    State(state): State<Arc<AppState>>,
    Query(params): Query<RagSearchQuery>,
) -> impl IntoResponse {
    match state.rag_store.build_context(&params.q, params.k) {
        Ok(ctx) => (StatusCode::OK, ctx),
        Err(e) => (StatusCode::INTERNAL_SERVER_ERROR, e.to_string()),
    }
}

// ─── Real Distributed Inference via WebSocket Chain ──────────────────────────

/// POST /api/v1/ws-inference
/// Tokenizes prompt, chains through registered WS nodes, returns generated text.
async fn ws_inference(
    State(state): State<Arc<AppState>>,
    Json(req): Json<serde_json::Value>,
) -> impl IntoResponse {
    let prompt = req["prompt"].as_str().unwrap_or("").to_string();
    let max_tokens = req["max_tokens"].as_u64().unwrap_or(64) as usize;
    let temperature = req["temperature"].as_f64().unwrap_or(0.7) as f32;
    let top_p = req.get("top_p").and_then(|v| v.as_f64()).map(|x| x as f32);
    let top_k = req.get("top_k").and_then(|v| v.as_u64()).map(|x| x as u32);

    if prompt.is_empty() {
        return Json(serde_json::json!({"error": "prompt required"})).into_response();
    }

    // Pin to the active generation before building the chain.
    let in_flight_guard = match state.ws_state.cluster_mgr.request_start().await {
        Some(g) => g,
        None => {
            return Json(serde_json::json!({
                "error": "Inference cluster not operational",
                "hint": "Wait until GET /api/v1/cluster/status reports operational (all GPU nodes ready on an active generation)"
            })).into_response();
        }
    };

    let chain = crate::ws_handler::build_inference_chain(&state.ws_state).await;
    if chain.is_empty() {
        crate::serving_clusters::ClusterManager::request_end(&in_flight_guard);
        return Json(serde_json::json!({
            "error": "No inference nodes connected via WebSocket",
            "hint": "Start Mac node with coordinator_url pointing to this coordinator, wait for nodes to download layers"
        })).into_response();
    }

    let node_count = chain.len();
    tracing::info!(nodes = node_count, "Starting distributed inference chain");

    // Tokenize prompt
    let token_ids = match tokenize_prompt(&prompt).await {
        Ok(ids) => ids,
        Err(e) => {
            crate::serving_clusters::ClusterManager::request_end(&in_flight_guard);
            return Json(serde_json::json!({"error": format!("Tokenize failed: {e}")})).into_response();
        }
    };

    let request_id = uuid::Uuid::new_v4().to_string();
    let (result_tx, result_rx) = tokio::sync::oneshot::channel::<crate::ws_handler::InferenceResult>();

    // Register pending forward
    {
        let mut fwds = state.ws_state.pending_forwards.write().await;
        fwds.insert(request_id.clone(), crate::ws_handler::PendingForward {
            hidden_data: vec![],
            chain: chain.clone(),
            current_step: 0,
            token_ids: token_ids.clone(),
            generated: vec![],
            max_tokens,
            temperature,
            top_p,
            top_k,
            result_tx: Some(result_tx),
            in_flight_guard: Some(in_flight_guard.clone()),
        });
    }

    // Send InferenceStart to first node
    let first = &chain[0];
    let sent = state.ws_state.send_to_node(&first.node_id, hyverk_comms::messages::CoordinatorMessage::InferenceStart {
        request_id: request_id.clone(),
        token_ids: token_ids.clone(),
        layer_start: first.layer_start,
        layer_end: first.layer_end,
        max_tokens,
        temperature: Some(temperature),
        top_p,
        top_k,
    }).await;

    if !sent {
        if let Some(pending) = state.ws_state.pending_forwards.write().await.remove(&request_id) {
            crate::ws_handler::release_in_flight(&pending, &state.ws_state.cluster_mgr);
        } else {
            crate::serving_clusters::ClusterManager::request_end(&in_flight_guard);
        }
        return Json(serde_json::json!({"error": "Failed to reach first node"})).into_response();
    }

    tracing::info!(
        request_id = %request_id,
        first_node = %first.node_id,
        layers = format!("{}-{}", first.layer_start, first.layer_end),
        "Inference chain started"
    );

    // Wait for result (2 min timeout; node failures now propagate immediately via dropped result_tx)
    let start = std::time::Instant::now();
    match tokio::time::timeout(std::time::Duration::from_secs(120), result_rx).await {
        Ok(Ok(result)) => {
            let elapsed = start.elapsed().as_secs_f64();

            // Decode generated tokens
            let text = decode_tokens(&result.generated_ids).await
                .unwrap_or_else(|_| format!("[{} tokens]", result.generated_ids.len()));

            let cluster_info: Vec<serde_json::Value> = chain.iter().map(|s| {
                serde_json::json!({
                    "node_id": s.node_id,
                    "layers": format!("{}-{}", s.layer_start, s.layer_end),
                    "position": if s.layer_start == 0 { "first" } else if s.is_last { "last" } else { "middle" },
                })
            }).collect();

            Json(serde_json::json!({
                "text": text,
                "tokens": result.tokens,
                "elapsed_secs": elapsed,
                "tokens_per_sec": result.tokens as f64 / elapsed.max(0.001),
                "distributed": true,
                "nodes_used": node_count,
                "cluster": cluster_info,
                "request_id": request_id,
            })).into_response()
        }
        Ok(Err(_)) => {
            let removed = {
                let mut fw = state.ws_state.pending_forwards.write().await;
                fw.remove(&request_id)
            };
            if let Some(pending) = removed {
                crate::ws_handler::release_in_flight(&pending, &state.ws_state.cluster_mgr);
                crate::ws_handler::broadcast_inference_end(&state.ws_state, &pending.chain, &request_id).await;
            }
            Json(serde_json::json!({"error": "Inference channel closed unexpectedly"})).into_response()
        }
        Err(_) => {
            let removed = {
                let mut fw = state.ws_state.pending_forwards.write().await;
                fw.remove(&request_id)
            };
            if let Some(pending) = removed {
                crate::ws_handler::release_in_flight(&pending, &state.ws_state.cluster_mgr);
                crate::ws_handler::broadcast_inference_end(&state.ws_state, &pending.chain, &request_id).await;
            }
            Json(serde_json::json!({
                "error": "Inference timed out (120s)",
                "hint": "Nodes may be downloading model weights or a node failed silently"
            })).into_response()
        }
    }
}

/// Tokenize text using the local tokenizer via Python
async fn tokenize_prompt(prompt: &str) -> Result<Vec<u32>, String> {
    // Format as Qwen2.5 ChatML
    let formatted = format!(
        "<|im_start|>system\nYou are a helpful coding assistant.<|im_end|>\n<|im_start|>user\n{}<|im_end|>\n<|im_start|>assistant\n",
        prompt
    );

    let tok_path = crate::model_paths::first_existing_tokenizer()
        .ok_or_else(|| {
            format!(
                "tokenizer.json not found (checked {:?})",
                crate::model_paths::tokenizer_candidates()
            )
        })?;
    let tok_path_str = tok_path.to_string_lossy().to_string();

    let script = r#"
import sys, json
from tokenizers import Tokenizer
t = Tokenizer.from_file(sys.argv[2])
enc = t.encode(sys.argv[1])
print(json.dumps(enc.ids))
"#;

    let out = tokio::process::Command::new("python3")
        .arg("-c")
        .arg(script)
        .arg(&formatted)
        .arg(&tok_path_str)
        .output()
        .await
        .map_err(|e| format!("Python error: {e}"))?;

    if !out.status.success() {
        let err = String::from_utf8_lossy(&out.stderr);
        return Err(format!("Tokenizer failed: {err}"));
    }

    let ids: Vec<u64> = serde_json::from_str(std::str::from_utf8(&out.stdout).unwrap_or("[]").trim())
        .map_err(|e| format!("JSON parse error: {e}"))?;

    Ok(ids.into_iter().map(|x| x as u32).collect())
}

/// Decode generated token IDs to text using Python
async fn decode_tokens(ids: &[u32]) -> Result<String, String> {
    if ids.is_empty() { return Ok(String::new()); }

    // Filter out special tokens (EOS, etc.)
    let clean_ids: Vec<u32> = ids.iter().copied()
        .filter(|&id| id < 151643)
        .collect();

    let ids_json = serde_json::to_string(&clean_ids).unwrap_or_default();

    let tok_path = crate::model_paths::first_existing_tokenizer()
        .ok_or_else(|| "tokenizer.json not found".to_string())?;
    let tok_path_str = tok_path.to_string_lossy().to_string();

    let script = r#"
import sys, json
from tokenizers import Tokenizer
t = Tokenizer.from_file(sys.argv[2])
ids = json.loads(sys.argv[1])
print(t.decode(ids), end='')
"#;

    let out = tokio::process::Command::new("python3")
        .arg("-c")
        .arg(script)
        .arg(&ids_json)
        .arg(&tok_path_str)
        .output()
        .await
        .map_err(|e| format!("Python error: {e}"))?;

    Ok(String::from_utf8_lossy(&out.stdout).to_string())
}

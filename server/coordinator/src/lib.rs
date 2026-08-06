pub mod dataset_store;
pub mod grpc_server;
pub mod http_api;
pub mod layer_training;
pub mod metrics;
pub mod model_paths;
pub mod node_stats;
pub mod registry;
pub mod router;
pub mod serving_clusters;
pub mod task_store;
pub mod training_store;
pub mod ws_handler;

use std::sync::Arc;
use dataset_store::DatasetStore;
use hyverk_core::config::CoordinatorConfig;
use hyverk_rag::{RagConfig, store::RagStore};
use metrics::LiveCounters;
use training_store::TrainingStore;
use grpc_server::GrpcService;
use http_api::AppState;
use registry::NodeRegistry;
use router::TaskRouter;
use task_store::TaskStore;
use hyverk_proto::node_coordinator_server::NodeCoordinatorServer;
use std::time::Duration;
use tokio_util::sync::CancellationToken;
use tonic::transport::Server;
use tracing::{error, info};

/// Start the coordinator (gRPC + HTTP). Returns when cancelled or on fatal error.
pub async fn run_coordinator(
    config: &CoordinatorConfig,
    shutdown: CancellationToken,
) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
    info!(
        grpc_port = config.grpc_port,
        http_port = config.http_port,
        "Starting coordinator"
    );

    let started_at_secs = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs();

    let registry = NodeRegistry::new();
    let task_store = TaskStore::new();
    let router = TaskRouter::new(registry.clone(), task_store.clone());
    // Data directory: HYVERK_DATA_DIR env if set (Fly.io volume), else ~/.hyverk/
    let data_dir = std::env::var("HYVERK_DATA_DIR")
        .map(std::path::PathBuf::from)
        .unwrap_or_else(|_| {
            dirs::home_dir()
                .unwrap_or_else(|| std::path::PathBuf::from("/tmp"))
                .join(".hyverk")
        });
    std::fs::create_dir_all(&data_dir).ok();
    let dataset_db_path = data_dir.join("dataset.db").to_string_lossy().to_string();
    let dataset_store = DatasetStore::open(&dataset_db_path);
    info!(path = %dataset_db_path, "Dataset store initialized (SQLite)");
    let training_store = TrainingStore::new();
    let training_db_path = data_dir.join("training.db").to_string_lossy().to_string();
    let layer_training = layer_training::LayerTrainingStore::open(&training_db_path);
    let counters = std::sync::Arc::new(LiveCounters::new());
    let rag_config = RagConfig {
        db_path: data_dir.join("rag.db").to_string_lossy().to_string(),
        ..RagConfig::default()
    };
    let rag_store = std::sync::Arc::new(
        RagStore::open(&rag_config.db_path)
            .unwrap_or_else(|e| {
                tracing::warn!("RAG store init failed ({e}), using in-memory fallback");
                RagStore::open(":memory:").expect("in-memory RAG store")
            })
    );

    // Background reaper
    let registry_reaper = registry.clone();
    let task_store_reaper = task_store.clone();
    let timeout = Duration::from_secs(config.heartbeat_timeout_secs);
    let shutdown_reaper = shutdown.clone();
    tokio::spawn(async move {
        loop {
            tokio::select! {
                _ = shutdown_reaper.cancelled() => break,
                _ = tokio::time::sleep(Duration::from_secs(10)) => {}
            }
            registry_reaper.remove_stale_nodes(timeout).await;
            task_store_reaper.reassign_stale_tasks(timeout * 3).await;
            task_store_reaper.cleanup_old_tasks().await;
        }
    });

    // gRPC server
    let grpc_counters = counters.clone();
    let grpc_service = GrpcService {
        registry: registry.clone(),
        task_store: task_store.clone(),
        router,
        heartbeat_interval_secs: config.heartbeat_timeout_secs / 3,
        counters: grpc_counters,
    };

    // gRPC on dedicated port (for local dev)
    let grpc_addr = format!("{}:{}", config.bind_addr, config.grpc_port).parse()?;
    let shutdown_grpc = shutdown.clone();
    let grpc_handle = tokio::spawn(async move {
        info!(%grpc_addr, "gRPC server listening");
        Server::builder()
            .add_service(NodeCoordinatorServer::new(grpc_service))
            .serve_with_shutdown(grpc_addr, shutdown_grpc.cancelled())
            .await
    });

    // HTTP server
    let http_addr = format!("{}:{}", config.bind_addr, config.http_port);
    // Stale shard reaper
    let training_reaper = training_store.clone();
    let timeout_training = timeout * 6; // 3x heartbeat timeout for training shards
    let shutdown_training = shutdown.clone();
    tokio::spawn(async move {
        loop {
            tokio::select! {
                _ = shutdown_training.cancelled() => break,
                _ = tokio::time::sleep(Duration::from_secs(60)) => {}
            }
            training_reaper.reassign_stale_shards(timeout_training).await;
        }
    });

    // Layer training: reassign stale shards + auto-advance rounds
    let lt_auto = layer_training.clone();
    let ds_auto = dataset_store.clone();
    let data_dir_auto = data_dir.to_string_lossy().to_string();
    let shutdown_auto = shutdown.clone();
    tokio::spawn(async move {
        loop {
            tokio::select! {
                _ = shutdown_auto.cancelled() => break,
                _ = tokio::time::sleep(Duration::from_secs(30)) => {}
            }
            lt_auto.reassign_stale(120).await;
            lt_auto.merge_completed_rounds(&data_dir_auto).await;
            let ds_size = ds_auto.stats().await.total_examples;
            lt_auto.auto_advance(ds_size).await;
        }
    });

    let http_router = TaskRouter::new(registry.clone(), task_store.clone());
    let ws_state = Arc::new(ws_handler::WsState::new());
    let model_dir = crate::model_paths::model_dir();
    info!(path = %model_dir.display(), "Coordinator model directory (HYVERK_MODEL_DIR)");
    let app = http_api::create_router(AppState {
        task_store,
        registry,
        router: http_router,
        dataset_store,
        training_store,
        layer_training,
        // Same Arc-backed manager as ws_state — keep a handle for future non-WS callers.
        cluster_manager: ws_state.cluster_mgr.clone(),
        pending_signals: Arc::new(tokio::sync::RwLock::new(std::collections::HashMap::new())),
        ws_state,
        rag_store,
        counters: counters.clone(),
        started_at_secs,
    });
    let listener = tokio::net::TcpListener::bind(&http_addr).await?;
    info!(%http_addr, "HTTP server listening");

    let shutdown_http = shutdown.clone();
    let http_handle = tokio::spawn(async move {
        axum::serve(listener, app)
            .with_graceful_shutdown(async move { shutdown_http.cancelled().await })
            .await
    });

    tokio::select! {
        r = grpc_handle => {
            error!("gRPC server exited: {:?}", r);
        }
        r = http_handle => {
            error!("HTTP server exited: {:?}", r);
        }
    }

    Ok(())
}

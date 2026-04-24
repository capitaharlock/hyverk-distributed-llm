mod model;
mod sampling;
mod server;
mod worker;

use anyhow::Result;
use candle_core::Device;
use clap::Parser;
use std::{net::SocketAddr, path::PathBuf, sync::Arc};
use tokio::sync::Mutex;
use tracing::info;

use crate::{
    model::{KvStore, QwenShard},
    server::{build_router, run_tcp_server, SharedWorker},
    worker::InferenceWorker,
};

#[derive(Parser, Debug)]
#[command(name = "hyverk-infer", about = "Hyverk Candle inference server")]
struct Args {
    /// Path to model directory (contains config.json + *.safetensors)
    #[arg(long, env = "HYVERK_MODEL_DIR")]
    model_dir: PathBuf,

    /// HTTP port (TCP server binds to port+1)
    #[arg(long, env = "HYVERK_PORT", default_value = "18100")]
    port: u16,

    /// Hostname/IP to bind
    #[arg(long, env = "HYVERK_HOST", default_value = "0.0.0.0")]
    host: String,

    /// First decoder layer to load (inclusive)
    #[arg(long, env = "HYVERK_LAYER_START", default_value = "0")]
    layer_start: usize,

    /// Last decoder layer to load (exclusive); 0 = load all
    #[arg(long, env = "HYVERK_LAYER_END", default_value = "0")]
    layer_end: usize,

    /// Max KV cache entries (per-request slots)
    #[arg(long, env = "HYVERK_KV_MAX_ENTRIES", default_value = "16")]
    kv_max_entries: usize,

    /// KV idle eviction timeout in seconds
    #[arg(long, env = "HYVERK_KV_IDLE_TIMEOUT_S", default_value = "300")]
    kv_idle_timeout_s: u64,

    /// Force CPU device
    #[arg(long)]
    cpu: bool,
}

#[tokio::main]
async fn main() -> Result<()> {
    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| tracing_subscriber::EnvFilter::new("info")),
        )
        .init();

    let args = Args::parse();

    let device = pick_device(args.cpu)?;
    let device_name = device_label(&device);
    info!("device={device_name}");

    info!("Loading model from {}", args.model_dir.display());
    let t0 = std::time::Instant::now();

    // Determine shard range
    let cfg_path = args.model_dir.join("config.json");
    let cfg: model::Qwen2Config = serde_json::from_str(&std::fs::read_to_string(&cfg_path)?)?;
    let layer_end = if args.layer_end == 0 { cfg.num_hidden_layers } else { args.layer_end };
    let layer_start = args.layer_start;

    info!("Loading layers {layer_start}..{layer_end} of {}", cfg.num_hidden_layers);
    let shard = QwenShard::load(&args.model_dir, layer_start, layer_end, device)?;
    info!("Model loaded in {:.1}s", t0.elapsed().as_secs_f32());

    let kv_store = KvStore::new(args.kv_max_entries, args.kv_idle_timeout_s);
    let worker = InferenceWorker::new(shard, kv_store, device_name.clone());
    let shared: SharedWorker = Arc::new(Mutex::new(worker));

    // KV idle reaper
    {
        let w = shared.clone();
        tokio::spawn(async move {
            let mut interval = tokio::time::interval(std::time::Duration::from_secs(60));
            loop {
                interval.tick().await;
                let mut locked = w.lock().await;
                let evicted = locked.kv_store.evict_idle();
                if evicted > 0 {
                    tracing::info!("kv_reaper: evicted {evicted} idle entries");
                }
            }
        });
    }

    let http_addr: SocketAddr = format!("{}:{}", args.host, args.port).parse()?;
    let tcp_addr: SocketAddr = format!("{}:{}", args.host, args.port + 1).parse()?;

    // Start TCP server
    {
        let w = shared.clone();
        tokio::spawn(async move {
            if let Err(e) = run_tcp_server(tcp_addr, w).await {
                tracing::error!("TCP server error: {e}");
            }
        });
    }

    // Print ready signal — Rust coordinator reads this from stdout
    let ready = serde_json::json!({
        "status": "ready",
        "device": device_name,
        "layer_start": layer_start,
        "layer_end": layer_end,
        "http_port": args.port,
        "tcp_port": args.port + 1,
        "host": args.host,
    });
    println!("{}", serde_json::to_string(&ready)?);

    // Start HTTP server (blocks)
    info!("HTTP listening on {http_addr}");
    let app = build_router(shared);
    let listener = tokio::net::TcpListener::bind(http_addr).await?;
    axum::serve(listener, app).await?;
    Ok(())
}

fn pick_device(force_cpu: bool) -> Result<Device> {
    if force_cpu {
        return Ok(Device::Cpu);
    }
    #[cfg(feature = "cuda")]
    {
        if candle_core::utils::cuda_is_available() {
            return Ok(Device::new_cuda(0)?);
        }
    }
    #[cfg(feature = "metal")]
    {
        if candle_core::utils::metal_is_available() {
            return Ok(Device::new_metal(0)?);
        }
    }
    Ok(Device::Cpu)
}

fn device_label(d: &Device) -> String {
    match d {
        Device::Cpu => "cpu".to_string(),
        Device::Cuda(_) => "cuda".to_string(),
        Device::Metal(_) => "metal".to_string(),
    }
}

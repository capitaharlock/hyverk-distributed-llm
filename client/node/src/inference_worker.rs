// Inference Worker — Tensor-Parallel Layer Server
//
// When assigned to a serving cluster by the coordinator:
// 1. Downloads assigned layer weights
// 2. Starts the Python serve_layers.py HTTP server
// 3. Accepts /forward requests from the previous node in the chain
// 4. Sends activations to the next node
//
// The coordinator tells each node:
//   - Which layers to serve (layer_start, layer_end)
//   - Its position (first/middle/last)
//   - The next node's URL (for forwarding activations)

use tracing::{info, error};

/// Start the inference layer server as a subprocess
pub async fn run_inference_worker(
    coordinator_base: &str,
    node_id: &str,
    model_dir: &str,
    adapter_path: &str,
    layer_start: usize,
    layer_end: usize,
    position: &str,
    listen_port: u16,
    next_node_url: &str,
    shutdown: tokio_util::sync::CancellationToken,
) {
    info!(
        layers = format!("{}-{}", layer_start, layer_end),
        position,
        port = listen_port,
        "Starting inference layer server"
    );

    let script = find_inference_script();
    let python = find_python();

    let mut cmd = tokio::process::Command::new(&python);
    cmd.arg(&script)
        .arg("--model-dir").arg(model_dir)
        .arg("--layer-start").arg(layer_start.to_string())
        .arg("--layer-end").arg(layer_end.to_string())
        .arg("--listen-port").arg(listen_port.to_string())
        .arg("--position").arg(position)
        .kill_on_drop(true);

    if !adapter_path.is_empty() {
        cmd.arg("--adapter").arg(adapter_path);
    }
    if !next_node_url.is_empty() {
        cmd.arg("--next-node").arg(next_node_url);
    }

    let mut child = match cmd.spawn() {
        Ok(c) => c,
        Err(e) => {
            error!("Failed to start inference server: {e}");
            return;
        }
    };

    info!("Inference server running on port {listen_port}");

    // Wait for shutdown signal or process exit
    tokio::select! {
        _ = shutdown.cancelled() => {
            info!("Shutting down inference server");
            child.kill().await.ok();
        }
        status = child.wait() => {
            match status {
                Ok(s) => info!("Inference server exited: {s}"),
                Err(e) => error!("Inference server error: {e}"),
            }
        }
    }
}

fn find_python() -> String {
    let candidates = [".venv/bin/python3", "../.venv/bin/python3", "python3"];
    for p in &candidates {
        if std::path::Path::new(p).exists() { return p.to_string(); }
    }
    "python3".to_string()
}

fn find_inference_script() -> String {
    let candidates = [
        "inference/serve_layers.py",
        "../inference/serve_layers.py",
        "/app/inference/serve_layers.py",
    ];
    for p in &candidates {
        if std::path::Path::new(p).exists() { return p.to_string(); }
    }
    "inference/serve_layers.py".to_string()
}

// Layer-Sharded LoRA Training Worker
//
// This runs on each node alongside the main poll loop.
// It claims layer assignments from the coordinator, downloads
// only the needed layers (~1GB), trains LoRA adapters, and
// uploads the result (~2MB).
//
// Flow:
//   1. POST /api/v1/layer-training/claim → get assignment
//   2. GET  /api/v1/layer-training/shard/{offset}/{size} → training data
//   3. Load layer weights from local cache or coordinator
//   4. Train LoRA for those layers
//   5. POST /api/v1/layer-training/submit → upload adapter

use serde::Deserialize;
use tracing::{info, error};

#[derive(Debug, Deserialize)]
pub struct LayerAssignment {
    pub assignment_id: String,
    pub layer_start: usize,
    pub layer_end: usize,
    pub data_shard_start: usize,
    pub data_shard_size: usize,
}

/// Run the layer training worker loop.
/// Polls for assignments, trains, submits results.
pub async fn run_layer_worker(
    coordinator_http_url: &str,
    node_id: &str,
    model_dir: &str,
    shutdown: tokio_util::sync::CancellationToken,
) {
    let client = reqwest::Client::builder()
        .timeout(std::time::Duration::from_secs(120))
        .build()
        .unwrap_or_else(|_| reqwest::Client::new());
    let base = coordinator_http_url.trim_end_matches('/');

    // Check if model weights exist locally — if not, skip training (synthesis-only node)
    let index_path = format!("{}/model.safetensors.index.json", model_dir);
    if !std::path::Path::new(&index_path).exists() {
        info!("No model weights found at {model_dir} — this node will do synthesis only, not training");
        return;
    }

    info!("Layer training worker started (model weights found)");

    loop {
        tokio::select! {
            _ = shutdown.cancelled() => {
                info!("Layer worker shutting down");
                break;
            }
            _ = tokio::time::sleep(std::time::Duration::from_secs(10)) => {}
        }

        // Try to claim an assignment
        let assignment = match claim_assignment(&client, base, node_id).await {
            Some(a) => a,
            None => continue, // No work available, keep polling
        };

        info!(
            assignment_id = %assignment.assignment_id,
            layers = format!("{}-{}", assignment.layer_start, assignment.layer_end),
            data_size = assignment.data_shard_size,
            "Claimed layer assignment"
        );

        // Download training data shard
        let data = match download_data_shard(
            &client, base,
            assignment.data_shard_start,
            assignment.data_shard_size,
        ).await {
            Ok(d) => d,
            Err(e) => {
                error!("Failed to download data shard: {e}");
                continue;
            }
        };

        info!(examples = data.len(), "Downloaded training data");

        // Train LoRA for assigned layers
        let result = train_layer_lora(
            model_dir,
            assignment.layer_start,
            assignment.layer_end,
            &data,
            base,
        ).await;

        match result {
            Ok((adapter_bytes, loss, steps)) => {
                info!(
                    loss = loss,
                    steps = steps,
                    adapter_size = adapter_bytes.len(),
                    "Layer training complete"
                );

                // Submit adapter
                if let Err(e) = submit_adapter(
                    &client, base,
                    &assignment.assignment_id,
                    adapter_bytes,
                    loss,
                    steps,
                ).await {
                    error!("Failed to submit adapter: {e}");
                }
            }
            Err(e) => {
                error!("Training failed: {e}");
            }
        }
    }
}

async fn claim_assignment(
    client: &reqwest::Client,
    base: &str,
    node_id: &str,
) -> Option<LayerAssignment> {
    let resp = client
        .post(format!("{base}/api/v1/layer-training/claim"))
        .json(&serde_json::json!({"node_id": node_id}))
        .send()
        .await
        .ok()?;

    if !resp.status().is_success() {
        return None;
    }

    let body: serde_json::Value = resp.json().await.ok()?;
    if body.get("error").is_some() {
        return None;
    }

    serde_json::from_value(body["assignment"].clone()).ok()
}

#[derive(Debug, Deserialize)]
struct TrainingExample {
    instruction: String,
    response: String,
}

async fn download_data_shard(
    client: &reqwest::Client,
    base: &str,
    offset: usize,
    size: usize,
) -> Result<Vec<TrainingExample>, Box<dyn std::error::Error + Send + Sync>> {
    let resp = client
        .get(format!("{base}/api/v1/layer-training/shard/{offset}/{size}"))
        .send()
        .await?;

    let body = resp.text().await?;
    let examples: Vec<TrainingExample> = body
        .lines()
        .filter(|l| !l.trim().is_empty())
        .filter_map(|l| serde_json::from_str(l).ok())
        .collect();

    Ok(examples)
}

/// Train LoRA adapters for specific layers using PyTorch (subprocess).
/// Returns (adapter_bytes, final_loss, total_steps)
async fn train_layer_lora(
    model_dir: &str,
    layer_start: usize,
    layer_end: usize,
    data: &[TrainingExample],
    coordinator_base: &str,
) -> Result<(Vec<u8>, f32, usize), Box<dyn std::error::Error + Send + Sync>> {
    info!(
        layer_start, layer_end,
        examples = data.len(),
        "Starting real LoRA training (PyTorch)"
    );

    // Write data shard to temp file
    let data_file = format!("/tmp/hyverk_shard_{}_{}.jsonl", layer_start, layer_end);
    let adapter_file = format!("/tmp/hyverk_adapter_{}_{}.safetensors", layer_start, layer_end);
    {
        use std::io::Write;
        let mut f = std::fs::File::create(&data_file)?;
        for ex in data {
            let line = serde_json::json!({"instruction": ex.instruction, "response": ex.response});
            writeln!(f, "{}", serde_json::to_string(&line)?)?;
        }
    }

    // Ensure tokenizer exists locally (download from coordinator if missing)
    let tokenizer_path = format!("{}/tokenizer.json", model_dir);
    if !std::path::Path::new(&tokenizer_path).exists() {
        info!("Downloading tokenizer from coordinator...");
        let tok_url = format!("{}/api/v1/layer-training/tokenizer", coordinator_base);
        let http = reqwest::Client::new();
        match http.get(&tok_url).send().await {
            Ok(r) if r.status().is_success() => {
                let bytes = r.bytes().await.unwrap_or_default();
                if bytes.len() > 1000 {
                    std::fs::create_dir_all(model_dir).ok();
                    std::fs::write(&tokenizer_path, &bytes).ok();
                    info!(size = bytes.len(), "Tokenizer downloaded");
                }
            }
            _ => return Err("Cannot download tokenizer from coordinator".into()),
        }
    }

    let script = find_training_script()?;

    // Call PyTorch training as subprocess
    // Use venv python if available, otherwise system python3
    let python = find_python();
    let output = tokio::process::Command::new(&python)
        .arg(&script)
        .arg("--model-dir").arg(model_dir)
        .arg("--layer-start").arg(layer_start.to_string())
        .arg("--layer-end").arg(layer_end.to_string())
        .arg("--data-file").arg(&data_file)
        .arg("--output").arg(&adapter_file)
        .arg("--lora-rank").arg("16")
        .arg("--epochs").arg("1")
        .arg("--max-seq-len").arg("256")
        .stderr(std::process::Stdio::piped())
        .stdout(std::process::Stdio::piped())
        .kill_on_drop(true)
        .output();

    // Timeout: 90 seconds max per shard (prevents stuck processes)
    let output = tokio::time::timeout(
        std::time::Duration::from_secs(90),
        output,
    ).await
        .map_err(|_| "Training timed out after 90 seconds")?
        .map_err(|e| format!("Failed to run training script: {e}. Is python3 + torch installed?"))?;

    // Log stderr (training progress)
    let stderr = String::from_utf8_lossy(&output.stderr);
    for line in stderr.lines() {
        info!(target: "pytorch", "{}", line);
    }

    if !output.status.success() {
        let stdout = String::from_utf8_lossy(&output.stdout);
        // Try to parse error from JSON stdout
        if let Ok(result) = serde_json::from_str::<serde_json::Value>(&stdout) {
            if let Some(err) = result.get("error").and_then(|e| e.as_str()) {
                return Err(format!("Training failed: {err}").into());
            }
        }
        return Err(format!("Training script exited with code {:?}: {stderr}", output.status.code()).into());
    }

    // Parse result from stdout (JSON)
    let stdout = String::from_utf8_lossy(&output.stdout);
    let result: serde_json::Value = serde_json::from_str(stdout.trim())
        .map_err(|e| format!("Failed to parse training result: {e}\nstdout: {stdout}"))?;

    let loss = result["loss"].as_f64().unwrap_or(0.0) as f32;
    let steps = result["steps"].as_u64().unwrap_or(0) as usize;
    let device = result["device"].as_str().unwrap_or("unknown");

    info!(loss, steps, device, "PyTorch training complete");

    // Read adapter file
    let adapter_bytes = std::fs::read(&adapter_file)
        .map_err(|e| format!("Failed to read adapter: {e}"))?;

    // Cleanup temp files
    let _ = std::fs::remove_file(&data_file);
    let _ = std::fs::remove_file(&adapter_file);

    Ok((adapter_bytes, loss, steps))
}

fn find_python() -> String {
    let candidates = [
        ".venv/bin/python3",
        "../.venv/bin/python3",
        "../../.venv/bin/python3",
        "/app/.venv/bin/python3",
        "python3",
    ];
    for path in &candidates {
        if std::path::Path::new(path).exists() {
            return path.to_string();
        }
    }
    // Check VIRTUAL_ENV env
    if let Ok(venv) = std::env::var("VIRTUAL_ENV") {
        let p = format!("{venv}/bin/python3");
        if std::path::Path::new(&p).exists() {
            return p;
        }
    }
    "python3".to_string()
}

fn find_training_script() -> Result<String, Box<dyn std::error::Error + Send + Sync>> {
    let candidates = [
        "training/train_layer_lora.py",
        "../training/train_layer_lora.py",
        "../../training/train_layer_lora.py",
        "/app/training/train_layer_lora.py",  // Docker/Fly.io
    ];
    for path in &candidates {
        if std::path::Path::new(path).exists() {
            return Ok(path.to_string());
        }
    }
    // Try finding relative to executable
    if let Ok(exe) = std::env::current_exe() {
        if let Some(dir) = exe.parent() {
            let script = dir.join("training").join("train_layer_lora.py");
            if script.exists() {
                return Ok(script.to_string_lossy().to_string());
            }
        }
    }
    Err("Training script not found. Expected: training/train_layer_lora.py".into())
}

async fn submit_adapter(
    client: &reqwest::Client,
    base: &str,
    assignment_id: &str,
    adapter_bytes: Vec<u8>,
    loss: f32,
    steps: usize,
) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
    // Compress adapter before base64 (2.8MB → ~200KB compressed)
    use std::io::Write;
    let mut encoder = flate2::write::GzEncoder::new(Vec::new(), flate2::Compression::fast());
    encoder.write_all(&adapter_bytes)?;
    let compressed = encoder.finish()?;

    use base64::Engine;
    let adapter_b64 = base64::engine::general_purpose::STANDARD.encode(&compressed);

    info!(
        raw_size = adapter_bytes.len(),
        compressed_size = compressed.len(),
        b64_size = adapter_b64.len(),
        "Uploading adapter (compressed)"
    );

    let resp = client
        .post(format!("{base}/api/v1/layer-training/submit"))
        .json(&serde_json::json!({
            "assignment_id": assignment_id,
            "loss": loss,
            "steps": steps,
            "adapter_base64": adapter_b64,
            "compressed": true,
        }))
        .send()
        .await?;

    if resp.status().is_success() {
        info!(assignment_id, "Adapter submitted successfully");
        Ok(())
    } else {
        let body = resp.text().await.unwrap_or_default();
        Err(format!("Submit failed: {body}").into())
    }
}

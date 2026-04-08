// LoRA training loop.
// Uses AdamW optimizer from candle-nn.
// Gradient flows only through LoRA vars (base weights are frozen Tensors, not Vars).
// Supports gradient accumulation for large effective batch sizes.

use crate::{
    dataset::{load_jsonl, make_batches, tokenize_examples},
    lora::LoraConfig,
    model::{Qwen2Config, Qwen2LoraModel},
    TrainingConfig,
};
use candle_core::{DType, Device, Result, Tensor};
use candle_nn::{AdamW, Optimizer, ParamsAdamW, VarBuilder, VarMap};
use tokenizers::Tokenizer;
use tracing::{info, warn};

/// Get current process RSS (resident set size) in bytes on macOS.
fn current_rss_bytes() -> u64 {
    let usage = unsafe {
        let mut info: libc::rusage = std::mem::zeroed();
        libc::getrusage(libc::RUSAGE_SELF, &mut info);
        info
    };
    // ru_maxrss is in bytes on macOS
    usage.ru_maxrss as u64
}

/// Hard memory guard. Returns Err if RSS exceeds the limit — training loop catches this and exits cleanly.
fn check_memory_limit(limit_gb: u64, context: &str) -> std::result::Result<(), Box<dyn std::error::Error + Send + Sync>> {
    let limit_bytes = limit_gb * 1024 * 1024 * 1024;
    let rss = current_rss_bytes();
    let rss_gb = rss as f64 / 1e9;
    if rss > limit_bytes {
        return Err(format!(
            "MEMORY LIMIT EXCEEDED during {context}: {rss_gb:.1}GB > {limit_gb}GB limit. \
             Reduce --batch-size, --max-seq-len, or use a smaller model (1.5B instead of 7B).",
        ).into());
    }
    if rss > limit_bytes * 3 / 4 {
        warn!(rss_gb = format!("{rss_gb:.1}"), limit_gb, "Memory usage above 75% of limit");
    }
    Ok(())
}

pub struct TrainingResult {
    pub steps: usize,
    pub final_loss: f32,
    pub duration_secs: u64,
}

/// Train LoRA adapters on a data shard.
/// Returns serialized adapter weights (safetensors bytes) on success.
pub async fn train_on_shard(
    model_dir: &std::path::Path,
    tokenizer: &Tokenizer,
    shard_content: &str,
    config: &TrainingConfig,
) -> std::result::Result<(Vec<u8>, TrainingResult), Box<dyn std::error::Error + Send + Sync>> {
    let start = std::time::Instant::now();

    // Select device: Metal (Apple Silicon) → CPU fallback
    let device = select_device()?;
    info!("Training device: {device:?}");

    // Base weights in F16 (14GB vs 28GB for 7B), LoRA vars in F32 (optimizer needs F32)
    let base_dtype = DType::F16;
    let lora_dtype = DType::F32;

    // Load dataset
    let mut examples = load_jsonl(shard_content);
    if examples.is_empty() {
        return Err("No examples in shard".into());
    }
    // Optional cap for fast iteration (max_examples=0 means no limit)
    if config.max_examples > 0 && examples.len() > config.max_examples {
        examples.truncate(config.max_examples);
        warn!(capped = config.max_examples, "Dataset capped for fast training");
    }
    info!(examples = examples.len(), "Loaded shard");

    // Tokenize
    let batch = tokenize_examples(&examples, tokenizer, config.max_seq_len)?;
    if batch.input_ids.is_empty() {
        return Err("All examples were filtered out during tokenization".into());
    }
    info!(sequences = batch.input_ids.len(), "Tokenized");

    // Estimate memory usage before loading
    let bytes_per_param: usize = match base_dtype {
        DType::F16 | DType::BF16 => 2,
        DType::F32 => 4,
        _ => 4,
    };
    let model_params = 7_000_000_000usize; // ~7B for Qwen2.5-7B
    let weight_gb = (model_params * bytes_per_param) as f64 / 1e9;
    let activation_gb = (config.batch_size * config.max_seq_len * 3584 * bytes_per_param * 28) as f64 / 1e9; // rough estimate
    let total_est_gb = weight_gb + activation_gb * 3.0; // 3x for forward+backward+optimizer
    info!(
        weights_gb = format!("{weight_gb:.1}"),
        activation_gb = format!("{activation_gb:.1}"),
        estimated_total_gb = format!("{total_est_gb:.1}"),
        batch_size = config.batch_size,
        seq_len = config.max_seq_len,
        base_dtype = ?base_dtype,
        "Memory estimate"
    );
    if total_est_gb > 48.0 {
        warn!("Estimated memory {total_est_gb:.0}GB may exceed system RAM. Consider: --batch-size 1 --max-seq-len 128");
    }

    // Build LoRA VarMap (trainable parameters — F32 for optimizer)
    let varmap = VarMap::new();
    let lora_vb = VarBuilder::from_varmap(&varmap, lora_dtype, &device);

    let lora_config = LoraConfig {
        rank: config.lora_rank,
        alpha: config.lora_alpha,
    };

    let mem_limit = config.max_memory_gb;
    check_memory_limit(mem_limit, "before model load")?;

    // Load model (base weights frozen + LoRA vars in varmap)
    let model_config = Qwen2Config::qwen25_7b(); // TODO: load from config.json
    let model = tokio::task::spawn_blocking({
        let model_dir = model_dir.to_path_buf();
        let device = device.clone();
        move || Qwen2LoraModel::load(&model_dir, model_config, &lora_config, lora_vb, &device, base_dtype)
    })
    .await
    .map_err(|e| format!("spawn_blocking: {e}"))??;

    check_memory_limit(mem_limit, "after model load")?;

    let all_vars = varmap.all_vars();
    info!(lora_params = all_vars.len(), "LoRA vars initialized");

    // Optimizer
    let mut optimizer = AdamW::new(
        all_vars,
        ParamsAdamW {
            lr: config.learning_rate,
            weight_decay: 0.01,
            ..Default::default()
        },
    )?;

    let batches = make_batches(&batch, config.batch_size);
    let mut total_steps = 0usize;
    let mut last_loss = f32::NAN;

    for epoch in 0..config.num_epochs {
        let mut epoch_loss = 0f32;
        let mut accum_loss: Option<Tensor> = None;
        let mut accum_count = 0usize;

        for (step, (input_batch, label_batch)) in batches.iter().enumerate() {
            let input_ids = batch_to_tensor(input_batch, &device, DType::U32)?;
            let labels = batch_to_tensor_i64(label_batch, &device)?;

            check_memory_limit(mem_limit, "before forward pass")?;

            // Forward pass
            let logits = model.forward(&input_ids)?;
            // logits: [batch, seq, vocab] — may be F16, cast to F32 for loss
            let logits = logits.to_dtype(DType::F32)?;

            check_memory_limit(mem_limit, "after forward pass")?;
            let (b, s, v) = logits.dims3()?;

            // Flatten for cross-entropy: [batch*seq, vocab]
            let logits_flat = logits.reshape((b * s, v))?;

            // Build target tensor: flatten labels, replace -100 with 0 for indexing
            // Then compute loss only on non-ignored positions
            let loss_val = masked_cross_entropy(&logits_flat, &labels, b * s, &device)?;

            // Gradient accumulation
            let normalized = (loss_val / config.grad_accum_steps as f64)?;
            accum_loss = Some(match accum_loss {
                None => normalized,
                Some(prev) => (prev + normalized)?,
            });
            accum_count += 1;

            if accum_count >= config.grad_accum_steps || step == batches.len() - 1 {
                if let Some(loss) = accum_loss.take() {
                    let loss_val_f = loss.to_scalar::<f32>()?;
                    optimizer.backward_step(&loss)?;
                    epoch_loss += loss_val_f;
                    total_steps += 1;
                    last_loss = loss_val_f;

                    if total_steps % 10 == 0 {
                        info!(epoch, step, loss = loss_val_f, "Training step");
                    }
                }
                accum_count = 0;
            }
        }

        let avg_loss = epoch_loss / batches.len() as f32;
        info!(epoch, avg_loss, "Epoch complete");
    }

    // Serialize LoRA adapter
    let adapter_bytes = crate::adapter::serialize_adapter(&varmap)?;

    let result = TrainingResult {
        steps: total_steps,
        final_loss: last_loss,
        duration_secs: start.elapsed().as_secs(),
    };

    info!(
        steps = total_steps,
        final_loss = last_loss,
        secs = result.duration_secs,
        "Training complete"
    );

    Ok((adapter_bytes, result))
}

/// Cross-entropy loss, ignoring positions where label == -100.
/// Manual implementation — per-position loss with mask.
fn masked_cross_entropy(
    logits: &Tensor,     // [N, vocab]
    labels: &Tensor,     // [batch, seq] with -100 for ignored
    n: usize,
    _device: &Device,
) -> Result<Tensor> {
    let labels_flat = labels.reshape(n)?; // [N] i64
    let logits_dtype = logits.dtype();

    // Create mask: 1.0 where label >= 0 (not ignored)
    let mask = labels_flat.ge(0i64)?.to_dtype(logits_dtype)?; // [N]
    let n_active = mask.sum_all()?.to_scalar::<f32>()?.max(1.0) as f64;

    // Replace -100 with 0 for safe gather (masked out anyway)
    let zeros_i64 = Tensor::zeros_like(&labels_flat)?;
    let valid_mask = labels_flat.ge(0i64)?;
    let safe_labels = valid_mask.where_cond(&labels_flat, &zeros_i64)?
        .to_dtype(candle_core::DType::U32)?; // [N] u32

    // Manual log-softmax + nll (per-position)
    // log_softmax = logits - log(sum(exp(logits)))
    let max_logits = logits.max_keepdim(candle_core::D::Minus1)?;
    let shifted = logits.broadcast_sub(&max_logits)?;
    let exp = shifted.exp()?;
    let sum_exp = exp.sum_keepdim(candle_core::D::Minus1)?;
    let log_sum_exp = sum_exp.log()?;
    let log_probs = shifted.broadcast_sub(&log_sum_exp)?; // [N, vocab]

    // Gather log probs at target positions: log_probs[i, safe_labels[i]]
    let target_indices = safe_labels.unsqueeze(1)?; // [N, 1]
    let target_log_probs = log_probs.gather(&target_indices, 1)?.squeeze(1)?; // [N]

    // NLL = -log_probs, masked
    let nll = target_log_probs.neg()?; // [N]
    let masked_nll = (nll * mask)?;

    masked_nll.sum_all()? / n_active
}

fn batch_to_tensor(data: &[Vec<u32>], device: &Device, dtype: DType) -> Result<Tensor> {
    let b = data.len();
    let s = data[0].len();
    let flat: Vec<u32> = data.iter().flatten().copied().collect();
    Tensor::from_vec(flat, (b, s), device)?.to_dtype(dtype)
}

fn batch_to_tensor_i64(data: &[Vec<i64>], device: &Device) -> Result<Tensor> {
    let b = data.len();
    let s = data[0].len();
    let flat: Vec<i64> = data.iter().flatten().copied().collect();
    Tensor::from_vec(flat, (b, s), device)
}

fn select_device() -> std::result::Result<Device, Box<dyn std::error::Error + Send + Sync>> {
    // candle-core is always compiled with metal feature — try it unconditionally
    if let Ok(device) = Device::new_metal(0) {
        info!("Using Metal GPU");
        return Ok(device);
    }
    warn!("Metal unavailable, falling back to CPU");
    Ok(Device::Cpu)
}

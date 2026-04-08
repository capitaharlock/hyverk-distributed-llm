// @llm-context: _rjj/context/modules/training/overview.md
// Phase 4: Distributed LoRA Fine-tuning
//
// Architecture:
//   - Coordinator distributes dataset shards (from hyverk-synthesis output)
//   - Each node downloads a shard, trains LoRA adapters locally
//   - Node uploads adapter weights to coordinator
//   - Coordinator aggregates via FedAvg → merged adapter
//
// Model: Qwen2.5-Coder-7B (or 32B for production)
// Framework: candle-core + candle-nn (Metal backend for Apple Silicon)
// LoRA: rank=16, applied to q/k/v/o projections in attention layers
// Training: AdamW optimizer, cross-entropy loss, ChatML format

pub mod adapter;
pub mod dataset;
pub mod lora;
pub mod model;
pub mod trainer;

use serde::{Deserialize, Serialize};

/// LoRA fine-tuning configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TrainingConfig {
    /// LoRA rank (r). Higher = more capacity, more params.
    pub lora_rank: usize,
    /// LoRA alpha. Effective scale = alpha/rank.
    pub lora_alpha: f64,
    /// AdamW learning rate
    pub learning_rate: f64,
    /// Training epochs over the shard
    pub num_epochs: usize,
    /// Mini-batch size (sequences per step)
    pub batch_size: usize,
    /// Max sequence length in tokens
    pub max_seq_len: usize,
    /// Gradient accumulation steps (effective batch = batch_size * accum)
    pub grad_accum_steps: usize,
    /// Modules to apply LoRA to (attention projections)
    pub target_modules: Vec<String>,
    /// Model architecture (for weight name mapping)
    pub model_family: ModelFamily,
    /// Cap dataset at N examples (0 = no limit). Use for fast dev iteration.
    pub max_examples: usize,
    /// Hard memory limit in GB. Process aborts if RSS exceeds this.
    pub max_memory_gb: u64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ModelFamily {
    Qwen2,
    Llama3,
    Mistral,
}

impl Default for TrainingConfig {
    fn default() -> Self {
        Self {
            lora_rank: 16,
            lora_alpha: 32.0,
            learning_rate: 2e-4,
            num_epochs: 3,
            batch_size: 1,       // Conservative default — 7B model is memory-hungry
            max_seq_len: 256,    // 256 tokens; use grad_accum to compensate small batch
            grad_accum_steps: 8, // Effective batch = 1 * 8 = 8
            target_modules: vec![
                "q_proj".to_string(),
                "k_proj".to_string(),
                "v_proj".to_string(),
                "o_proj".to_string(),
            ],
            model_family: ModelFamily::Qwen2,
            max_examples: 0,    // 0 = use all examples
            max_memory_gb: 16,  // Hard limit — abort if exceeded
        }
    }
}

/// Training job status sent/received from coordinator
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TrainingJobStatus {
    pub job_id: String,
    pub status: String,
    pub shards_total: usize,
    pub shards_completed: usize,
    pub adapters_received: usize,
}

/// A dataset shard assigned to this node
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DataShard {
    pub shard_id: String,
    pub job_id: String,
    /// JSONL content (instruction + response pairs)
    pub content: String,
    pub example_count: usize,
}

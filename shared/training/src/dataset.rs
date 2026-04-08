// Dataset loading, tokenization, and batching for LoRA fine-tuning.
// Input format: JSONL with {"instruction": "...", "response": "..."}
// Tokenization: ChatML format for Qwen2.5-Instruct
// Labels: -100 for prompt tokens (not trained on), token ids for response

use serde::Deserialize;
use tokenizers::Tokenizer;

#[derive(Debug, Deserialize)]
pub struct TrainingExample {
    pub instruction: String,
    pub response: String,
    #[serde(default)]
    pub category: String,
}

/// ChatML system prompt that defines the Hyverk persona
pub const HYVERK_SYSTEM: &str = "You are Hyverk, an expert software engineer specializing in Rust, TypeScript, distributed systems, and DevOps. You write clean, idiomatic, production-quality code with proper error handling.";

/// Format a training example as ChatML tokens
/// Returns: (full_text, prompt_len_chars) — prompt_len used to mask labels
pub fn format_chatml(example: &TrainingExample) -> String {
    format!(
        "<|im_start|>system\n{}<|im_end|>\n<|im_start|>user\n{}<|im_end|>\n<|im_start|>assistant\n{}<|im_end|>",
        HYVERK_SYSTEM, example.instruction, example.response
    )
}

pub fn format_chatml_prompt_only(example: &TrainingExample) -> String {
    format!(
        "<|im_start|>system\n{}<|im_end|>\n<|im_start|>user\n{}<|im_end|>\n<|im_start|>assistant\n",
        HYVERK_SYSTEM, example.instruction
    )
}

/// A tokenized batch ready for the training loop
pub struct TokenizedBatch {
    /// Shape: [batch_size, seq_len]
    pub input_ids: Vec<Vec<u32>>,
    /// Shape: [batch_size, seq_len], -100 = ignore in loss
    pub labels: Vec<Vec<i64>>,
}

pub fn load_jsonl(content: &str) -> Vec<TrainingExample> {
    content
        .lines()
        .filter(|l| !l.trim().is_empty())
        .filter_map(|l| serde_json::from_str(l).ok())
        .collect()
}

/// Tokenize examples into batches using parallel batch encoding.
/// Masks prompt tokens with -100 (not trained, only trained on response tokens).
/// Uses encode_batch internally to parallelize across rayon threads.
pub fn tokenize_examples(
    examples: &[TrainingExample],
    tokenizer: &Tokenizer,
    max_seq_len: usize,
) -> Result<TokenizedBatch, Box<dyn std::error::Error + Send + Sync>> {
    let total = examples.len();

    // Build all texts upfront for batch encoding
    let prompts: Vec<tokenizers::EncodeInput> = examples
        .iter()
        .map(|ex| format_chatml_prompt_only(ex).into())
        .collect();
    let fulls: Vec<tokenizers::EncodeInput> = examples
        .iter()
        .map(|ex| format_chatml(ex).into())
        .collect();

    // Parallel batch encoding — uses rayon internally
    tracing::info!(examples = total, "Tokenizing (batch mode)...");
    let prompt_encs = tokenizer
        .encode_batch(prompts, false)
        .map_err(|e| format!("batch tokenizer error (prompts): {e}"))?;
    let full_encs = tokenizer
        .encode_batch(fulls, false)
        .map_err(|e| format!("batch tokenizer error (fulls): {e}"))?;

    let mut input_ids = Vec::with_capacity(total);
    let mut labels = Vec::with_capacity(total);

    for (prompt_enc, full_enc) in prompt_encs.iter().zip(full_encs.iter()) {
        let prompt_len = prompt_enc.get_ids().len().min(max_seq_len);
        let full_ids: Vec<u32> = full_enc
            .get_ids()
            .iter()
            .copied()
            .take(max_seq_len)
            .collect();

        // Labels: -100 for prompt, token id for response
        let lbl: Vec<i64> = full_ids
            .iter()
            .enumerate()
            .map(|(i, &id)| if i < prompt_len { -100i64 } else { id as i64 })
            .collect();

        // Pad to max_seq_len
        let mut padded_ids = full_ids;
        padded_ids.resize(max_seq_len, 0);
        let mut padded_lbl = lbl;
        padded_lbl.resize(max_seq_len, -100);

        // Skip examples where all labels are -100 (response too long, truncated)
        if padded_lbl.iter().all(|&l| l == -100) {
            tracing::debug!("Skipping example: response entirely truncated");
            continue;
        }

        input_ids.push(padded_ids);
        labels.push(padded_lbl);
    }

    tracing::info!(kept = input_ids.len(), filtered = total - input_ids.len(), "Tokenization complete");
    Ok(TokenizedBatch { input_ids, labels })
}

/// Split a flat list of examples into mini-batches
pub fn make_batches(
    batch: &TokenizedBatch,
    batch_size: usize,
) -> Vec<(Vec<Vec<u32>>, Vec<Vec<i64>>)> {
    let n = batch.input_ids.len();
    (0..n)
        .step_by(batch_size)
        .map(|start| {
            let end = (start + batch_size).min(n);
            (
                batch.input_ids[start..end].to_vec(),
                batch.labels[start..end].to_vec(),
            )
        })
        .collect()
}

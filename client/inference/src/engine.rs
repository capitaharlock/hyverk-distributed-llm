// @llm-context: _rjj/stack.md
// @llm-critical: This wraps llama.cpp — inference runs on a blocking thread, not async
// @llm-critical: Models are cached in memory after first load. Cache is keyed by model name.

use hyverk_core::error::HyverkError;
use llama_cpp_2::context::params::LlamaContextParams;
use llama_cpp_2::llama_backend::LlamaBackend;
use llama_cpp_2::llama_batch::LlamaBatch;
use llama_cpp_2::model::params::LlamaModelParams;
use llama_cpp_2::model::{AddBos, LlamaModel};
use llama_cpp_2::sampling::LlamaSampler;
use std::collections::HashMap;
use std::path::PathBuf;
use std::sync::{Arc, Mutex};
use tracing::{info, warn};

pub struct InferenceEngine {
    backend: Arc<LlamaBackend>,
    models_dir: PathBuf,
    cache: Mutex<HashMap<String, Arc<LlamaModel>>>,
}

#[derive(Debug, Clone)]
pub struct GenerateRequest {
    pub model: String,
    pub prompt: String,
    pub temperature: f32,
    pub max_tokens: u32,
}

#[derive(Debug, Clone)]
pub struct GenerateResponse {
    pub text: String,
    pub tokens_generated: u32,
}

impl InferenceEngine {
    pub fn new(models_dir: PathBuf) -> Result<Self, HyverkError> {
        let backend = LlamaBackend::init()
            .map_err(|e| HyverkError::Inference(format!("Failed to init llama backend: {e}")))?;
        Ok(Self {
            backend: Arc::new(backend),
            models_dir,
            cache: Mutex::new(HashMap::new()),
        })
    }

    /// List available GGUF models in the models directory.
    pub fn list_models(&self) -> Vec<String> {
        let mut models = Vec::new();
        let dir = match std::fs::read_dir(&self.models_dir) {
            Ok(d) => d,
            Err(e) => {
                warn!("Cannot read models dir {:?}: {}", self.models_dir, e);
                return models;
            }
        };
        for entry in dir.flatten() {
            let path = entry.path();
            if path.extension().map_or(false, |ext| ext == "gguf") {
                if let Some(stem) = path.file_stem() {
                    models.push(stem.to_string_lossy().to_string());
                }
            }
        }
        models.sort();
        models
    }

    fn model_path(&self, model: &str) -> PathBuf {
        self.models_dir.join(format!("{model}.gguf"))
    }

    /// Get or load a model from cache. Models stay in memory after first load.
    fn get_or_load_model(&self, model_name: &str) -> Result<Arc<LlamaModel>, HyverkError> {
        let mut cache = self.cache.lock().unwrap();
        if let Some(model) = cache.get(model_name) {
            return Ok(model.clone());
        }

        let model_path = self.model_path(model_name);
        if !model_path.exists() {
            return Err(HyverkError::Inference(format!(
                "Model not found: {}",
                model_path.display()
            )));
        }

        info!(model = %model_name, path = %model_path.display(), "Loading model into cache");
        let model_params = LlamaModelParams::default();
        let model = LlamaModel::load_from_file(&self.backend, &model_path, &model_params)
            .map_err(|e| HyverkError::Inference(format!("Failed to load model: {e}")))?;

        let model = Arc::new(model);
        cache.insert(model_name.to_string(), model.clone());
        info!(model = %model_name, "Model cached");
        Ok(model)
    }

    /// Run inference. CPU-bound work runs on a blocking tokio thread.
    pub async fn generate(&self, req: GenerateRequest) -> Result<GenerateResponse, HyverkError> {
        // Load/cache model on the current thread (fast if cached)
        let model = self.get_or_load_model(&req.model)?;
        let backend = self.backend.clone();

        let result = tokio::task::spawn_blocking(move || {
            Self::run_inference(&backend, &model, &req)
        })
        .await
        .map_err(|e| HyverkError::Inference(format!("Task join error: {e}")))?;

        result
    }

    fn run_inference(
        backend: &LlamaBackend,
        model: &LlamaModel,
        req: &GenerateRequest,
    ) -> Result<GenerateResponse, HyverkError> {
        let ctx_params = LlamaContextParams::default();
        let mut ctx = model
            .new_context(backend, ctx_params)
            .map_err(|e| HyverkError::Inference(format!("Failed to create context: {e}")))?;

        let tokens = model
            .str_to_token(&req.prompt, AddBos::Always)
            .map_err(|e| HyverkError::Inference(format!("Tokenization failed: {e}")))?;

        if tokens.is_empty() {
            return Ok(GenerateResponse {
                text: String::new(),
                tokens_generated: 0,
            });
        }

        let mut batch = LlamaBatch::new(ctx.n_ctx() as usize, 1);

        let last_idx = tokens.len() - 1;
        for (i, token) in tokens.iter().enumerate() {
            batch
                .add(*token, i as i32, &[0], i == last_idx)
                .map_err(|e| HyverkError::Inference(format!("Batch add failed: {e}")))?;
        }

        ctx.decode(&mut batch)
            .map_err(|e| HyverkError::Inference(format!("Decode failed: {e}")))?;

        let mut sampler = LlamaSampler::chain_simple([
            LlamaSampler::temp(req.temperature),
            LlamaSampler::dist(0),
        ]);

        let mut decoder = encoding_rs::UTF_8.new_decoder();
        let mut output = String::new();
        let mut n_cur = tokens.len();
        let max = req.max_tokens as usize;
        let mut generated = 0u32;

        for _ in 0..max {
            let token = sampler.sample(&ctx, batch.n_tokens() - 1);

            if model.is_eog_token(token) {
                break;
            }

            let piece = model
                .token_to_piece(token, &mut decoder, true, None)
                .map_err(|e| HyverkError::Inference(format!("Token to piece failed: {e}")))?;
            output.push_str(&piece);
            generated += 1;

            batch.clear();
            batch
                .add(token, n_cur as i32, &[0], true)
                .map_err(|e| HyverkError::Inference(format!("Batch add failed: {e}")))?;

            ctx.decode(&mut batch)
                .map_err(|e| HyverkError::Inference(format!("Decode failed: {e}")))?;

            n_cur += 1;
        }

        info!(tokens = generated, "Inference complete");

        Ok(GenerateResponse {
            text: output,
            tokens_generated: generated,
        })
    }
}

// Qwen2.5 transformer with LoRA adapters on attention projections.
// Supports Qwen2.5-7B and Qwen2.5-32B (same architecture, different sizes).
//
// Architecture:
//   - RMSNorm (pre-norm)
//   - Multi-head attention with GQA (Group Query Attention)
//   - Rotary Position Embedding (RoPE, theta=1_000_000)
//   - SwiGLU MLP
//   - LoRA on: q_proj, k_proj, v_proj, o_proj (attention)
//
// Weight names (from Qwen2.5 HuggingFace safetensors):
//   model.embed_tokens.weight
//   model.layers.{i}.input_layernorm.weight
//   model.layers.{i}.self_attn.q_proj.{weight,bias}
//   model.layers.{i}.self_attn.k_proj.{weight,bias}
//   model.layers.{i}.self_attn.v_proj.{weight,bias}
//   model.layers.{i}.self_attn.o_proj.weight
//   model.layers.{i}.post_attention_layernorm.weight
//   model.layers.{i}.mlp.{gate,up,down}_proj.weight
//   model.norm.weight
//   lm_head.weight

use crate::lora::{frozen_linear, LoraConfig, LoraLinear};
use candle_core::{DType, Device, Result, Tensor, D};
use candle_nn::{Embedding, Module, VarBuilder};
use std::collections::HashMap;
use std::sync::Arc;

// Manual RmsNorm — candle_nn::RmsNorm uses a kernel not available on Metal.
// RmsNorm(x) = x / sqrt(mean(x^2) + eps) * weight
struct RmsNorm {
    weight: Tensor,
    eps: f64,
}

impl RmsNorm {
    fn new(weight: Tensor, eps: f64) -> Self {
        Self { weight, eps }
    }
}

impl Module for RmsNorm {
    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        // xs: [batch, seq, hidden] or [batch, hidden]
        let variance = xs.sqr()?.mean_keepdim(D::Minus1)?;
        // Add eps via affine transform: variance * 1.0 + eps
        let rms = variance.affine(1.0, self.eps)?.sqrt()?;
        let normed = xs.broadcast_div(&rms)?;
        normed.broadcast_mul(&self.weight)
    }
}

/// Qwen2.5-7B configuration (hardcoded — loaded from config.json at runtime)
#[derive(Debug, Clone)]
pub struct Qwen2Config {
    pub hidden_size: usize,
    pub intermediate_size: usize,
    pub num_hidden_layers: usize,
    pub num_attention_heads: usize,
    pub num_key_value_heads: usize,
    pub vocab_size: usize,
    pub rms_norm_eps: f64,
    pub rope_theta: f64,
    pub max_position_embeddings: usize,
}

impl Qwen2Config {
    /// Qwen2.5-Coder-7B defaults
    pub fn qwen25_7b() -> Self {
        Self {
            hidden_size: 3584,
            intermediate_size: 18944,
            num_hidden_layers: 28,
            num_attention_heads: 28,
            num_key_value_heads: 4,
            vocab_size: 151936,
            rms_norm_eps: 1e-6,
            rope_theta: 1_000_000.0,
            max_position_embeddings: 131072,
        }
    }

    /// Qwen2.5-Coder-1.5B (for quick tests)
    pub fn qwen25_1_5b() -> Self {
        Self {
            hidden_size: 1536,
            intermediate_size: 8960,
            num_hidden_layers: 28,
            num_attention_heads: 12,
            num_key_value_heads: 2,
            vocab_size: 151936,
            rms_norm_eps: 1e-6,
            rope_theta: 1_000_000.0,
            max_position_embeddings: 131072,
        }
    }

    pub fn head_dim(&self) -> usize {
        self.hidden_size / self.num_attention_heads
    }
}

// ─── Rotary Position Embedding ───────────────────────────────────────────────

struct RotaryEmbedding {
    cos: Tensor,
    sin: Tensor,
}

impl RotaryEmbedding {
    fn new(cfg: &Qwen2Config, dtype: DType, device: &Device) -> Result<Self> {
        let dim = cfg.head_dim();
        let max_seq = cfg.max_position_embeddings.min(8192); // cap for memory

        let inv_freq: Vec<f32> = (0..dim)
            .step_by(2)
            .map(|i| 1.0_f32 / (cfg.rope_theta as f32).powf(i as f32 / dim as f32))
            .collect();

        let inv_freq = Tensor::new(inv_freq.as_slice(), device)?;
        let positions = Tensor::arange(0u32, max_seq as u32, device)?.to_dtype(DType::F32)?;

        // freqs: [max_seq, dim/2]
        let freqs = positions.unsqueeze(1)?.broadcast_mul(&inv_freq.unsqueeze(0)?)?;
        // emb: [max_seq, dim] — cat(freqs, freqs)
        let emb = Tensor::cat(&[&freqs, &freqs], D::Minus1)?;

        Ok(Self {
            cos: emb.cos()?.to_dtype(dtype)?,
            sin: emb.sin()?.to_dtype(dtype)?,
        })
    }

    fn apply(&self, q: &Tensor, k: &Tensor, offset: usize) -> Result<(Tensor, Tensor)> {
        let seq_len = q.dim(2)?;
        let cos = self.cos.narrow(0, offset, seq_len)?;
        let sin = self.sin.narrow(0, offset, seq_len)?;

        let q_rot = apply_rope(q, &cos, &sin)?;
        let k_rot = apply_rope(k, &cos, &sin)?;
        Ok((q_rot, k_rot))
    }
}

fn rotate_half(xs: &Tensor) -> Result<Tensor> {
    let last = xs.dim(D::Minus1)?;
    let half = last / 2;
    let x1 = xs.narrow(D::Minus1, 0, half)?;
    let x2 = xs.narrow(D::Minus1, half, half)?;
    Tensor::cat(&[&x2.neg()?, &x1], D::Minus1)
}

fn apply_rope(xs: &Tensor, cos: &Tensor, sin: &Tensor) -> Result<Tensor> {
    // xs: [batch, heads, seq, head_dim]
    // cos/sin: [seq, head_dim] → broadcast
    let (b, h, s, d) = xs.dims4()?;
    let cos = cos.unsqueeze(0)?.unsqueeze(0)?.broadcast_as((b, h, s, d))?;
    let sin = sin.unsqueeze(0)?.unsqueeze(0)?.broadcast_as((b, h, s, d))?;
    xs.broadcast_mul(&cos)? + rotate_half(xs)?.broadcast_mul(&sin)?
}

// ─── Causal Mask ─────────────────────────────────────────────────────────────

fn causal_mask(seq_len: usize, dtype: DType, device: &Device) -> Result<Tensor> {
    let mut data = vec![0f32; seq_len * seq_len];
    for i in 0..seq_len {
        for j in (i + 1)..seq_len {
            data[i * seq_len + j] = f32::NEG_INFINITY;
        }
    }
    Tensor::from_vec(data, (seq_len, seq_len), device)?.to_dtype(dtype)
}

/// Manual softmax — candle's Metal backend lacks the softmax-last-dim kernel.
/// softmax(x) = exp(x - max(x)) / sum(exp(x - max(x)))
fn manual_softmax(xs: &Tensor) -> Result<Tensor> {
    let max = xs.max_keepdim(D::Minus1)?;
    let shifted = xs.broadcast_sub(&max)?;
    let exp = shifted.exp()?;
    let sum = exp.sum_keepdim(D::Minus1)?;
    exp.broadcast_div(&sum)
}

// ─── Attention ───────────────────────────────────────────────────────────────

struct Attention {
    q_proj: LoraLinear,
    k_proj: LoraLinear,
    v_proj: LoraLinear,
    o_proj: LoraLinear,
    rope: Arc<RotaryEmbedding>,
    n_heads: usize,
    n_kv_heads: usize,
    head_dim: usize,
    scale: f64,
}

impl Attention {
    fn load(
        weights: &HashMap<String, Tensor>,
        layer_idx: usize,
        cfg: &Qwen2Config,
        lora_config: &LoraConfig,
        lora_vb: VarBuilder,
        rope: Arc<RotaryEmbedding>,
    ) -> Result<Self> {
        let p = |name: &str| format!("model.layers.{layer_idx}.self_attn.{name}");
        let get = |name: &str| -> Result<Tensor> {
            weights.get(name).cloned().ok_or_else(|| {
                candle_core::Error::Msg(format!("missing weight: {name}"))
            })
        };

        let lora_vb = lora_vb.pp(format!("layers.{layer_idx}.self_attn"));

        let q_proj = LoraLinear::new(
            get(&p("q_proj.weight"))?,
            weights.get(&p("q_proj.bias")).cloned(),
            lora_config,
            lora_vb.pp("q_proj"),
        )?;
        let k_proj = LoraLinear::new(
            get(&p("k_proj.weight"))?,
            weights.get(&p("k_proj.bias")).cloned(),
            lora_config,
            lora_vb.pp("k_proj"),
        )?;
        let v_proj = LoraLinear::new(
            get(&p("v_proj.weight"))?,
            weights.get(&p("v_proj.bias")).cloned(),
            lora_config,
            lora_vb.pp("v_proj"),
        )?;
        let o_proj = LoraLinear::new(
            get(&p("o_proj.weight"))?,
            None,
            lora_config,
            lora_vb.pp("o_proj"),
        )?;

        Ok(Self {
            q_proj, k_proj, v_proj, o_proj,
            rope,
            n_heads: cfg.num_attention_heads,
            n_kv_heads: cfg.num_key_value_heads,
            head_dim: cfg.head_dim(),
            scale: 1.0 / (cfg.head_dim() as f64).sqrt(),
        })
    }

    fn forward(&self, xs: &Tensor, mask: &Tensor, pos_offset: usize) -> Result<Tensor> {
        let (b, s, _) = xs.dims3()?;

        let q = self.q_proj.forward(xs)?
            .reshape((b, s, self.n_heads, self.head_dim))?
            .transpose(1, 2)?;
        let k = self.k_proj.forward(xs)?
            .reshape((b, s, self.n_kv_heads, self.head_dim))?
            .transpose(1, 2)?;
        let v = self.v_proj.forward(xs)?
            .reshape((b, s, self.n_kv_heads, self.head_dim))?
            .transpose(1, 2)?;

        let (q, k) = self.rope.apply(&q, &k, pos_offset)?;

        // Repeat k/v heads to match q heads (GQA → MHA)
        let reps = self.n_heads / self.n_kv_heads;
        let k = k.repeat((1, reps, 1, 1))?;
        let v = v.repeat((1, reps, 1, 1))?;

        // Scaled dot-product attention
        let attn = (q.matmul(&k.transpose(2, 3)?)? * self.scale)?;
        // Add causal mask: [1, 1, s, s]
        let mask = mask.unsqueeze(0)?.unsqueeze(0)?;
        let attn = attn.broadcast_add(&mask)?;
        let attn = manual_softmax(&attn)?;

        let out = attn.matmul(&v)?           // (b, heads, s, head_dim)
            .transpose(1, 2)?               // (b, s, heads, head_dim)
            .reshape((b, s, self.n_heads * self.head_dim))?;

        self.o_proj.forward(&out)
    }
}

// ─── MLP (SwiGLU) ────────────────────────────────────────────────────────────

struct Mlp {
    gate_proj: candle_nn::Linear,
    up_proj: candle_nn::Linear,
    down_proj: candle_nn::Linear,
}

impl Mlp {
    fn load(weights: &HashMap<String, Tensor>, layer_idx: usize) -> Result<Self> {
        let p = |name: &str| format!("model.layers.{layer_idx}.mlp.{name}");
        let get = |name: &str| -> Result<Tensor> {
            weights.get(name).cloned().ok_or_else(|| {
                candle_core::Error::Msg(format!("missing weight: {name}"))
            })
        };
        Ok(Self {
            gate_proj: frozen_linear(get(&p("gate_proj.weight"))?, None),
            up_proj: frozen_linear(get(&p("up_proj.weight"))?, None),
            down_proj: frozen_linear(get(&p("down_proj.weight"))?, None),
        })
    }

    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        let gate = candle_nn::ops::silu(&self.gate_proj.forward(xs)?)?;
        let up = self.up_proj.forward(xs)?;
        self.down_proj.forward(&(gate * up)?)
    }
}

// ─── Decoder Layer ───────────────────────────────────────────────────────────

struct DecoderLayer {
    input_norm: RmsNorm,
    attn: Attention,
    post_attn_norm: RmsNorm,
    mlp: Mlp,
}

impl DecoderLayer {
    fn load(
        weights: &HashMap<String, Tensor>,
        layer_idx: usize,
        cfg: &Qwen2Config,
        lora_config: &LoraConfig,
        lora_vb: VarBuilder,
        rope: Arc<RotaryEmbedding>,
    ) -> Result<Self> {
        let get = |name: &str| -> Result<Tensor> {
            weights.get(name).cloned().ok_or_else(|| {
                candle_core::Error::Msg(format!("missing weight: {name}"))
            })
        };

        let input_norm = RmsNorm::new(
            get(&format!("model.layers.{layer_idx}.input_layernorm.weight"))?,
            cfg.rms_norm_eps,
        );
        let post_attn_norm = RmsNorm::new(
            get(&format!("model.layers.{layer_idx}.post_attention_layernorm.weight"))?,
            cfg.rms_norm_eps,
        );

        Ok(Self {
            input_norm,
            attn: Attention::load(weights, layer_idx, cfg, lora_config, lora_vb, rope)?,
            post_attn_norm,
            mlp: Mlp::load(weights, layer_idx)?,
        })
    }

    fn forward(&self, xs: &Tensor, mask: &Tensor) -> Result<Tensor> {
        let residual = xs;
        let xs = self.input_norm.forward(xs)?;
        let xs = self.attn.forward(&xs, mask, 0)?;
        let xs = (xs + residual)?;

        let residual = &xs;
        let xs = self.post_attn_norm.forward(&xs)?;
        let xs = self.mlp.forward(&xs)?;
        xs + residual
    }
}

// ─── Full Model ───────────────────────────────────────────────────────────────

pub struct Qwen2LoraModel {
    embed_tokens: Embedding,
    layers: Vec<DecoderLayer>,
    norm: RmsNorm,
    lm_head: candle_nn::Linear,
    #[allow(dead_code)]
    cfg: Qwen2Config,
    dtype: DType,
}

impl Qwen2LoraModel {
    /// Load the model from a directory containing:
    ///   - model.safetensors (or model-*.safetensors shards)
    ///   - model.safetensors.index.json (if sharded)
    /// LoRA adapters are created fresh in lora_vb (trainable).
    pub fn load(
        model_dir: &std::path::Path,
        cfg: Qwen2Config,
        lora_config: &LoraConfig,
        lora_vb: VarBuilder,
        device: &Device,
        dtype: DType,
    ) -> Result<Self> {
        tracing::info!("Loading base model weights from {:?}", model_dir);
        let weights = load_safetensors(model_dir, device, dtype)?;
        tracing::info!(params = weights.len(), "Weights loaded");

        let rope = Arc::new(RotaryEmbedding::new(&cfg, dtype, device)?);

        let embed_tokens = Embedding::new(
            weights.get("model.embed_tokens.weight")
                .ok_or_else(|| candle_core::Error::Msg("missing embed_tokens".into()))?
                .clone(),
            cfg.hidden_size,
        );

        let layers: Result<Vec<_>> = (0..cfg.num_hidden_layers)
            .map(|i| {
                tracing::debug!("Loading layer {i}/{}", cfg.num_hidden_layers);
                DecoderLayer::load(&weights, i, &cfg, lora_config, lora_vb.clone(), Arc::clone(&rope))
            })
            .collect();
        let layers = layers?;

        let norm = RmsNorm::new(
            weights.get("model.norm.weight")
                .ok_or_else(|| candle_core::Error::Msg("missing model.norm.weight".into()))?
                .clone(),
            cfg.rms_norm_eps,
        );

        let lm_head = frozen_linear(
            weights.get("lm_head.weight")
                .ok_or_else(|| candle_core::Error::Msg("missing lm_head.weight".into()))?
                .clone(),
            None,
        );

        Ok(Self { embed_tokens, layers, norm, lm_head, cfg, dtype })
    }

    /// Forward pass: input_ids → logits
    /// input_ids: [batch_size, seq_len]
    /// Returns: [batch_size, seq_len, vocab_size]
    pub fn forward(&self, input_ids: &Tensor) -> Result<Tensor> {
        let (_, seq_len) = input_ids.dims2()?;
        let device = input_ids.device();
        let dtype = self.dtype;

        let mut xs = self.embed_tokens.forward(input_ids)?;

        let mask = causal_mask(seq_len, dtype, device)?;

        for layer in &self.layers {
            xs = layer.forward(&xs, &mask)?;
        }

        xs = self.norm.forward(&xs)?;
        self.lm_head.forward(&xs)
    }
}

// ─── Safetensors loader ───────────────────────────────────────────────────────

pub fn load_safetensors(
    dir: &std::path::Path,
    device: &Device,
    dtype: DType,
) -> Result<HashMap<String, Tensor>> {
    use std::collections::HashMap as HM;

    let index_path = dir.join("model.safetensors.index.json");
    let mut tensors: HM<String, Tensor> = HM::new();

    if index_path.exists() {
        // Sharded model — load from index
        let index_json = std::fs::read_to_string(&index_path)
            .map_err(|e| candle_core::Error::Msg(format!("read index: {e}")))?;
        let index: serde_json::Value = serde_json::from_str(&index_json)
            .map_err(|e| candle_core::Error::Msg(format!("parse index: {e}")))?;

        let weight_map = index["weight_map"].as_object()
            .ok_or_else(|| candle_core::Error::Msg("invalid index".into()))?;

        let mut seen_shards = std::collections::HashSet::new();
        for (_, filename) in weight_map {
            let filename = filename.as_str().unwrap_or_default();
            if !seen_shards.insert(filename.to_string()) { continue; }

            let path = dir.join(filename);
            tracing::debug!("Loading shard {:?}", path);
            let shard = candle_core::safetensors::load(&path, device)?;
            for (k, v) in shard {
                tensors.insert(k, v.to_dtype(dtype)?);
            }
        }
    } else {
        // Single file
        let path = dir.join("model.safetensors");
        let shard = candle_core::safetensors::load(&path, device)?;
        for (k, v) in shard {
            tensors.insert(k, v.to_dtype(dtype)?);
        }
    }

    Ok(tensors)
}

/// Layer-sharded Qwen2.5 inference in Candle.
///
/// fp32-through-softmax on every attention op — required for MPS/Metal safety
/// (fp16 Q@K^T overflows at ~65504 with the Qwen2.5-7B residual magnitudes we
/// measured on M4 Max, layer 27 k_proj.bias max=442, residual ~2000-3500).
///
/// DecoderLayer/Attention/Mlp/RmsNorm are all private in candle-transformers
/// 0.8.x, so we implement them from scratch here.
use anyhow::{bail, Result};
use candle_core::{DType, Device, IndexOp, Tensor, D};
use candle_nn::{linear, linear_no_bias, embedding, Module, VarBuilder};
use serde::Deserialize;
use std::collections::BTreeMap;

// ── Config ────────────────────────────────────────────────────────────────────

#[derive(Debug, Clone, Deserialize)]
pub struct Qwen2Config {
    pub hidden_size: usize,
    pub intermediate_size: usize,
    pub num_hidden_layers: usize,
    pub num_attention_heads: usize,
    pub num_key_value_heads: usize,
    pub max_position_embeddings: usize,
    #[serde(default = "default_rope_theta")]
    pub rope_theta: f64,
    pub rms_norm_eps: f64,
    pub vocab_size: usize,
    pub sliding_window: Option<usize>,
}

fn default_rope_theta() -> f64 { 1_000_000.0 }

impl Qwen2Config {
    pub fn head_dim(&self) -> usize { self.hidden_size / self.num_attention_heads }
}

// ── RMSNorm ───────────────────────────────────────────────────────────────────

struct RmsNorm {
    weight: Tensor,
    eps: f64,
}

impl RmsNorm {
    fn load(vb: VarBuilder, size: usize, eps: f64) -> Result<Self> {
        let weight = vb.get(size, "weight")?;
        Ok(Self { weight, eps })
    }

    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let orig_dtype = x.dtype();
        let x_f32 = x.to_dtype(DType::F32)?;
        let hidden = x_f32.dim(D::Minus1)?;
        // variance: sum(x^2) / n
        let variance = (x_f32.sqr()?.sum_keepdim(D::Minus1)? / hidden as f64)?;
        let normed = x_f32.broadcast_div(&(variance + self.eps)?.sqrt()?)?;
        let out = normed.to_dtype(orig_dtype)?.broadcast_mul(&self.weight)?;
        Ok(out)
    }
}

// ── RotaryEmbedding ───────────────────────────────────────────────────────────

struct RotaryEmbedding {
    sin: Tensor,
    cos: Tensor,
}

impl RotaryEmbedding {
    fn new(head_dim: usize, max_seq: usize, theta: f64, device: &Device) -> Result<Self> {
        let half = head_dim / 2;
        let inv_freq: Vec<f32> = (0..half)
            .map(|i| 1.0 / (theta as f32).powf(i as f32 * 2.0 / head_dim as f32))
            .collect();
        let inv_freq = Tensor::from_vec(inv_freq, half, device)?.to_dtype(DType::F32)?;
        let positions = Tensor::arange(0u32, max_seq as u32, device)?
            .to_dtype(DType::F32)?
            .reshape((max_seq, 1))?;
        let freqs = positions.broadcast_mul(&inv_freq.reshape((1, half))?)?;
        let emb = Tensor::cat(&[&freqs, &freqs], 1)?;
        Ok(Self { sin: emb.sin()?, cos: emb.cos()? })
    }

    /// Apply RoPE to [batch, heads, seq, head_dim] tensors.
    fn apply(&self, x: &Tensor, offset: usize) -> Result<Tensor> {
        let (b, h, s, d) = x.dims4()?;
        let dtype = x.dtype();
        // cos/sin are stored as F32; cast to input dtype to avoid Metal dtype mismatch
        let cos = self.cos.i(offset..offset + s)?.reshape((1, 1, s, d))?.to_dtype(dtype)?;
        let sin = self.sin.i(offset..offset + s)?.reshape((1, 1, s, d))?.to_dtype(dtype)?;
        let cos = cos.broadcast_as((b, h, s, d))?;
        let sin = sin.broadcast_as((b, h, s, d))?;
        let half = d / 2;
        let x1 = x.i((.., .., .., ..half))?;
        let x2 = x.i((.., .., .., half..))?;
        let x_rot = Tensor::cat(&[&x2.neg()?, &x1], D::Minus1)?;
        Ok((x.broadcast_mul(&cos)? + x_rot.broadcast_mul(&sin)?)?)
    }
}

// ── Attention ─────────────────────────────────────────────────────────────────

pub type KvEntry = (Tensor, Tensor);

struct Attention {
    q_proj: candle_nn::Linear,
    k_proj: candle_nn::Linear,
    v_proj: candle_nn::Linear,
    o_proj: candle_nn::Linear,
    num_q_heads: usize,
    num_kv_heads: usize,
    head_dim: usize,
    scale: f64,
}

impl Attention {
    fn load(vb: VarBuilder, cfg: &Qwen2Config) -> Result<Self> {
        let h = cfg.hidden_size;
        let nq = cfg.num_attention_heads;
        let nkv = cfg.num_key_value_heads;
        let hd = cfg.head_dim();
        Ok(Self {
            q_proj: linear(h, nq * hd, vb.pp("q_proj"))?,
            k_proj: linear(h, nkv * hd, vb.pp("k_proj"))?,
            v_proj: linear(h, nkv * hd, vb.pp("v_proj"))?,
            o_proj: linear_no_bias(nq * hd, h, vb.pp("o_proj"))?,
            num_q_heads: nq,
            num_kv_heads: nkv,
            head_dim: hd,
            scale: 1.0 / (hd as f64).sqrt(),
        })
    }

    fn forward(
        &self,
        x: &Tensor,
        rope: &RotaryEmbedding,
        kv_cache: Option<&KvEntry>,
        position_offset: usize,
    ) -> Result<(Tensor, KvEntry)> {
        let (b, s, _) = x.dims3()?;
        let q = self.q_proj.forward(x)?
            .reshape((b, s, self.num_q_heads, self.head_dim))?.transpose(1, 2)?;
        let k = self.k_proj.forward(x)?
            .reshape((b, s, self.num_kv_heads, self.head_dim))?.transpose(1, 2)?;
        let v = self.v_proj.forward(x)?
            .reshape((b, s, self.num_kv_heads, self.head_dim))?.transpose(1, 2)?;

        let q = rope.apply(&q, position_offset)?;
        let k = rope.apply(&k, position_offset)?;

        // Append to KV cache
        let (k_full, v_full) = if let Some((k_prev, v_prev)) = kv_cache {
            (Tensor::cat(&[k_prev, &k], 2)?, Tensor::cat(&[v_prev, &v], 2)?)
        } else {
            (k, v)
        };
        let new_cache: KvEntry = (k_full.clone(), v_full.clone());

        // GQA: repeat KV heads
        let repeats = self.num_q_heads / self.num_kv_heads;
        let k_rep = k_full.repeat(&[1, repeats, 1, 1])?;
        let v_rep = v_full.repeat(&[1, repeats, 1, 1])?;

        // fp32 attention — MPS safety
        let q_f32 = q.to_dtype(DType::F32)?;
        let k_f32 = k_rep.to_dtype(DType::F32)?;
        let v_f32 = v_rep.to_dtype(DType::F32)?;
        let attn = (q_f32.matmul(&k_f32.transpose(2, 3)?)? * self.scale)?;
        let attn = candle_nn::ops::softmax(&attn, D::Minus1)?;
        let out = attn.matmul(&v_f32)?.to_dtype(x.dtype())?;

        let out = out.transpose(1, 2)?.reshape((b, s, self.num_q_heads * self.head_dim))?;
        let out = self.o_proj.forward(&out)?;
        Ok((out, new_cache))
    }
}

// ── MLP ───────────────────────────────────────────────────────────────────────

struct Mlp {
    gate_proj: candle_nn::Linear,
    up_proj: candle_nn::Linear,
    down_proj: candle_nn::Linear,
}

impl Mlp {
    fn load(vb: VarBuilder, cfg: &Qwen2Config) -> Result<Self> {
        Ok(Self {
            gate_proj: linear_no_bias(cfg.hidden_size, cfg.intermediate_size, vb.pp("gate_proj"))?,
            up_proj: linear_no_bias(cfg.hidden_size, cfg.intermediate_size, vb.pp("up_proj"))?,
            down_proj: linear_no_bias(cfg.intermediate_size, cfg.hidden_size, vb.pp("down_proj"))?,
        })
    }

    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let orig_dtype = x.dtype();
        // silu on F16 may return F32 on Metal — cast back before multiply
        let gate = candle_nn::ops::silu(&self.gate_proj.forward(x)?)?.to_dtype(orig_dtype)?;
        let up = self.up_proj.forward(x)?;
        Ok(self.down_proj.forward(&(gate * up)?)?)
    }
}

// ── DecoderLayer ──────────────────────────────────────────────────────────────

struct DecoderLayer {
    self_attn: Attention,
    mlp: Mlp,
    input_ln: RmsNorm,
    post_attn_ln: RmsNorm,
}

impl DecoderLayer {
    fn load(vb: VarBuilder, cfg: &Qwen2Config) -> Result<Self> {
        Ok(Self {
            self_attn: Attention::load(vb.pp("self_attn"), cfg)?,
            mlp: Mlp::load(vb.pp("mlp"), cfg)?,
            input_ln: RmsNorm::load(vb.pp("input_layernorm"), cfg.hidden_size, cfg.rms_norm_eps)?,
            post_attn_ln: RmsNorm::load(vb.pp("post_attention_layernorm"), cfg.hidden_size, cfg.rms_norm_eps)?,
        })
    }

    fn forward(
        &self,
        x: &Tensor,
        rope: &RotaryEmbedding,
        kv_cache: Option<&KvEntry>,
        position_offset: usize,
    ) -> Result<(Tensor, KvEntry)> {
        let normed = self.input_ln.forward(x)?;
        let (attn_out, new_kv) = self.self_attn.forward(&normed, rope, kv_cache, position_offset)?;
        let x2 = (x + attn_out)?;
        let normed2 = self.post_attn_ln.forward(&x2)?;
        let mlp_out = self.mlp.forward(&normed2)?;
        Ok(((x2 + mlp_out)?, new_kv))
    }
}

// ── Full shard ────────────────────────────────────────────────────────────────

pub struct QwenShard {
    embed_tokens: Option<candle_nn::Embedding>,
    layers: Vec<DecoderLayer>,
    norm: Option<RmsNorm>,
    lm_head: Option<candle_nn::Linear>,
    rope: RotaryEmbedding,
    pub cfg: Qwen2Config,
    pub layer_start: usize,
    pub layer_end: usize,
    pub device: Device,
}

impl QwenShard {
    /// Load model from `model_dir`, materialising only `layer_start..layer_end`.
    pub fn load(
        model_dir: &std::path::Path,
        layer_start: usize,
        layer_end: usize,
        device: Device,
    ) -> Result<Self> {
        let cfg: Qwen2Config =
            serde_json::from_str(&std::fs::read_to_string(model_dir.join("config.json"))?)?;

        let mut st_files: Vec<_> = std::fs::read_dir(model_dir)?
            .filter_map(|e| e.ok())
            .map(|e| e.path())
            .filter(|p| p.extension().and_then(|s| s.to_str()) == Some("safetensors"))
            .collect();
        st_files.sort();
        if st_files.is_empty() {
            bail!("No safetensors files found in {}", model_dir.display());
        }

        let dtype = if matches!(device, Device::Cpu) { DType::F32 } else { DType::F16 };
        let vb = unsafe { VarBuilder::from_mmaped_safetensors(&st_files, dtype, &device)? };
        let vb_model = vb.pp("model");

        let embed_tokens = if layer_start == 0 {
            Some(embedding(cfg.vocab_size, cfg.hidden_size, vb_model.pp("embed_tokens"))?)
        } else {
            None
        };

        let mut layers = Vec::with_capacity(layer_end - layer_start);
        for i in layer_start..layer_end {
            layers.push(DecoderLayer::load(vb_model.pp(format!("layers.{i}")), &cfg)?);
        }

        let is_last = layer_end == cfg.num_hidden_layers;
        let norm = if is_last {
            Some(RmsNorm::load(vb_model.pp("norm"), cfg.hidden_size, cfg.rms_norm_eps)?)
        } else {
            None
        };
        let lm_head = if is_last {
            Some(linear_no_bias(cfg.hidden_size, cfg.vocab_size, vb.pp("lm_head"))?)
        } else {
            None
        };

        let rope = RotaryEmbedding::new(
            cfg.head_dim(),
            cfg.max_position_embeddings,
            cfg.rope_theta,
            &device,
        )?;

        Ok(Self { embed_tokens, layers, norm, lm_head, rope, cfg, layer_start, layer_end, device })
    }

    pub fn layers_len(&self) -> usize { self.layer_end - self.layer_start }

    pub fn forward(
        &self,
        input: ForwardInput,
        kv_caches: &[Option<KvEntry>],
        position_offset: usize,
    ) -> Result<ForwardOutput> {
        if kv_caches.len() != self.layers.len() {
            bail!("kv_caches len {} != layers len {}", kv_caches.len(), self.layers.len());
        }

        let mut hidden = match input {
            ForwardInput::Tokens(ids) => {
                self.embed_tokens
                    .as_ref()
                    .ok_or_else(|| anyhow::anyhow!("embed_tokens not loaded — not first shard"))?
                    .forward(&ids)?
            }
            ForwardInput::Hidden(h) => h,
        };

        let mut new_kvs = Vec::with_capacity(self.layers.len());
        for (layer, kv) in self.layers.iter().zip(kv_caches.iter()) {
            let (h, new_kv) = layer.forward(&hidden, &self.rope, kv.as_ref(), position_offset)?;
            hidden = h;
            new_kvs.push(new_kv);
        }

        let logits = if let (Some(norm), Some(lm_head)) = (&self.norm, &self.lm_head) {
            Some(lm_head.forward(&norm.forward(&hidden)?)?)
        } else {
            None
        };

        Ok(ForwardOutput { hidden, new_kvs, logits })
    }
}

// ── Input/Output ──────────────────────────────────────────────────────────────

pub enum ForwardInput {
    Tokens(Tensor),
    Hidden(Tensor),
}

pub struct ForwardOutput {
    pub hidden: Tensor,
    pub new_kvs: Vec<KvEntry>,
    pub logits: Option<Tensor>,
}

// ── KV cache store ────────────────────────────────────────────────────────────

pub struct KvStore {
    entries: BTreeMap<String, (Vec<KvEntry>, std::time::Instant)>,
    max_entries: usize,
    idle_timeout: std::time::Duration,
}

impl KvStore {
    pub fn new(max_entries: usize, idle_timeout_s: u64) -> Self {
        Self {
            entries: BTreeMap::new(),
            max_entries,
            idle_timeout: std::time::Duration::from_secs(idle_timeout_s),
        }
    }

    pub fn get(&self, request_id: &str) -> Option<&Vec<KvEntry>> {
        self.entries.get(request_id).map(|(kv, _)| kv)
    }

    pub fn insert(&mut self, request_id: String, kvs: Vec<KvEntry>) {
        if self.entries.len() >= self.max_entries && !self.entries.contains_key(&request_id) {
            let oldest = self
                .entries
                .iter()
                .min_by_key(|(_, (_, t))| *t)
                .map(|(k, _)| k.clone());
            if let Some(k) = oldest { self.entries.remove(&k); }
        }
        self.entries.insert(request_id, (kvs, std::time::Instant::now()));
    }

    pub fn touch(&mut self, request_id: &str) {
        if let Some((_, t)) = self.entries.get_mut(request_id) {
            *t = std::time::Instant::now();
        }
    }

    pub fn evict_idle(&mut self) -> usize {
        let now = std::time::Instant::now();
        let timeout = self.idle_timeout;
        let before = self.entries.len();
        self.entries.retain(|_, (_, t)| now.duration_since(*t) < timeout);
        before - self.entries.len()
    }
}

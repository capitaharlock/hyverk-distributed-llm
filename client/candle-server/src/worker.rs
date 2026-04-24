use anyhow::{bail, Result};
use axum::http::{HeaderName, HeaderValue};
use candle_core::{DType, Device, Tensor};
use serde_json::json;
use std::str::FromStr;
use tracing::info;

use crate::{
    model::{ForwardInput, KvEntry, KvStore, QwenShard},
    sampling::{sample_next_token, SamplingParams},
};

pub struct InferenceWorker {
    pub shard: QwenShard,
    pub kv_store: KvStore,
    pub device_name: String,
}

impl InferenceWorker {
    pub fn new(shard: QwenShard, kv_store: KvStore, device_name: String) -> Self {
        Self { shard, kv_store, device_name }
    }

    pub fn health_json(&self) -> serde_json::Value {
        json!({
            "status": "ready",
            "device": self.device_name,
            "layers": self.shard.layers_len(),
            "layer_start": self.shard.layer_start,
            "layer_end": self.shard.layer_end,
        })
    }

    /// Dispatch inference by mode. Returns (response_headers, response_body_bytes).
    ///
    /// - "forward"  → hidden-state hop (pipeline parallelism)
    /// - "generate" → token IDs in, next_token out (single-node full pass)
    /// - "decode"   → single token in, next_token out (decode step)
    pub fn run_inference(
        &mut self,
        mode: &str,
        request_id: &str,
        shape: &[usize],
        body: &[u8],
        sampling: &SamplingParams,
    ) -> Result<(Vec<(HeaderName, HeaderValue)>, Vec<u8>)> {
        let t0 = std::time::Instant::now();

        let result = match mode {
            "forward"  => self.handle_forward(request_id, shape, body)?,
            "generate" => self.handle_generate(request_id, shape, body, sampling)?,
            "decode"   => self.handle_decode(request_id, shape, body, sampling)?,
            other => bail!("unknown mode: {other}"),
        };

        let elapsed_ms = t0.elapsed().as_secs_f64() * 1000.0;
        info!(mode, request_id, elapsed_ms, "inference ok");

        let mut headers = vec![h("X-Elapsed-Ms", &format!("{elapsed_ms:.1}"))];
        headers.extend(result.extra_headers);
        Ok((headers, result.body))
    }

    // ── forward ───────────────────────────────────────────────────────────────

    fn handle_forward(&mut self, request_id: &str, shape: &[usize], body: &[u8]) -> Result<InferResult> {
        if shape.len() < 3 { bail!("forward: need shape [b, seq, hidden], got {shape:?}"); }
        let (b, seq, hs) = (shape[0], shape[1], shape[2]);
        let hidden = bytes_to_f16_tensor(body, &[b, seq, hs], &self.shard.device)?;

        let (kv_vec, offset) = self.take_kv(request_id);
        let out = self.shard.forward(ForwardInput::Hidden(hidden), &kv_vec, offset)?;
        self.kv_store.insert(request_id.to_string(), out.new_kvs);

        let resp_bytes = tensor_to_f16_bytes(&out.hidden)?;
        let (rb, rseq, rhs) = (b, seq, self.shard.cfg.hidden_size);
        Ok(InferResult {
            body: resp_bytes,
            extra_headers: vec![h("X-Shape", &format!("[{rb},{rseq},{rhs}]"))],
        })
    }

    // ── generate ──────────────────────────────────────────────────────────────

    fn handle_generate(
        &mut self,
        request_id: &str,
        shape: &[usize],
        body: &[u8],
        sampling: &SamplingParams,
    ) -> Result<InferResult> {
        let (b, seq) = match shape.len() {
            l if l >= 2 => (shape[0], shape[1]),
            _ => (1, shape[0]),
        };
        let tokens = bytes_to_u32_tensor(body, b, seq, &self.shard.device)?;
        let (kv_vec, offset) = self.take_kv(request_id);
        let out = self.shard.forward(ForwardInput::Tokens(tokens), &kv_vec, offset)?;
        self.kv_store.insert(request_id.to_string(), out.new_kvs);

        let logits = out.logits.ok_or_else(|| anyhow::anyhow!("generate requires last shard"))?;
        let next_token = sample_next_token(&logits, sampling)?;
        Ok(InferResult {
            body: next_token.to_le_bytes().to_vec(),
            extra_headers: vec![h("X-Next-Token", &next_token.to_string())],
        })
    }

    // ── decode ────────────────────────────────────────────────────────────────

    fn handle_decode(
        &mut self,
        request_id: &str,
        _shape: &[usize],
        body: &[u8],
        sampling: &SamplingParams,
    ) -> Result<InferResult> {
        if body.len() < 4 { bail!("decode: need ≥4 bytes for token id"); }
        let token_id = u32::from_le_bytes(body[..4].try_into().unwrap());
        let token_tensor = Tensor::from_vec(vec![token_id], (1usize, 1usize), &self.shard.device)?;
        let (kv_vec, offset) = self.take_kv(request_id);
        let out = self.shard.forward(ForwardInput::Tokens(token_tensor), &kv_vec, offset)?;
        self.kv_store.insert(request_id.to_string(), out.new_kvs);

        let logits = out.logits.ok_or_else(|| anyhow::anyhow!("decode requires last shard"))?;
        let next_token = sample_next_token(&logits, sampling)?;
        Ok(InferResult {
            body: next_token.to_le_bytes().to_vec(),
            extra_headers: vec![h("X-Next-Token", &next_token.to_string())],
        })
    }

    // ── KV helpers ─────────────────────────────────────────────────────────────

    fn take_kv(&mut self, request_id: &str) -> (Vec<Option<KvEntry>>, usize) {
        let n = self.shard.layers_len();
        // Clone KVs while holding immutable borrow, then drop before touch()
        let cache: Option<Vec<KvEntry>> = self.kv_store.get(request_id).map(|v| v.clone());
        if let Some(kvs) = cache {
            let offset = kvs[0].0.dim(2).unwrap_or(0);
            self.kv_store.touch(request_id);
            (kvs.into_iter().map(Some).collect(), offset)
        } else {
            (vec![None; n], 0)
        }
    }
}

struct InferResult {
    body: Vec<u8>,
    extra_headers: Vec<(HeaderName, HeaderValue)>,
}

// ── tensor serialisation ──────────────────────────────────────────────────────

/// Interpret `data` as packed fp16 LE and load onto `device`.
fn bytes_to_f16_tensor(data: &[u8], shape: &[usize], device: &Device) -> Result<Tensor> {
    let n_elems: usize = shape.iter().product();
    if data.len() != n_elems * 2 {
        bail!("expected {} bytes for shape {shape:?} fp16, got {}", n_elems * 2, data.len());
    }
    let f32s: Vec<f32> = data
        .chunks_exact(2)
        .map(|b| half::f16::from_le_bytes([b[0], b[1]]).to_f32())
        .collect();
    let t = Tensor::from_vec(f32s, shape.to_vec(), &Device::Cpu)?.to_device(device)?;
    Ok(if matches!(device, Device::Cpu) {
        t.to_dtype(DType::F32)?
    } else {
        t.to_dtype(DType::F16)?
    })
}

fn bytes_to_u32_tensor(data: &[u8], b: usize, seq: usize, device: &Device) -> Result<Tensor> {
    let n = b * seq;
    if data.len() < n * 4 { bail!("not enough bytes for u32 tokens"); }
    let ids: Vec<u32> = data.chunks_exact(4).take(n)
        .map(|c| u32::from_le_bytes(c.try_into().unwrap()))
        .collect();
    Ok(Tensor::from_vec(ids, (b, seq), device)?)
}

/// Serialise tensor as fp16 LE bytes (matching Python's np.float16 wire format).
fn tensor_to_f16_bytes(t: &Tensor) -> Result<Vec<u8>> {
    let f32s: Vec<f32> = t.to_dtype(DType::F32)?.flatten_all()?.to_vec1()?;
    let mut out = Vec::with_capacity(f32s.len() * 2);
    for f in f32s {
        out.extend_from_slice(&half::f16::from_f32(f).to_le_bytes());
    }
    Ok(out)
}

fn h(name: &str, value: &str) -> (HeaderName, HeaderValue) {
    (HeaderName::from_str(name).unwrap(), HeaderValue::from_str(value).unwrap())
}

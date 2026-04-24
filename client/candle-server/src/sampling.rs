use anyhow::Result;
use candle_core::{DType, IndexOp, Tensor};
use rand::distributions::Distribution;

#[derive(Debug, Clone, serde::Deserialize, Default)]
pub struct SamplingParams {
    pub temperature: Option<f64>,
    pub top_p: Option<f64>,
    pub top_k: Option<usize>,
}

/// Sample from logits tensor of shape [1, vocab] or [1, seq, vocab].
/// Always reads from the last position.
pub fn sample_next_token(logits: &Tensor, params: &SamplingParams) -> Result<u32> {
    let logits = logits.to_dtype(DType::F32)?;
    let logits = match logits.dims().len() {
        3 => {
            let seq = logits.dim(1)?;
            logits.i((0, seq - 1, ..))?
        }
        _ => logits.i((0, ..))?,
    };

    let temperature = params.temperature.unwrap_or(1.0);
    if temperature < 1e-6 {
        return argmax(&logits);
    }

    let logits = (&logits / temperature)?;
    let mut probs: Vec<f32> = {
        let v: Vec<f32> = logits.to_vec1()?;
        let max = v.iter().copied().fold(f32::NEG_INFINITY, f32::max);
        let mut p: Vec<f32> = v.iter().map(|&l| (l - max).exp()).collect();
        let sum: f32 = p.iter().sum();
        p.iter_mut().for_each(|x| *x /= sum);
        p
    };

    let vocab = probs.len();

    // top-k
    let top_k = params.top_k.unwrap_or(vocab).min(vocab);
    if top_k < vocab {
        let mut indexed: Vec<(usize, f32)> = probs.iter().copied().enumerate().collect();
        indexed.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
        for (idx, _) in &indexed[top_k..] { probs[*idx] = 0.0; }
        let sum: f32 = probs.iter().sum();
        if sum > 0.0 { probs.iter_mut().for_each(|p| *p /= sum); }
    }

    // top-p (nucleus)
    if let Some(top_p) = params.top_p {
        let mut indexed: Vec<(usize, f32)> = probs.iter().copied().enumerate().collect();
        indexed.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
        let mut cum = 0.0f32;
        let mut cutoff = indexed.len();
        for (i, (_, p)) in indexed.iter().enumerate() {
            cum += p;
            if cum >= top_p as f32 { cutoff = i + 1; break; }
        }
        for (idx, _) in &indexed[cutoff..] { probs[*idx] = 0.0; }
        let sum: f32 = probs.iter().sum();
        if sum > 0.0 { probs.iter_mut().for_each(|p| *p /= sum); }
    }

    let dist = rand::distributions::WeightedIndex::new(&probs)?;
    Ok(dist.sample(&mut rand::thread_rng()) as u32)
}

fn argmax(logits: &Tensor) -> Result<u32> {
    let v: Vec<f32> = logits.to_vec1()?;
    Ok(v.iter().enumerate().max_by(|a, b| a.1.partial_cmp(b.1).unwrap()).map(|(i, _)| i).unwrap_or(0) as u32)
}

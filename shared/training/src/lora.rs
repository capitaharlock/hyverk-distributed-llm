// LoRA (Low-Rank Adaptation) layer implementation.
// For each target linear layer W ∈ ℝ^(out×in):
//   W_new = W + (B @ A) * scale
//   where A ∈ ℝ^(rank×in), B ∈ ℝ^(out×rank), scale = alpha/rank
// Only A and B are trainable (W is frozen).
//
// At initialization: B = 0 → delta = 0 → model is identical to base.
// During training: gradients flow only through A and B.

use candle_core::{Result, Tensor};
use candle_nn::{Linear, Module, VarBuilder};

#[derive(Debug, Clone)]
pub struct LoraConfig {
    pub rank: usize,
    pub alpha: f64,
}

impl LoraConfig {
    pub fn scale(&self) -> f64 {
        self.alpha / self.rank as f64
    }
}

/// A linear layer with a LoRA adapter attached.
/// base_weight is frozen; lora_a and lora_b are trainable Variables.
pub struct LoraLinear {
    base: Linear,
    lora_a: Tensor,  // (rank, in_features) — trainable
    lora_b: Tensor,  // (out_features, rank) — trainable, init=0
    scale: f64,
}

impl LoraLinear {
    /// Create a LoraLinear from frozen base weights + new trainable LoRA vars.
    ///
    /// vb: VarBuilder backed by a VarMap (trainable vars)
    /// base_weight: frozen weight tensor from pre-loaded model
    /// base_bias: optional frozen bias
    pub fn new(
        base_weight: Tensor,
        base_bias: Option<Tensor>,
        config: &LoraConfig,
        vb: VarBuilder,
    ) -> Result<Self> {
        let (out_features, in_features) = base_weight.dims2()?;
        let rank = config.rank;
        let scale = config.scale();

        // A: Kaiming uniform init (standard LoRA init)
        let lora_a = vb.get_with_hints(
            (rank, in_features),
            "lora_a",
            candle_nn::init::DEFAULT_KAIMING_UNIFORM,
        )?;

        // B: Zero init — ensures delta = B@A = 0 at start
        let lora_b = vb.get_with_hints(
            (out_features, rank),
            "lora_b",
            candle_nn::init::Init::Const(0.0),
        )?;

        let base = Linear::new(base_weight, base_bias);
        Ok(Self { base, lora_a, lora_b, scale })
    }
}

impl Module for LoraLinear {
    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        // Base output (no gradient through frozen weights — may be F16)
        let base_out = self.base.forward(xs)?;

        // LoRA delta: xs @ A^T @ B^T * scale
        // Cast xs to LoRA dtype (F32) if base is F16
        let lora_dtype = self.lora_a.dtype();
        let xs_lora = if xs.dtype() != lora_dtype {
            xs.to_dtype(lora_dtype)?
        } else {
            xs.clone()
        };

        // Flatten to 2D for matmul (candle Metal requires matching dims)
        let shape = xs_lora.shape().clone();
        let dims = shape.dims();
        let in_features = dims[dims.len() - 1];
        let batch_flat = xs_lora.elem_count() / in_features;

        let xs_2d = xs_lora.reshape((batch_flat, in_features))?;
        let lora_out = xs_2d
            .matmul(&self.lora_a.t()?)?    // (batch_flat, rank)
            .matmul(&self.lora_b.t()?)?;   // (batch_flat, out_features)
        let lora_out = (lora_out * self.scale)?;

        // Reshape back to original batch dims + out_features
        let out_features = lora_out.dim(1)?;
        let mut out_shape: Vec<usize> = dims[..dims.len() - 1].to_vec();
        out_shape.push(out_features);
        let lora_out = lora_out.reshape(out_shape)?;

        // Cast lora_out back to base dtype for addition
        let lora_out = if lora_out.dtype() != base_out.dtype() {
            lora_out.to_dtype(base_out.dtype())?
        } else {
            lora_out
        };

        base_out + lora_out
    }
}

/// Create a frozen Linear from a tensor (no gradient tracking)
pub fn frozen_linear(weight: Tensor, bias: Option<Tensor>) -> Linear {
    Linear::new(weight, bias)
}

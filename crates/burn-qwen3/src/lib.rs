#![forbid(unsafe_code)]

mod model;
use anyhow::{Result, ensure};
pub use model::{Qwen3Model, Qwen3ModelConfig};
use serde::Deserialize;

#[derive(Debug, Deserialize)]
pub struct RopeParameters {
    pub rope_theta: f64,
}

/// The dense Qwen3 topology shared by audio and text path adapters.
#[derive(Debug, Deserialize)]
pub struct LLMConfig {
    pub attention_bias: bool,
    pub head_dim: usize,
    pub hidden_act: String,
    pub hidden_size: usize,
    pub intermediate_size: usize,
    pub num_attention_heads: usize,
    pub num_hidden_layers: usize,
    pub num_key_value_heads: usize,
    pub rms_norm_eps: f64,
    pub rope_parameters: Option<RopeParameters>,
    pub rope_theta: Option<f64>,
    pub vocab_size: usize,
}

impl LLMConfig {
    pub fn rope_base(&self) -> f64 {
        self.rope_parameters
            .as_ref()
            .map(|r| r.rope_theta)
            .or(self.rope_theta)
            .unwrap_or(0.0)
    }

    pub fn validate(&self) -> Result<()> {
        ensure!(self.hidden_act == "silu", "unsupported Qwen3 activation");
        ensure!(
            (1..=96).contains(&self.num_hidden_layers)
                && (1..=8192).contains(&self.hidden_size)
                && (1..=32768).contains(&self.intermediate_size),
            "invalid Qwen3 dimensions"
        );
        ensure!(
            (2..=256).contains(&self.head_dim)
                && self.head_dim.is_multiple_of(2)
                && (1..=64).contains(&self.num_attention_heads)
                && (1..=self.num_attention_heads).contains(&self.num_key_value_heads)
                && self
                    .num_attention_heads
                    .is_multiple_of(self.num_key_value_heads),
            "invalid Qwen3 attention layout"
        );
        ensure!(
            (1..=200_000).contains(&self.vocab_size)
                && self.rope_base().is_finite()
                && self.rope_base() > 1.0
                && self.rms_norm_eps.is_finite()
                && self.rms_norm_eps > 0.0,
            "invalid Qwen3 vocabulary or normalization"
        );
        Ok(())
    }
}

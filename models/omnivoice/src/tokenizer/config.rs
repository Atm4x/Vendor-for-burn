use serde::{Deserialize, Serialize};

use crate::{tokenizer::dac::config::DacConfig, tokenizer::hubert::config::HubertConfig};

#[derive(Serialize, Deserialize)]
pub struct HiggsAudioV2TokenizerConfig {
    pub acoustic_model_config: DacConfig,
    pub semantic_model_config: HubertConfig,
    pub architectures: Vec<String>,
    pub block_dilations: Vec<usize>,
    pub channel_ratios: Vec<usize>,
    pub codebook_dim: usize,
    pub codebook_size: usize,
    pub downsample_factor: usize,
    pub dtype: String,
    pub initializer_range: f64,
    pub kernel_size: usize,
    pub model_type: String,
    pub sample_rate: usize,
    pub semantic_sample_rate: usize,
    pub strides: Vec<usize>,
    pub target_bandwidths: Vec<f64>,
    pub transformers_version: String,
    pub unit_kernel_size: usize,
}

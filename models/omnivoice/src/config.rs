pub use native_burn_qwen3::LLMConfig;
use serde::Deserialize;

#[derive(Deserialize)]
pub struct OmniVoiceConfig {
    pub audio_codebook_weights: Vec<usize>,
    pub audio_mask_id: usize,
    pub audio_vocab_size: usize,
    pub llm_config: LLMConfig,
    pub num_audio_codebook: usize,
}

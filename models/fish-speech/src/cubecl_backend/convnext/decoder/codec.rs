//! Synthesis-facing decoder adapter and prepared Firefly reference/audio codec.

use super::*;
use crate::weights::{LoadedWeights, Result as WeightResult, WeightError};
use fish_speech_core::{audio::FIREFLY_GROUPS, synthesis::AcousticDecoder};
use std::sync::Arc;

/// A model-scoped adapter for the standard synthesis pipeline.
/// The acoustic codebooks are validated before any CubeCL command is submitted.
pub struct CubeClFireflyAcousticDecoder {
    executor: CubeClFireflyDecoder,
    weights: Arc<BTreeMap<String, Vec<f32>>>,
}

impl CubeClFireflyAcousticDecoder {
    pub fn from_checkpoint(client: Client, checkpoint: &LoadedWeights) -> WeightResult<Self> {
        let graph = FireflyGraph::waveform_decode(1, 1, 1)
            .map_err(|e| WeightError::Invalid(format!("Firefly decoder: {e}")))?;
        let weights = checkpoint.firefly_cpu_weights(&graph)?;
        Ok(Self::new(client, weights))
    }

    pub fn new(client: Client, weights: BTreeMap<String, Vec<f32>>) -> Self {
        Self {
            executor: CubeClFireflyDecoder::new(client),
            weights: Arc::new(weights),
        }
    }

    pub fn decode_indices(&self, indices: &[i32], frames: usize) -> Result<Vec<f32>> {
        if frames == 0 || indices.len() != FIREFLY_GROUPS * frames {
            return Err(FishError::InvalidShape(
                "Firefly acoustic codes must be [8, frames]",
            ));
        }
        self.executor.decode_codes(&self.weights, indices, frames)
    }
}

impl AcousticDecoder for CubeClFireflyAcousticDecoder {
    fn decode(&mut self, acoustic_codes: &[Vec<u32>]) -> Result<Vec<f32>> {
        if acoustic_codes.len() != FIREFLY_GROUPS {
            return Err(FishError::InvalidShape(
                "Firefly requires eight acoustic codebooks",
            ));
        }
        let frames = acoustic_codes[0].len();
        if frames == 0 || acoustic_codes.iter().any(|codes| codes.len() != frames) {
            return Err(FishError::InvalidShape(
                "inconsistent acoustic codebook frames",
            ));
        }
        let mut indices = Vec::with_capacity(FIREFLY_GROUPS * frames);
        for codebook in acoustic_codes {
            for &token in codebook {
                if token >= 1000 {
                    return Err(FishError::InvalidShape("invalid FSQ acoustic token"));
                }
                indices.push(token as i32);
            }
        }
        self.decode_indices(&indices, frames)
    }
}

/// A unified Firefly reference encoder and acoustic decoder built from one checkpoint.
/// The reference encoding profile is immutable; the decoder accepts any valid code-frame length.
pub struct CubeClFireflyCodec {
    encoder: CubeClPreparedFireflyEncoder,
    decoder: CubeClFireflyAcousticDecoder,
}

impl CubeClFireflyCodec {
    pub fn from_checkpoint(
        client: Client,
        checkpoint: &LoadedWeights,
        reference_samples: usize,
    ) -> WeightResult<Self> {
        let encoder_graph = FireflyGraph::reference_encode(1, reference_samples)
            .map_err(|e| WeightError::Invalid(format!("Firefly encoder: {e}")))?;
        let encoder_weights = checkpoint.firefly_cpu_weights(&encoder_graph)?;
        let encoder =
            CubeClPreparedFireflyEncoder::new(client.clone(), encoder_graph, encoder_weights)
                .map_err(|e| WeightError::Invalid(format!("CubeCL encoder: {e}")))?;
        let decoder = CubeClFireflyAcousticDecoder::from_checkpoint(client, checkpoint)?;
        Ok(Self { encoder, decoder })
    }

    pub fn encode_reference_pcm(&self, pcm: &[f32]) -> Result<Vec<i32>> {
        self.encoder.encode_mono_pcm(pcm)
    }

    pub fn decode_codes(&mut self, codes: &[Vec<u32>]) -> Result<Vec<f32>> {
        self.decoder.decode(codes)
    }

    pub fn decoder_mut(&mut self) -> &mut CubeClFireflyAcousticDecoder {
        &mut self.decoder
    }
}

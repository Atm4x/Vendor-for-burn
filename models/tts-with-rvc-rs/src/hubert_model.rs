use burn::tensor::{Device, Tensor};

use crate::{checkpoint::RvcCheckpoint, hubert::HubertFrontend, hubert_position::HubertEncoder};

/// HuBERT feature extraction from 16-kHz mono PCM to content embeddings.
/// The caller supplies a valid-frame mask after convolutional downsampling if needed.
pub struct HubertFeatureModel {
    frontend: HubertFrontend,
    encoder: HubertEncoder,
}

impl HubertFeatureModel {
    pub fn load(checkpoint: &RvcCheckpoint, layers: usize, heads: usize, device: &Device) -> Result<Self, String> {
        Ok(Self {
            frontend: HubertFrontend::load(checkpoint, device)?,
            encoder: HubertEncoder::load(checkpoint, layers, heads, device)?,
        })
    }

    pub fn forward(&self, mono_waveform: Tensor<3>, valid_frames: Option<Tensor<2>>) -> Tensor<3> {
        let features = self.frontend.forward(mono_waveform);
        self.encoder.forward(features, valid_frames)
    }

    pub fn output_layers(&self) -> usize {
        self.encoder.layers()
    }
}

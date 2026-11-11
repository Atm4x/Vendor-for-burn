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
        let device = mono_waveform.device();
        let started = std::time::Instant::now();
        let features = self.frontend.forward(mono_waveform);
        crate::diagnostic::measure(&device, "hubert.frontend_conv", started, serde_json::json!({"shape":features.dims()}));
        let started = std::time::Instant::now();
        let encoded = self.encoder.forward(features, valid_frames);
        crate::diagnostic::measure(&device, "hubert.transformer", started, serde_json::json!({"shape":encoded.dims(),"layers":self.output_layers()}));
        encoded
    }

    pub fn output_layers(&self) -> usize {
        self.encoder.layers()
    }
}

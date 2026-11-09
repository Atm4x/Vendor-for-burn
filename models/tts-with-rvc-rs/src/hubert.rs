use burn::tensor::{activation, module, ops::ConvOptions, Device, Tensor};

use crate::checkpoint::RvcCheckpoint;

/// The first HuBERT Base feature-extractor layer: Conv1d -> GroupNorm -> GELU.
/// This is not yet the seven-layer feature extractor or the transformer encoder.
pub struct HubertFirstConv {
    weight: Tensor<3>,
    norm_weight: Tensor<1>,
    norm_bias: Tensor<1>,
}

impl HubertFirstConv {
    pub fn load(checkpoint: &RvcCheckpoint, device: &Device) -> Result<Self, String> {
        let weight = checkpoint.tensor::<3>("feature_extractor.conv_layers.0.0.weight", device)?;
        let norm_weight = checkpoint.tensor::<1>("feature_extractor.conv_layers.0.2.weight", device)?;
        let norm_bias = checkpoint.tensor::<1>("feature_extractor.conv_layers.0.2.bias", device)?;
        if weight.dims() != [512, 1, 10]
            || norm_weight.dims() != [512]
            || norm_bias.dims() != [512]
        {
            return Err("expected HuBERT Base first Conv1d (512,1,10) and GroupNorm 512".into());
        }
        Ok(Self { weight, norm_weight, norm_bias })
    }

    pub fn forward(&self, waveform: Tensor<3>) -> Tensor<3> {
        assert_eq!(waveform.dims()[1], 1);
        let x = module::conv1d(
            waveform,
            self.weight.clone(),
            None,
            ConvOptions::new_with_padding([5], [(0, 0)], [1], 1),
        );
        let mean = x.clone().mean_dim(2);
        let centered = x - mean;
        let variance = centered.clone().powf_scalar(2.0).mean_dim(2);
        let normalized = centered / (variance + 1e-5).sqrt();
        activation::gelu(
            normalized * self.norm_weight.clone().reshape([1, 512, 1])
                + self.norm_bias.clone().reshape([1, 512, 1]),
        )
    }
}

use burn::tensor::{activation, module, ops::ConvOptions, Device, Tensor};

use crate::{checkpoint::RvcCheckpoint, conv1d_layout::{default_conv1d_layout, load_conv1d_weight}};

/// The first HuBERT Base feature-extractor layer: Conv1d -> GroupNorm -> GELU.
/// This is not yet the seven-layer feature extractor or the transformer encoder.
pub struct HubertFirstConv {
    weight: Tensor<3>,
    norm_weight: Tensor<1>,
    norm_bias: Tensor<1>,
}

impl HubertFirstConv {
    pub fn load(checkpoint: &RvcCheckpoint, device: &Device) -> Result<Self, String> {
        let weight = load_conv1d_weight(
            checkpoint, "feature_extractor.conv_layers.0.0", false,
            default_conv1d_layout(), device)?;
        let norm_weight = checkpoint.tensor::<1>("feature_extractor.conv_layers.0.2.weight", device)?;
        let norm_bias = checkpoint.tensor::<1>("feature_extractor.conv_layers.0.2.bias", device)?;
        let [channels, input_channels, kernel] = weight.dims();
        if input_channels != 1 || kernel != 10
            || norm_weight.dims() != [channels]
            || norm_bias.dims() != [channels]
        {
            return Err("HuBERT first Conv1d must have one input channel, kernel 10 and matching GroupNorm".into());
        }
        Ok(Self { weight, norm_weight, norm_bias })
    }

    pub fn forward(&self, waveform: Tensor<3>) -> Tensor<3> {
        assert_eq!(waveform.dims()[1], 1);
        let options = ConvOptions::new_with_padding([5], [(0, 0)], [1], 1);
        let x = if std::env::var("RVC_HUBERT_FRONTEND_GEMM").as_deref() == Ok("1") {
            crate::cubecl_conv::conv1d(waveform, self.weight.clone(), None, options)
        } else {
            module::conv1d(waveform, self.weight.clone(), None, options)
        };
        let mean = x.clone().mean_dim(2);
        let centered = x - mean;
        let variance = centered.clone().powf_scalar(2.0).mean_dim(2);
        let normalized = centered / (variance + 1e-5).sqrt();
        let channels = self.norm_weight.dims()[0];
        activation::gelu(
            normalized * self.norm_weight.clone().reshape([1, channels, 1])
                + self.norm_bias.clone().reshape([1, channels, 1]),
        )
    }
}

/// Seven convolutional layers and the original post-extraction LayerNorm/Linear.
/// The HuBERT Transformer encoder is a separate, unfinished component.
pub struct HubertFrontend {
    first: HubertFirstConv,
    rest: Vec<Tensor<3>>,
    layer_norm_weight: Tensor<1>,
    layer_norm_bias: Tensor<1>,
    projection_weight: Tensor<2>,
    projection_bias: Tensor<1>,
}

impl HubertFrontend {
    pub fn load(checkpoint: &RvcCheckpoint, device: &Device) -> Result<Self, String> {
        let first = HubertFirstConv::load(checkpoint, device)?;
        let channels = first.norm_weight.dims()[0];
        let mut rest = Vec::with_capacity(6);
        for layer in 1..=6 {
            let weight = load_conv1d_weight(
                checkpoint, &format!("feature_extractor.conv_layers.{layer}.0"), false,
                default_conv1d_layout(), device)?;
            let kernel = if layer <= 4 { 3 } else { 2 };
            if weight.dims() != [channels, channels, kernel] {
                return Err(format!("HuBERT conv layer {layer}: invalid channels or kernel"));
            }
            rest.push(weight);
        }
        let layer_norm_weight = checkpoint.tensor::<1>("layer_norm.weight", device)?;
        let layer_norm_bias = checkpoint.tensor::<1>("layer_norm.bias", device)?;
        let projection_weight = checkpoint.tensor::<2>("post_extract_proj.weight", device)?;
        let projection_bias = checkpoint.tensor::<1>("post_extract_proj.bias", device)?;
        if layer_norm_weight.dims() != [channels]
            || layer_norm_bias.dims() != [channels]
            || projection_weight.dims()[1] != channels
            || projection_weight.dims()[0] != projection_bias.dims()[0]
        {
            return Err("HuBERT LayerNorm/post_extract_proj dimensions are inconsistent".into());
        }
        Ok(Self { first, rest, layer_norm_weight, layer_norm_bias, projection_weight, projection_bias })
    }

    pub fn conv_features(&self, waveform: Tensor<3>) -> Tensor<3> {
        let mut x = self.first.forward(waveform);
        for weight in &self.rest {
            let options = ConvOptions::new_with_padding([2], [(0, 0)], [1], 1);
            let conv = if std::env::var("RVC_HUBERT_FRONTEND_GEMM").as_deref() == Ok("1") {
                crate::cubecl_conv::conv1d(x, weight.clone(), None, options)
            } else {
                module::conv1d(x, weight.clone(), None, options)
            };
            x = activation::gelu(conv);
        }
        x
    }

    pub fn forward(&self, waveform: Tensor<3>) -> Tensor<3> {
        let extracted = self.conv_features(waveform).permute([0, 2, 1]);
        let [batch, frames, channels] = extracted.dims();
        let mean = extracted.clone().mean_dim(2);
        let centered = extracted - mean;
        let variance = centered.clone().powf_scalar(2.0).mean_dim(2);
        let normalized = centered / (variance + 1e-5).sqrt();
        let normal = normalized * self.layer_norm_weight.clone().reshape([1, 1, channels])
            + self.layer_norm_bias.clone().reshape([1, 1, channels]);
        let output_channels = self.projection_bias.dims()[0];
        normal.reshape([batch * frames, channels])
            .matmul(self.projection_weight.clone().transpose())
            .reshape([batch, frames, output_channels])
            + self.projection_bias.clone().reshape([1, 1, output_channels])
    }
}

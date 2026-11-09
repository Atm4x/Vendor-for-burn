use burn::tensor::{activation, module, ops::ConvOptions, Device, Tensor};

use crate::checkpoint::RvcCheckpoint;
use crate::hubert_transformer::HubertTransformerStack;

/// Fairseq HuBERT positional convolution with weight normalization along the kernel axis.
/// An invalid frame is zeroed before positional convolution, as in Fairseq's encoder.
pub struct HubertPosition {
    weight: Tensor<3>,
    bias: Tensor<1>,
    layer_norm_weight: Tensor<1>,
    layer_norm_bias: Tensor<1>,
    groups: usize,
}

impl HubertPosition {
    pub fn load(checkpoint: &RvcCheckpoint, device: &Device) -> Result<Self, String> {
        let weight = checkpoint.weight_norm_conv1d_dim2("encoder.pos_conv.0", device)?;
        let bias = checkpoint.tensor::<1>("encoder.pos_conv.0.bias", device)?;
        let layer_norm_weight = checkpoint.tensor::<1>("encoder.layer_norm.weight", device)?;
        let layer_norm_bias = checkpoint.tensor::<1>("encoder.layer_norm.bias", device)?;
        let [channels, per_group, kernel] = weight.dims();
        if channels == 0 || per_group == 0 || kernel == 0 || channels % per_group != 0
            || bias.dims() != [channels]
            || layer_norm_weight.dims() != [channels]
            || layer_norm_bias.dims() != [channels]
        {
            return Err("HuBERT position weight, groups or layer normalization dimensions are invalid".into());
        }
        let groups = channels / per_group;
        if channels % groups != 0 {
            return Err("HuBERT positional groups must divide the embedding size".into());
        }
        Ok(Self { weight, bias, layer_norm_weight, layer_norm_bias, groups })
    }

    pub fn forward(&self, mut input: Tensor<3>, valid_keys: Option<Tensor<2>>) -> Tensor<3> {
        let [batch, time, channels] = input.dims();
        assert_eq!(self.weight.dims()[0], channels);
        if let Some(valid) = valid_keys {
            assert_eq!(valid.dims(), [batch, time]);
            input = input * valid.reshape([batch, time, 1]);
        }
        let kernel = self.weight.dims()[2];
        let mut position = module::conv1d(
            input.clone().permute([0, 2, 1]),
            self.weight.clone(),
            Some(self.bias.clone()),
            ConvOptions::new_with_padding([1], [(kernel / 2, kernel / 2)], [1], self.groups),
        );
        if kernel % 2 == 0 {
            position = position.slice([0..batch, 0..channels, 0..time]);
        }
        assert_eq!(position.dims(), [batch, channels, time]);
        let combined = input + activation::gelu(position.permute([0, 2, 1]));
        let mean = combined.clone().mean_dim(2);
        let centered = combined - mean;
        let variance = centered.clone().powf_scalar(2.0).mean_dim(2);
        centered / (variance + 1e-5).sqrt()
            * self.layer_norm_weight.clone().reshape([1, 1, channels])
            + self.layer_norm_bias.clone().reshape([1, 1, channels])
    }
}

/// Inference-only composition: positional embedding, outer LayerNorm, and N transformer layers.
pub struct HubertEncoder {
    position: HubertPosition,
    stack: HubertTransformerStack,
}

impl HubertEncoder {
    pub fn load(
        checkpoint: &RvcCheckpoint,
        layer_count: usize,
        heads: usize,
        device: &Device,
    ) -> Result<Self, String> {
        Ok(Self {
            position: HubertPosition::load(checkpoint, device)?,
            stack: HubertTransformerStack::load(checkpoint, layer_count, heads, device)?,
        })
    }

    pub fn layers(&self) -> usize { self.stack.len() }

    pub fn position_only(&self, input: Tensor<3>, valid_keys: Option<Tensor<2>>) -> Tensor<3> {
        self.position.forward(input, valid_keys)
    }

    pub fn forward(&self, input: Tensor<3>, valid_keys: Option<Tensor<2>>) -> Tensor<3> {
        let input = self.position.forward(input, valid_keys.clone());
        self.stack.forward(input, valid_keys)
    }
}

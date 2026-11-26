use burn::tensor::{activation, module, ops::ConvOptions, Device, Tensor};

use crate::{checkpoint::RvcCheckpoint, conv1d_layout::{default_conv1d_layout, from_host_values}};
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
        let (shape, values) = checkpoint.weight_norm_conv1d_dim2_values("encoder.pos_conv.0")?;
        let weight = from_host_values(shape, values, device, default_conv1d_layout());
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
        let device = input.device();
        let started = std::time::Instant::now();
        let input = self.position.forward(input, valid_keys.clone());
        crate::diagnostic::measure(&device, "hubert.position_conv", started, serde_json::json!({"output_shape":input.dims()}));
        let started = std::time::Instant::now();
        let out = self.stack.forward(input, valid_keys);
        crate::diagnostic::measure(&device, "hubert.transformer_stack", started, serde_json::json!({"layers":self.layers(),"output_shape":out.dims()}));
        out
    }
}

#[cfg(all(test, feature = "flex"))]
mod layout_parity_tests {
    use super::*;
    use crate::conv1d_layout::{Conv1dWeightLayout, from_host_values};
    use burn::tensor::TensorData;

    #[test]
    fn grouped_positional_conv_prepack_parity_over_kernel_and_group_matrix() {
        let device = Device::flex();
        for (channels, per_group, kernel, time) in [
            (8, 8, 1, 13),
            (16, 4, 3, 31),
            (32, 4, 16, 41),
            (32, 2, 64, 33),
            (48, 3, 128, 24),
        ] {
            let batch = 1;
            let input_values: Vec<f32> = (0..batch * channels * time)
                .map(|i| ((i * 17 % 73) as f32 - 36.0) / 57.0)
                .collect();
            let weight_values: Vec<f32> = (0..channels * per_group * kernel)
                .map(|i| ((i * 13 % 101) as f32 - 50.0) / (101.0 * kernel as f32))
                .collect();
            let input = Tensor::from_data(TensorData::new(input_values, [batch, channels, time]), &device);
            let normal = from_host_values([channels, per_group, kernel], weight_values.clone(), &device, Conv1dWeightLayout::Standard);
            let prepacked = from_host_values([channels, per_group, kernel], weight_values, &device, Conv1dWeightLayout::Prepacked);
            let groups = channels / per_group;
            let options = ConvOptions::new_with_padding([1], [(kernel / 2, kernel / 2)], [1], groups);
            let reference = module::conv1d(input.clone(), normal, None, options.clone())
                .into_data().try_into_vec::<f32>().unwrap();
            let actual = module::conv1d(input, prepacked, None, options)
                .into_data().try_into_vec::<f32>().unwrap();
            assert_eq!(reference.len(), actual.len());
            for (lhs, rhs) in reference.into_iter().zip(actual) {
                assert!((lhs - rhs).abs() < 1e-5, "groups={groups}, kernel={kernel}, time={time}: {lhs} vs {rhs}");
            }
        }
    }

    #[test]
    fn full_positional_encoder_masked_path_preserves_values_with_host_prepack() {
        let device = Device::flex();
        let (channels, per_group, kernel, batch, time) = (16, 4, 16, 2, 17);
        let values: Vec<f32> = (0..channels * per_group * kernel)
            .map(|i| ((i * 7 % 29) as f32 - 14.0) / 210.0).collect();
        let weights = |layout| from_host_values([channels, per_group, kernel], values.clone(), &device, layout);
        let position = |layout| HubertPosition {
            weight: weights(layout),
            bias: Tensor::from_data(TensorData::new(vec![0.01; channels], [channels]), &device),
            layer_norm_weight: Tensor::ones([channels], &device),
            layer_norm_bias: Tensor::zeros([channels], &device),
            groups: channels / per_group,
        };
        let input = Tensor::from_data(TensorData::new(
            (0..batch * time * channels).map(|i| ((i * 11 % 37) as f32 - 18.0) / 73.0).collect(),
            [batch, time, channels]
        ), &device);
        let mask = Tensor::from_data(TensorData::new(
            (0..batch * time).map(|i| if i % 13 == 0 { 0.0 } else { 1.0 }).collect(),
            [batch, time],
        ), &device);
        let expected = position(Conv1dWeightLayout::Standard).forward(input.clone(), Some(mask.clone()))
            .into_data().try_into_vec::<f32>().unwrap();
        let actual = position(Conv1dWeightLayout::Prepacked).forward(input, Some(mask))
            .into_data().try_into_vec::<f32>().unwrap();
        assert_eq!(expected.len(), actual.len());
        for (index, (reference, current)) in expected.into_iter().zip(actual).enumerate() {
            assert!((reference - current).abs() < 1e-5, "position={index}: {reference} vs {current}");
        }
    }
}

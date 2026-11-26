use burn::tensor::{Tensor, TensorData};

use crate::{flex_depthwise::FlexDepthwiseNorm, flex_pointwise::FlexPointwise};

/// Runs the native ConvNeXt block entirely in host buffers.
pub(crate) struct FlexConvNeXt {
    preprocess: FlexDepthwiseNorm,
    pointwise: FlexPointwise,
}

impl FlexConvNeXt {
    pub(crate) fn new(preprocess: FlexDepthwiseNorm, pointwise: FlexPointwise) -> Self {
        Self {
            preprocess,
            pointwise,
        }
    }

    pub(crate) fn forward_host(
        &self,
        samples: &[f32],
        batch: usize,
        channels: usize,
        frames: usize,
    ) -> Vec<f32> {
        let timer = std::time::Instant::now();
        let normalized = self
            .preprocess
            .forward_host(samples, batch, channels, frames);
        if std::env::var_os("SILERO_VOCODER_TIMING").is_some() {
            eprintln!(
                "[vocoder-block] fused_depthwise_norm_ms={:.3}",
                timer.elapsed().as_secs_f64() * 1000.0
            );
        }
        self.pointwise
            .forward_host(&normalized, samples, batch, channels, frames)
    }

    pub(crate) fn forward(&self, input: Tensor<3>) -> Tensor<3> {
        let [batch, channels, frames] = input.dims();
        let device = input.device();
        let samples = input
            .into_data()
            .try_to_vec::<f32>()
            .expect("Flex ConvNeXt input must be f32");
        let output = self.forward_host(&samples, batch, channels, frames);
        Tensor::from_data(TensorData::new(output, [batch, channels, frames]), &device)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use burn::tensor::{Device, TensorData};

    #[test]
    fn host_chain_matches_tensor_boundary_bitwise() {
        let device = Device::flex();
        for (batch, channels, frames) in [(1, 8, 13), (2, 16, 35), (1, 32, 75)] {
            let input = Tensor::<3>::from_data(
                TensorData::new(
                    (0..batch * channels * frames)
                        .map(|i| ((i * 7 % 97) as f32 - 48.0) * 0.007)
                        .collect(),
                    [batch, channels, frames],
                ),
                &device,
            );
            let weight = Tensor::<3>::from_data(
                TensorData::new(
                    (0..channels * 7)
                        .map(|i| ((i * 13 % 41) as f32 - 20.0) * 0.009)
                        .collect(),
                    [channels, 1, 7],
                ),
                &device,
            );
            let bias = Tensor::<1>::from_data(
                TensorData::new(vec![0.015f32; channels], [channels]),
                &device,
            );
            let gamma = Tensor::<1>::from_data(
                TensorData::new(vec![1.0f32; channels], [channels]),
                &device,
            );
            let beta = Tensor::<1>::from_data(
                TensorData::new(vec![0.01f32; channels], [channels]),
                &device,
            );
            let hidden = channels * 3;
            let pw1 = Tensor::<2>::from_data(
                TensorData::new(
                    (0..channels * hidden)
                        .map(|i| ((i * 17 % 109) as f32 - 54.0) * 0.0006)
                        .collect(),
                    [channels, hidden],
                ),
                &device,
            );
            let pw2 = Tensor::<2>::from_data(
                TensorData::new(
                    (0..hidden * channels)
                        .map(|i| ((i * 19 % 131) as f32 - 65.0) * 0.0006)
                        .collect(),
                    [hidden, channels],
                ),
                &device,
            );
            let pw1_bias =
                Tensor::<1>::from_data(TensorData::new(vec![0.002f32; hidden], [hidden]), &device);
            let pw2_bias = Tensor::<1>::from_data(
                TensorData::new(vec![0.003f32; channels], [channels]),
                &device,
            );
            let residual_gamma = Tensor::<1>::from_data(
                TensorData::new(vec![0.9f32; channels], [channels]),
                &device,
            );
            let staged_preprocess =
                FlexDepthwiseNorm::new(weight.clone(), bias.clone(), gamma.clone(), beta.clone())
                    .unwrap();
            let staged_pointwise = FlexPointwise::new(
                pw1.clone(),
                pw1_bias.clone(),
                pw2.clone(),
                pw2_bias.clone(),
                residual_gamma.clone(),
            )
            .unwrap();
            let original = staged_pointwise
                .forward(staged_preprocess.forward(input.clone()), input.clone())
                .into_data()
                .try_to_vec::<f32>()
                .unwrap();
            let native = FlexConvNeXt::new(
                FlexDepthwiseNorm::new(weight, bias, gamma, beta).unwrap(),
                FlexPointwise::new(pw1, pw1_bias, pw2, pw2_bias, residual_gamma).unwrap(),
            );
            let actual = native
                .forward(input)
                .into_data()
                .try_to_vec::<f32>()
                .unwrap();
            assert_eq!(original.len(), actual.len());
            for (idx, (a, b)) in original.iter().zip(&actual).enumerate() {
                assert_eq!(a.to_bits(), b.to_bits(), "element {idx}: {a} vs {b}");
            }
        }
    }
}

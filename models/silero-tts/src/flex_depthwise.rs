use anyhow::{Result, ensure};
use burn::tensor::{Tensor, TensorData};
use rayon::prelude::*;

pub(crate) struct FlexDepthwise {
    channels: usize,
    kernel: usize,
    weights: Vec<f32>,
    bias: Vec<f32>,
}

impl FlexDepthwise {
    pub(crate) fn new(weight: Tensor<3>, bias: Tensor<1>) -> Result<Self> {
        let [channels, per_group, kernel] = weight.dims();
        ensure!(
            channels > 0
                && per_group == 1
                && kernel > 0
                && kernel % 2 == 1
                && bias.dims() == [channels],
            "Invalid depthwise convolution weights"
        );
        Ok(Self {
            channels,
            kernel,
            weights: weight.into_data().try_to_vec::<f32>()?,
            bias: bias.into_data().try_to_vec::<f32>()?,
        })
    }

    pub(crate) fn forward(&self, input: Tensor<3>) -> Tensor<3> {
        let [batch, channels, frames] = input.dims();
        assert_eq!(channels, self.channels);
        let device = input.device();
        let samples = input
            .into_data()
            .try_to_vec::<f32>()
            .expect("Flex depthwise requires f32");
        let mut result = vec![0.0f32; batch * channels * frames];
        let kernel = self.kernel;
        let pad = (kernel / 2) as isize;
        let process = |index: usize, output: &mut [f32]| {
            let channel = index % channels;
            let signal = &samples[index * frames..(index + 1) * frames];
            let weight = &self.weights[channel * kernel..(channel + 1) * kernel];
            for (t, value) in output.iter_mut().enumerate() {
                let mut acc = 0.0f32;
                for (k, w) in weight.iter().enumerate() {
                    let position = t as isize + k as isize - pad;
                    if position >= 0 && position < frames as isize {
                        acc += signal[position as usize] * w;
                    }
                }
                *value = acc + self.bias[channel];
            }
        };
        if batch * channels * frames >= 32_768 {
            result
                .par_chunks_mut(frames)
                .enumerate()
                .for_each(|(index, output)| process(index, output));
        } else {
            for (index, output) in result.chunks_mut(frames).enumerate() {
                process(index, output);
            }
        }
        Tensor::from_data(TensorData::new(result, [batch, channels, frames]), &device)
    }
}

/// Performs channel-wise convolution and layer normalization without an intermediate Burn tensor.
pub(crate) struct FlexDepthwiseNorm {
    conv: FlexDepthwise,
    gamma: Vec<f32>,
    beta: Vec<f32>,
}

impl FlexDepthwiseNorm {
    pub(crate) fn new(
        weight: Tensor<3>,
        bias: Tensor<1>,
        gamma: Tensor<1>,
        beta: Tensor<1>,
    ) -> Result<Self> {
        let conv = FlexDepthwise::new(weight, bias)?;
        ensure!(
            gamma.dims() == [conv.channels] && beta.dims() == [conv.channels],
            "Invalid layer norm parameters"
        );
        Ok(Self {
            conv,
            gamma: gamma.into_data().try_to_vec::<f32>()?,
            beta: beta.into_data().try_to_vec::<f32>()?,
        })
    }

    pub(crate) fn forward(&self, input: Tensor<3>) -> Tensor<3> {
        let [batch, channels, frames] = input.dims();
        assert_eq!(channels, self.conv.channels);
        let device = input.device();
        let samples = input
            .into_data()
            .try_to_vec::<f32>()
            .expect("Flex fused depthwise requires f32");
        let mut output = vec![0.0f32; batch * frames * channels];
        let kernel = self.conv.kernel;
        let pad = (kernel / 2) as isize;
        let work = |row: usize, values: &mut [f32]| {
            let b = row / frames;
            let t = row % frames;
            for (c, value) in values.iter_mut().enumerate() {
                let signal = &samples[(b * channels + c) * frames..(b * channels + c + 1) * frames];
                let weight = &self.conv.weights[c * kernel..(c + 1) * kernel];
                let mut acc = 0.0f32;
                for (k, w) in weight.iter().enumerate() {
                    let position = t as isize + k as isize - pad;
                    if position >= 0 && position < frames as isize {
                        acc += signal[position as usize] * w;
                    }
                }
                *value = acc + self.conv.bias[c];
            }
            let mean = values.iter().copied().sum::<f32>() / channels as f32;
            let variance =
                values.iter().map(|x| (x - mean) * (x - mean)).sum::<f32>() / channels as f32;
            let inv_std = 1.0f32 / (variance + 1.0e-6).sqrt();
            for (c, value) in values.iter_mut().enumerate() {
                *value = (*value - mean) * (inv_std * self.gamma[c]) + self.beta[c];
            }
        };
        if output.len() >= 32_768 {
            output
                .par_chunks_mut(channels)
                .enumerate()
                .for_each(|(r, values)| work(r, values));
        } else {
            for (r, values) in output.chunks_mut(channels).enumerate() {
                work(r, values);
            }
        }
        Tensor::from_data(TensorData::new(output, [batch, frames, channels]), &device)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use burn::tensor::{Device, module::conv1d, ops::ConvOptions};

    #[test]
    fn fused_depthwise_layer_norm_matches_staged_flex() {
        use burn::tensor::module::layer_norm;
        let device = Device::flex();
        for (batch, channels, frames, kernel) in [
            (1, 8, 13, 7),
            (2, 16, 35, 7),
            (1, 64, 127, 3),
            (1, 512, 97, 7),
            (2, 3, 1, 7),
        ] {
            let input = Tensor::<3>::from_data(
                TensorData::new(
                    (0..batch * channels * frames)
                        .map(|i| (i * 7 % 97) as f32 * 0.007 - 0.28)
                        .collect(),
                    [batch, channels, frames],
                ),
                &device,
            );
            let weight = Tensor::<3>::from_data(
                TensorData::new(
                    (0..channels * kernel)
                        .map(|i| (i * 13 % 37) as f32 * 0.009 - 0.16)
                        .collect(),
                    [channels, 1, kernel],
                ),
                &device,
            );
            let bias = Tensor::<1>::from_data(
                TensorData::new(
                    (0..channels).map(|i| i as f32 * 0.001 - 0.2).collect(),
                    [channels],
                ),
                &device,
            );
            let gamma = Tensor::<1>::from_data(
                TensorData::new(
                    (0..channels)
                        .map(|i| (i * 29 % 107) as f32 * 0.007)
                        .collect(),
                    [channels],
                ),
                &device,
            );
            let beta = Tensor::<1>::from_data(
                TensorData::new(
                    (0..channels).map(|i| i as f32 * 0.0005 - 0.1).collect(),
                    [channels],
                ),
                &device,
            );
            let expected = layer_norm(
                FlexDepthwise::new(weight.clone(), bias.clone())
                    .unwrap()
                    .forward(input.clone())
                    .swap_dims(1, 2),
                gamma.clone(),
                Some(beta.clone()),
                1.0e-6,
            )
            .into_data()
            .try_to_vec::<f32>()
            .unwrap();
            let actual = FlexDepthwiseNorm::new(weight, bias, gamma, beta)
                .unwrap()
                .forward(input)
                .into_data()
                .try_to_vec::<f32>()
                .unwrap();
            let max_error = expected
                .iter()
                .zip(actual)
                .map(|(a, b)| (a - b).abs())
                .fold(0.0f32, f32::max);
            assert!(
                max_error <= 2.0e-5,
                "shape={batch},{channels},{frames}, kernel={kernel} max={max_error}"
            );
        }
    }

    #[test]
    fn native_depthwise_matches_burn_for_batches_and_temporal_boundaries() {
        let device = Device::flex();
        for (batch, channels, frames, kernel) in [
            (1, 8, 13, 7),
            (2, 16, 35, 7),
            (1, 64, 127, 3),
            (1, 512, 97, 7),
            (2, 3, 1, 7),
        ] {
            let input = Tensor::<3>::from_data(
                TensorData::new(
                    (0..batch * channels * frames)
                        .map(|i| (i * 7 % 97) as f32 * 0.007 - 0.28)
                        .collect(),
                    [batch, channels, frames],
                ),
                &device,
            );
            let weight = Tensor::<3>::from_data(
                TensorData::new(
                    (0..channels * kernel)
                        .map(|i| (i * 13 % 37) as f32 * 0.009 - 0.16)
                        .collect(),
                    [channels, 1, kernel],
                ),
                &device,
            );
            let bias = Tensor::<1>::from_data(
                TensorData::new(
                    (0..channels).map(|i| i as f32 * 0.001 - 0.2).collect(),
                    [channels],
                ),
                &device,
            );
            let expected = conv1d(
                input.clone(),
                weight.clone(),
                Some(bias.clone()),
                ConvOptions::new([1], [kernel / 2], [1], channels),
            )
            .into_data()
            .try_to_vec::<f32>()
            .unwrap();
            let got = FlexDepthwise::new(weight, bias)
                .unwrap()
                .forward(input)
                .into_data()
                .try_to_vec::<f32>()
                .unwrap();
            let max_delta = got
                .iter()
                .zip(&expected)
                .map(|(a, b)| (a - b).abs())
                .fold(0.0f32, f32::max);
            assert!(
                max_delta < 0.00002,
                "shape={batch},{channels},{frames}, k={kernel}: {max_delta}"
            );
        }
    }
}

use anyhow::{Result, ensure};
use burn::tensor::{Device, Tensor, TensorData};
use rayon::prelude::*;
use std::time::Instant;

#[inline]
pub(crate) fn fast_gelu_scalar(x: f32) -> f32 {
    let z = x * std::f32::consts::FRAC_1_SQRT_2;
    let t = 1.0 / (1.0 + 0.3275911 * z.abs());
    let mut poly = 1.061405429 * t - 1.453152027;
    poly = poly * t + 1.421413741;
    poly = poly * t - 0.284496736;
    poly = poly * t + 0.254829592;
    poly *= t * (-z * z).exp();
    let erf = (1.0 - poly).copysign(z);
    0.5 * x * (1.0 + erf)
}

pub(crate) fn apply_fast_gelu(data: &mut [f32], parallel: bool) {
    if parallel && data.len() >= 32_768 {
        data.par_iter_mut()
            .for_each(|value| *value = fast_gelu_scalar(*value));
    } else {
        for value in data {
            *value = fast_gelu_scalar(*value);
        }
    }
}

struct Projection {
    in_channels: usize,
    out_channels: usize,
    weights: Vec<f32>,
    output_contiguous: bool,
    bias: Vec<f32>,
}

impl Projection {
    fn new(weight: Tensor<2>, bias: Tensor<1>) -> Result<Self> {
        let output_contiguous = std::env::var_os("SILERO_FLEX_GEMM_DIRECT_WEIGHTS").is_some();
        Self::new_with_layout(weight, bias, output_contiguous)
    }

    fn new_with_layout(
        weight: Tensor<2>,
        bias: Tensor<1>,
        output_contiguous: bool,
    ) -> Result<Self> {
        let [in_channels, out_channels] = weight.dims();
        ensure!(
            bias.dims() == [out_channels],
            "Invalid pointwise bias shape"
        );
        let weights = if output_contiguous {
            weight.into_data().try_to_vec::<f32>()?
        } else {
            weight.transpose().into_data().try_to_vec::<f32>()?
        };
        let bias = bias.into_data().try_to_vec::<f32>()?;
        Ok(Self {
            in_channels,
            out_channels,
            weights,
            output_contiguous,
            bias,
        })
    }

    fn forward_raw(&self, input: &[f32], rows: usize) -> Vec<f32> {
        assert_eq!(input.len(), rows * self.in_channels);
        let mut output = vec![0.0; rows * self.out_channels];
        let parallel = if rows * self.in_channels * self.out_channels >= 192 * 192 * 192 {
            gemm::Parallelism::Rayon(0)
        } else {
            gemm::Parallelism::None
        };
        unsafe {
            gemm::gemm(
                rows,
                self.out_channels,
                self.in_channels,
                output.as_mut_ptr(),
                1,
                self.out_channels as isize,
                false,
                input.as_ptr(),
                1,
                self.in_channels as isize,
                self.weights.as_ptr(),
                if self.output_contiguous {
                    1
                } else {
                    self.in_channels as isize
                },
                if self.output_contiguous {
                    self.out_channels as isize
                } else {
                    1
                },
                0.0f32,
                1.0f32,
                false,
                false,
                false,
                parallel,
            );
        }
        output
    }

    fn forward(&self, input: &[f32], rows: usize) -> Vec<f32> {
        let mut output = self.forward_raw(input, rows);
        if output.len() >= 32_768 {
            output.par_chunks_mut(self.out_channels).for_each(|row| {
                for (value, bias) in row.iter_mut().zip(&self.bias) {
                    *value += *bias;
                }
            });
        } else {
            for row in output.chunks_mut(self.out_channels) {
                for (value, bias) in row.iter_mut().zip(&self.bias) {
                    *value += *bias;
                }
            }
        }
        output
    }
}

fn bias_gelu_epilogue(output: &mut [f32], bias: &[f32], parallel: bool) {
    let width = bias.len();
    if parallel && output.len() >= 32_768 {
        output.par_chunks_mut(width).for_each(|row| {
            for (value, bias) in row.iter_mut().zip(bias) {
                *value = fast_gelu_scalar(*value + *bias);
            }
        });
    } else {
        for row in output.chunks_mut(width) {
            for (value, bias) in row.iter_mut().zip(bias) {
                *value = fast_gelu_scalar(*value + *bias);
            }
        }
    }
}

fn bias_gamma_residual_epilogue(
    output: &mut [f32],
    bias: &[f32],
    gamma: &[f32],
    residual: &[f32],
    batch: usize,
    frames: usize,
    parallel: bool,
) {
    let channels = bias.len();
    assert_eq!(output.len(), batch * frames * channels);
    assert_eq!(residual.len(), batch * frames * channels);
    let apply_row = |row: usize, values: &mut [f32]| {
        let b = row / frames;
        let t = row % frames;
        for c in 0..channels {
            values[c] =
                (values[c] + bias[c]) * gamma[c] + residual[(b * channels + c) * frames + t];
        }
    };
    if parallel && output.len() >= 32_768 {
        output
            .par_chunks_mut(channels)
            .enumerate()
            .for_each(|(r, values)| apply_row(r, values));
    } else {
        for (r, values) in output.chunks_mut(channels).enumerate() {
            apply_row(r, values);
        }
    }
}

pub(crate) struct FlexPointwise {
    first: Projection,
    second: Projection,
    gamma: Vec<f32>,
    fused_epilogue: bool,
}

impl FlexPointwise {
    pub(crate) fn new(
        first_weight: Tensor<2>,
        first_bias: Tensor<1>,
        second_weight: Tensor<2>,
        second_bias: Tensor<1>,
        gamma: Tensor<1>,
    ) -> Result<Self> {
        let first = Projection::new(first_weight, first_bias)?;
        let second = Projection::new(second_weight, second_bias)?;
        let gamma = gamma.into_data().try_to_vec::<f32>()?;
        ensure!(
            first.out_channels == second.in_channels
                && first.in_channels == second.out_channels
                && gamma.len() == second.out_channels,
            "Invalid ConvNeXt pointwise dimensions"
        );
        Ok(Self {
            first,
            second,
            gamma,
            fused_epilogue: std::env::var_os("SILERO_FLEX_STAGED_EPILOGUE").is_none(),
        })
    }

    pub(crate) fn forward(&self, norm: Tensor<3>, residual: Tensor<3>) -> Tensor<3> {
        self.forward_impl(norm, residual, self.fused_epilogue)
    }

    fn forward_impl(&self, norm: Tensor<3>, residual: Tensor<3>, fused: bool) -> Tensor<3> {
        let [batch, frames, channels] = norm.dims();
        assert_eq!(residual.dims(), [batch, channels, frames]);
        assert_eq!(channels, self.first.in_channels);
        let device = Device::flex();
        let input = norm
            .into_data()
            .try_to_vec::<f32>()
            .expect("Flex norm must use f32");
        let residual = residual
            .into_data()
            .try_to_vec::<f32>()
            .expect("Flex residual must use f32");
        let output = self.forward_host_time_major(&input, &residual, batch, frames, fused);
        Tensor::from_data(TensorData::new(output, [batch, frames, channels]), &device)
            .swap_dims(1, 2)
    }

    pub(crate) fn forward_host(
        &self,
        input: &[f32],
        residual: &[f32],
        batch: usize,
        channels: usize,
        frames: usize,
    ) -> Vec<f32> {
        assert_eq!(channels, self.first.in_channels);
        let temporal =
            self.forward_host_time_major(input, residual, batch, frames, self.fused_epilogue);
        let mut output = vec![0.0f32; batch * channels * frames];
        for b in 0..batch {
            for c in 0..channels {
                for t in 0..frames {
                    output[(b * channels + c) * frames + t] =
                        temporal[(b * frames + t) * channels + c];
                }
            }
        }
        output
    }

    fn forward_host_time_major(
        &self,
        input: &[f32],
        residual: &[f32],
        batch: usize,
        frames: usize,
        fused: bool,
    ) -> Vec<f32> {
        let channels = self.first.in_channels;
        assert_eq!(input.len(), batch * frames * channels);
        assert_eq!(residual.len(), batch * channels * frames);
        let rows = batch * frames;
        let timed = std::env::var_os("SILERO_VOCODER_TIMING").is_some();
        let start = Instant::now();
        let mut hidden = if fused {
            self.first.forward_raw(&input, rows)
        } else {
            self.first.forward(&input, rows)
        };
        if timed {
            eprintln!(
                "[vocoder-block] pwconv1_ms={:.3}",
                start.elapsed().as_secs_f64() * 1000.0
            );
        }
        let start = Instant::now();
        let parallel = std::env::var_os("SILERO_SERIAL_GELU").is_none();
        if fused {
            bias_gelu_epilogue(&mut hidden, &self.first.bias, parallel);
        } else {
            apply_fast_gelu(&mut hidden, parallel);
        }
        if timed {
            eprintln!(
                "[vocoder-block] gelu_ms={:.3}",
                start.elapsed().as_secs_f64() * 1000.0
            );
        }
        let start = Instant::now();
        let mut output = if fused {
            self.second.forward_raw(&hidden, rows)
        } else {
            self.second.forward(&hidden, rows)
        };
        if timed {
            eprintln!(
                "[vocoder-block] pwconv2_ms={:.3}",
                start.elapsed().as_secs_f64() * 1000.0
            );
        }
        if fused {
            bias_gamma_residual_epilogue(
                &mut output,
                &self.second.bias,
                &self.gamma,
                &residual,
                batch,
                frames,
                parallel,
            );
        } else {
            for b in 0..batch {
                for t in 0..frames {
                    for c in 0..channels {
                        let i = (b * frames + t) * channels + c;
                        output[i] =
                            output[i] * self.gamma[c] + residual[(b * channels + c) * frames + t];
                    }
                }
            }
        }
        output
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use burn::tensor::module::linear;

    #[test]
    fn direct_weights_match_transposed_weights() {
        let device = Device::flex();
        for (rows, in_channels, out_channels) in [
            (1, 9, 17),
            (7, 16, 32),
            (97, 512, 1536),
            (173, 1536, 512),
            (257, 512, 1536),
        ] {
            let values = (0..in_channels * out_channels)
                .map(|i| ((i * 29 % 211) as f32 - 105.0) * 0.0004)
                .collect();
            let weights = Tensor::<2>::from_data(
                TensorData::new(values, [in_channels, out_channels]),
                &device,
            );
            let bias = Tensor::<1>::from_data(
                TensorData::new(
                    (0..out_channels).map(|i| i as f32 * 0.001).collect(),
                    [out_channels],
                ),
                &device,
            );
            let normal = Projection::new_with_layout(weights.clone(), bias.clone(), false).unwrap();
            let direct = Projection::new_with_layout(weights, bias, true).unwrap();
            let input: Vec<f32> = (0..rows * in_channels)
                .map(|i| ((i * 17 % 97) as f32 - 48.0) * 0.002)
                .collect();
            let a = normal.forward_raw(&input, rows);
            let b = direct.forward_raw(&input, rows);
            let max_error = a
                .iter()
                .zip(b.iter())
                .map(|(a, b)| (a - b).abs())
                .fold(0.0f32, f32::max);
            assert!(
                max_error < 0.0001,
                "{rows}x{in_channels}x{out_channels}: {max_error}"
            );
        }
    }

    #[test]
    fn fused_epilogues_match_staged_operations_bitwise() {
        let rows = 67usize;
        let channels = 512usize;
        let hidden = 1536usize;
        let mut original = (0..rows * hidden)
            .map(|i| ((i * 13 % 511) as f32 - 255.0) * 0.004)
            .collect::<Vec<_>>();
        let mut combined = original.clone();
        let bias = (0..hidden)
            .map(|i| ((i * 17 % 97) as f32 - 48.0) * 0.002)
            .collect::<Vec<_>>();
        for row in original.chunks_mut(hidden) {
            for (v, b) in row.iter_mut().zip(&bias) {
                *v += *b;
            }
        }
        apply_fast_gelu(&mut original, true);
        bias_gelu_epilogue(&mut combined, &bias, true);
        assert!(
            original
                .iter()
                .zip(&combined)
                .all(|(a, b)| a.to_bits() == b.to_bits())
        );

        let mut original = (0..rows * channels)
            .map(|i| ((i * 19 % 499) as f32 - 249.0) * 0.006)
            .collect::<Vec<_>>();
        let mut combined = original.clone();
        let bias = (0..channels)
            .map(|i| (i as f32 - 256.0) * 0.0003)
            .collect::<Vec<_>>();
        let gamma = (0..channels)
            .map(|i| 0.001 * (i + 1) as f32)
            .collect::<Vec<_>>();
        let residual = (0..rows * channels)
            .map(|i| ((i * 13 % 29) as f32 - 14.0) * 0.02)
            .collect::<Vec<_>>();
        for row in original.chunks_mut(channels) {
            for (v, b) in row.iter_mut().zip(&bias) {
                *v += *b;
            }
        }
        for t in 0..rows {
            for c in 0..channels {
                let i = t * channels + c;
                original[i] = original[i] * gamma[c] + residual[c * rows + t];
            }
        }
        bias_gamma_residual_epilogue(&mut combined, &bias, &gamma, &residual, 1, rows, true);
        assert!(
            original
                .iter()
                .zip(&combined)
                .all(|(a, b)| a.to_bits() == b.to_bits())
        );
    }

    #[test]
    fn native_projection_matches_tensor_linear_for_uneven_rows() {
        let device = Device::flex();
        for (rows, in_channels, out_channels) in [(1, 8, 12), (7, 32, 48), (33, 64, 24)] {
            let weights = (0..in_channels * out_channels)
                .map(|i| ((i * 19 % 103) as f32 - 51.0) * 0.007)
                .collect::<Vec<_>>();
            let biases = (0..out_channels)
                .map(|i| (i as f32 - 11.0) * 0.001)
                .collect::<Vec<_>>();
            let input = (0..rows * in_channels)
                .map(|i| ((i * 13 % 109) as f32 - 54.0) * 0.009)
                .collect::<Vec<_>>();
            let weight = Tensor::<2>::from_data(
                TensorData::new(weights, [out_channels, in_channels]),
                &device,
            )
            .transpose();
            let bias = Tensor::<1>::from_data(TensorData::new(biases, [out_channels]), &device);
            let expected = linear(
                Tensor::<2>::from_data(
                    TensorData::new(input.clone(), [rows, in_channels]),
                    &device,
                ),
                weight.clone(),
                Some(bias.clone()),
            )
            .into_data()
            .try_to_vec::<f32>()
            .unwrap();
            let actual = Projection::new(weight, bias).unwrap().forward(&input, rows);
            assert_eq!(expected.len(), actual.len());
            for (a, b) in expected.into_iter().zip(actual) {
                assert!(
                    (a - b).abs() < 1e-5,
                    "Native pointwise mismatch: expected {a}, got {b}"
                );
            }
        }
    }

    #[test]
    fn fused_pointwise_matches_tensor_path_with_multiple_batches() {
        let device = Device::flex();
        let (batch, frames, channels, hidden) = (2, 7, 8, 20);
        let norm_data = (0..batch * frames * channels)
            .map(|i| ((i * 7 % 37) as f32 - 18.0) * 0.035)
            .collect::<Vec<_>>();
        let residual_data = (0..batch * channels * frames)
            .map(|i| ((i * 11 % 41) as f32 - 20.0) * 0.02)
            .collect::<Vec<_>>();
        let w1 = Tensor::<2>::from_data(
            TensorData::new(
                (0..hidden * channels)
                    .map(|i| ((i * 3 % 53) as f32 - 26.0) * 0.01)
                    .collect::<Vec<_>>(),
                [hidden, channels],
            ),
            &device,
        )
        .transpose();
        let b1 = Tensor::<1>::from_data(TensorData::new(vec![0.03; hidden], [hidden]), &device);
        let w2 = Tensor::<2>::from_data(
            TensorData::new(
                (0..channels * hidden)
                    .map(|i| ((i * 5 % 47) as f32 - 23.0) * 0.008)
                    .collect::<Vec<_>>(),
                [channels, hidden],
            ),
            &device,
        )
        .transpose();
        let b2 = Tensor::<1>::from_data(TensorData::new(vec![0.02; channels], [channels]), &device);
        let gamma =
            Tensor::<1>::from_data(TensorData::new(vec![0.2; channels], [channels]), &device);
        let norm = Tensor::<3>::from_data(
            TensorData::new(norm_data, [batch, frames, channels]),
            &device,
        );
        let residual = Tensor::<3>::from_data(
            TensorData::new(residual_data, [batch, channels, frames]),
            &device,
        );
        let mut expected_hidden = linear(norm.clone(), w1.clone(), Some(b1.clone()))
            .into_data()
            .try_to_vec::<f32>()
            .unwrap();
        apply_fast_gelu(&mut expected_hidden, false);
        let expected = linear(
            Tensor::<3>::from_data(
                TensorData::new(expected_hidden, [batch, frames, hidden]),
                &device,
            ),
            w2.clone(),
            Some(b2.clone()),
        )
        .mul(gamma.clone().reshape([1, 1, channels]))
        .swap_dims(1, 2)
            + residual.clone();
        let got = FlexPointwise::new(w1, b1, w2, b2, gamma)
            .unwrap()
            .forward(norm, residual);
        assert_eq!(expected.dims(), got.dims());
        let expected = expected.into_data().try_to_vec::<f32>().unwrap();
        let got = got.into_data().try_to_vec::<f32>().unwrap();
        for (a, b) in expected.into_iter().zip(got) {
            assert!(
                (a - b).abs() < 1e-5,
                "Fused pointwise mismatch: expected {a}, got {b}"
            );
        }
    }
}

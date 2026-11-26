use anyhow::{Result, ensure};
use burn::tensor::{Tensor, TensorData};
use rayon::prelude::*;

/// Dense same-length FP32 convolution for English HiFi-GAN. The residue classes
/// of a dilated convolution are independent contiguous sequences. Each padded
/// sequence is a read-only sliding-window GEMM operand, without im2col.
pub(crate) struct FlexEnglishConv {
    channels: usize,
    kernel: usize,
    dilation: usize,
    weight: Vec<f32>,
    bias: Vec<f32>,
}

impl FlexEnglishConv {
    pub(crate) fn new(weight: Tensor<3>, bias: Tensor<1>, dilation: usize) -> Result<Self> {
        let [channels, inputs, kernel] = weight.dims();
        ensure!(
            channels == inputs && channels > 0 && kernel > 0 && kernel % 2 == 1
                && dilation > 0 && bias.dims() == [channels],
            "Invalid English residual convolution"
        );
        let source = weight.into_data().try_into_vec::<f32>()?;
        let mut weight = vec![0.0; source.len()];
        for o in 0..channels {
            for k in 0..kernel {
                for c in 0..channels {
                    weight[(k * channels + c) * channels + o] =
                        source[(o * channels + c) * kernel + k];
                }
            }
        }
        Ok(Self {
            channels,
            kernel,
            dilation,
            weight,
            bias: bias.into_data().try_into_vec()?,
        })
    }

    fn residual(&self, input: &[f32], frames: usize) -> Vec<f32> {
        let c = self.channels;
        let d = self.dilation;
        let pad = self.kernel / 2;
        assert_eq!(input.len(), frames * c);
        let mut output = vec![0.0; input.len()];
        if frames == 0 {
            return output;
        }
        let mut scratch = vec![0.0f32; (frames.div_ceil(d) + 2 * pad) * c];
        for phase in 0..d.min(frames) {
            let rows = (frames - phase).div_ceil(d);
            let padded = &mut scratch[..(rows + 2 * pad) * c];
            padded[..pad * c].fill(0.0);
            padded[(pad + rows) * c..].fill(0.0);
            padded[pad * c..(pad + rows) * c]
                .par_chunks_mut(c)
                .enumerate()
                .for_each(|(i, dst)| {
                    let start = (phase + i * d) * c;
                    for (v, &x) in dst.iter_mut().zip(&input[start..start + c]) {
                        *v = if x >= 0.0 { x } else { x * 0.1 };
                    }
                });
            // SAFETY: A is a column-major [C,K*C] matrix. B is a read-only
            // overlapping [K*C,rows] view of padded; its final window ends at
            // padded.len(). Output columns are disjoint phase frames and the
            // final column ends at or before output.len(). Buffers do not alias.
            unsafe {
                gemm::gemm(
                    c, rows, self.kernel * c,
                    output.as_mut_ptr().add(phase * c), (d * c) as isize, 1, false,
                    self.weight.as_ptr(), c as isize, 1,
                    padded.as_ptr(), c as isize, 1,
                    0.0f32, 1.0f32, false, false, false, gemm::Parallelism::Rayon(0),
                );
            }
        }
        output.par_chunks_mut(c).zip(input.par_chunks(c)).for_each(|(dst, src)| {
            for ((v, &x), &b) in dst.iter_mut().zip(src).zip(&self.bias) {
                *v = (*v + b) + x;
            }
        });
        output
    }
}

fn transpose(input: &[f32], rows: usize, cols: usize) -> Vec<f32> {
    assert_eq!(input.len(), rows * cols);
    let mut output = vec![0.0; input.len()];
    if input.is_empty() {
        return output;
    }
    output.par_chunks_mut(rows * 32).enumerate().for_each(|(tile, dst)| {
        let start = tile * 32;
        let width = cols.min(start + 32) - start;
        for r in (0..rows).step_by(32) {
            for j in 0..width {
                for i in r..rows.min(r + 32) {
                    dst[j * rows + i] = input[i * cols + start + j];
                }
            }
        }
    });
    output
}

pub(crate) struct FlexEnglishResidualStage(pub [[FlexEnglishConv; 2]; 3]);

impl FlexEnglishResidualStage {
    fn forward_batch(&self, src: &[f32], channels: usize, frames: usize) -> Vec<f32> {
        let x = transpose(src, channels, frames);
        let [a, b, c] = &self.0;
        let run = |block: &[FlexEnglishConv; 2]| {
            block[1].residual(&block[0].residual(&x, frames), frames)
        };
        let mut sum = run(a);
        for block in [b, c] {
            let part = run(block);
            sum.par_iter_mut().zip(&part).for_each(|(s, p)| *s += *p);
        }
        sum.par_iter_mut().for_each(|v| *v /= 3.0);
        transpose(&sum, frames, channels)
    }

    pub(crate) fn forward(&self, input: Tensor<3>) -> Tensor<3> {
        let [batch, channels, frames] = input.dims();
        let device = input.device();
        let input = input.into_data().try_into_vec::<f32>().expect("English Flex requires FP32");
        let result = if input.is_empty() {
            Vec::new()
        } else if batch == 1 {
            self.forward_batch(&input, channels, frames)
        } else {
            let mut result = Vec::with_capacity(input.len());
            for src in input.chunks_exact(channels * frames) {
                result.extend(self.forward_batch(src, channels, frames));
            }
            result
        };
        Tensor::from_data(TensorData::new(result, [batch, channels, frames]), &device)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use burn::tensor::{Device, activation::leaky_relu, module::conv1d, ops::ConvOptions};

    fn data(count: usize, seed: usize, scale: f32) -> Vec<f32> {
        (0..count).map(|i| (((i * 37 + seed * 53) % 101) as f32 - 50.0) * scale).collect()
    }

    fn conv(channels: usize, kernel: usize, dilation: usize, seed: usize) -> (FlexEnglishConv, Tensor<3>, Tensor<1>) {
        let device = Device::flex();
        let w = Tensor::from_data(TensorData::new(data(channels * channels * kernel, seed, 0.0007), [channels, channels, kernel]), &device);
        let b = Tensor::from_data(TensorData::new(data(channels, seed + 1, 0.001), [channels]), &device);
        (FlexEnglishConv::new(w.clone(), b.clone(), dilation).unwrap(), w, b)
    }

    #[test]
    fn phase_residual_matches_tensor_convolution_at_boundaries() {
        let device = Device::flex();
        for channels in [1, 3, 17, 64] {
            for kernel in [3, 7, 11] {
                for dilation in [1, 3] {
                    let (native, w, b) = conv(channels, kernel, dilation, 1);
                    for frames in [1, 2, 3, 4, 17, 31, 32, 33, 129] {
                        let host = data(channels * frames, 3, 0.012);
                        let x = Tensor::from_data(TensorData::new(transpose(&host, frames, channels), [1, channels, frames]), &device);
                        let expected = (conv1d(leaky_relu(x.clone(), 0.1), w.clone(), Some(b.clone()), ConvOptions::new([1], [(kernel / 2) * dilation], [dilation], 1)) + x).into_data().try_into_vec::<f32>().unwrap();
                        let actual = transpose(&native.residual(&host, frames), frames, channels);
                        let error = actual.iter().zip(expected).map(|(a,b)| (a-b).abs()).fold(0.0f32, f32::max);
                        assert!(error < 2e-4, "C={channels} K={kernel} D={dilation} T={frames}: {error}");
                    }
                }
            }
        }
    }

    #[test]
    fn complete_stage_matches_three_tensor_branches_for_batch_two() {
        let device = Device::flex();
        let (batch, channels, frames) = (2, 7, 65);
        let x = Tensor::from_data(TensorData::new(data(batch * channels * frames, 5, 0.009), [batch, channels, frames]), &device);
        let mut reference = Vec::new();
        let blocks = [3, 7, 11].map(|kernel| {
            let (a, aw, ab) = conv(channels, kernel, 1, kernel);
            let (b, bw, bb) = conv(channels, kernel, 3, kernel + 1);
            let y = conv1d(leaky_relu(x.clone(), 0.1), aw, Some(ab), ConvOptions::new([1], [kernel / 2], [1], 1)) + x.clone();
            let z = conv1d(leaky_relu(y.clone(), 0.1), bw, Some(bb), ConvOptions::new([1], [kernel / 2 * 3], [3], 1)) + y;
            reference.push(z);
            [a, b]
        });
        let expected = ((reference[0].clone() + reference[1].clone() + reference[2].clone()) / 3.0).into_data().try_into_vec::<f32>().unwrap();
        let actual = FlexEnglishResidualStage(blocks).forward(x).into_data().try_into_vec::<f32>().unwrap();
        assert_eq!(actual.len(), expected.len());
        let error = actual.iter().zip(expected).map(|(a,b)| (a-b).abs()).fold(0.0f32, f32::max);
        assert!(error < 2e-4, "stage error {error}");
    }
}

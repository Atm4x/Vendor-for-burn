use anyhow::{Result, ensure};
use burn::tensor::{Tensor, TensorData};
use rayon::prelude::*;

const KERNEL: usize = 9;
const PAD: usize = KERNEL / 2;

/// Native channels-last transformer feed-forward residual. The first GEMM
/// reads overlapping windows of the padded input; the second is pointwise.
pub(crate) struct FlexTransformerFfn {
    hidden: usize,
    inner: usize,
    first: Vec<f32>,
    second: Vec<f32>,
    first_bias: Vec<f32>,
    second_bias: Vec<f32>,
}

fn parallelism(work: usize) -> gemm::Parallelism {
    if work >= 192 * 192 * 192 {
        gemm::Parallelism::Rayon(0)
    } else {
        gemm::Parallelism::None
    }
}

impl FlexTransformerFfn {
    pub(crate) fn new(
        first: Tensor<3>,
        first_bias: Tensor<1>,
        second: Tensor<3>,
        second_bias: Tensor<1>,
    ) -> Result<Self> {
        let [inner, hidden, kernel] = first.dims();
        ensure!(
            hidden > 0
                && inner > 0
                && kernel == KERNEL
                && first_bias.dims() == [inner]
                && second.dims() == [hidden, inner, 1]
                && second_bias.dims() == [hidden],
            "Invalid transformer FFN shapes"
        );
        let source = first.into_data().try_into_vec::<f32>()?;
        let mut first = vec![0.0; source.len()];
        for o in 0..inner {
            for k in 0..KERNEL {
                for c in 0..hidden {
                    first[(k * hidden + c) * inner + o] = source[(o * hidden + c) * KERNEL + k];
                }
            }
        }
        let source = second.into_data().try_into_vec::<f32>()?;
        let mut second = vec![0.0; source.len()];
        for o in 0..hidden {
            for c in 0..inner {
                second[c * hidden + o] = source[o * inner + c];
            }
        }
        Ok(Self {
            hidden,
            inner,
            first,
            second,
            first_bias: first_bias.into_data().try_into_vec()?,
            second_bias: second_bias.into_data().try_into_vec()?,
        })
    }

    fn apply(
        &self,
        input: &[f32],
        output: &mut [f32],
        padded: &mut [f32],
        intermediate: &mut [f32],
        frames: usize,
    ) {
        let (h, f) = (self.hidden, self.inner);
        assert_eq!(input.len(), frames * h);
        assert_eq!(output.len(), frames * h);
        assert_eq!(padded.len(), (frames + 2 * PAD) * h);
        assert_eq!(intermediate.len(), frames * f);
        if frames == 0 {
            return;
        }
        padded[PAD * h..(PAD + frames) * h].copy_from_slice(input);
        // SAFETY: first is [F,K*H] column-major, padded is a read-only
        // [K*H,T] overlapping window matrix with column stride H. The last
        // window ends at padded.len(). Destination columns do not overlap.
        unsafe {
            gemm::gemm(
                f,
                frames,
                KERNEL * h,
                intermediate.as_mut_ptr(),
                f as isize,
                1,
                false,
                self.first.as_ptr(),
                f as isize,
                1,
                padded.as_ptr(),
                h as isize,
                1,
                0.0f32,
                1.0f32,
                false,
                false,
                false,
                parallelism(f * frames * KERNEL * h),
            );
        }
        let relu = |row: &mut [f32]| {
            for (v, &b) in row.iter_mut().zip(&self.first_bias) {
                *v = (*v + b).max(0.0);
            }
        };
        if intermediate.len() >= 32_768 {
            intermediate.par_chunks_mut(f).for_each(relu);
        } else {
            intermediate.chunks_mut(f).for_each(relu);
        }
        // SAFETY: second is [H,F], intermediate is [F,T], output is [H,T];
        // all contiguous column-major matrices with independent storage.
        unsafe {
            gemm::gemm(
                h,
                frames,
                f,
                output.as_mut_ptr(),
                h as isize,
                1,
                false,
                self.second.as_ptr(),
                h as isize,
                1,
                intermediate.as_ptr(),
                f as isize,
                1,
                0.0f32,
                1.0f32,
                false,
                false,
                false,
                parallelism(h * frames * f),
            );
        }
        let residual = |(row, source): (&mut [f32], &[f32])| {
            for ((v, &b), &x) in row.iter_mut().zip(&self.second_bias).zip(source) {
                *v = (*v + b) + x;
            }
        };
        if output.len() >= 32_768 {
            output
                .par_chunks_mut(h)
                .zip(input.par_chunks(h))
                .for_each(residual);
        } else {
            output.chunks_mut(h).zip(input.chunks(h)).for_each(residual);
        }
    }

    pub(crate) fn forward(&self, input: Tensor<3>) -> Tensor<3> {
        let [batch, frames, hidden] = input.dims();
        assert_eq!(hidden, self.hidden);
        let device = input.device();
        let input = input
            .into_data()
            .try_into_vec::<f32>()
            .expect("Flex transformer FFN requires FP32");
        let mut output = vec![0.0; input.len()];
        if frames != 0 {
            let mut padded = vec![0.0; (frames + 2 * PAD) * hidden];
            let mut intermediate = vec![0.0; frames * self.inner];
            for (source, dest) in input
                .chunks_exact(frames * hidden)
                .zip(output.chunks_exact_mut(frames * hidden))
            {
                self.apply(source, dest, &mut padded, &mut intermediate, frames);
            }
        }
        Tensor::from_data(TensorData::new(output, [batch, frames, hidden]), &device)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use burn::tensor::{Device, activation::relu, module::conv1d, ops::ConvOptions};

    fn data(n: usize, seed: usize, scale: f32) -> Vec<f32> {
        (0..n)
            .map(|i| (((i * 37 + seed * 53) % 101) as f32 - 50.0) * scale)
            .collect()
    }

    #[test]
    fn fused_ffn_matches_tensor_graph_with_batch_and_padding() {
        let device = Device::flex();
        for (hidden, inner) in [(1, 1), (3, 7), (64, 64), (128, 512)] {
            let a = Tensor::from_data(
                TensorData::new(
                    data(inner * hidden * KERNEL, 1, 0.0007),
                    [inner, hidden, KERNEL],
                ),
                &device,
            );
            let ab = Tensor::from_data(TensorData::new(data(inner, 2, 0.001), [inner]), &device);
            let b = Tensor::from_data(
                TensorData::new(data(hidden * inner, 3, 0.0007), [hidden, inner, 1]),
                &device,
            );
            let bb = Tensor::from_data(TensorData::new(data(hidden, 4, 0.001), [hidden]), &device);
            let native =
                FlexTransformerFfn::new(a.clone(), ab.clone(), b.clone(), bb.clone()).unwrap();
            for batch in [1, 2] {
                for frames in [1, 2, 4, 9, 17, 33, 129] {
                    let x = Tensor::from_data(
                        TensorData::new(
                            data(batch * frames * hidden, 5, 0.01),
                            [batch, frames, hidden],
                        ),
                        &device,
                    );
                    let y = relu(conv1d(
                        x.clone().swap_dims(1, 2),
                        a.clone(),
                        Some(ab.clone()),
                        ConvOptions::new([1], [PAD], [1], 1),
                    ));
                    let expected = (x.clone()
                        + conv1d(
                            y,
                            b.clone(),
                            Some(bb.clone()),
                            ConvOptions::new([1], [0], [1], 1),
                        )
                        .swap_dims(1, 2))
                    .into_data()
                    .try_into_vec::<f32>()
                    .unwrap();
                    let actual = native.forward(x).into_data().try_into_vec::<f32>().unwrap();
                    assert_eq!(expected.len(), actual.len());
                    let max = actual
                        .iter()
                        .zip(expected)
                        .map(|(a, b)| (a - b).abs())
                        .fold(0.0f32, f32::max);
                    assert!(
                        max < 2e-4,
                        "H={hidden} F={inner} T={frames} B={batch}: {max}"
                    );
                }
            }
        }
    }
}

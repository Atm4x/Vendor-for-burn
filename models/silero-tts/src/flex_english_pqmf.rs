use anyhow::{Result, ensure};
use burn::tensor::{Tensor, TensorData};

const BANDS: usize = 4;
const PAD: usize = 8;
const WINDOW: usize = 2 * PAD + 1;
const TAPS: usize = 63;

/// Combines the stride-four synthesis upsampler and FIR into a four-phase
/// filter bank. Each column of the GEMM represents four consecutive samples.
pub(crate) struct FlexEnglishPqmf {
    weights: [f32; BANDS * WINDOW * BANDS],
}

impl FlexEnglishPqmf {
    pub(crate) fn new(synthesis: Tensor<3>, updown: Tensor<3>) -> Result<Self> {
        ensure!(
            synthesis.dims() == [1, BANDS, TAPS] && updown.dims() == [BANDS, BANDS, BANDS],
            "Invalid PQMF shapes"
        );
        let g = synthesis.into_data().try_into_vec::<f32>()?;
        let u = updown.into_data().try_into_vec::<f32>()?;
        let mut weights = [0.0; BANDS * WINDOW * BANDS];
        for offset in 0..WINDOW {
            let delta = offset as isize - PAD as isize;
            for i in 0..BANDS {
                for p in 0..BANDS {
                    let mut sum = 0.0;
                    for q in 0..BANDS {
                        let k =
                            delta * BANDS as isize + q as isize - p as isize + (TAPS / 2) as isize;
                        if !(0..TAPS as isize).contains(&k) {
                            continue;
                        }
                        for o in 0..BANDS {
                            sum += g[o * TAPS + k as usize] * u[(i * BANDS + o) * BANDS + q];
                        }
                    }
                    weights[(offset * BANDS + i) * BANDS + p] = sum;
                }
            }
        }
        Ok(Self { weights })
    }

    fn apply(&self, input: &[f32], output: &mut [f32], scratch: &mut [f32], frames: usize) {
        assert_eq!(input.len(), BANDS * frames);
        assert_eq!(output.len(), BANDS * frames);
        assert_eq!(scratch.len(), (frames + 2 * PAD) * BANDS);
        for (t, row) in scratch[PAD * BANDS..(PAD + frames) * BANDS]
            .chunks_exact_mut(BANDS)
            .enumerate()
        {
            for c in 0..BANDS {
                row[c] = input[c * frames + t];
            }
        }
        if frames == 0 {
            return;
        }
        // SAFETY: weights are [4,68] column-major; scratch is padded by eight
        // frames at both ends. Its overlapping windows are read-only and the
        // final 68-element window ends at scratch.len(). Output is disjoint.
        unsafe {
            gemm::gemm(
                BANDS,
                frames,
                BANDS * WINDOW,
                output.as_mut_ptr(),
                BANDS as isize,
                1,
                false,
                self.weights.as_ptr(),
                BANDS as isize,
                1,
                scratch.as_ptr(),
                BANDS as isize,
                1,
                0.0f32,
                1.0f32,
                false,
                false,
                false,
                if frames >= 8192 {
                    gemm::Parallelism::Rayon(0)
                } else {
                    gemm::Parallelism::None
                },
            );
        }
    }

    pub(crate) fn forward(&self, input: Tensor<3>) -> Tensor<3> {
        let [batch, bands, frames] = input.dims();
        assert_eq!(bands, BANDS);
        let device = input.device();
        let input = input
            .into_data()
            .try_into_vec::<f32>()
            .expect("English PQMF requires FP32");
        let mut result = vec![0.0; input.len()];
        if frames != 0 {
            let mut scratch = vec![0.0; (frames + 2 * PAD) * BANDS];
            for (source, dest) in input
                .chunks_exact(BANDS * frames)
                .zip(result.chunks_exact_mut(BANDS * frames))
            {
                self.apply(source, dest, &mut scratch, frames);
            }
        }
        Tensor::from_data(TensorData::new(result, [batch, 1, frames * BANDS]), &device)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use burn::tensor::{
        Device,
        module::{conv_transpose1d, conv1d},
        ops::{ConvOptions, ConvTransposeOptions},
    };

    fn data(n: usize, seed: usize) -> Vec<f32> {
        (0..n)
            .map(|i| (((i * 37 + seed * 53) % 101) as f32 - 50.0) * 0.004)
            .collect()
    }

    #[test]
    fn fused_polyphase_matches_dense_pqmf_at_all_boundaries() {
        let device = Device::flex();
        let g = Tensor::from_data(
            TensorData::new(data(BANDS * TAPS, 1), [1, BANDS, TAPS]),
            &device,
        );
        // Deliberately dense, non-diagonal upsampling weights: the algebra is
        // not restricted to the sparse filter used by the official checkpoint.
        let u = Tensor::from_data(TensorData::new(data(BANDS.pow(3), 2), [BANDS; 3]), &device);
        let fused = FlexEnglishPqmf::new(g.clone(), u.clone()).unwrap();
        for batch in [1, 2] {
            for frames in [1, 2, 7, 8, 9, 16, 17, 33, 129, 1025] {
                let x = Tensor::from_data(
                    TensorData::new(data(batch * BANDS * frames, 3), [batch, BANDS, frames]),
                    &device,
                );
                let upsampled = conv_transpose1d(
                    x.clone(),
                    u.clone(),
                    None,
                    ConvTransposeOptions::new([4], [0], [0], [1], 1),
                );
                let expected = conv1d(
                    upsampled,
                    g.clone(),
                    None,
                    ConvOptions::new([1], [31], [1], 1),
                )
                .into_data()
                .try_into_vec::<f32>()
                .unwrap();
                let actual = fused.forward(x).into_data().try_into_vec::<f32>().unwrap();
                assert_eq!(expected.len(), actual.len());
                let max = actual
                    .iter()
                    .zip(expected)
                    .map(|(a, b)| (a - b).abs())
                    .fold(0.0f32, f32::max);
                assert!(max < 1e-5, "batch={batch}, frames={frames}: {max}");
            }
        }
    }
}

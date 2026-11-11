use burn::tensor::{Device, Int, Tensor, activation};

use crate::checkpoint::RvcCheckpoint;

pub const F0_BINS: usize = 360;
const CENTS_START: f64 = 1997.3794084376191;
const CENTS_STEP: f64 = 20.0;

/// Center bin plus nine contiguous values (offsets -4..=4).
pub(crate) const COMPACT_F0_WIDTH: usize = 10;

/// Final RMVPE inference head after the bidirectional GRU.
pub struct RmvpeHead {
    weight: Tensor<2>,
    bias: Tensor<1>,
}

impl RmvpeHead {
    pub fn load(checkpoint: &RvcCheckpoint, device: &Device) -> Result<Self, String> {
        let weight = checkpoint.tensor::<2>("fc.1.weight", device)?;
        let bias = checkpoint.tensor::<1>("fc.1.bias", device)?;
        if weight.dims() != [F0_BINS, 512] || bias.dims() != [F0_BINS] {
            return Err("RMVPE output projection must have shape (360, 512) and bias (360)".into());
        }
        Ok(Self { weight, bias })
    }

    pub fn forward(&self, gru_output: Tensor<3>) -> Tensor<3> {
        let [batch, time, channels] = gru_output.dims();
        assert_eq!(
            channels, 512,
            "RMVPE bidirectional GRU must emit 512 features"
        );
        let projected = gru_output
            .reshape([batch * time, 512])
            .matmul(self.weight.clone().transpose())
            + self.bias.clone().reshape([1, F0_BINS]);
        activation::sigmoid(projected).reshape([batch, time, F0_BINS])
    }
}

/// Reproduces RMVPE's 9-bin weighted local-centroid F0 decoder, per frame.
/// `salience` is the 360-class sigmoid output from RMVPE.
pub fn salience_to_f0(salience: &[f32], threshold: f32) -> Result<Vec<f32>, String> {
    if salience.len() % F0_BINS != 0 || !threshold.is_finite() || threshold < 0.0 {
        return Err(
            "RMVPE requires complete 360-bin frames and a finite nonnegative threshold".into(),
        );
    }
    let mut output = Vec::with_capacity(salience.len() / F0_BINS);
    for frame in salience.chunks_exact(F0_BINS) {
        if frame.iter().any(|value| !value.is_finite()) {
            return Err("RMVPE salience contains a non-finite value".into());
        }
        let mut center = 0;
        for index in 1..F0_BINS {
            if frame[index] > frame[center] {
                center = index;
            }
        }
        if frame[center] <= threshold {
            output.push(0.0);
            continue;
        }
        let start = center.saturating_sub(4);
        let end = (center + 4).min(F0_BINS - 1);
        output.push(decode_center_f0(
            (start..=end).map(|index| (index, frame[index])),
        ));
    }
    Ok(output)
}

/// Preserves the original CPU FP64 accumulation order; the input iterator must be ascending.
fn decode_center_f0(neighbors: impl Iterator<Item = (usize, f32)>) -> f32 {
    let mut weighted = 0.0_f64;
    let mut total = 0.0_f64;
    for (index, value) in neighbors {
        let value = value as f64;
        weighted += value * (CENTS_START + CENTS_STEP * index as f64);
        total += value;
    }
    let cents = if total > 0.0 { weighted / total } else { 0.0 };
    if cents == 0.0 {
        0.0
    } else {
        (10.0 * 2.0_f64.powf(cents / 1200.0)) as f32
    }
}

/// Compact RMVPE salience on-device, before host readback. The format is
/// [center, value(center-4), ..., value(center+4)], with boundary samples clamped
/// in the gather and ignored by the CPU decoder. A nonfinite input frame has
/// center=-1, so the decoder still rejects NaN/Inf anywhere in all 360 bins.
/// A tie is resolved by the lowest bin index, independently of backend argmax behavior.
pub(crate) fn compact_salience(salience: Tensor<3>) -> Tensor<3> {
    let [batch, frames, bins] = salience.dims();
    assert_eq!(bins, F0_BINS, "RMVPE salience must have 360 bins");
    let device = salience.device();
    let maxima = salience.clone().max_dim(2);
    let is_max = salience.clone().equal(maxima.expand([batch, frames, bins]));
    let indices = Tensor::<1, Int>::arange(0..F0_BINS as i64, &device)
        .reshape([1, 1, F0_BINS])
        .expand([batch, frames, bins]);
    let centers = indices
        .mask_fill(is_max.bool_not(), F0_BINS as i64)
        .min_dim(2);
    let finite = salience.clone().is_finite().all_dim(2);
    let centers = centers.mask_fill(finite.bool_not(), -1_i64);
    let offsets = Tensor::<1, Int>::from_ints([-4, -3, -2, -1, 0, 1, 2, 3, 4], &device)
        .reshape([1, 1, 9])
        .expand([batch, frames, 9]);
    let gather_indices =
        (centers.clone().repeat_dim(2, 9) + offsets).clamp(0, (F0_BINS - 1) as i64);
    let values = salience.gather(2, gather_indices);
    Tensor::cat(vec![centers.float(), values], 2)
}

/// Decode a compact GPU salience frame on CPU, retaining the original FP64 formula.
pub(crate) fn compact_salience_to_f0(compact: &[f32], threshold: f32) -> Result<Vec<f32>, String> {
    if compact.len() % COMPACT_F0_WIDTH != 0 || !threshold.is_finite() || threshold < 0.0 {
        return Err(
            "RMVPE compact F0 requires complete frames and a finite nonnegative threshold".into(),
        );
    }
    compact
        .chunks_exact(COMPACT_F0_WIDTH)
        .map(|frame| {
            let center = frame[0];
            if !center.is_finite()
                || center < 0.0
                || center >= F0_BINS as f32
                || center.fract() != 0.0
            {
                return Err(
                    "RMVPE salience contains a non-finite value or invalid compact center".into(),
                );
            }
            let center = center as usize;
            if !frame[1..].iter().all(|v| v.is_finite()) {
                return Err("RMVPE compact salience contains a non-finite value".into());
            }
            if frame[5] <= threshold {
                return Ok(0.0);
            }
            let neighbors = (-4isize..=4).filter_map(|offset| {
                let index = center as isize + offset;
                (index >= 0 && index < F0_BINS as isize)
                    .then_some((index as usize, frame[(offset + 5) as usize]))
            });
            Ok(decode_center_f0(neighbors))
        })
        .collect()
}

#[cfg(test)]
mod compact_tests {
    use super::*;
    use burn::tensor::{Device, TensorData};

    #[test]
    fn compact_salience_matches_dense_cpu_on_boundaries_ties_and_noise() {
        let device = Device::flex();
        for batch in [1, 2, 4] {
            for frames in [1, 2, 7, 31, 33] {
                let mut values = Vec::with_capacity(batch * frames * F0_BINS);
                for row in 0..batch * frames {
                    let mut data: Vec<f32> = (0..F0_BINS)
                        .map(|bin| (((bin * 97 + row * 17) % 331) as f32 + 1.0) / 700.0)
                        .collect();
                    match row % 7 {
                        0 => {
                            data[0] = 1.0;
                            data[359] = 0.9;
                        }
                        1 => {
                            data[359] = 1.0;
                        }
                        2 => {
                            data[0] = 0.99;
                            data[359] = 0.99;
                        }
                        3 => {
                            data.fill(0.0);
                        }
                        4 => {
                            data[1] = 1.0;
                            data[4] = 1.0;
                        }
                        5 => {
                            data[358] = 1.0;
                        }
                        _ => {
                            data[179] = 1.0;
                        }
                    }
                    values.extend(data);
                }
                let tensor = Tensor::<3>::from_data(
                    TensorData::new(values.clone(), [batch, frames, F0_BINS]),
                    &device,
                );
                let compact = compact_salience(tensor)
                    .into_data()
                    .try_into_vec::<f32>()
                    .unwrap();
                assert_eq!(compact.len(), batch * frames * COMPACT_F0_WIDTH);
                for threshold in [0.0, 0.03, 0.5, 1.0] {
                    let dense = salience_to_f0(&values, threshold).unwrap();
                    let sparse = compact_salience_to_f0(&compact, threshold).unwrap();
                    assert!(
                        dense
                            .iter()
                            .zip(&sparse)
                            .all(|(a, b)| a.to_bits() == b.to_bits()),
                        "batch={batch} frames={frames} threshold={threshold:?}"
                    );
                }
                for (row, packed) in compact.chunks_exact(COMPACT_F0_WIDTH).enumerate() {
                    let input = &values[row * F0_BINS..(row + 1) * F0_BINS];
                    let expected = input
                        .iter()
                        .position(|&v| v == input.iter().copied().fold(f32::NEG_INFINITY, f32::max))
                        .unwrap();
                    assert_eq!(packed[0], expected as f32, "row={row}");
                }
            }
        }
    }

    #[test]
    #[ignore = "opt-in CPU-only readback benchmark; GPU runtime evidence required"]
    fn benchmark_compact_readback_vs_dense_cpu() {
        use std::hint::black_box;
        use std::time::Instant;
        let device = Device::flex();
        for batch in [1, 4] {
            for frames in [35, 70, 200, 1000] {
                let values: Vec<f32> = (0..batch * frames * F0_BINS)
                    .map(|i| ((i.wrapping_mul(619) % 991) as f32 + 1.0) / 992.0)
                    .collect();
                let tensor = Tensor::<3>::from_data(
                    TensorData::new(values, [batch, frames, F0_BINS]),
                    &device,
                );
                let mut dense_ms = Vec::new();
                let mut compact_ms = Vec::new();
                for repetition in 0..12 {
                    let compact_first = repetition % 2 == 0;
                    for use_compact in [compact_first, !compact_first] {
                        let instant = Instant::now();
                        let output = if use_compact {
                            compact_salience(tensor.clone())
                                .into_data()
                                .try_into_vec::<f32>()
                                .unwrap()
                        } else {
                            tensor.clone().into_data().try_into_vec::<f32>().unwrap()
                        };
                        black_box(output);
                        let ms = instant.elapsed().as_secs_f64() * 1000.0;
                        if use_compact {
                            compact_ms.push(ms)
                        } else {
                            dense_ms.push(ms)
                        }
                    }
                }
                dense_ms.sort_by(f64::total_cmp);
                compact_ms.sort_by(f64::total_cmp);
                eprintln!(
                    "RMVPE CPU B={batch} T={frames} dense_median_ms={:.4} compact_median_ms={:.4}",
                    dense_ms[6], compact_ms[6]
                );
            }
        }
    }

    #[test]
    fn compact_salience_rejects_nonfinite_anywhere_and_invalid_thresholds() {
        let device = Device::flex();
        for bad in [f32::NAN, f32::INFINITY, f32::NEG_INFINITY] {
            let mut values = vec![0.1f32; F0_BINS];
            values[200] = bad;
            let tensor = Tensor::<3>::from_data(TensorData::new(values, [1, 1, F0_BINS]), &device);
            let compact = compact_salience(tensor)
                .into_data()
                .try_into_vec::<f32>()
                .unwrap();
            assert_eq!(compact[0], -1.0);
            assert!(compact_salience_to_f0(&compact, 0.03).is_err());
        }
        assert!(compact_salience_to_f0(&[0.0, 1.0], 0.03).is_err());
        assert!(compact_salience_to_f0(&[0.0; COMPACT_F0_WIDTH], f32::NAN).is_err());
    }
}

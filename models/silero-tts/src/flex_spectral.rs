use burn::tensor::{Device, Tensor, TensorData};
use rayon::prelude::*;

/// Converts the channel-last log-magnitude/phase projection to three channel-first tensors.
pub(crate) fn split_spectrum(spectrum: Tensor<3>) -> (Tensor<3>, Tensor<3>, Tensor<3>) {
    let [batch, frames, channels] = spectrum.dims();
    assert!(channels % 2 == 0 && frames > 0);
    let bins = channels / 2;
    let source = spectrum
        .into_data()
        .try_to_vec::<f32>()
        .expect("Flex spectral projection must use f32");
    let mut mag = vec![0.0f32; batch * frames * bins];
    let mut cosine = vec![0.0f32; batch * frames * bins];
    let mut sine = vec![0.0f32; batch * frames * bins];

    let apply_row = |row: usize, mag: &mut [f32], cos: &mut [f32], sin: &mut [f32]| {
        let input = &source[row * channels..(row + 1) * channels];
        for k in 0..bins {
            mag[k] = input[k].exp().min(100.0);
            let (s, c) = input[bins + k].sin_cos();
            cos[k] = c;
            sin[k] = s;
        }
    };
    if mag.len() >= 32_768 {
        mag.par_chunks_mut(bins)
            .zip(cosine.par_chunks_mut(bins))
            .zip(sine.par_chunks_mut(bins))
            .enumerate()
            .for_each(|(row, ((mag, cos), sin))| apply_row(row, mag, cos, sin));
    } else {
        for (row, ((mag, cos), sin)) in mag
            .chunks_mut(bins)
            .zip(cosine.chunks_mut(bins))
            .zip(sine.chunks_mut(bins))
            .enumerate()
        {
            apply_row(row, mag, cos, sin);
        }
    }
    let device = Device::flex();
    let tensor = |data| {
        Tensor::from_data(TensorData::new(data, [batch, frames, bins]), &device).swap_dims(1, 2)
    };
    (tensor(mag), tensor(cosine), tensor(sine))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn fused_spectrum_matches_tensor_path() {
        let device = Device::flex();
        for (batch, frames, bins) in [(1, 5, 7), (2, 9, 11), (1, 64, 257)] {
            let channels = 2 * bins;
            let input: Vec<f32> = (0..batch * frames * channels)
                .map(|i| (((i * 971) % 9001) as f32 - 4500.) * 0.0007)
                .collect();
            let data =
                Tensor::<3>::from_data(TensorData::new(input, [batch, frames, channels]), &device);
            let swapped = data.clone().swap_dims(1, 2);
            let mag = swapped
                .clone()
                .slice([0..batch, 0..bins, 0..frames])
                .exp()
                .clamp_max(100.0);
            let phase = swapped.slice([0..batch, bins..channels, 0..frames]);
            let expected = [mag, phase.clone().cos(), phase.sin()];
            let (a, b, c) = split_spectrum(data);
            for (old, new) in expected.into_iter().zip([a, b, c]) {
                let reference = old.into_data().try_to_vec::<f32>().unwrap();
                let actual = new.into_data().try_to_vec::<f32>().unwrap();
                let max_error = reference
                    .into_iter()
                    .zip(actual)
                    .map(|(a, b)| (a - b).abs())
                    .fold(0.0f32, f32::max);
                assert!(max_error <= 0.00002, "max diff: {max_error}");
            }
        }
    }
}

use crate::{F5Error, Result};

const TARGET_RATE: usize = 24_000;
const LOWPASS_FILTER_WIDTH: f32 = 6.0;
const ROLLOFF: f32 = 0.99;

fn gcd(mut a: usize, mut b: usize) -> usize {
    while b != 0 {
        (a, b) = (b, a % b);
    }
    a
}

/// Numerically reproduces the default `torchaudio.functional.resample` Hann-windowed sinc kernel.
/// The reference frontend is always 24 kHz; no arbitrary-rate filter bank is stored in the model.
pub fn resample_to_24khz(input: &[f32], source_rate: usize) -> Result<Vec<f32>> {
    if source_rate == 0 || source_rate > 384_000 || input.is_empty() {
        return Err(F5Error::InvalidConfiguration(
            "Invalid source sample rate or empty audio",
        ));
    }
    if input.iter().any(|sample| !sample.is_finite()) {
        return Err(F5Error::InvalidConfiguration(
            "Reference audio contains non-finite samples",
        ));
    }
    if source_rate == TARGET_RATE {
        return Ok(input.to_vec());
    }
    let factor = gcd(source_rate, TARGET_RATE);
    let orig = source_rate / factor;
    let target = TARGET_RATE / factor;
    let base = orig.min(target) as f32 * ROLLOFF;
    let width = (LOWPASS_FILTER_WIDTH * orig as f32 / base).ceil() as usize;
    let kernel_width = 2 * width + orig;
    let estimated_coefficients =
        target
            .checked_mul(kernel_width)
            .ok_or(F5Error::InvalidConfiguration(
                "Resampling phase count overflow",
            ))?;
    if estimated_coefficients > 4_000_000 {
        return Err(F5Error::InvalidConfiguration(
            "Nonstandard source rate requires an excessively large polyphase filter",
        ));
    }
    let result_length = input
        .len()
        .checked_mul(target)
        .and_then(|length| length.checked_add(orig - 1))
        .ok_or(F5Error::InvalidShape("Resampled audio length overflow"))?
        / orig;
    let mut kernels = Vec::with_capacity(estimated_coefficients);
    for phase in 0..target {
        for index in 0..kernel_width {
            let offset = index as isize - width as isize;
            let time = ((offset as f32 / orig as f32) - (phase as f32 / target as f32)) * base;
            let time = time.clamp(-LOWPASS_FILTER_WIDTH, LOWPASS_FILTER_WIDTH);
            let window = (time * core::f32::consts::PI / (2.0 * LOWPASS_FILTER_WIDTH))
                .cos()
                .powi(2);
            let scaled = time * core::f32::consts::PI;
            let sinc = if scaled == 0.0 {
                1.0
            } else {
                scaled.sin() / scaled
            };
            kernels.push(sinc * window * (base / orig as f32));
        }
    }
    let mut output = Vec::with_capacity(result_length);
    for index in 0..result_length {
        let phase = index % target;
        let base_offset = index / target * orig;
        let mut value = 0.0_f32;
        for tap in 0..kernel_width {
            let sample_index = base_offset as isize + tap as isize - width as isize;
            if sample_index >= 0 && (sample_index as usize) < input.len() {
                value += input[sample_index as usize] * kernels[phase * kernel_width + tap];
            }
        }
        output.push(value);
    }
    Ok(output)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn resampling_identity_and_empty_audio() {
        let samples = vec![0.0, 0.1, -0.2, 0.7];
        assert_eq!(resample_to_24khz(&samples, 24_000).unwrap(), samples);
        assert!(resample_to_24khz(&[], 48_000).is_err());
        assert!(resample_to_24khz(&[0.2], 0).is_err());
    }

    #[test]
    fn torchaudio_sinc_resampling_parity() {
        let fixture: serde_json::Value = serde_json::from_str(include_str!(
            "../tests/fixtures/resample_torchaudio_expected.json"
        ))
        .unwrap();
        for case in fixture.as_array().unwrap() {
            let source_rate = case["source_rate"].as_u64().unwrap() as usize;
            let input: Vec<f32> = case["input"]
                .as_array()
                .unwrap()
                .iter()
                .map(|v| v.as_f64().unwrap() as f32)
                .collect();
            let expected: Vec<f32> = case["output"]
                .as_array()
                .unwrap()
                .iter()
                .map(|v| v.as_f64().unwrap() as f32)
                .collect();
            let actual = resample_to_24khz(&input, source_rate).unwrap();
            assert_eq!(actual.len(), expected.len(), "sample rate {source_rate}");
            let max_abs = actual
                .iter()
                .zip(expected.iter())
                .map(|(a, b)| (a - b).abs())
                .fold(0.0_f32, f32::max);
            assert!(
                max_abs <= 5e-5,
                "torchaudio parity at {source_rate} Hz: max_abs={max_abs}"
            );
        }
    }
}

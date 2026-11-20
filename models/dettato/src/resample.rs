use std::f64::consts::PI;

/// Band-limited, deterministic resampling for mono audio. The cutoff is reduced
/// when downsampling to avoid aliasing. Unlike Python's soxr HQ, this is a
/// windowed-sinc approximation; DSP parity is tracked separately.
pub fn resample(input: &[f32], source_rate: u32, target_rate: f64) -> Result<Vec<f32>, String> {
    if source_rate == 0 || !target_rate.is_finite() || target_rate <= 0.0
        || input.iter().any(|x| !x.is_finite())
    { return Err("invalid resampling input or rate".into()); }
    if input.is_empty() { return Ok(Vec::new()); }
    if (target_rate - source_rate as f64).abs() < 1e-8 { return Ok(input.to_vec()); }
    let ratio = target_rate / source_rate as f64;
    let length = ((input.len() as f64) * ratio).round() as usize;
    if length > 100_000_000 { return Err("resampled audio too large".into()); }
    let cutoff = ratio.min(1.0) * 0.98;
    const RADIUS: isize = 48;
    let mut output = Vec::with_capacity(length);
    for i in 0..length {
        let position = i as f64 / ratio;
        let centre = position.floor() as isize;
        let mut numerator = 0.0f64;
        let mut denominator = 0.0f64;
        for j in centre - RADIUS + 1..=centre + RADIUS {
            let d = position - j as f64;
            let a = d.abs() / RADIUS as f64;
            if a >= 1.0 { continue; }
            let x = PI * d * cutoff;
            let sinc = if x.abs() < 1e-10 { 1.0 } else { x.sin() / x };
            let window = 0.5 + 0.5 * (PI * a).cos();
            let coefficient = cutoff * sinc * window;
            if j >= 0 && (j as usize) < input.len() {
                numerator += input[j as usize] as f64 * coefficient;
            }
            denominator += coefficient;
        }
        output.push((numerator / denominator.max(1e-16)) as f32);
    }
    Ok(output)
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn resampler_handles_identity_short_and_silence() {
        let original = [0.0, 0.1, -0.2, 0.4];
        assert_eq!(resample(&original, 16000, 16000.0).unwrap(), original);
        assert!(resample(&[0.0; 100], 44100, 16000.0).unwrap().iter().all(|v| v.abs() < 1e-7));
        assert_eq!(resample(&[], 44100, 16000.0).unwrap(), Vec::<f32>::new());
        assert!(resample(&original, 0, 16000.0).is_err());
    }
    #[test]
    fn downsampling_constant_preserves_level_interior() {
        let signal = vec![0.12; 5000];
        let output = resample(&signal, 44100, 16000.0).unwrap();
        for &x in &output[200..output.len()-200] { assert!((x - 0.12).abs() < 1e-4); }
    }
    #[test]
    fn downsampling_edges_have_zero_padding_not_renormalized_boundary() {
        let signal = vec![0.12; 5_000];
        let output = resample(&signal, 44_100, 16_000.0).unwrap();
        assert!(output[0] > 0.04 && output[0] < 0.10, "first sample: {}", output[0]);
        assert!((output[300] - 0.12).abs() < 1e-4);
        assert!(output[output.len() - 1] < 0.14);
    }
}

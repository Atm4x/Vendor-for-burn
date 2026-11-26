use crate::{F5Error, Result};

pub const FFT_LEN: usize = 1024;

pub(crate) fn transform(real: &mut [f32; FFT_LEN], imag: &mut [f32; FFT_LEN], inverse: bool) {
    let mut j = 0;
    for i in 1..FFT_LEN {
        let mut bit = FFT_LEN >> 1;
        while j & bit != 0 {
            j ^= bit;
            bit >>= 1;
        }
        j ^= bit;
        if i < j {
            real.swap(i, j);
            imag.swap(i, j);
        }
    }
    let mut width = 2;
    while width <= FFT_LEN {
        let angle = if inverse { 2.0 } else { -2.0 } * core::f32::consts::PI / width as f32;
        let step_r = angle.cos();
        let step_i = angle.sin();
        for start in (0..FFT_LEN).step_by(width) {
            let mut wr = 1.0;
            let mut wi = 0.0;
            for offset in 0..width / 2 {
                let left = start + offset;
                let right = left + width / 2;
                let tr = wr * real[right] - wi * imag[right];
                let ti = wr * imag[right] + wi * real[right];
                real[right] = real[left] - tr;
                imag[right] = imag[left] - ti;
                real[left] += tr;
                imag[left] += ti;
                let next_r = wr * step_r - wi * step_i;
                wi = wr * step_i + wi * step_r;
                wr = next_r;
            }
        }
        width <<= 1;
    }
    if inverse {
        let normalization = 1.0 / FFT_LEN as f32;
        for (r, i) in real.iter_mut().zip(imag.iter_mut()) {
            *r *= normalization;
            *i *= normalization;
        }
    }
}

#[derive(Debug, Clone)]
pub struct IStft {
    window: [f32; FFT_LEN],
}

impl Default for IStft {
    fn default() -> Self {
        Self::new()
    }
}

impl IStft {
    pub fn new() -> Self {
        let mut window = [0.0; FFT_LEN];
        for (index, value) in window.iter_mut().enumerate() {
            *value =
                0.5 - 0.5 * (2.0 * core::f32::consts::PI * index as f32 / FFT_LEN as f32).cos();
        }
        Self { window }
    }

    /// Inverts [frames, 513] Hermitian half-spectra in the Vocos `same` alignment.
    pub fn reconstruct(&self, real_spectrum: &[f32], imag_spectrum: &[f32]) -> Result<Vec<f32>> {
        const BINS: usize = FFT_LEN / 2 + 1;
        const HOP: usize = 256;
        if real_spectrum.is_empty()
            || real_spectrum.len() % BINS != 0
            || imag_spectrum.len() != real_spectrum.len()
            || real_spectrum
                .iter()
                .chain(imag_spectrum)
                .any(|x| !x.is_finite())
        {
            return Err(F5Error::InvalidShape(
                "ISTFT expects finite [frames, 513] half-spectra",
            ));
        }
        let frames = real_spectrum.len() / BINS;
        if frames > 65536 {
            return Err(F5Error::InvalidShape("ISTFT exceeds maximum frame count"));
        }
        let samples = frames
            .checked_mul(HOP)
            .and_then(|n| n.checked_add(FFT_LEN))
            .ok_or(F5Error::InvalidShape("ISTFT output size overflow"))?;
        let mut output = vec![0.0_f64; samples];
        let mut envelope = vec![0.0_f64; samples];
        let mut real = [0.0; FFT_LEN];
        let mut imag = [0.0; FFT_LEN];
        // Repeat the last spectral frame to match the 'same' STFT-head alignment.
        for frame in 0..=frames {
            let index = frame.min(frames - 1);
            let src = index * BINS;
            real[..BINS].copy_from_slice(&real_spectrum[src..src + BINS]);
            imag[..BINS].copy_from_slice(&imag_spectrum[src..src + BINS]);
            imag[0] = 0.0;
            imag[FFT_LEN / 2] = 0.0;
            for bin in 1..FFT_LEN / 2 {
                real[FFT_LEN - bin] = real[bin];
                imag[FFT_LEN - bin] = -imag[bin];
            }
            transform(&mut real, &mut imag, true);
            let offset = frame * HOP;
            for n in 0..FFT_LEN {
                let w = f64::from(self.window[n]);
                output[offset + n] += f64::from(real[n]) * w;
                envelope[offset + n] += w * w;
            }
        }
        let start = FFT_LEN / 2;
        let mut result = Vec::with_capacity(frames * HOP);
        for n in start..start + frames * HOP {
            let norm = envelope[n];
            if norm <= 1e-11 {
                return Err(F5Error::ModelExecution("ISTFT uncovered frame".to_string()));
            }
            result.push((output[n] / norm) as f32);
        }
        Ok(result)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn fft_roundtrip_matches_samples() {
        let mut r = [0.0; FFT_LEN];
        let mut i = [0.0; FFT_LEN];
        for (n, v) in r.iter_mut().enumerate() {
            *v = (n as f32 * 0.237).sin() + 0.3 * (n as f32 * 0.083).cos();
        }
        let expected = r;
        transform(&mut r, &mut i, false);
        transform(&mut r, &mut i, true);
        assert!(r.iter().zip(expected).all(|(a, b)| (a - b).abs() < 2e-4));
        assert!(i.iter().all(|v| v.abs() < 2e-4));
    }

    #[test]
    fn istft_matches_independent_pytorch() {
        let fixture: serde_json::Value =
            serde_json::from_str(include_str!("../tests/fixtures/vocos_istft_expected.json"))
                .unwrap();
        let as_f32 = |field: &str| -> Vec<f32> {
            fixture[field]
                .as_array()
                .unwrap()
                .iter()
                .map(|v| v.as_f64().unwrap() as f32)
                .collect()
        };
        let reconstructed = IStft::new()
            .reconstruct(&as_f32("real"), &as_f32("imag"))
            .unwrap();
        let expected = as_f32("wave");
        assert_eq!(
            reconstructed.len(),
            fixture["frames"].as_u64().unwrap() as usize * 256
        );
        let max_abs = reconstructed
            .iter()
            .zip(expected)
            .map(|(a, b)| (a - b).abs())
            .fold(0.0_f32, f32::max);
        assert!(max_abs < 1e-5, "ISTFT PyTorch parity max_abs={max_abs}");
    }

    #[test]
    fn istft_rejects_invalid_and_returns_correct_length() {
        let decoder = IStft::new();
        assert!(decoder.reconstruct(&[], &[]).is_err());
        assert!(decoder.reconstruct(&[0.0], &[0.0]).is_err());
        let mut real = vec![0.0; 513 * 7];
        let imag = vec![0.0; real.len()];
        for row in real.chunks_exact_mut(513) {
            row[0] = 1.0;
        }
        let wave = decoder.reconstruct(&real, &imag).unwrap();
        assert_eq!(wave.len(), 7 * 256);
        assert!(wave.iter().all(|v| v.is_finite()));
    }
}

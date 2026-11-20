use crate::{AUDIO_SAMPLE_RATE, HOP_SAMPLES};
use rustfft::{FftPlanner, num_complex::Complex32};
use std::f64::consts::PI;

pub const DEFAULT_HARMONICS: [f64; 8] = [0.5, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0];

#[derive(Clone, Debug)]
pub struct HcqtConfig {
    pub n_fft: usize,
    pub n_bins: usize,
    pub bins_per_semitone: usize,
    pub fmin: f64,
    pub harmonics: Vec<f64>,
    pub extra_nffts: Vec<usize>,
    pub log_eps: f32,
}

impl Default for HcqtConfig {
    fn default() -> Self {
        Self {
            n_fft: 2048,
            n_bins: 288,
            bins_per_semitone: 3,
            fmin: 27.5,
            harmonics: DEFAULT_HARMONICS.to_vec(),
            extra_nffts: vec![512],
            log_eps: 1e-5,
        }
    }
}

#[derive(Debug)]
struct FilterBank {
    rows: Vec<Vec<(usize, f32)>>,
}

impl FilterBank {
    fn new(n_bins: usize, n_fft: usize, fmin: f64, bps: usize) -> Self {
        let fft_step = AUDIO_SAMPLE_RATE as f64 / n_fft as f64;
        let mut rows = Vec::with_capacity(n_bins);
        for lane in 0..n_bins {
            let center = fmin * 2f64.powf(lane as f64 / (12 * bps) as f64);
            let lo = center * 2f64.powf(-1.0 / (12 * bps) as f64);
            let hi = center * 2f64.powf(1.0 / (12 * bps) as f64);
            let width = (hi - lo).max(2.0 * fft_step);
            let left = center - width * 0.5;
            let right = center + width * 0.5;
            let mut row = Vec::new();
            let mut sum = 0.0f64;
            for k in 0..=n_fft / 2 {
                let f = k as f64 * fft_step;
                let up = (f - left) / (center - left).max(1e-9);
                let down = (right - f) / (right - center).max(1e-9);
                let weight = up.min(down).max(0.0);
                if weight > 0.0 {
                    row.push((k, weight as f32));
                    sum += weight;
                }
            }
            if sum > 0.0 {
                for (_, weight) in &mut row { *weight = (*weight as f64 / sum) as f32; }
            }
            rows.push(row);
        }
        Self { rows }
    }

    fn project(&self, power: &[f32], result: &mut [f32], log_eps: f32) {
        for (out, row) in result.iter_mut().zip(&self.rows) {
            let mut sum = 0.0;
            for &(fft_bin, weight) in row {
                sum += weight * power[fft_bin];
            }
            *out = (sum + log_eps).ln();
        }
    }
}

#[derive(Clone, Debug)]
pub struct Hcqt {
    config: HcqtConfig,
    shifts: Vec<isize>,
    low_shift: isize,
}

#[derive(Clone, Debug, PartialEq)]
pub struct Features {
    pub channels: usize,
    pub frames: usize,
    pub bins: usize,
    pub data: Vec<f32>,
}

impl Features {
    pub fn get(&self, channel: usize, frame: usize, bin: usize) -> f32 {
        self.data[(channel * self.frames + frame) * self.bins + bin]
    }
}

impl Hcqt {
    pub fn new(config: HcqtConfig) -> Result<Self, String> {
        if config.n_fft < 2 || config.n_fft % 2 != 0
            || config.bins_per_semitone == 0 || config.n_bins == 0
            || config.harmonics.is_empty() || !config.fmin.is_finite() || config.fmin <= 0.0
            || !config.log_eps.is_finite() || config.log_eps <= 0.0
            || config.extra_nffts.iter().any(|&n| n < 2 || n % 2 != 0)
            || config.harmonics.iter().any(|&h| !h.is_finite() || h <= 0.0)
        {
            return Err("invalid HCQT configuration".into());
        }
        let shifts = config.harmonics.iter().map(|&harmonic| {
            (12.0 * config.bins_per_semitone as f64 * harmonic.log2()).round() as isize
        }).collect::<Vec<_>>();
        let low_shift = shifts.iter().copied().min().unwrap().min(0);
        Ok(Self { config, shifts, low_shift })
    }

    pub fn default_16k() -> Self {
        Self::new(HcqtConfig::default()).expect("valid built-in HCQT configuration")
    }

    pub fn channels(&self) -> usize {
        self.shifts.len() + self.config.extra_nffts.len()
    }

    /// Extended linear HCQT power for Grouper harmonic-demixing, indexed [bin, frame].
    /// Uses exactly the reference long-window triangular filter bank.
    pub fn extended_power(&self, audio16k: &[f32]) -> (usize, isize, Vec<f32>) {
        let frames = audio16k.len() / HOP_SAMPLES;
        let hi = self.shifts.iter().copied().max().unwrap().max(0);
        let bins = (self.config.n_bins as isize + hi - self.low_shift) as usize;
        let fmin = self.config.fmin
            * 2f64.powf(self.low_shift as f64 / (12 * self.config.bins_per_semitone) as f64);
        let bank = FilterBank::new(bins, self.config.n_fft, fmin, self.config.bins_per_semitone);
        let log_power = spectrogram(audio16k, self.config.n_fft, frames, &bank, self.config.log_eps);
        let mut power = vec![0.0; bins * frames];
        for frame in 0..frames {
            for bin in 0..bins {
                power[bin * frames + frame] = (log_power[frame * bins + bin].exp() - self.config.log_eps).max(0.0);
            }
        }
        (bins, self.low_shift, power)
    }

    pub fn extract(&self, audio16k: &[f32]) -> Features {
        let frames = audio16k.len() / HOP_SAMPLES;
        let channels = self.channels();
        let bins = self.config.n_bins;
        let mut data = vec![0.0; frames * bins * channels];
        if frames == 0 { return Features { channels, frames, bins, data }; }

        let hi = self.shifts.iter().copied().max().unwrap().max(0);
        let long_bins = (bins as isize + hi - self.low_shift) as usize;
        let fmin = self.config.fmin
            * 2f64.powf(self.low_shift as f64 / (12 * self.config.bins_per_semitone) as f64);
        let long = FilterBank::new(long_bins, self.config.n_fft, fmin, self.config.bins_per_semitone);
        let long_features = spectrogram(audio16k, self.config.n_fft, frames, &long, self.config.log_eps);
        for (channel, shift) in self.shifts.iter().enumerate() {
            let start = (*shift - self.low_shift) as usize;
            for frame in 0..frames {
                let input = &long_features[frame * long_bins + start..frame * long_bins + start + bins];
                data[(channel * frames + frame) * bins..(channel * frames + frame + 1) * bins]
                    .copy_from_slice(input);
            }
        }
        for (index, &nfft) in self.config.extra_nffts.iter().enumerate() {
            let bank = FilterBank::new(bins, nfft, self.config.fmin, self.config.bins_per_semitone);
            let extra = spectrogram(audio16k, nfft, frames, &bank, self.config.log_eps);
            let channel = index + self.shifts.len();
            data[channel * frames * bins..(channel + 1) * frames * bins].copy_from_slice(&extra);
        }
        Features { channels, frames, bins, data }
    }
}

fn spectrogram(audio: &[f32], nfft: usize, frames: usize, bank: &FilterBank, log_eps: f32) -> Vec<f32> {
    let mut planner = FftPlanner::<f32>::new();
    let fft = planner.plan_fft_forward(nfft);
    let hann = (0..nfft).map(|i| {
        (0.5 - 0.5 * (2.0 * PI * i as f64 / nfft as f64).cos()) as f32
    }).collect::<Vec<_>>();
    let mut input = vec![Complex32::new(0.0, 0.0); nfft];
    let mut power = vec![0.0; nfft / 2 + 1];
    let mut output = vec![0.0; frames * bank.rows.len()];
    for frame in 0..frames {
        let offset = frame as isize * HOP_SAMPLES as isize - nfft as isize / 2;
        for (i, sample) in input.iter_mut().enumerate() {
            let index = offset + i as isize;
            let x = if index >= 0 { audio.get(index as usize).copied().unwrap_or(0.0) } else { 0.0 };
            *sample = Complex32::new(x * hann[i], 0.0);
        }
        fft.process(&mut input);
        for (i, p) in power.iter_mut().enumerate() { *p = input[i].norm_sqr(); }
        bank.project(&power, &mut output[frame * bank.rows.len()..(frame + 1) * bank.rows.len()], log_eps);
    }
    output
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn default_hcqt_has_expected_time_and_frequency_shape() {
        let features = Hcqt::default_16k().extract(&vec![0.0; AUDIO_SAMPLE_RATE]);
        assert_eq!((features.channels, features.frames, features.bins), (9, 100, 288));
        assert_eq!(features.data.len(), 9 * 100 * 288);
        assert!(features.data.iter().all(|&x| (x - 1e-5f32.ln()).abs() < 1e-5));
    }

    #[test]
    fn hcqt_matches_original_torch_reference_samples() {
        let fixture: serde_json::Value = serde_json::from_str(
            include_str!("../tests/fixtures/hcqt_samples.json")
        ).unwrap();
        let count = fixture["samples"].as_u64().unwrap() as usize;
        let mut samples = (0..count).map(|i| {
            let x = i as f64 / AUDIO_SAMPLE_RATE as f64;
            (0.2 * (2.0 * PI * 220.0 * x).sin()
                + 0.35 * (2.0 * PI * 330.0 * x).sin()) as f32
        }).collect::<Vec<_>>();
        samples[0] += 0.4;
        let actual = Hcqt::default_16k().extract(&samples);
        for (position, expected) in fixture["positions"].as_array().unwrap()
            .iter().zip(fixture["values"].as_array().unwrap()) {
            let c = position[0].as_u64().unwrap() as usize;
            let t = position[1].as_u64().unwrap() as usize;
            let b = position[2].as_u64().unwrap() as usize;
            let reference = expected.as_f64().unwrap() as f32;
            let observed = actual.get(c, t, b);
            assert!((observed - reference).abs() < 0.01,
                "HCQT differs from Python at ({c},{t},{b}): {observed} vs {reference}");
        }
    }

    #[test]
    fn hcqt_uses_zero_center_padding_and_10ms_grid() {
        let extractor = Hcqt::default_16k();
        assert_eq!(extractor.extract(&vec![0.0; 159]).frames, 0);
        assert_eq!(extractor.extract(&vec![0.0; 160]).frames, 1);
        let mut input = vec![0.0; 160 * 6];
        input[0] = 1.0;
        let features = extractor.extract(&input);
        assert!(features.data.iter().all(|x| x.is_finite()));
        assert!(features.get(0, 0, 0) > 1e-5f32.ln());
    }
}

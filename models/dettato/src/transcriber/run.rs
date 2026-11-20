use super::{decode::{DecodeConfig, DecodedNote, FrameMaps, decode}, net::{LaneNet, Settings}};
use crate::{AUDIO_SAMPLE_RATE, PITCH_COUNT, checkpoint::{Checkpoint, ModelPaths}, hcqt::{Hcqt, HcqtConfig}};
use burn::tensor::Device;
use serde_json::Value;
use std::path::Path;

pub const WINDOW_SECONDS: usize = 8;
pub const WINDOW_SAMPLES: usize = WINDOW_SECONDS * AUDIO_SAMPLE_RATE;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Window {
    pub start_sample: usize,
    pub first_song_frame: isize,
}

pub fn windows(n_samples: usize, hop_seconds: f64, overlap_step: f64) -> Result<(usize, Vec<Window>), String> {
    if n_samples == 0 || !hop_seconds.is_finite() || hop_seconds <= 0.0
        || !overlap_step.is_finite() || overlap_step <= 0.0 || overlap_step > 1.0
    {
        return Err("invalid transcriber window layout".into());
    }
    let frame_count = (n_samples as f64 / AUDIO_SAMPLE_RATE as f64 / hop_seconds).ceil() as usize;
    let pad = WINDOW_SAMPLES / 2;
    let step = ((WINDOW_SAMPLES as f64 * overlap_step).round() as usize).max(1);
    let global_pad_frames = (pad as f64 / AUDIO_SAMPLE_RATE as f64 / hop_seconds).round() as isize;
    let mut result = Vec::new();
    for start in (0..n_samples.saturating_add(pad)).step_by(step) {
        let first = (start as f64 / AUDIO_SAMPLE_RATE as f64 / hop_seconds).round() as isize
            - global_pad_frames;
        result.push(Window { start_sample: start, first_song_frame: first });
    }
    Ok((frame_count, result))
}

pub struct Transcriber {
    frontend: Hcqt,
    network: LaneNet,
    decoder: DecodeConfig,
}

impl Transcriber {
    pub fn from_directory(weights_root: impl AsRef<Path>, device: &Device) -> Result<Self, String> {
        let paths = ModelPaths::in_directory(weights_root, "transcriber")?;
        let config = paths.load_config()?;
        let settings = Settings::from_config(&config)?;
        let decoder = settings.decoder(&config)?;
        let frontend = Hcqt::new(hcqt_config(&config)?)?;
        let checkpoint = Checkpoint::open(&paths.weights)?;
        let network = LaneNet::load(&checkpoint, device, settings)?;
        Ok(Self { frontend, network, decoder })
    }

    pub fn maps(&self, audio16k: &[f32]) -> Result<FrameMaps, String> {
        if audio16k.is_empty() || audio16k.iter().any(|x| !x.is_finite()) {
            return Err("audio must contain finite mono PCM16k samples".into());
        }
        let (frames, plan) = windows(audio16k.len(), self.decoder.hop_seconds, 0.25)?;
        let frame_len = frames.checked_mul(PITCH_COUNT).ok_or("transcriber frame count overflow")?;
        let mut accumulator: [Vec<f32>; 6] = std::array::from_fn(|_| vec![0.0; frame_len]);
        let mut weights = vec![0.0f32; frames];
        let pad = WINDOW_SAMPLES / 2;
        let mut audio_window = vec![0.0f32; WINDOW_SAMPLES];
        for window in plan {
            for (i, sample) in audio_window.iter_mut().enumerate() {
                let pos = window.start_sample + i;
                *sample = if pos >= pad { audio16k.get(pos - pad).copied().unwrap_or(0.0) } else { 0.0 };
            }
            let features = self.frontend.extract(&audio_window);
            let prediction = self.network.infer_features(&features)?;
            let local_frames = prediction.maps.frames();
            for local in 0..local_frames {
                let global = window.first_song_frame + local as isize;
                if global < 0 || global >= frames as isize { continue; }
                let g = global as usize;
                let weight = (0.5 - 0.5 * (2.0 * std::f64::consts::PI
                    * (local + 1) as f64 / (local_frames + 1) as f64).cos()) as f32;
                weights[g] += weight;
                let local_start = local * PITCH_COUNT;
                let global_start = g * PITCH_COUNT;
                for h in 0..6 {
                    let src = &prediction.maps.heads[h][local_start..local_start + PITCH_COUNT];
                    let dst = &mut accumulator[h][global_start..global_start + PITCH_COUNT];
                    for (out, &val) in dst.iter_mut().zip(src) { *out += weight * val; }
                }
            }
        }
        for (g, weight) in weights.iter().copied().enumerate() {
            let denom = weight.max(1e-6);
            for head in &mut accumulator {
                for v in &mut head[g * PITCH_COUNT..(g + 1) * PITCH_COUNT] { *v /= denom; }
            }
        }
        FrameMaps::new(frames, accumulator)
    }

    pub fn notes(&self, audio16k: &[f32]) -> Result<Vec<DecodedNote>, String> {
        decode(&self.maps(audio16k)?, &self.decoder)
    }
}

fn hcqt_config(config: &Value) -> Result<HcqtConfig, String> {
    let model = config.get("model").ok_or("missing transcriber model configuration")?;
    let mut c = HcqtConfig::default();
    for (name, target) in [("n_fft", &mut c.n_fft), ("n_bins", &mut c.n_bins)] {
        if let Some(value) = model.get(name) {
            *target = value.as_u64().ok_or_else(|| format!("{name} must be integer"))? as usize;
        }
    }
    if let Some(value) = model.get("harmonics") {
        c.harmonics = value.as_array().ok_or("harmonics must be a list")?
            .iter().map(|v| v.as_f64().ok_or_else(|| "invalid harmonic".to_string()))
            .collect::<Result<Vec<_>, _>>()?;
    }
    if let Some(value) = model.get("extra_nffts") {
        c.extra_nffts = value.as_array().ok_or("extra_nffts must be a list")?
            .iter().map(|v| v.as_u64().map(|n| n as usize)
                .ok_or_else(|| "invalid extra_nffts entry".to_string()))
            .collect::<Result<Vec<_>, _>>()?;
    }
    Ok(c)
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn eight_second_windows_use_two_second_steps_with_half_window_padding() {
        let (frames, plan) = windows(4 * AUDIO_SAMPLE_RATE, 0.02, 0.25).unwrap();
        assert_eq!(frames, 200);
        assert_eq!(plan.iter().map(|p| p.start_sample).collect::<Vec<_>>(), [0, 32000, 64000, 96000]);
        assert_eq!(plan.iter().map(|p| p.first_song_frame).collect::<Vec<_>>(), [-200, -100, 0, 100]);
    }

    #[test]
    fn all_song_frames_receive_positive_hann_weight() {
        for samples in [160, 1600, 32000, 64000, 129999, 160000, 320000] {
            let (frames, plan) = windows(samples, 0.02, 0.25).unwrap();
            let mut weight = vec![0.0; frames];
            for window in plan {
                for t in 0..400 {
                    let g = window.first_song_frame + t;
                    if (0..frames as isize).contains(&g) {
                        weight[g as usize] += 0.5 - 0.5 * (2.0 * std::f64::consts::PI
                            * (t + 1) as f64 / 401.0).cos();
                    }
                }
            }
            assert!(weight.into_iter().all(|w| w > 0.0));
        }
    }
}

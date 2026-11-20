use super::{decode::{DecodeConfig, DecodedNote, FrameMaps, decode}, net::{LaneNet, Settings}};
use crate::{AUDIO_SAMPLE_RATE, PITCH_COUNT, checkpoint::{Checkpoint, ModelPaths}, hcqt::{Hcqt, HcqtConfig}, evidence::features::{self, EVIDENCE_FEATURES, HIDDEN_CHANNELS}};
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
        self.maps_with_step(audio16k, 0.25)
    }

    /// The original Grouper uses the 4-second stride (hop_frac=0.5)
    /// while the primary Transcriber maps use a 2-second stride.
    pub fn maps_with_step(&self, audio16k: &[f32], overlap_step: f64) -> Result<FrameMaps, String> {
        if audio16k.is_empty() || audio16k.iter().any(|x| !x.is_finite()) {
            return Err("audio must contain finite mono PCM16k samples".into());
        }
        let (frames, plan) = windows(audio16k.len(), self.decoder.hop_seconds, overlap_step)?;
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

    pub fn notes_with_evidence(&self, audio16k: &[f32])
        -> Result<(Vec<DecodedNote>, Vec<[f32; EVIDENCE_FEATURES]>), String>
    {
        let maps = self.maps(audio16k)?;
        let notes = decode(&maps, &self.decoder)?;
        let frames = features::note_frames(&notes, maps.frames(), self.decoder.hop_seconds)?;
        if notes.is_empty() { return Ok((notes, Vec::new())); }
        let map_rows = features::map_evidence(&maps, &frames)?;
        let (song_frames, plan) = windows(audio16k.len(), self.decoder.hop_seconds, 0.25)?;
        let window_frames = (WINDOW_SECONDS as f64 / self.decoder.hop_seconds).round() as usize;
        let hann = (0..window_frames).map(|t| {
            (0.5 - 0.5 * (2.0 * std::f64::consts::PI
                * (t + 1) as f64 / (window_frames + 1) as f64).cos()) as f32
        }).collect::<Vec<_>>();
        let mut weight_sum = vec![0.0f32; song_frames];
        for window in &plan {
            for (local, &weight) in hann.iter().enumerate() {
                let global = window.first_song_frame + local as isize;
                if (0..song_frames as isize).contains(&global) {
                    weight_sum[global as usize] += weight;
                }
            }
        }
        let mut at = vec![vec![0.0; HIDDEN_CHANNELS]; notes.len()];
        let mut mean = vec![vec![0.0; HIDDEN_CHANNELS]; notes.len()];
        let mut samples = vec![0.0f32; WINDOW_SAMPLES];
        let pad = WINDOW_SAMPLES / 2;
        for window in &plan {
            for (i, sample) in samples.iter_mut().enumerate() {
                let pos = window.start_sample + i;
                *sample = if pos >= pad { audio16k.get(pos - pad).copied().unwrap_or(0.0) } else { 0.0 };
            }
            let prediction = self.network.infer_features(&self.frontend.extract(&samples))?;
            let [batch, channels, time, lanes] = prediction.hidden.dims();
            if batch != 1 || channels != HIDDEN_CHANNELS || lanes != PITCH_COUNT || time != window_frames {
                return Err(format!("unexpected hidden tap: [{batch}, {channels}, {time}, {lanes}]"));
            }
            let tap: Vec<f32> = prediction.hidden.into_data().try_into_vec::<f32>()
                .map_err(|e| format!("reading hidden tap: {e}"))?;
            let lo = window.first_song_frame.max(0) as usize;
            let hi = (window.first_song_frame + window_frames as isize)
                .min(song_frames as isize).max(0) as usize;
            if hi <= lo { continue; }
            for (index, note) in frames.iter().enumerate() {
                let left = lo.max(note.start);
                let right = hi.min(note.end);
                if left >= right { continue; }
                let span = (note.end - note.start) as f32;
                for global in left..right {
                    let local = (global as isize - window.first_song_frame) as usize;
                    let weight = hann[local] / weight_sum[global].max(1e-6);
                    for channel in 0..channels {
                        let x = tap[(channel * time + local) * lanes + note.pitch_index];
                        mean[index][channel] += weight * x / span;
                        if global == note.start { at[index][channel] += weight * x; }
                    }
                }
            }
        }
        let evidence = features::assemble(&map_rows, &at, &mean)?;
        Ok((notes, evidence))
    }

    /// Extract the 12 note scalars and three 368-dimensional taps used by Grouper.
    /// A second 4-second-stride pass accumulates only frames touched by requested
    /// notes, rather than materializing three whole-song hidden maps.
    pub fn grouper_features(&self, audio16k: &[f32], notes: &[DecodedNote])
        -> Result<(Vec<[f32; 12]>, Vec<[[f32; HIDDEN_CHANNELS]; 3]>), String>
    {
        let maps = self.maps_with_step(audio16k, 0.5)?;
        let decoded = decode(&maps, &self.decoder)?;
        let total_frames = maps.frames();
        if total_frames == 0 { return Err("empty Grouper frame map".into()); }
        let hop = self.decoder.hop_seconds;
        let mut sorted = notes.to_vec();
        sorted.sort_by(|a, b| a.onset.total_cmp(&b.onset).then_with(|| a.pitch.cmp(&b.pitch)));
        let mut meta = Vec::<[f32; 12]>::with_capacity(sorted.len());
        let mut intervals = Vec::<(usize, usize, usize)>::with_capacity(sorted.len());
        for note in &sorted {
            if !note.onset.is_finite()
                || !note.offset.is_finite() || note.onset < 0.0 || note.offset <= note.onset
            { return Err("Grouper input requires valid piano-range notes".into()); }
            let pitch = (note.pitch as isize - 21).clamp(0, PITCH_COUNT as isize - 1) as usize;
            let t0 = ((note.onset / hop).round() as isize).clamp(0, total_frames as isize - 1) as usize;
            let te = ((note.offset.max(note.onset + 0.02) / hop).ceil() as isize).clamp(t0 as isize + 1, total_frames as isize) as usize;
            let slot = decoded.iter().filter(|n| n.pitch == note.pitch)
                .min_by(|a, b| (a.onset - note.onset).abs().total_cmp(&(b.onset - note.onset).abs()))
                .filter(|n| (n.onset - note.onset).abs() <= 0.1)
                .map(|n| (n.slot - 1) as f32).unwrap_or(0.0);
            let mut row = [0.0; 12];
            row[0] = note.onset as f32;
            row[1] = note.offset.max(note.onset + 0.02) as f32;
            row[2] = note.pitch as f32;
            row[3] = slot;
            for (i, head) in [0usize, 3].iter().enumerate() {
                let m = &maps.heads[*head];
                row[4+i] = (t0.saturating_sub(1)..=(t0+1).min(total_frames-1))
                    .map(|t| m[t * PITCH_COUNT + pitch]).fold(0.0f32, f32::max);
                row[6+i] = maps.heads[*head+1][t0 * PITCH_COUNT + pitch];
                row[8+i] = (t0..te).map(|t| maps.heads[*head+1][t * PITCH_COUNT + pitch])
                    .sum::<f32>() / (te - t0) as f32;
                row[10+i] = ((te-1).saturating_sub(1)..=(te).min(total_frames-1))
                    .map(|t| maps.heads[*head+2][t * PITCH_COUNT + pitch]).fold(0.0f32, f32::max);
            }
            meta.push(row);
            intervals.push((t0, te, pitch));
        }
        let mut taps = vec![[[0.0f32; HIDDEN_CHANNELS]; 3]; sorted.len()];
        if sorted.is_empty() { return Ok((meta, taps)); }
        let (_, plan) = windows(audio16k.len(), hop, 0.5)?;
        let frames_per_window = (WINDOW_SECONDS as f64 / hop).round() as usize;
        let hann = (0..frames_per_window).map(|t| {
            (0.5 - 0.5 * (2.0 * std::f64::consts::PI * (t+1) as f64
                / (frames_per_window+1) as f64).cos()) as f32
        }).collect::<Vec<_>>();
        let mut sums = vec![0.0f32; total_frames];
        for window in &plan {
            for (local, &w) in hann.iter().enumerate() {
                let global = window.first_song_frame + local as isize;
                if (0..total_frames as isize).contains(&global) { sums[global as usize] += w; }
            }
        }
        let mut samples = vec![0.0f32; WINDOW_SAMPLES];
        for window in plan {
            for (i, sample) in samples.iter_mut().enumerate() {
                let absolute = window.start_sample + i;
                *sample = if absolute >= WINDOW_SAMPLES/2 {
                    audio16k.get(absolute - WINDOW_SAMPLES/2).copied().unwrap_or(0.0)
                } else { 0.0 };
            }
            let prediction = self.network.infer_features(&self.frontend.extract(&samples))?;
            let [_, channels, time, hidden_lanes] = prediction.hidden.dims();
            let [_, stem_channels, stem_time, stem_lanes] = prediction.stem.dims();
            if channels != HIDDEN_CHANNELS || stem_channels != HIDDEN_CHANNELS
                || time != frames_per_window || stem_time != time
                || hidden_lanes != PITCH_COUNT || stem_lanes < PITCH_COUNT
            { return Err("unexpected Transcriber tap shape".into()); }
            let hidden = prediction.hidden.into_data().try_into_vec::<f32>()
                .map_err(|e| format!("Grouper hidden tap readback: {e}"))?;
            let stem = prediction.stem.into_data().try_into_vec::<f32>()
                .map_err(|e| format!("Grouper stem tap readback: {e}"))?;
            for (i, &(start, end, pitch)) in intervals.iter().enumerate() {
                for (tap_ix, last) in [(0, start+1), (1, end.min(start+50)), (2, end.min(start+10))] {
                    let first = start.max(window.first_song_frame.max(0) as usize);
                    let last = last.min((window.first_song_frame + frames_per_window as isize)
                        .max(0) as usize).min(total_frames);
                    if last <= first { continue; }
                    for global in first..last {
                        let local = (global as isize - window.first_song_frame) as usize;
                        let weight = hann[local] / sums[global].max(1e-6);
                        let norm = if tap_ix == 0 { 1.0 }
                            else if tap_ix == 1 { (end-start).min(50).max(1) as f32 }
                            else { (end-start).min(10).max(1) as f32 };
                        for c in 0..channels {
                            let (source, lanes) = if tap_ix == 2 { (&stem, stem_lanes) } else { (&hidden, hidden_lanes) };
                            taps[i][tap_ix][c] += source[(c*time + local)*lanes + pitch] * weight / norm;
                        }
                    }
                }
            }
        }
        Ok((meta, taps))
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

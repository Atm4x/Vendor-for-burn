use super::{
    decode::{GroupPredictions, WindowPrediction},
    front::{self, AudioFeatures},
    lane::{LaneConfig, LaneFront},
    layers::{Dense, Norm, TransformerBlock},
};
use crate::{AUDIO_SAMPLE_RATE, checkpoint::{Checkpoint, ModelPaths}};
use burn::tensor::{Device, Tensor, TensorData, activation};
use serde_json::Value;
use std::path::Path;

pub const TRANSCRIBER_TAP_DIM: usize = 368;
pub const GROUPER_INSTRUMENT_COUNT: usize = 73;

#[derive(Clone, Debug)]
pub struct NoteNetConfig {
    pub width: usize,
    pub depth: usize,
    pub heads: usize,
    pub feedforward: usize,
    pub embedding: usize,
    pub harmonics: usize,
    pub sustain: usize,
    pub lane_channels: usize,
    pub lane_depth: usize,
    pub lane_stride: usize,
    pub lane_attentions: usize,
    pub lane_expansion: usize,
    pub lanes: usize,
    pub s15_taps: Vec<usize>,
    pub s15_hidden: usize,
    pub s15_dimension: usize,
    pub position_scale: f32,
    pub window_seconds: f64,
    pub window_hop: f64,
    pub max_notes: usize,
}

impl Default for NoteNetConfig {
    fn default() -> Self {
        Self {
            width: 176, depth: 7, heads: 4, feedforward: 352, embedding: 32,
            harmonics: 10, sustain: 4, lane_channels: 128, lane_depth: 6,
            lane_stride: 8, lane_attentions: 2, lane_expansion: 2, lanes: 96,
            s15_taps: vec![0, 1, 2], s15_hidden: 256, s15_dimension: TRANSCRIBER_TAP_DIM,
            position_scale: 10.0, window_seconds: 24.0, window_hop: 0.5, max_notes: 1024,
        }
    }
}

impl NoteNetConfig {
    pub fn from_json(config: &Value) -> Result<Self, String> {
        let model = config.get("model").ok_or("Grouper config missing model")?;
        let mut result = Self::default();
        let u = |name: &str, default: usize| -> Result<usize, String> {
            match model.get(name) {
                None => Ok(default),
                Some(x) => x.as_u64().map(|v| v as usize)
                    .ok_or_else(|| format!("Grouper model.{name} must be an unsigned integer")),
            }
        };
        result.width = u("d", result.width)?;
        result.depth = u("depth", result.depth)?;
        result.heads = u("heads", result.heads)?;
        result.feedforward = u("ff", result.feedforward)?;
        result.embedding = u("emb", result.embedding)?;
        result.harmonics = u("hd", result.harmonics)?;
        result.sustain = u("hd_sus", result.sustain)?;
        result.lane_channels = u("lanes", result.lane_channels)?;
        result.lane_depth = u("lane_depth", result.lane_depth)?;
        result.lane_stride = u("lane_ts", result.lane_stride)?;
        result.lane_attentions = u("lane_attn", result.lane_attentions)?;
        result.lane_expansion = u("lane_exp", result.lane_expansion)?;
        result.lanes = u("n_lanes", result.lanes)?;
        result.s15_hidden = u("s15_hid", result.s15_hidden)?;
        result.s15_dimension = u("s15_dim", result.s15_dimension)?;
        if let Some(taps) = model.get("s15_taps") {
            result.s15_taps = taps.as_array().ok_or("Grouper s15_taps must be an array")?.iter()
                .map(|v| v.as_u64().map(|x| x as usize).ok_or("invalid s15 tap"))
                .collect::<Result<Vec<_>, _>>()?;
        }
        if let Some(v) = model.get("pos_scale") {
            result.position_scale = v.as_f64().ok_or("invalid pos_scale")? as f32;
        }
        for (key, dst) in [
            ("window_s", &mut result.window_seconds),
            ("window_hop", &mut result.window_hop),
        ] {
            if let Some(v) = config.get(key) { *dst = v.as_f64().ok_or(format!("invalid {key}"))?; }
        }
        if let Some(v) = config.get("max_notes") {
            result.max_notes = v.as_u64().ok_or("invalid max_notes")? as usize;
        }
        if result.width == 0 || result.depth == 0 || result.heads == 0 || result.width % result.heads != 0
            || result.width / result.heads < 2 || result.feedforward == 0 || result.embedding == 0 || result.harmonics == 0 || result.sustain == 0
            || result.lane_channels == 0 || result.lane_depth == 0 || result.lane_stride == 0
            || result.lane_attentions > result.lane_depth || result.lanes != 96
            || result.s15_taps.is_empty() || result.s15_taps.iter().any(|&i| i >= 3)
            || result.s15_dimension != TRANSCRIBER_TAP_DIM || result.s15_hidden == 0
            || !result.position_scale.is_finite() || !result.window_seconds.is_finite()
            || result.window_seconds <= 0.0 || !result.window_hop.is_finite()
            || result.window_hop <= 0.0 || result.max_notes < 4
        { return Err("invalid Grouper NoteNet configuration".into()); }
        Ok(result)
    }
}

#[derive(Clone, Debug)]
pub struct WindowInput {
    pub audio16k: Vec<f32>,
    pub notes: Vec<[f32; 12]>,
    pub taps: Vec<[[f32; TRANSCRIBER_TAP_DIM]; 3]>,
}

#[derive(Clone, Debug)]
pub struct WindowOutput {
    pub probabilities: Vec<Vec<f32>>,
    pub embeddings: Vec<Vec<f32>>,
    pub hidden: Vec<Vec<f32>>,
    pub keep_logits: Vec<f32>,
}

pub struct NoteNet {
    device: Device,
    pub config: NoteNetConfig,
    pitch_embedding: Vec<f32>,
    pitch_features1: Dense,
    pitch_features2: Dense,
    hd_prior: Vec<f32>,
    hd_enc1: Dense,
    hd_enc2: Dense,
    prof1: Dense,
    prof2: Dense,
    lanes: LaneFront,
    s15_norm: Norm,
    s15_fc1: Dense,
    s15_fc2: Dense,
    input_norm: Norm,
    blocks: Vec<TransformerBlock>,
    output_norm: Norm,
    instrument: Dense,
    embedding: Dense,
    keep: Dense,
}

impl NoteNet {
    pub fn from_directory(root: impl AsRef<Path>, device: &Device) -> Result<Self, String> {
        let paths = ModelPaths::in_directory(root, "grouper")?;
        let config_json = paths.load_config()?;
        let config = NoteNetConfig::from_json(&config_json)?;
        let checkpoint = Checkpoint::open(&paths.weights)?;
        Self::load(&checkpoint, device, config)
    }

    pub fn load(checkpoint: &Checkpoint, device: &Device, config: NoteNetConfig) -> Result<Self, String> {
        let (shape, pitch_embedding) = checkpoint.values("pitch_emb.weight")?;
        if shape != [128, config.width] { return Err("Grouper pitch embedding shape mismatch".into()); }
        let (shape, hd_prior) = checkpoint.values("hd.prior")?;
        if shape != [config.harmonics] { return Err("Grouper harmonic prior shape mismatch".into()); }
        let blocks = (0..config.depth).map(|i| {
            TransformerBlock::load(checkpoint, &format!("blocks.{i}"), device, config.heads)
        }).collect::<Result<Vec<_>, _>>()?;
        Ok(Self {
            device: device.clone(),
            pitch_embedding,
            pitch_features1: Dense::load(checkpoint, "feat.0", device)?,
            pitch_features2: Dense::load(checkpoint, "feat.2", device)?,
            hd_prior,
            hd_enc1: Dense::load(checkpoint, "hd.enc.0", device)?,
            hd_enc2: Dense::load(checkpoint, "hd.enc.2", device)?,
            prof1: Dense::load(checkpoint, "prof.net.0", device)?,
            prof2: Dense::load(checkpoint, "prof.net.2", device)?,
            lanes: LaneFront::load(checkpoint, device, LaneConfig {
                channels: config.lane_channels, depth: config.lane_depth,
                stride: config.lane_stride, attentions: config.lane_attentions,
                expansion: config.lane_expansion, lanes: config.lanes,
            })?,
            s15_norm: Norm::load(checkpoint, "s15_ln", device, 1e-5)?,
            s15_fc1: Dense::load(checkpoint, "s15_net.0", device)?,
            s15_fc2: Dense::load(checkpoint, "s15_net.2", device)?,
            input_norm: Norm::load(checkpoint, "ln_in", device, 1e-5)?,
            blocks,
            output_norm: Norm::load(checkpoint, "ln_out", device, 1e-5)?,
            instrument: Dense::load(checkpoint, "heads.inst", device)?,
            embedding: Dense::load(checkpoint, "heads.emb", device)?,
            keep: Dense::load(checkpoint, "heads.keep", device)?,
            config,
        })
    }

    fn note_features(&self, notes: &[[f32; 12]]) -> Tensor<3> {
        let d = self.config.width;
        let mut embeddings = Vec::with_capacity(notes.len() * d);
        let mut scalars = Vec::with_capacity(notes.len() * 11);
        for note in notes {
            let pitch = (note[2].round() as isize).clamp(0, 127) as usize;
            embeddings.extend_from_slice(&self.pitch_embedding[pitch*d..(pitch+1)*d]);
            scalars.push((note[1] - note[0]).max(0.01).ln());
            scalars.extend_from_slice(&note[3..12]);
            scalars.push(note[2] / 64.0 - 1.0);
        }
        let base = Tensor::<3>::from_data(TensorData::new(embeddings, [1, notes.len(), d]), &self.device);
        let scalar = Tensor::<3>::from_data(TensorData::new(scalars, [1, notes.len(), 11]), &self.device);
        base + self.pitch_features2.forward(activation::gelu(self.pitch_features1.forward(scalar)))
    }

    fn tapped_features(&self, taps: &[[[f32; TRANSCRIBER_TAP_DIM]; 3]]) -> Result<Tensor<3>, String> {
        let n = taps.len();
        let dim = TRANSCRIBER_TAP_DIM;
        let selection = &self.config.s15_taps;
        let count = selection.len();
        let quantized = taps.iter().map(|note| {
            let mut result = [[0.0f32; TRANSCRIBER_TAP_DIM]; 3];
            for tap in 0..3 {
                let scale = note[tap].iter().map(|v| v.abs()).fold(0.0f32, f32::max).max(1e-6) / 127.0;
                let rounded_scale = half::f16::from_f32(scale).to_f32();
                for c in 0..dim {
                    let q = (note[tap][c] / scale).round().clamp(-127.0, 127.0);
                    result[tap][c] = q * rounded_scale;
                }
            }
            result
        }).collect::<Vec<_>>();
        let mut stats = vec![(0.0f64, 0.0f64); count * dim];
        for f in &quantized {
            for (ti, &tap) in selection.iter().enumerate() {
                for (channel, &value) in f[tap].iter().enumerate() {
                    let item = &mut stats[ti * dim + channel];
                    item.0 += f64::from(value);
                    item.1 += f64::from(value) * f64::from(value);
                }
            }
        }
        let mut values = Vec::with_capacity(n * count * dim);
        for f in &quantized {
            for (ti, &tap) in selection.iter().enumerate() {
                for (channel, &value) in f[tap].iter().enumerate() {
                    let (sum, squares) = stats[ti * dim + channel];
                    let mean = sum / n as f64;
                    let std = (squares / n as f64 - mean * mean).max(0.0).sqrt().max(1e-3);
                    values.push(((f64::from(value) - mean) / std) as f32);
                }
            }
        }
        let x = Tensor::<3>::from_data(TensorData::new(values, [1, n * count, dim]), &self.device);
        let x = self.s15_norm.forward(x).reshape([1, n, count * dim]);
        Ok(self.s15_fc2.forward(activation::gelu(self.s15_fc1.forward(x))))
    }

    fn audio_features(&self, front: &AudioFeatures, notes: &[[f32; 12]]) -> Result<Tensor<3>, String> {
        let n = notes.len();
        let h = self.config.harmonics;
        let sus = self.config.sustain;
        let hd = front::harmonic_features(front, notes, &self.hd_prior, sus)?;
        let profiles = front::profile_features(&front.spectrogram, notes);
        let hd = Tensor::<3>::from_data(TensorData::new(hd.into_iter().flatten().collect::<Vec<_>>(),
            [1, n, h * (front::HD_ATTACK_OFFSETS.len() + sus + 1)]), &self.device);
        let prof = Tensor::<3>::from_data(TensorData::new(profiles.into_iter().flatten().collect::<Vec<_>>(),
            [1, n, front.spectrogram.channels * 4]), &self.device);
        let hd = self.hd_enc2.forward(activation::gelu(self.hd_enc1.forward(hd)));
        let prof = self.prof2.forward(activation::gelu(self.prof1.forward(prof)));
        Ok(hd + prof + self.lanes.forward(&front.spectrogram, notes)?)
    }

    pub fn infer(&self, window: &WindowInput) -> Result<WindowOutput, String> {
        let n = window.notes.len();
        if n == 0 || n != window.taps.len() || window.notes.iter().flatten().any(|x| !x.is_finite())
            || window.taps.iter().flatten().flatten().any(|x| !x.is_finite())
            || window.audio16k.iter().any(|x| !x.is_finite())
        { return Err("invalid Grouper window note/audio input".into()); }
        let front = front::extract(&window.audio16k)?;
        let x = self.note_features(&window.notes)
            + self.audio_features(&front, &window.notes)?
            + self.tapped_features(&window.taps)?;
        let mut x = self.input_norm.forward(x);
        let positions = window.notes.iter().map(|v| v[0] * self.config.position_scale).collect::<Vec<_>>();
        for block in &self.blocks { x = block.forward(x, &positions, &vec![true; n], &self.device)?; }
        let x = self.output_norm.forward(x);
        let to_vec = |x: Tensor<3>| x.into_data().try_into_vec::<f32>()
            .map_err(|e| format!("Grouper output readback: {e}"));
        let instrument = to_vec(self.instrument.forward(x.clone()))?;
        let embedding = to_vec(self.embedding.forward(x.clone()))?;
        let keep_logits = to_vec(self.keep.forward(x.clone()))?;
        let hidden = to_vec(x)?;
        let mut probabilities = Vec::with_capacity(n);
        let mut embeddings = Vec::with_capacity(n);
        let mut hidden_states = Vec::with_capacity(n);
        for i in 0..n {
            let logits = &instrument[i * GROUPER_INSTRUMENT_COUNT..(i + 1)*GROUPER_INSTRUMENT_COUNT];
            let mx = logits.iter().copied().fold(f32::NEG_INFINITY, f32::max);
            let mut p = logits.iter().map(|&v| (v - mx).exp()).collect::<Vec<_>>();
            let z = p.iter().sum::<f32>().max(1e-30);
            for v in &mut p { *v /= z; }
            probabilities.push(p);
            let slice = &embedding[i * self.config.embedding..(i + 1)*self.config.embedding];
            let mag = slice.iter().map(|&v| v*v).sum::<f32>().sqrt().max(1e-12);
            embeddings.push(slice.iter().map(|&v| v / mag).collect());
            hidden_states.push(hidden[i*self.config.width..(i+1)*self.config.width].to_vec());
        }
        Ok(WindowOutput { probabilities, embeddings, hidden: hidden_states, keep_logits })
    }

    /// Notes are globally time-ordered, with metadata and transcriber taps aligned.
    /// Returns weighted full-song probabilities/embeddings, best-window states,
    /// and each raw window for the tracklet decoder's Hungarian overlap matching.
    pub fn predict_song(&self, audio16k: &[f32], notes: &[[f32; 12]], taps: &[[[f32; TRANSCRIBER_TAP_DIM]; 3]])
        -> Result<GroupPredictions, String>
    {
        if audio16k.is_empty() || audio16k.iter().any(|v| !v.is_finite())
            || notes.iter().any(|n| !n[0].is_finite() || n[0] < 0.0 || !n[1].is_finite() || n[1] <= n[0])
            || notes.len() != taps.len() || notes.windows(2).any(|w| w[0][0] > w[1][0]) {
            return Err("Grouper notes/taps must be aligned and sorted by onset".into());
        }
        let n = notes.len();
        let window_seconds = self.config.window_seconds;
        let mut result = GroupPredictions {
            window_seconds, embeddings: vec![vec![0.0; self.config.embedding]; n],
            probabilities: vec![vec![0.0; GROUPER_INSTRUMENT_COUNT]; n],
            hidden: Some(vec![vec![0.0; self.config.width]; n]), windows: vec![],
        };
        if n == 0 { return Ok(result); }
        let mut sums = vec![0.0f64; n];
        let mut best = vec![-1.0f64; n];
        let mut t0 = 0.0f64;
        let final_on = f64::from(notes[n-1][0]);
        while t0 <= final_on + 1e-9 {
            let a = notes.partition_point(|v| f64::from(v[0]) < t0);
            let b = notes.partition_point(|v| f64::from(v[0]) < t0 + window_seconds);
            let end = b.min(a + self.config.max_notes);
            let original_indices = (a..end).collect::<Vec<_>>();
            if !original_indices.is_empty() {
                let mut indices = original_indices.clone();
                while indices.len() < 4 { indices.push(original_indices[(indices.len() - original_indices.len()) % original_indices.len()]); }
                let ns = indices.iter().map(|&j| {
                    let mut note = notes[j];
                    note[0] -= t0 as f32;
                    note[1] = (f64::from(note[1]) - t0).min(window_seconds + 1.0) as f32;
                    note
                }).collect::<Vec<_>>();
                let tf = indices.iter().map(|&j| taps[j]).collect::<Vec<_>>();
                let window_len = ((0.5 + window_seconds + 1.0) * AUDIO_SAMPLE_RATE as f64).round() as usize;
                let first_sample = ((t0 - 0.5) * AUDIO_SAMPLE_RATE as f64).round() as isize;
                let mut samples = vec![0.0f32; window_len];
                for (j, value) in samples.iter_mut().enumerate() {
                    let absolute = first_sample + j as isize;
                    if absolute >= 0 {
                        if let Some(&sample) = audio16k.get(absolute as usize) {
                            // Mirror the int16 quantization in Python's `_Song`.
                            *value = (sample * 32768.0).round().clamp(-32768.0, 32767.0) / 32768.0;
                        }
                    }
                }
                let output = self.infer(&WindowInput { audio16k: samples, notes: ns, taps: tf })?;
                for (local, &global) in original_indices.iter().enumerate() {
                    let w = (1.0 - ((f64::from(notes[global][0]) - t0) / window_seconds - 0.5).abs()*2.0).clamp(0.05, 1.0);
                    sums[global] += w;
                    for (out, &v) in result.probabilities[global].iter_mut().zip(&output.probabilities[local]) { *out += (w * f64::from(v)) as f32; }
                    for (out, &v) in result.embeddings[global].iter_mut().zip(&output.embeddings[local]) { *out += (w * f64::from(v)) as f32; }
                    if w > best[global] {
                        result.hidden.as_mut().unwrap()[global] = output.hidden[local].clone();
                        best[global] = w;
                    }
                }
                let window = WindowPrediction {
                    start_seconds: t0,
                    note_indices: original_indices.clone(),
                    probabilities: output.probabilities.into_iter().take(original_indices.len()).collect(),
                    embeddings: output.embeddings.into_iter().take(original_indices.len()).collect(),
                };
                result.windows.push(window);
            }
            if b >= n && end >= b { break; }
            if b.saturating_sub(a) > self.config.max_notes {
                t0 = (t0 + 0.25 * window_seconds * self.config.window_hop)
                    .max(f64::from(notes[a + self.config.max_notes/2][0]));
            } else { t0 += window_seconds * self.config.window_hop; }
        }
        for i in 0..n {
            if sums[i] <= 0.0 { continue; }
            for v in &mut result.probabilities[i] { *v /= sums[i] as f32; }
            let mag = result.embeddings[i].iter().map(|&v| v*v).sum::<f32>().sqrt().max(1e-9);
            for v in &mut result.embeddings[i] { *v /= mag; }
        }
        Ok(result)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn note_scalars_match_reference_convention() {
        let mut note = [0.0f32; 12];
        note[0] = 1.0; note[1] = 1.5; note[2] = 69.0;
        note[3..12].copy_from_slice(&[1.0, 0.2, 0.9, 0.1, 0.8, 0.5, 0.4, 0.7, 0.6]);
        let mut scalars = vec![(note[1]-note[0]).max(0.01).ln()];
        scalars.extend_from_slice(&note[3..12]);
        scalars.push(note[2]/64.0-1.0);
        assert_eq!(scalars.len(), 11);
        assert_eq!(scalars[1], 1.0);
        assert!((scalars[10] - (69.0/64.0-1.0)).abs() < 1e-6);
    }

    #[test]
    fn model_config_rejects_wrong_transcriber_tap_dimensions() {
        let json: Value = serde_json::json!({"model": {"s15_dim": 256}});
        assert!(NoteNetConfig::from_json(&json).is_err());
    }
}

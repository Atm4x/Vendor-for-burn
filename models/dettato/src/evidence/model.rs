use super::{candidates::{ABSOLUTE_DURATION_BINS, CandidateLogits, DURATION_RATIO_BINS, INSERTION_PITCH_BINS, INSERTION_TIME_BINS, InsertionLogits, NoteLogits, ONSET_BINS}, embedding::{CONTINUOUS_FEATURES, continuous_features}, features::EVIDENCE_FEATURES, repair::RepairInput};
use crate::{checkpoint::{Checkpoint, ModelPaths}};
use burn::tensor::{activation, Device, Tensor, TensorData};
use serde_json::Value;
use std::path::Path;

#[derive(Clone, Debug)]
pub struct RepairModelConfig {
    pub width: usize,
    pub layers: usize,
    pub heads: usize,
    pub feedforward_multiplier: usize,
    pub evidence_features: usize,
    pub keep_threshold: f32,
    pub insert_threshold: f32,
}

impl Default for RepairModelConfig {
    fn default() -> Self {
        Self { width: 192, layers: 9, heads: 6, feedforward_multiplier: 3,
            evidence_features: EVIDENCE_FEATURES, keep_threshold: 0.4, insert_threshold: 0.6 }
    }
}

impl RepairModelConfig {
    pub fn from_json(json: &Value) -> Result<Self, String> {
        let mut result = Self::default();
        let model = json.get("model").ok_or("Evidence config missing model")?;
        let integer = |key: &str, fallback: usize| -> Result<usize, String> {
            model.get(key).map(|v| v.as_u64().map(|v| v as usize)
                .ok_or_else(|| format!("Evidence model.{key} must be integer"))).unwrap_or(Ok(fallback))
        };
        result.width = integer("d", result.width)?;
        result.layers = integer("layers", result.layers)?;
        result.heads = integer("heads", result.heads)?;
        result.feedforward_multiplier = integer("ff_mult", result.feedforward_multiplier)?;
        result.evidence_features = integer("n_ev", result.evidence_features)?;
        if let Some(v) = json.get("keep_th") {
            result.keep_threshold = v.as_f64().ok_or("invalid keep_th")? as f32;
        }
        if let Some(v) = json.get("insert_th") {
            result.insert_threshold = v.as_f64().ok_or("invalid insert_th")? as f32;
        }
        if result.width == 0 || result.heads == 0 || result.width % result.heads != 0
            || result.width / result.heads < 28 || result.layers == 0
            || result.evidence_features != EVIDENCE_FEATURES
            || result.feedforward_multiplier == 0
        { return Err("invalid Evidence network parameters".into()); }
        Ok(result)
    }
}

#[derive(Clone)]
struct Linear {
    weight: Tensor<2>,
    bias: Option<Tensor<1>>,
}

impl Linear {
    fn load(checkpoint: &Checkpoint, name: &str, device: &Device, bias: bool) -> Result<Self, String> {
        let weight = checkpoint.tensor::<2>(&format!("{name}.weight"), device)?.transpose();
        let bias = if bias { Some(checkpoint.tensor::<1>(&format!("{name}.bias"), device)?) }
            else { None };
        if bias.as_ref().is_some_and(|b| b.dims()[0] != weight.dims()[1]) {
            return Err(format!("{name}: incompatible bias"));
        }
        Ok(Self { weight, bias })
    }

    fn forward(&self, x: Tensor<3>) -> Tensor<3> {
        let [input, output] = self.weight.dims();
        let y = x.matmul(self.weight.clone().reshape([1, input, output]));
        if let Some(bias) = &self.bias { y + bias.clone().reshape([1, 1, output]) } else { y }
    }
}

#[derive(Clone)]
struct RmsNorm {
    weight: Tensor<1>,
}

impl RmsNorm {
    fn load(checkpoint: &Checkpoint, name: &str, device: &Device) -> Result<Self, String> {
        Ok(Self { weight: checkpoint.tensor::<1>(&format!("{name}.w"), device)? })
    }

    fn forward(&self, x: Tensor<3>) -> Tensor<3> {
        let d = self.weight.dims()[0];
        let root = (x.clone().powf_scalar(2.0).mean_dim(2) + 1e-6).sqrt().recip();
        x * root * self.weight.clone().reshape([1, 1, d])
    }
}

#[derive(Clone)]
struct Swiglu {
    gate: Linear,
    out: Linear,
    hidden: usize,
}

impl Swiglu {
    fn load(checkpoint: &Checkpoint, name: &str, device: &Device, hidden: usize) -> Result<Self, String> {
        Ok(Self {
            gate: Linear::load(checkpoint, &format!("{name}.w1"), device, false)?,
            out: Linear::load(checkpoint, &format!("{name}.w2"), device, false)?,
            hidden,
        })
    }

    fn forward(&self, x: Tensor<3>) -> Tensor<3> {
        let y = self.gate.forward(x);
        let a = y.clone().narrow(2, 0, self.hidden);
        let b = y.narrow(2, self.hidden, self.hidden);
        self.out.forward(a.clone() * activation::sigmoid(a) * b)
    }
}

#[derive(Clone)]
struct Attention {
    query: Linear,
    key_value: Linear,
    output: Linear,
    heads: usize,
    head_dim: usize,
}

impl Attention {
    fn load(checkpoint: &Checkpoint, name: &str, device: &Device, width: usize, heads: usize) -> Result<Self, String> {
        Ok(Self {
            query: Linear::load(checkpoint, &format!("{name}.q"), device, false)?,
            key_value: Linear::load(checkpoint, &format!("{name}.kv"), device, false)?,
            output: Linear::load(checkpoint, &format!("{name}.o"), device, false)?,
            heads, head_dim: width / heads,
        })
    }

    fn rotate(x: Tensor<4>, c: &Tensor<5>, s: &Tensor<5>) -> Tensor<4> {
        let [batch, heads, notes, hd] = x.dims();
        let used = 28;
        let rotations: Tensor<5> = x.clone().narrow(3, 0, used)
            .reshape([batch, heads, notes, used / 2, 2]);
        let a = rotations.clone().narrow(4, 0, 1);
        let b = rotations.narrow(4, 1, 1);
        let one = a.clone() * c.clone() - b.clone() * s.clone();
        let two = a * s.clone() + b * c.clone();
        let combined: Tensor<4> = Tensor::cat(vec![one, two], 4)
            .reshape([batch, heads, notes, used]);
        if hd == used { combined }
        else { Tensor::cat(vec![combined, x.narrow(3, used, hd - used)], 3) }
    }

    fn forward(&self, x: Tensor<3>, cos: &Tensor<5>, sin: &Tensor<5>) -> Tensor<3> {
        let [batch, n, width] = x.dims();
        let query = self.query.forward(x.clone())
            .reshape([batch, n, self.heads, self.head_dim]).permute([0, 2, 1, 3]);
        let kv = self.key_value.forward(x)
            .reshape([batch, n, 2, self.heads, self.head_dim]);
        let key = kv.clone().narrow(2, 0, 1)
            .reshape([batch, n, self.heads, self.head_dim]).permute([0, 2, 1, 3]);
        let val = kv.narrow(2, 1, 1)
            .reshape([batch, n, self.heads, self.head_dim]).permute([0, 2, 1, 3]);
        let q = Self::rotate(query, cos, sin);
        let k = Self::rotate(key, cos, sin);
        let logits = q.matmul(k.transpose()) * (self.head_dim as f32).powf(-0.5);
        let prob = activation::softmax(logits, 3);
        let z = prob.matmul(val).permute([0, 2, 1, 3]).reshape([batch, n, width]);
        self.output.forward(z)
    }
}

struct Block {
    norm_attention: RmsNorm,
    attention: Attention,
    norm_ff: RmsNorm,
    feedforward: Swiglu,
}

impl Block {
    fn load(checkpoint: &Checkpoint, name: &str, device: &Device, cfg: &RepairModelConfig) -> Result<Self, String> {
        let hidden = ((cfg.width * cfg.feedforward_multiplier) as f64 / 32.0)
            .round_ties_even() as usize * 32;
        Ok(Self {
            norm_attention: RmsNorm::load(checkpoint, &format!("{name}.n1"), device)?,
            attention: Attention::load(checkpoint, &format!("{name}.att"), device, cfg.width, cfg.heads)?,
            norm_ff: RmsNorm::load(checkpoint, &format!("{name}.n2"), device)?,
            feedforward: Swiglu::load(checkpoint, &format!("{name}.ff"), device, hidden)?,
        })
    }

    fn forward(&self, mut x: Tensor<3>, c: &Tensor<5>, s: &Tensor<5>) -> Tensor<3> {
        x = x.clone() + self.attention.forward(self.norm_attention.forward(x), c, s);
        x.clone() + self.feedforward.forward(self.norm_ff.forward(x))
    }
}

struct Embedding {
    instruments: Vec<f32>,
    pitches: Vec<f32>,
    velocities: Vec<f32>,
    cont_one: Linear,
    cont_two: Linear,
    evidence_one: Linear,
    evidence_two: Linear,
    norm: RmsNorm,
    width: usize,
}

impl Embedding {
    fn load(checkpoint: &Checkpoint, device: &Device, width: usize) -> Result<Self, String> {
        let table = |name: &str, rows: usize| -> Result<Vec<f32>, String> {
            let (shape, values) = checkpoint.values(name)?;
            if shape != [rows, width] { return Err(format!("invalid Evidence embedding {name} shape: {shape:?}")); }
            Ok(values)
        };
        Ok(Self {
            instruments: table("emb.inst.weight", 130)?,
            pitches: table("emb.pitch.weight", 256)?,
            velocities: table("emb.vel.weight", 16)?,
            cont_one: Linear::load(checkpoint, "emb.cont.0", device, true)?,
            cont_two: Linear::load(checkpoint, "emb.cont.2", device, true)?,
            evidence_one: Linear::load(checkpoint, "emb.evm.0", device, true)?,
            evidence_two: Linear::load(checkpoint, "emb.evm.2", device, true)?,
            norm: RmsNorm::load(checkpoint, "emb.norm", device)?,
            width,
        })
    }

    fn forward(&self, notes: &[RepairInput], evidence: &[[f32; EVIDENCE_FEATURES]], window_start: f64, device: &Device)
        -> Result<Tensor<3>, String>
    {
        if notes.len() != evidence.len() || notes.is_empty() {
            return Err("Evidence embedding requires non-empty aligned notes/evidence".into());
        }
        let features = continuous_features(notes, window_start)?;
        let n = notes.len();
        let mut lookup = vec![0.0f32; n * self.width];
        for (i, note) in notes.iter().enumerate() {
            let velocity = (note.velocity / 8).min(15) as usize;
            let pitch = note.pitch as usize;
            for j in 0..self.width {
                lookup[i * self.width + j] = self.instruments[j]
                    + self.pitches[pitch * self.width + j]
                    + self.velocities[velocity * self.width + j];
            }
        }
        let continuous = features.iter().flat_map(|row| row.iter().copied()).collect::<Vec<_>>();
        let cont = Tensor::<3>::from_data(TensorData::new(continuous, [1, n, CONTINUOUS_FEATURES]), device);
        let ev = evidence.iter().flat_map(|row| row.iter().copied()).collect::<Vec<_>>();
        let ev = Tensor::<3>::from_data(TensorData::new(ev, [1, n, EVIDENCE_FEATURES]), device);
        let proj = |x: Tensor<3>, a: &Linear, b: &Linear| {
            let y = a.forward(x);
            b.forward(y.clone() * activation::sigmoid(y))
        };
        let base = Tensor::<3>::from_data(TensorData::new(lookup, [1, n, self.width]), device);
        Ok(self.norm.forward(base + proj(cont, &self.cont_one, &self.cont_two)
            + proj(ev, &self.evidence_one, &self.evidence_two)))
    }
}

pub struct EvidenceNet {
    device: Device,
    config: RepairModelConfig,
    embedding: Embedding,
    blocks: Vec<Block>,
    norm: RmsNorm,
    keep: Linear,
    onset: Linear,
    duration: Linear,
    inserts: [Linear; 2],
}

impl EvidenceNet {
    pub fn from_directory(root: impl AsRef<Path>, device: &Device) -> Result<Self, String> {
        let paths = ModelPaths::in_directory(root, "evidence")?;
        Self::from_paths(&paths, device)
    }

    pub fn from_paths(paths: &ModelPaths, device: &Device) -> Result<Self, String> {
        let config = RepairModelConfig::from_json(&paths.load_config()?)?;
        let checkpoint = Checkpoint::open(&paths.weights)?;
        Self::load(&checkpoint, device, config)
    }

    pub fn load(checkpoint: &Checkpoint, device: &Device, config: RepairModelConfig) -> Result<Self, String> {
        let embedding = Embedding::load(checkpoint, device, config.width)?;
        let blocks = (0..config.layers)
            .map(|i| Block::load(checkpoint, &format!("bb.blocks.{i}"), device, &config))
            .collect::<Result<Vec<_>, _>>()?;
        Ok(Self {
            device: device.clone(),
            embedding, blocks,
            norm: RmsNorm::load(checkpoint, "bb.norm", device)?,
            keep: Linear::load(checkpoint, "keep", device, true)?,
            onset: Linear::load(checkpoint, "don", device, true)?,
            duration: Linear::load(checkpoint, "dur", device, true)?,
            inserts: [
                Linear::load(checkpoint, "ins.0", device, true)?,
                Linear::load(checkpoint, "ins.1", device, true)?,
            ],
            config,
        })
    }

    fn rope(&self, notes: &[RepairInput], start: f64) -> (Tensor<5>, Tensor<5>) {
        let n = notes.len();
        let mut cos = vec![0.0; n * 14];
        let mut sin = vec![0.0; n * 14];
        for (i, note) in notes.iter().enumerate() {
            for (axis, (n_freq, lo, hi, value)) in [
                (10, 0.02f64, 40.0f64, note.onset - start),
                (4, 2.0f64, 96.0f64, f64::from(note.pitch)),
            ].into_iter().enumerate() {
                let offset = if axis == 0 { 0 } else { 10 };
                for j in 0..n_freq {
                    let ratio = j as f64 / (n_freq - 1) as f64;
                    let wavelength = (lo.ln() + ratio * (hi.ln() - lo.ln())).exp();
                    let angle = 2.0 * std::f64::consts::PI * value / wavelength;
                    cos[i * 14 + offset + j] = angle.cos() as f32;
                    sin[i * 14 + offset + j] = angle.sin() as f32;
                }
            }
        }
        (
            Tensor::<5>::from_data(TensorData::new(cos, [1, 1, n, 14, 1]), &self.device),
            Tensor::<5>::from_data(TensorData::new(sin, [1, 1, n, 14, 1]), &self.device),
        )
    }

    pub fn infer(&self, notes: &[RepairInput], evidence: &[[f32; EVIDENCE_FEATURES]], start: f64)
        -> Result<CandidateLogits, String>
    {
        if notes.is_empty() { return Ok(CandidateLogits { notes: Vec::new() }); }
        let (cos, sin) = self.rope(notes, start);
        let mut h = self.embedding.forward(notes, evidence, start, &self.device)?;
        for block in &self.blocks { h = block.forward(h, &cos, &sin); }
        h = self.norm.forward(h);
        let n = notes.len();
        let take = |x: Tensor<3>| -> Result<Vec<f32>, String> {
            x.into_data().try_into_vec::<f32>().map_err(|e| format!("reading Evidence heads: {e}"))
        };
        let keep = take(self.keep.forward(h.clone()))?;
        let on = take(self.onset.forward(h.clone()))?;
        let dur = take(self.duration.forward(h.clone()))?;
        let ins0 = take(self.inserts[0].forward(h.clone()))?;
        let ins1 = take(self.inserts[1].forward(h))?;
        let ins_size = 1 + INSERTION_PITCH_BINS + INSERTION_TIME_BINS + ABSOLUTE_DURATION_BINS;
        if keep.len() != n || on.len() != n * ONSET_BINS || dur.len() != n * DURATION_RATIO_BINS
            || ins0.len() != n * ins_size || ins1.len() != n * ins_size
        { return Err("Evidence head projection shape mismatch".into()); }
        let insertion = |flat: &[f32], i: usize| -> InsertionLogits {
            let pos = i * ins_size;
            InsertionLogits {
                exists: flat[pos],
                pitch: flat[pos + 1..pos + 1 + INSERTION_PITCH_BINS].to_vec(),
                onset: flat[pos + 1 + INSERTION_PITCH_BINS..pos + 1 + INSERTION_PITCH_BINS + INSERTION_TIME_BINS].to_vec(),
                duration: flat[pos + 1 + INSERTION_PITCH_BINS + INSERTION_TIME_BINS..pos + ins_size].to_vec(),
            }
        };
        Ok(CandidateLogits { notes: (0..n).map(|i| NoteLogits {
            keep: keep[i],
            onset: on[i * ONSET_BINS..(i + 1) * ONSET_BINS].to_vec(),
            duration_ratio: dur[i * DURATION_RATIO_BINS..(i + 1) * DURATION_RATIO_BINS].to_vec(),
            insertion: [insertion(&ins0, i), insertion(&ins1, i)],
        }).collect() })
    }

    pub fn config(&self) -> &RepairModelConfig { &self.config }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn model_configuration_parses_original_parameter_names() {
        let value = serde_json::json!({"model":{"d":192,"layers":9,"heads":6,"ff_mult":3,"n_ev":769}});
        let cfg = RepairModelConfig::from_json(&value).unwrap();
        assert_eq!(cfg.width, 192);
        assert_eq!(cfg.layers, 9);
        assert_eq!(cfg.evidence_features, 769);
    }
}

use super::{front::note_rasters, layers::{Dense, Norm}};
use crate::{checkpoint::Checkpoint, hcqt::Features, transcriber::model::Conv2d};
use burn::tensor::{Device, Tensor, TensorData, activation};

const PITCH_OFFSETS: [isize; 14] = [-2, -1, 1, 2, -12, 12, -19, 19, -24, 24, -28, 28, -31, 31];

pub(super) struct LaneConfig {
    pub channels: usize,
    pub depth: usize,
    pub lanes: usize,
    pub stride: usize,
    pub attentions: usize,
    pub expansion: usize,
}

struct LayerNorm2d {
    weight: Tensor<1>,
    bias: Tensor<1>,
}

impl LayerNorm2d {
    fn load(ckpt: &Checkpoint, key: &str, device: &Device) -> Result<Self, String> {
        let weight = ckpt.tensor::<1>(&format!("{key}.w"), device)?;
        let bias = ckpt.tensor::<1>(&format!("{key}.b"), device)?;
        if weight.dims() != bias.dims() { return Err(format!("{key}: invalid LayerNorm2d")); }
        Ok(Self { weight, bias })
    }
    fn forward(&self, x: Tensor<4>) -> Tensor<4> {
        let channels = self.weight.dims()[0];
        let centered = x.clone() - x.mean_dim(1);
        let variance = centered.clone().powf_scalar(2.0).mean_dim(1);
        centered * (variance + 1e-6).sqrt().recip()
            * self.weight.clone().reshape([1, channels, 1, 1])
            + self.bias.clone().reshape([1, channels, 1, 1])
    }
}

struct LaneBlock {
    dw: Conv2d,
    taps: Tensor<2>,
    norm: LayerNorm2d,
    pw1: Conv2d,
    pw2: Conv2d,
    gamma: Tensor<1>,
}

impl LaneBlock {
    fn load(ckpt: &Checkpoint, key: &str, device: &Device, dilation: usize, channels: usize) -> Result<Self, String> {
        let taps = ckpt.tensor::<2>(&format!("{key}.pm.taps"), device)?;
        let gamma = ckpt.tensor::<1>(&format!("{key}.gamma"), device)?;
        if taps.dims() != [channels, PITCH_OFFSETS.len()] || gamma.dims() != [channels] {
            return Err(format!("{key}: invalid lane-block channels"));
        }
        Ok(Self {
            dw: Conv2d::load(ckpt, &format!("{key}.dw_t"), device, [1, 1], [3*dilation, 0], [dilation, 1], channels)?,
            taps,
            norm: LayerNorm2d::load(ckpt, &format!("{key}.norm"), device)?,
            pw1: Conv2d::load(ckpt, &format!("{key}.pw1"), device, [1, 1], [0, 0], [1, 1], 1)?,
            pw2: Conv2d::load(ckpt, &format!("{key}.pw2"), device, [1, 1], [0, 0], [1, 1], 1)?,
            gamma,
        })
    }

    fn pitch_mix(&self, x: Tensor<4>) -> Tensor<4> {
        let [batch, channels, frames, lanes] = x.dims();
        let device = x.device();
        let mut out = Tensor::<4>::zeros([batch, channels, frames, lanes], &device);
        for (index, &offset) in PITCH_OFFSETS.iter().enumerate() {
            let skip = offset.unsigned_abs();
            if skip >= lanes { continue; }
            let moved = if offset > 0 {
                Tensor::cat(vec![
                    x.clone().narrow(3, skip, lanes - skip),
                    Tensor::<4>::zeros([batch, channels, frames, skip], &device),
                ], 3)
            } else {
                Tensor::cat(vec![
                    Tensor::<4>::zeros([batch, channels, frames, skip], &device),
                    x.clone().narrow(3, 0, lanes - skip),
                ], 3)
            };
            out = out + moved * self.taps.clone().narrow(1, index, 1).reshape([1, channels, 1, 1]);
        }
        out
    }

    fn forward(&self, x: Tensor<4>) -> Tensor<4> {
        let channels = self.gamma.dims()[0];
        let conv = self.dw.forward(x.clone());
        let mixed = self.norm.forward(conv.clone() + self.pitch_mix(conv));
        x + self.pw2.forward(activation::gelu(self.pw1.forward(mixed)))
            * self.gamma.clone().reshape([1, channels, 1, 1])
    }
}

struct PitchAttention {
    channels: usize,
    heads: usize,
    lanes: usize,
    norm: Norm,
    qkv: Dense,
    out: Dense,
    gamma: Tensor<1>,
    rel_bias: Tensor<2>,
}

impl PitchAttention {
    fn load(ckpt: &Checkpoint, key: &str, device: &Device, channels: usize, lanes: usize) -> Result<Self, String> {
        let gamma = ckpt.tensor::<1>(&format!("{key}.gamma"), device)?;
        let bias = ckpt.tensor::<2>(&format!("{key}.rel_bias"), device)?;
        let [heads, offsets] = bias.dims();
        if offsets != lanes * 2 - 1 || heads == 0 || channels % heads != 0 || gamma.dims() != [channels] {
            return Err(format!("{key}: invalid pitch-attention dimensions"));
        }
        Ok(Self {
            channels, heads, lanes,
            norm: Norm::load(ckpt, &format!("{key}.ln"), device, 1e-5)?,
            qkv: Dense::load(ckpt, &format!("{key}.qkv"), device)?,
            out: Dense::load(ckpt, &format!("{key}.out"), device)?,
            gamma,
            rel_bias: bias,
        })
    }

    fn forward(&self, x: Tensor<4>) -> Result<Tensor<4>, String> {
        let [batch, channels, t, p] = x.dims();
        if channels != self.channels || p != self.lanes { return Err("pitch attention input width mismatch".into()); }
        let n = batch * t;
        let hd = channels / self.heads;
        let seq = x.clone().permute([0, 2, 3, 1]).reshape([n, p, channels]);
        let qkv = self.qkv.forward(self.norm.forward(seq)).reshape([n, p, 3, self.heads, hd]);
        let get = |part: usize| qkv.clone().narrow(2, part, 1)
            .reshape([n, p, self.heads, hd]).permute([0, 2, 1, 3]);
        let q = get(0);
        let k = get(1);
        let v = get(2);
        let device = x.device();
        let bias = self.rel_bias.clone().into_data().try_into_vec::<f32>()
            .map_err(|e| format!("reading pitch-relative-bias: {e}"))?;
        let mut expanded = Vec::with_capacity(self.heads * p * p);
        for h in 0..self.heads {
            for i in 0..p {
                for j in 0..p {
                    expanded.push(bias[h * (2*p-1) + (j as isize - i as isize + p as isize - 1) as usize]);
                }
            }
        }
        let bias = Tensor::<4>::from_data(TensorData::new(expanded, [1, self.heads, p, p]), &device);
        let weights = activation::softmax(q.matmul(k.transpose()) * (hd as f32).powf(-0.5) + bias, 3);
        let context = weights.matmul(v).permute([0, 2, 1, 3]).reshape([n, p, channels]);
        let result = self.out.forward(context) * self.gamma.clone().reshape([1, 1, channels]);
        Ok(x + result.reshape([batch, t, p, channels]).permute([0, 3, 1, 2]))
    }
}

pub(super) struct LaneFront {
    stem: Conv2d,
    occupancy: Conv2d,
    blocks: Vec<LaneBlock>,
    attentions: Vec<Option<PitchAttention>>,
    read: Dense,
    config: LaneConfig,
    device: Device,
}

impl LaneFront {
    pub fn load(ckpt: &Checkpoint, device: &Device, config: LaneConfig) -> Result<Self, String> {
        if config.stride == 0 || config.lanes == 0 || config.channels == 0 || config.depth == 0 {
            return Err("invalid LaneFront configuration".into());
        }
        let mut blocks = Vec::with_capacity(config.depth);
        let mut attentions = Vec::with_capacity(config.depth);
        let interval = (config.depth / config.attentions.max(1)).max(1);
        let mut selected = Vec::new();
        for i in (0..config.depth).rev().step_by(interval).take(config.attentions) { selected.push(i); }
        for i in 0..config.depth {
            blocks.push(LaneBlock::load(ckpt, &format!("lanes.blocks.{i}"), device,
                [1, 2, 4, 8][i % 4], config.channels)?);
            attentions.push(if selected.contains(&i) { Some(PitchAttention::load(ckpt, &format!("lanes.attn.{i}"), device,
                config.channels, config.lanes)?) } else { None });
        }
        Ok(Self {
            stem: Conv2d::load(ckpt, "lanes.stem", device,
                [config.stride, 3], [0, 1], [1, 1], 1)?,
            occupancy: Conv2d::load(ckpt, "lanes.occ", device, [1, 1], [0, 0], [1, 1], 1)?,
            read: Dense::load(ckpt, "lanes.read", device)?,
            blocks, attentions, config,
            device: device.clone(),
        })
    }

    pub fn forward(&self, features: &Features, notes: &[[f32; 12]]) -> Result<Tensor<3>, String> {
        if features.frames < self.config.stride || features.bins / 3 != self.config.lanes {
            return Err("LaneFront feature grid must have 3 frequency bins per lane".into());
        }
        let x = Tensor::<4>::from_data(TensorData::new(features.data.clone(),
            [1, features.channels, features.frames, features.bins]), &self.device);
        let mut y = self.stem.forward(x);
        let [_, c, t, p] = y.dims();
        if p != self.config.lanes || c != self.config.channels { return Err("LaneFront stem mismatch".into()); }
        let rasters = note_rasters(notes, t, p, self.config.stride);
        y = y + self.occupancy.forward(Tensor::<4>::from_data(TensorData::new(rasters, [1, 4, t, p]), &self.device));
        for (block, attention) in self.blocks.iter().zip(&self.attentions) {
            y = block.forward(y);
            if let Some(attention) = attention { y = attention.forward(y)?; }
        }
        let flat = y.into_data().try_into_vec::<f32>().map_err(|e| format!("reading LaneFront states: {e}"))?;
        let mut read = Vec::with_capacity(notes.len() * 2 * c);
        for note in notes {
            let fps = 100.0 / self.config.stride as f32;
            let on = ((note[0] + 0.5) * fps).trunc().clamp(0.0, (t - 1) as f32) as usize;
            let off = (((note[1] + 0.5) * fps).trunc() as usize).clamp(on + 1, t);
            let lane = (note[2].round() as isize - 21).clamp(0, (p-1) as isize) as usize;
            for ch in 0..c { read.push(flat[(ch * t + on) * p + lane]); }
            for ch in 0..c {
                let sum = (on..off).map(|frame| flat[(ch * t + frame) * p + lane]).sum::<f32>();
                read.push(sum / (off - on) as f32);
            }
        }
        Ok(self.read.forward(Tensor::<3>::from_data(TensorData::new(read, [1, notes.len(), 2*c]), &self.device)))
    }
}

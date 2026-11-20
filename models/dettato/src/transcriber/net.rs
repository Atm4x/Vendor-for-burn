use super::{decode::{DecodeConfig, FrameMaps}, model::{BatchNorm, Conv2d, LaneBlock, Stem}, sequence::{AxialAttention, Axis, LaneGru}};
use crate::{PITCH_COUNT, checkpoint::Checkpoint, hcqt::Features};
use burn::tensor::{Device, Tensor};
use serde_json::Value;

const DEFAULT_MIX_OFFSETS: [isize; 14] = [-2, -1, 1, 2, -12, 12, -19, 19, -24, 24, -28, 28, -31, 31];

#[derive(Clone, Debug)]
pub struct Settings {
    pub channels: usize,
    pub depth: usize,
    pub refine: usize,
    pub time_stride: usize,
    pub lanes: usize,
    pub dilations: Vec<usize>,
    pub attn_at: Vec<usize>,
    pub gru_at: Vec<usize>,
    pub attn_axes: Vec<String>,
    pub attn_heads: usize,
    pub pitch_offsets: Vec<isize>,
}

impl Default for Settings {
    fn default() -> Self {
        Self {
            channels: 368,
            depth: 15,
            refine: 2,
            time_stride: 2,
            lanes: 88,
            dilations: vec![1, 2, 4, 8],
            attn_at: vec![5, 10, 15],
            gru_at: vec![5, 10, 15],
            attn_axes: vec!["pitch".into(), "time".into()],
            attn_heads: 8,
            pitch_offsets: DEFAULT_MIX_OFFSETS.to_vec(),
        }
    }
}

impl Settings {
    pub fn from_config(config: &Value) -> Result<Self, String> {
        let model = config.get("model").ok_or("transcriber config missing 'model'")?;
        let mut settings = Self::default();
        let number = |name: &str, fallback| -> Result<usize, String> {
            match model.get(name) {
                None => Ok(fallback),
                Some(value) => value.as_u64().map(|n| n as usize)
                    .ok_or_else(|| format!("model.{name} must be a positive integer")),
            }
        };
        let list = |name: &str, fallback: Vec<usize>| -> Result<Vec<usize>, String> {
            match model.get(name) {
                None => Ok(fallback),
                Some(value) => value.as_array().ok_or_else(|| format!("model.{name} must be an array"))?
                    .iter().map(|v| v.as_u64().map(|n| n as usize)
                        .ok_or_else(|| format!("model.{name} must contain positive integers")))
                    .collect(),
            }
        };
        settings.channels = number("c", settings.channels)?;
        settings.depth = number("depth", settings.depth)?;
        settings.refine = number("refine", settings.refine)?;
        settings.time_stride = number("t_stride", settings.time_stride)?;
        settings.lanes = number("n_lanes", settings.lanes)?;
        settings.attn_heads = number("attn_heads", settings.attn_heads)?;
        settings.dilations = list("dilations", settings.dilations)?;
        settings.attn_at = list("attn_at", settings.attn_at)?;
        settings.gru_at = list("gru_at", settings.gru_at)?;
        if let Some(value) = model.get("pmix") {
            settings.pitch_offsets = value.as_array().ok_or("model.pmix is not an array")?
                .iter().map(|v| v.as_i64().map(|n| n as isize)
                    .ok_or_else(|| "model.pmix must contain integer offsets".to_string()))
                .collect::<Result<Vec<_>, _>>()?;
        }
        if let Some(value) = model.get("attn_axes") {
            settings.attn_axes = value.as_array().ok_or("model.attn_axes is not an array")?
                .iter().map(|v| v.as_str().map(str::to_owned)
                    .ok_or_else(|| "model.attn_axes requires strings".to_string()))
                .collect::<Result<Vec<_>, _>>()?;
        }
        if let Some(heads) = model.get("heads") {
            let names = heads.as_array().ok_or("model.heads must be a list")?
                .iter().map(|v| v.as_str().ok_or("model.heads must contain strings"))
                .collect::<Result<Vec<_>, _>>()?;
            if names.as_slice() != super::decode::HEAD_NAMES.as_slice() {
                return Err("transcriber heads must be onset/frame/offset for two voices".into());
            }
        }
        if settings.time_stride == 0 || settings.depth == 0 || settings.lanes < PITCH_COUNT
            || settings.lanes > 128 || settings.dilations.is_empty() || settings.dilations.contains(&0)
            || settings.attn_heads == 0 || settings.channels % settings.attn_heads != 0
            || settings.attn_axes.iter().any(|v| v != "pitch" && v != "time")
        {
            return Err("invalid transcriber model settings".into());
        }
        Ok(settings)
    }

    pub fn decoder(&self, config: &Value) -> Result<DecodeConfig, String> {
        let thresholds = config.get("thresholds").ok_or("missing transcriber thresholds")?;
        let float = |name: &str, fallback: f64| -> Result<f64, String> {
            match thresholds.get(name) {
                None => Ok(fallback),
                Some(v) => v.as_f64().ok_or_else(|| format!("thresholds.{name} is not numeric")),
            }
        };
        let integer = |name: &str, fallback: usize| -> Result<usize, String> {
            match thresholds.get(name) {
                None => Ok(fallback),
                Some(v) => v.as_u64().map(|v| v as usize)
                    .ok_or_else(|| format!("thresholds.{name} is not integer")),
            }
        };
        let defaults = DecodeConfig::default();
        Ok(DecodeConfig {
            on_th: float("on_th", defaults.on_th as f64)? as f32,
            v2_th: match thresholds.get("v2_th") {
                Some(Value::Null) | None => None,
                Some(value) => Some(value.as_f64()
                    .ok_or("thresholds.v2_th is not numeric")? as f32),
            },
            off_th: float("off_th", defaults.off_th as f64)? as f32,
            fr_th: float("fr_th", defaults.fr_th as f64)? as f32,
            fr_low_frames: integer("fr_low_frames", defaults.fr_low_frames)?,
            on_nms: integer("on_nms", defaults.on_nms)?,
            hop_seconds: 0.01 * self.time_stride as f64,
            ..defaults
        })
    }
}

enum NetworkLayer {
    Block(LaneBlock),
    Attention(AxialAttention),
    Gru(LaneGru),
}

impl NetworkLayer {
    fn forward(&self, x: Tensor<4>) -> Tensor<4> {
        match self {
            Self::Block(block) => block.forward(x),
            Self::Attention(block) => block.forward(x),
            Self::Gru(block) => block.forward(x),
        }
    }
}

pub struct LaneNet {
    device: Device,
    pub settings: Settings,
    stem: Stem,
    down: Conv2d,
    layers: Vec<NetworkLayer>,
    up_norm: BatchNorm,
    refine: Vec<LaneBlock>,
    head: Conv2d,
}

pub struct NetworkOutput {
    pub maps: FrameMaps,
    pub stem: Tensor<4>,
    pub hidden: Tensor<4>,
}

impl LaneNet {
    pub fn load(ckpt: &Checkpoint, device: &Device, settings: Settings) -> Result<Self, String> {
        let stem = Stem::load(ckpt, device, settings.time_stride)?;
        let down = Conv2d::load(ckpt, "down", device, [2, 1], [1, 0], [1, 1], settings.channels)?;
        let up_norm = BatchNorm::load(ckpt, "up_norm", device)?;
        let head = Conv2d::load(ckpt, "head", device, [1, 1], [0, 0], [1, 1], 1)?;
        let mut layers = Vec::new();
        for index in 0..settings.depth {
            let key = format!("layers.{}", layers.len());
            layers.push(NetworkLayer::Block(LaneBlock::load(
                ckpt, &key, device,
                settings.dilations[index % settings.dilations.len()], settings.pitch_offsets.clone(),
            )?));
            if settings.attn_at.contains(&(index + 1)) {
                for axis in &settings.attn_axes {
                    let key = format!("layers.{}", layers.len());
                    let axis = if axis == "time" { Axis::Time } else { Axis::Pitch };
                    layers.push(NetworkLayer::Attention(AxialAttention::load(
                        ckpt, &key, device, axis, settings.attn_heads,
                    )?));
                }
            }
            if settings.gru_at.contains(&(index + 1)) {
                let key = format!("layers.{}", layers.len());
                layers.push(NetworkLayer::Gru(LaneGru::load(ckpt, &key, device)?));
            }
        }
        let refine = (0..settings.refine).map(|i| {
            LaneBlock::load(ckpt, &format!("refine.{i}"), device, 1, settings.pitch_offsets.clone())
        }).collect::<Result<Vec<_>, _>>()?;
        Ok(Self { device: device.clone(), settings, stem, down, layers, up_norm, refine, head })
    }

    pub fn infer_features(&self, input: &Features) -> Result<NetworkOutput, String> {
        let stem = self.stem.features(input, &self.device)?;
        let [batch, _, _, frequencies] = stem.dims();
        if batch != 1 || frequencies < self.settings.lanes {
            return Err("invalid stem shape".into());
        }
        let mut h = stem.clone().narrow(3, 0, self.settings.lanes);
        let frames = h.dims()[2];
        let skip = h.clone();
        h = self.down.forward(h);
        for layer in &self.layers { h = layer.forward(h); }
        let down_frames = h.dims()[2];
        let mut interleaved = Vec::with_capacity(down_frames * 2);
        for i in 0..down_frames {
            let frame = h.clone().narrow(2, i, 1);
            interleaved.push(frame.clone());
            interleaved.push(frame);
        }
        h = Tensor::cat(interleaved, 2).narrow(2, 0, frames);
        h = self.up_norm.forward(h + skip);
        for layer in &self.refine { h = layer.forward(h); }
        let hidden = h.narrow(3, 0, PITCH_COUNT);
        let logits = self.head.forward(hidden.clone());
        let [batch, heads, time, pitch] = logits.dims();
        if batch != 1 || heads != 6 || pitch != PITCH_COUNT {
            return Err(format!("invalid transcriber head output: [{batch},{heads},{time},{pitch}]"));
        }
        let flat: Vec<f32> = logits.into_data().try_into_vec::<f32>()
            .map_err(|e| format!("reading Dettato logits: {e}"))?;
        let length = time * PITCH_COUNT;
        let maps = FrameMaps::from_logits(time, std::array::from_fn(|h| {
            flat[h * length..(h + 1) * length].to_vec()
        }))?;
        Ok(NetworkOutput { maps, stem, hidden })
    }
}

use crate::audio::Features;
use crate::weights::Weights;
use anyhow::{Context, Result, ensure};
use burn::tensor::ops::ConvOptions;
use burn::tensor::{Device, Tensor, TensorData, activation, module};
use serde::{Deserialize, Serialize};

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(default)]
pub struct Config {
    pub n_mels: usize,
    pub n_layers: usize,
    pub d_model: usize,
    pub n_heads: usize,
    pub ff_expansion: usize,
    pub conv_kernel: usize,
    pub subsample_kernel: usize,
    pub subsample_channels: usize,
    pub num_classes: usize,
    pub self_attention_model: String,
    pub conv_norm_type: String,
    pub subsampling: String,
    pub subsampling_factor: usize,
    pub pos_emb_max_len: usize,
}
impl Default for Config {
    fn default() -> Self {
        Self {
            n_mels: 64,
            n_layers: 16,
            d_model: 768,
            n_heads: 16,
            ff_expansion: 4,
            conv_kernel: 31,
            subsample_kernel: 3,
            subsample_channels: 768,
            num_classes: 34,
            self_attention_model: "rotary".to_owned(),
            conv_norm_type: "batch_norm".to_owned(),
            subsampling: "conv2d".to_owned(),
            subsampling_factor: 4,
            pos_emb_max_len: 5000,
        }
    }
}
impl Config {
    pub fn validate(&self) -> Result<()> {
        ensure!(
            self.n_mels > 0 && self.n_layers > 0 && self.d_model > 0 && self.num_classes > 1,
            "invalid model dimensions"
        );
        ensure!(self.pos_emb_max_len > 1, "pos_emb_max_len must exceed 1");
        ensure!(
            self.n_heads > 0 && self.d_model % self.n_heads == 0,
            "d_model must be divisible by n_heads"
        );
        ensure!(
            self.conv_kernel % 2 == 1 && self.subsample_kernel % 2 == 1,
            "convolution kernels must be odd"
        );
        ensure!(
            self.subsample_channels > 0 && self.ff_expansion > 0,
            "invalid subsample or feed-forward dimensions"
        );
        if self.self_attention_model == "rotary" {
            ensure!(
                (self.d_model / self.n_heads) % 2 == 0,
                "rotary attention requires even head dimension"
            );
        }
        ensure!(
            matches!(self.self_attention_model.as_str(), "rotary" | "rel_pos"),
            "unsupported attention mode {:?}",
            self.self_attention_model
        );
        ensure!(
            matches!(self.conv_norm_type.as_str(), "batch_norm" | "layer_norm"),
            "unsupported convolution norm {:?}",
            self.conv_norm_type
        );
        ensure!(
            matches!(self.subsampling.as_str(), "conv1d" | "conv2d")
                && self.subsampling_factor == 4,
            "unsupported subsampling {:?}/{}: only factor 4 is implemented",
            self.subsampling,
            self.subsampling_factor
        );
        ensure!(
            self.subsample_channels == self.d_model,
            "subsample_channels must equal d_model for current encoder"
        );
        Ok(())
    }
}

struct Linear {
    weight: Tensor<2>,
    bias: Option<Tensor<1>>,
}
impl Linear {
    fn load(w: &mut Weights, key: &str, d: &Device, bias: bool) -> Result<Self> {
        let weight = w.tensor(&format!("{key}.weight"), d)?;
        let bias = if bias {
            Some(w.tensor(&format!("{key}.bias"), d)?)
        } else {
            None
        };
        Ok(Self { weight, bias })
    }
    fn forward(&self, x: Tensor<3>) -> Tensor<3> {
        let [batch, time, features] = x.dims();
        let [outputs, inputs] = self.weight.dims();
        assert_eq!(features, inputs);
        let mut output = x
            .reshape([batch * time, inputs])
            .matmul(self.weight.clone().transpose())
            .reshape([batch, time, outputs]);
        if let Some(bias) = &self.bias {
            output = output + bias.clone().reshape([1, 1, outputs]);
        };
        output
    }
}
struct Norm {
    weight: Tensor<1>,
    bias: Tensor<1>,
}
impl Norm {
    fn load(w: &mut Weights, key: &str, d: &Device) -> Result<Self> {
        Ok(Self {
            weight: w.tensor(&format!("{key}.weight"), d)?,
            bias: w.tensor(&format!("{key}.bias"), d)?,
        })
    }
    fn forward<const D: usize>(&self, x: Tensor<D>) -> Tensor<D> {
        module::layer_norm(x, self.weight.clone(), Some(self.bias.clone()), 1e-5)
    }
}
struct Conv1 {
    w: Tensor<3>,
    b: Tensor<1>,
    stride: usize,
    pad: usize,
    groups: usize,
}
impl Conv1 {
    fn load(
        w: &mut Weights,
        key: &str,
        d: &Device,
        stride: usize,
        pad: usize,
        groups: usize,
    ) -> Result<Self> {
        Ok(Self {
            w: w.tensor(&format!("{key}.weight"), d)?,
            b: w.tensor(&format!("{key}.bias"), d)?,
            stride,
            pad,
            groups,
        })
    }
    fn forward(&self, x: Tensor<3>) -> Tensor<3> {
        module::conv1d(
            x,
            self.w.clone(),
            Some(self.b.clone()),
            ConvOptions::new([self.stride], [self.pad], [1], self.groups),
        )
    }
}
struct Conv2 {
    w: Tensor<4>,
    b: Tensor<1>,
    stride: usize,
    pad: usize,
}
impl Conv2 {
    fn load(w: &mut Weights, key: &str, d: &Device, stride: usize, pad: usize) -> Result<Self> {
        Ok(Self {
            w: w.tensor(&format!("{key}.weight"), d)?,
            b: w.tensor(&format!("{key}.bias"), d)?,
            stride,
            pad,
        })
    }
    fn forward(&self, x: Tensor<4>) -> Tensor<4> {
        module::conv2d(
            x,
            self.w.clone(),
            Some(self.b.clone()),
            ConvOptions::new([self.stride; 2], [self.pad; 2], [1; 2], 1),
        )
    }
}
struct BatchNorm {
    weight: Tensor<1>,
    bias: Tensor<1>,
    mean: Tensor<1>,
    var: Tensor<1>,
}
impl BatchNorm {
    fn load(w: &mut Weights, key: &str, d: &Device) -> Result<Self> {
        Ok(Self {
            weight: w.tensor(&format!("{key}.weight"), d)?,
            bias: w.tensor(&format!("{key}.bias"), d)?,
            mean: w.tensor(&format!("{key}.running_mean"), d)?,
            var: w.tensor(&format!("{key}.running_var"), d)?,
        })
    }
    fn forward(&self, x: Tensor<3>) -> Tensor<3> {
        let c = self.weight.dims()[0];
        let m = self.mean.clone().reshape([1, c, 1]);
        let v = self.var.clone().reshape([1, c, 1]);
        let g = self.weight.clone().reshape([1, c, 1]);
        let b = self.bias.clone().reshape([1, c, 1]);
        (x - m) / (v + 1e-5).sqrt() * g + b
    }
}
struct FeedForward {
    a: Linear,
    b: Linear,
}
impl FeedForward {
    fn load(w: &mut Weights, key: &str, d: &Device) -> Result<Self> {
        Ok(Self {
            a: Linear::load(w, &format!("{key}.linear1"), d, true)?,
            b: Linear::load(w, &format!("{key}.linear2"), d, true)?,
        })
    }
    fn forward(&self, x: Tensor<3>) -> Tensor<3> {
        self.b.forward(activation::silu(self.a.forward(x)))
    }
}
struct Attention {
    q: Linear,
    k: Linear,
    v: Linear,
    o: Linear,
    heads: usize,
    head_dim: usize,
    position: PositionKind,
}

enum PositionKind {
    Rotary {
        base: f32,
    },
    Relative {
        linear_pos: Linear,
        u: Tensor<2>,
        v: Tensor<2>,
    },
}

fn relative_shift(x: Tensor<4>) -> Tensor<4> {
    let [batch, heads, time, positions] = x.dims();
    assert_eq!(positions, 2 * time - 1);
    let padded = Tensor::cat(
        vec![Tensor::zeros([batch, heads, time, 1], &x.device()), x],
        3,
    );
    padded
        .reshape([batch, heads, 2 * time, time])
        .slice([0..batch, 0..heads, 1..(2 * time), 0..time])
        .reshape([batch, heads, time, positions])
        .slice([0..batch, 0..heads, 0..time, 0..time])
}

impl Attention {
    fn load(w: &mut Weights, key: &str, d: &Device, config: &Config) -> Result<Self> {
        Ok(Self {
            q: Linear::load(w, &format!("{key}.linear_q"), d, true)?,
            k: Linear::load(w, &format!("{key}.linear_k"), d, true)?,
            v: Linear::load(w, &format!("{key}.linear_v"), d, true)?,
            o: Linear::load(w, &format!("{key}.linear_out"), d, true)?,
            heads: config.n_heads,
            head_dim: config.d_model / config.n_heads,
            position: if config.self_attention_model == "rotary" {
                PositionKind::Rotary {
                    base: config.pos_emb_max_len as f32,
                }
            } else {
                PositionKind::Relative {
                    linear_pos: Linear::load(w, &format!("{key}.linear_pos"), d, false)?,
                    u: w.tensor(&format!("{key}.pos_bias_u"), d)?,
                    v: w.tensor(&format!("{key}.pos_bias_v"), d)?,
                }
            },
        })
    }
    fn forward(&self, x: Tensor<3>) -> Tensor<3> {
        let [batch, time, dim] = x.dims();
        match &self.position {
            PositionKind::Rotary { base } => self.forward_rotary(x, batch, time, dim, *base),
            PositionKind::Relative { linear_pos, u, v } => {
                self.forward_relative(x, batch, time, dim, linear_pos, u, v)
            }
        }
    }

    fn forward_relative(
        &self,
        x: Tensor<3>,
        batch: usize,
        time: usize,
        dim: usize,
        linear_pos: &Linear,
        bias_u: &Tensor<2>,
        bias_v: &Tensor<2>,
    ) -> Tensor<3> {
        let q = self
            .q
            .forward(x.clone())
            .reshape([batch, time, self.heads, self.head_dim])
            .permute([0, 2, 1, 3]);
        let k = self
            .k
            .forward(x.clone())
            .reshape([batch, time, self.heads, self.head_dim])
            .permute([0, 2, 3, 1]);
        let v = self
            .v
            .forward(x)
            .reshape([batch, time, self.heads, self.head_dim])
            .permute([0, 2, 1, 3]);
        let device = q.device();
        let mut values = Vec::with_capacity((2 * time - 1) * dim);
        for pos in (-(time as isize - 1)..=(time as isize - 1)).rev() {
            for channel in 0..dim {
                let rate = 10000f32.powf((2 * (channel / 2)) as f32 / dim as f32);
                let phase = pos as f32 / rate;
                values.push(if channel % 2 == 0 {
                    phase.sin()
                } else {
                    phase.cos()
                });
            }
        }
        let pe = Tensor::<3>::from_data(TensorData::new(values, [1, 2 * time - 1, dim]), &device);
        let p = linear_pos
            .forward(pe)
            .reshape([1, 2 * time - 1, self.heads, self.head_dim])
            .permute([0, 2, 3, 1]);
        let p = if batch == 1 {
            p
        } else {
            p.repeat_dim(0, batch)
        };
        let u = bias_u.clone().reshape([1, self.heads, 1, self.head_dim]);
        let v_bias = bias_v.clone().reshape([1, self.heads, 1, self.head_dim]);
        let content = (q.clone() + u).matmul(k);
        let positional = relative_shift((q + v_bias).matmul(p));
        let attention =
            activation::softmax((content + positional) / (self.head_dim as f64).sqrt(), 3);
        let attended = attention
            .matmul(v)
            .permute([0, 2, 1, 3])
            .reshape([batch, time, dim]);
        self.o.forward(attended)
    }

    fn forward_rotary(
        &self,
        x: Tensor<3>,
        batch: usize,
        time: usize,
        dim: usize,
        base: f32,
    ) -> Tensor<3> {
        let device = x.device();
        let half = self.head_dim / 2;
        assert_eq!(self.head_dim % 2, 0, "RoPE needs even head dimension");
        // The upstream applies RoPE to the INPUT before Q/K projections, not to the projected tensors.
        let mut cos = Vec::with_capacity(time * self.head_dim);
        let mut sin = Vec::with_capacity(time * self.head_dim);
        for t in 0..time {
            let phases: Vec<f32> = (0..half)
                .map(|i| t as f32 / base.powf((i * 2) as f32 / self.head_dim as f32))
                .collect();
            for _ in 0..2 {
                for a in &phases {
                    cos.push(a.cos());
                    sin.push(a.sin());
                }
            }
        }
        let cs = Tensor::<4>::from_data(TensorData::new(cos, [1, time, 1, self.head_dim]), &device);
        let sn = Tensor::<4>::from_data(TensorData::new(sin, [1, time, 1, self.head_dim]), &device);
        let initial = x.clone().reshape([batch, time, self.heads, self.head_dim]);
        let rot = Tensor::cat(
            vec![
                initial
                    .clone()
                    .slice([0..batch, 0..time, 0..self.heads, half..self.head_dim])
                    .neg(),
                initial
                    .clone()
                    .slice([0..batch, 0..time, 0..self.heads, 0..half]),
            ],
            3,
        );
        let rotated = (initial * cs + rot * sn).reshape([batch, time, dim]);
        let q = self
            .q
            .forward(rotated.clone())
            .reshape([batch, time, self.heads, self.head_dim])
            .permute([0, 2, 1, 3]);
        let k = self
            .k
            .forward(rotated)
            .reshape([batch, time, self.heads, self.head_dim])
            .permute([0, 2, 3, 1]);
        let v = self
            .v
            .forward(x)
            .reshape([batch, time, self.heads, self.head_dim])
            .permute([0, 2, 1, 3]);
        let probs = activation::softmax(q.matmul(k) / (self.head_dim as f64).sqrt(), 3);
        let attended = probs
            .matmul(v)
            .permute([0, 2, 1, 3])
            .reshape([batch, time, dim]);
        self.o.forward(attended)
    }
}
struct Convolution {
    pointwise1: Conv1,
    depthwise: Conv1,
    pointwise2: Conv1,
    norm: ConvolutionNorm,
}
enum ConvolutionNorm {
    Batch(BatchNorm),
    Layer(Norm),
}
impl ConvolutionNorm {
    fn forward(&self, x: Tensor<3>) -> Tensor<3> {
        match self {
            Self::Batch(n) => n.forward(x),
            Self::Layer(n) => n.forward(x.permute([0, 2, 1])).permute([0, 2, 1]),
        }
    }
}
impl Convolution {
    fn load(w: &mut Weights, key: &str, d: &Device, c: &Config) -> Result<Self> {
        Ok(Self {
            pointwise1: Conv1::load(w, &format!("{key}.pointwise_conv1"), d, 1, 0, 1)?,
            depthwise: Conv1::load(
                w,
                &format!("{key}.depthwise_conv"),
                d,
                1,
                c.conv_kernel / 2,
                c.d_model,
            )?,
            pointwise2: Conv1::load(w, &format!("{key}.pointwise_conv2"), d, 1, 0, 1)?,
            norm: if c.conv_norm_type == "batch_norm" {
                ConvolutionNorm::Batch(BatchNorm::load(w, &format!("{key}.batch_norm"), d)?)
            } else {
                ConvolutionNorm::Layer(Norm::load(w, &format!("{key}.batch_norm"), d)?)
            },
        })
    }
    fn forward(&self, x: Tensor<3>) -> Tensor<3> {
        let [batch, time, channels] = x.dims();
        let projected = self.pointwise1.forward(x.permute([0, 2, 1]));
        let a = projected.clone().slice([0..batch, 0..channels, 0..time]);
        let b = projected.slice([0..batch, channels..channels * 2, 0..time]);
        let gated = a * activation::sigmoid(b);
        let depth = self.depthwise.forward(gated);
        self.pointwise2
            .forward(activation::silu(self.norm.forward(depth)))
            .permute([0, 2, 1])
    }
}
struct Block {
    n1: Norm,
    ff1: FeedForward,
    na: Norm,
    att: Attention,
    nc: Norm,
    conv: Convolution,
    n2: Norm,
    ff2: FeedForward,
    no: Norm,
}
impl Block {
    fn load(w: &mut Weights, key: &str, d: &Device, c: &Config) -> Result<Self> {
        Ok(Self {
            n1: Norm::load(w, &format!("{key}.norm_feed_forward1"), d)?,
            ff1: FeedForward::load(w, &format!("{key}.feed_forward1"), d)?,
            na: Norm::load(w, &format!("{key}.norm_self_att"), d)?,
            att: Attention::load(w, &format!("{key}.self_attn"), d, c)?,
            nc: Norm::load(w, &format!("{key}.norm_conv"), d)?,
            conv: Convolution::load(w, &format!("{key}.conv"), d, c)?,
            n2: Norm::load(w, &format!("{key}.norm_feed_forward2"), d)?,
            ff2: FeedForward::load(w, &format!("{key}.feed_forward2"), d)?,
            no: Norm::load(w, &format!("{key}.norm_out"), d)?,
        })
    }
    fn forward(&self, x: Tensor<3>) -> Tensor<3> {
        let residual = x.clone() + self.ff1.forward(self.n1.forward(x)) * 0.5;
        let residual = residual.clone() + self.att.forward(self.na.forward(residual.clone()));
        let residual = residual.clone() + self.conv.forward(self.nc.forward(residual.clone()));
        let residual = residual.clone() + self.ff2.forward(self.n2.forward(residual.clone())) * 0.5;
        self.no.forward(residual)
    }
}

pub struct EncoderModel {
    config: Config,
    subsampling: Subsampling,
    layers: Vec<Block>,
}

enum Subsampling {
    Conv2 {
        first: Conv2,
        second: Conv2,
        projection: Linear,
    },
    Conv1 {
        first: Conv1,
        second: Conv1,
    },
}

impl EncoderModel {
    pub fn load(w: &mut Weights, config: Config, device: &Device) -> Result<Self> {
        config.validate()?;
        let k = config.subsample_kernel;
        let p = k / 2;
        let subsampling = if config.subsampling == "conv2d" {
            Subsampling::Conv2 {
                first: Conv2::load(w, "encoder.pre_encode.conv.0", device, 2, p)?,
                second: Conv2::load(w, "encoder.pre_encode.conv.2", device, 2, p)?,
                projection: Linear::load(w, "encoder.pre_encode.out", device, true)?,
            }
        } else {
            Subsampling::Conv1 {
                first: Conv1::load(w, "encoder.pre_encode.conv.0", device, 2, p, 1)?,
                second: Conv1::load(w, "encoder.pre_encode.conv.2", device, 2, p, 1)?,
            }
        };
        let mut layers = Vec::with_capacity(config.n_layers);
        for i in 0..config.n_layers {
            layers.push(
                Block::load(w, &format!("encoder.layers.{i}"), device, &config)
                    .with_context(|| format!("loading Conformer block {i}"))?,
            );
        }
        Ok(Self {
            config,
            subsampling,
            layers,
        })
    }

    pub fn encode(&self, features: &Features, device: &Device) -> Result<Tensor<3>> {
        ensure!(features.n_mels == self.config.n_mels, "mel-bin mismatch");
        let mut seq = match &self.subsampling {
            Subsampling::Conv2 {
                first,
                second,
                projection,
            } => {
                let mut x = Tensor::<4>::from_data(
                    TensorData::new(
                        features.values.clone(),
                        [1, 1, features.n_mels, features.frames],
                    ),
                    device,
                )
                .permute([0, 1, 3, 2]);
                x = activation::relu(first.forward(x));
                x = activation::relu(second.forward(x));
                let [batch, channels, time, frequency] = x.dims();
                let seq = x
                    .permute([0, 2, 1, 3])
                    .reshape([batch, time, frequency * channels]);
                ensure!(
                    projection.weight.dims()[1] == frequency * channels,
                    "unexpected subsampling projection input {} != {}",
                    frequency * channels,
                    projection.weight.dims()[1]
                );
                projection.forward(seq)
            }
            Subsampling::Conv1 { first, second } => {
                let x = Tensor::<3>::from_data(
                    TensorData::new(
                        features.values.clone(),
                        [1, features.n_mels, features.frames],
                    ),
                    device,
                );
                activation::relu(second.forward(activation::relu(first.forward(x))))
                    .permute([0, 2, 1])
            }
        };
        for layer in &self.layers {
            seq = layer.forward(seq);
        }
        Ok(seq)
    }

    pub fn n_mels(&self) -> usize {
        self.config.n_mels
    }
    pub fn d_model(&self) -> usize {
        self.config.d_model
    }
}

pub struct CtcModel {
    encoder: EncoderModel,
    head: Conv1,
    num_classes: usize,
}
impl CtcModel {
    pub fn load(w: &mut Weights, config: Config, device: &Device) -> Result<Self> {
        let head = Conv1::load(w, "head.decoder_layers.0", device, 1, 0, 1)?;
        ensure!(
            head.w.dims()[0] == config.num_classes,
            "CTC head classes mismatch"
        );
        Ok(Self {
            encoder: EncoderModel::load(w, config.clone(), device)?,
            head,
            num_classes: config.num_classes,
        })
    }

    pub fn infer_logits(&self, features: &Features, device: &Device) -> Result<(Vec<f32>, usize)> {
        let encoded = self.encoder.encode(features, device)?;
        let logits = self
            .head
            .forward(encoded.permute([0, 2, 1]))
            .permute([0, 2, 1]);
        let [_, out_t, classes] = logits.dims();
        ensure!(classes == self.num_classes, "CTC class mismatch");
        let values = logits
            .into_data()
            .try_to_vec::<f32>()
            .context("retrieving model logits")?;
        Ok((values, out_t))
    }

    pub fn n_mels(&self) -> usize {
        self.encoder.n_mels()
    }
}

/// Utterance-level emotion classifier used by GigaAMEmo:
/// mean-pool the Conformer time dimension and apply a linear classifier.
pub struct EmotionModel {
    encoder: EncoderModel,
    weight: Tensor<2>,
    bias: Tensor<1>,
    num_emotions: usize,
}

impl EmotionModel {
    pub fn load(
        weights: &mut Weights,
        config: Config,
        num_emotions: usize,
        device: &Device,
    ) -> Result<Self> {
        ensure!(
            num_emotions > 1,
            "emotion classifier requires at least two labels"
        );
        let (weight, bias) = if weights.values("head.weight").is_ok() {
            (
                weights.tensor::<2>("head.weight", device)?,
                weights.tensor::<1>("head.bias", device)?,
            )
        } else {
            (
                weights
                    .tensor::<2>("head.0.weight", device)
                    .context("emotion classifier must have a linear head")?,
                weights.tensor::<1>("head.0.bias", device)?,
            )
        };
        ensure!(
            weight.dims() == [num_emotions, config.d_model],
            "emotion head shape {:?} != [{num_emotions}, {}]",
            weight.dims(),
            config.d_model
        );
        ensure!(
            bias.dims() == [num_emotions],
            "emotion head bias size mismatch"
        );
        Ok(Self {
            encoder: EncoderModel::load(weights, config, device)?,
            weight,
            bias,
            num_emotions,
        })
    }

    pub fn infer_probs(&self, features: &Features, device: &Device) -> Result<Vec<f32>> {
        let hidden = self.encoder.encode(features, device)?;
        let [batch, time, channels] = hidden.dims();
        ensure!(
            batch == 1 && time > 0,
            "expected nonempty single-sample encoder output"
        );
        let pooled = hidden.mean_dim(1).reshape([batch, channels]);
        let logits = pooled.matmul(self.weight.clone().transpose())
            + self.bias.clone().reshape([1, self.num_emotions]);
        let probs = activation::softmax(logits, 1)
            .into_data()
            .try_to_vec::<f32>()
            .context("retrieving emotion probabilities")?;
        ensure!(
            probs.iter().all(|v| v.is_finite()),
            "nonfinite emotion probability"
        );
        Ok(probs)
    }
}

#[cfg(all(test, feature = "flex"))]
mod legacy_tests {
    use super::*;

    #[test]
    fn rotary_frequency_base_matches_python_encoder() {
        let defaults = Config::default();
        assert_eq!(defaults.pos_emb_max_len, 5000);
        let parsed: Config = serde_json::from_str(r#"{"pos_emb_max_len": 8192}"#).unwrap();
        assert_eq!(parsed.pos_emb_max_len, 8192);
        let omitted: Config = serde_json::from_str("{}").unwrap();
        assert_eq!(omitted.pos_emb_max_len, 5000);
        let rate = (5000f32).powf(2.0 / 4.0);
        assert!((rate - 5000f32.sqrt()).abs() < 1e-5);
    }
    #[test]
    fn relative_shift_matches_pytorch_reshape() {
        let device = Device::flex();
        let input = Tensor::<4>::from_data(
            TensorData::new((0..15).map(|x| x as f32).collect::<Vec<_>>(), [1, 1, 3, 5]),
            &device,
        );
        let actual = relative_shift(input)
            .into_data()
            .try_to_vec::<f32>()
            .unwrap();
        assert_eq!(actual, [2., 3., 4., 6., 7., 8., 10., 11., 12.]);
    }
}

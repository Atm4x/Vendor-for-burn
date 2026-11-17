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
        }
    }
}
impl Config {
    pub fn validate(&self) -> Result<()> {
        ensure!(
            self.n_mels > 0 && self.n_layers > 0 && self.d_model > 0 && self.num_classes > 1,
            "invalid model dimensions"
        );
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
        })
    }
    fn forward(&self, x: Tensor<3>) -> Tensor<3> {
        let [batch, time, dim] = x.dims();
        let device = x.device();
        let half = self.head_dim / 2;
        assert_eq!(self.head_dim % 2, 0, "RoPE needs even head dimension");
        // The upstream applies RoPE to the INPUT before Q/K projections, not to the projected tensors.
        let mut cos = Vec::with_capacity(time * self.head_dim);
        let mut sin = Vec::with_capacity(time * self.head_dim);
        for t in 0..time {
            let phases: Vec<f32> = (0..half)
                .map(|i| t as f32 / 10000f32.powf((i * 2) as f32 / self.head_dim as f32))
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
    bn: BatchNorm,
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
            bn: BatchNorm::load(w, &format!("{key}.batch_norm"), d)?,
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
            .forward(activation::silu(self.bn.forward(depth)))
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
    sub1: Conv2,
    sub2: Conv2,
    projection: Linear,
    layers: Vec<Block>,
}
impl EncoderModel {
    pub fn load(w: &mut Weights, config: Config, device: &Device) -> Result<Self> {
        config.validate()?;
        let k = config.subsample_kernel;
        let p = k / 2;
        let sub1 = Conv2::load(w, "encoder.pre_encode.conv.0", device, 2, p)?;
        let sub2 = Conv2::load(w, "encoder.pre_encode.conv.2", device, 2, p)?;
        let projection = Linear::load(w, "encoder.pre_encode.out", device, true)?;
        let mut layers = Vec::with_capacity(config.n_layers);
        for i in 0..config.n_layers {
            layers.push(
                Block::load(w, &format!("encoder.layers.{i}"), device, &config)
                    .with_context(|| format!("loading Conformer block {i}"))?,
            );
        }
        Ok(Self {
            config,
            sub1,
            sub2,
            projection,
            layers,
        })
    }

    pub fn encode(&self, features: &Features, device: &Device) -> Result<Tensor<3>> {
        ensure!(features.n_mels == self.config.n_mels, "mel-bin mismatch");
        let mut x = Tensor::<4>::from_data(
            TensorData::new(
                features.values.clone(),
                [1, 1, features.n_mels, features.frames],
            ),
            device,
        )
        .permute([0, 1, 3, 2]);
        x = activation::relu(self.sub1.forward(x));
        x = activation::relu(self.sub2.forward(x));
        let [batch, channels, t, frequency] = x.dims();
        let seq = x
            .permute([0, 2, 1, 3])
            .reshape([batch, t, frequency * channels]);
        ensure!(
            self.projection.weight.dims()[1] == frequency * channels,
            "unexpected subsampling projection input {} != {}",
            frequency * channels,
            self.projection.weight.dims()[1]
        );
        let mut seq = self.projection.forward(seq);
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

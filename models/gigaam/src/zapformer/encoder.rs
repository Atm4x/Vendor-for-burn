use super::language::LanguageRequest;
use super::{
    config::{Activation, StackConfig, ZapformerConfig},
    frontend::ZapformerFrontend,
    layers::{FeedForward, RmsNorm, invalid_mask},
};
use crate::{model::Linear, weights::Weights};
use anyhow::{Context, Result, ensure};
use burn::tensor::{
    Bool, Device, Tensor, TensorData, activation, module,
    ops::{ConvOptions, PadMode},
};
use std::collections::BTreeMap;

struct RopeTable {
    cos: Tensor<4>,
    sin: Tensor<4>,
}
impl RopeTable {
    fn load(w: &mut Weights, key: &str, d: &Device, dim: usize, max_seq: usize) -> Result<Self> {
        let name = format!("{key}.ed{dim}_S{max_seq}_b10000");
        Ok(Self {
            cos: w.checked_tensor(&format!("{name}.cos"), [1, 1, max_seq, dim / 2], d)?,
            sin: w.checked_tensor(&format!("{name}.sin"), [1, 1, max_seq, dim / 2], d)?,
        })
    }
    fn positions(&self, time: usize) -> (Tensor<4>, Tensor<4>) {
        let width = self.cos.dims()[3];
        (
            self.cos.clone().slice([0..1, 0..1, 0..time, 0..width]),
            self.sin.clone().slice([0..1, 0..1, 0..time, 0..width]),
        )
    }
}

fn apply_rope(x: Tensor<4>, cos: &Tensor<4>, sin: &Tensor<4>) -> Tensor<4> {
    let [b, h, t, d] = x.dims();
    let pair = x.reshape([b, h, t, d / 2, 2]);
    let x0 = pair
        .clone()
        .slice([0..b, 0..h, 0..t, 0..d / 2, 0..1])
        .squeeze_dim::<4>(4);
    let x1 = pair
        .slice([0..b, 0..h, 0..t, 0..d / 2, 1..2])
        .squeeze_dim::<4>(4);
    let y0 = x0.clone() * cos.clone() - x1.clone() * sin.clone();
    let y1 = x0 * sin.clone() + x1 * cos.clone();
    Tensor::stack::<5>(vec![y0, y1], 4).reshape([b, h, t, d])
}

struct Attention {
    norm: RmsNorm,
    qk: Linear,
    vg: Linear,
    out: Linear,
    heads: usize,
    qdim: usize,
    vdim: usize,
}
impl Attention {
    fn load(w: &mut Weights, key: &str, d: &Device, s: &StackConfig) -> Result<Self> {
        let q = s.heads * s.query_dim;
        let v = s.heads * s.value_dim;
        Ok(Self {
            norm: RmsNorm::load(w, &format!("{key}.in_norm"), d, s.dim)?,
            qk: Linear::load_checked(w, &format!("{key}.qk_in_proj"), d, s.dim, 2 * q)?,
            vg: Linear::load_checked(w, &format!("{key}.vg_in_proj"), d, s.dim, 3 * v)?,
            out: Linear::load_checked(w, &format!("{key}.out_proj"), d, v, s.dim)?,
            heads: s.heads,
            qdim: s.query_dim,
            vdim: s.value_dim,
        })
    }
    fn forward(
        &self,
        qk: Tensor<3>,
        vg: Tensor<3>,
        rope: &(Tensor<4>, Tensor<4>),
        mask: &Tensor<2, Bool>,
    ) -> Tensor<3> {
        let [b, t, _] = qk.dims();
        let qwidth = self.heads * self.qdim;
        let vwidth = self.heads * self.vdim;
        let qk = self.qk.forward(self.norm.forward(qk));
        let q = qk.clone().slice([0..b, 0..t, 0..qwidth]);
        let k = qk.slice([0..b, 0..t, qwidth..2 * qwidth]);
        let q = apply_rope(
            q.reshape([b, t, self.heads, self.qdim])
                .permute([0, 2, 1, 3]),
            &rope.0,
            &rope.1,
        );
        let k = apply_rope(
            k.reshape([b, t, self.heads, self.qdim])
                .permute([0, 2, 1, 3]),
            &rope.0,
            &rope.1,
        );
        let vg = self.vg.forward(vg);
        let v = vg.clone().slice([0..b, 0..t, 0..vwidth]);
        let gate_in = vg.clone().slice([0..b, 0..t, vwidth..2 * vwidth]);
        let gate_out = vg.slice([0..b, 0..t, 2 * vwidth..3 * vwidth]);
        let v = (v * activation::sigmoid(gate_in))
            .reshape([b, t, self.heads, self.vdim])
            .permute([0, 2, 1, 3]);
        let scores = (q / (self.qdim as f64).sqrt()).matmul(k.transpose());
        let scores = scores.mask_fill(mask.clone().unsqueeze_dims::<4>(&[1, 2]), -10000.0);
        let values = activation::softmax(scores, 3)
            .matmul(v)
            .permute([0, 2, 1, 3])
            .reshape([b, t, vwidth]);
        self.out.forward(values * activation::sigmoid(gate_out))
    }
}

struct Convolution {
    input: Linear,
    weight: Tensor<3>,
    bias: Tensor<1>,
    mean_weight: Tensor<1>,
    output: Linear,
    dim: usize,
    kernel: usize,
    activation: Activation,
}
impl Convolution {
    fn load(
        w: &mut Weights,
        key: &str,
        d: &Device,
        s: &StackConfig,
        activation: Activation,
    ) -> Result<Self> {
        let dim = s.dim;
        Ok(Self {
            input: Linear::load_checked(w, &format!("{key}.in_proj"), d, dim, 2 * dim)?,
            weight: w.checked_tensor(
                &format!("{key}.depthwise_conv.weight"),
                [dim, 1, s.conv_kernel],
                d,
            )?,
            bias: w.checked_tensor(&format!("{key}.depthwise_conv.bias"), [dim], d)?,
            mean_weight: w.checked_tensor(&format!("{key}.weighted_mean.weights"), [dim], d)?,
            output: Linear::load_checked(w, &format!("{key}.out_proj"), d, dim, dim)?,
            dim,
            kernel: s.conv_kernel,
            activation,
        })
    }
    fn forward(&self, x: Tensor<3>, mask: &Tensor<2, Bool>, counts: &Tensor<3>) -> Tensor<3> {
        let [b, t, d] = x.dims();
        let projected = self.input.forward(x);
        let values = projected
            .clone()
            .slice([0..b, 0..t, 0..d])
            .mask_fill(mask.clone().unsqueeze_dim::<3>(2), 0.0);
        let gate = projected.slice([0..b, 0..t, d..2 * d]);
        // Full context uses the same valid-key mean for every query; no quadratic context tensor is needed.
        let mean = values.clone().sum_dim(1) / counts.clone()
            * self.mean_weight.clone().reshape([1, 1, d]);
        let convolved = module::conv1d(
            values.swap_dims(1, 2),
            self.weight.clone(),
            Some(self.bias.clone()),
            ConvOptions::new([1], [self.kernel / 2], [1], self.dim),
        )
        .swap_dims(1, 2);
        self.output.forward(
            self.activation
                .forward((convolved + mean) * activation::sigmoid(gate), false),
        )
    }
}

struct Layer {
    ff1: FeedForward,
    ff2: Option<FeedForward>,
    attention: Attention,
    conv: Option<Convolution>,
    norm: RmsNorm,
}
impl Layer {
    fn load(
        w: &mut Weights,
        key: &str,
        d: &Device,
        s: &StackConfig,
        activation: Activation,
    ) -> Result<Self> {
        Ok(Self {
            ff1: FeedForward::load(
                w,
                &format!("{key}.feed_forward1"),
                d,
                s.dim,
                s.ff_multiple,
                activation,
            )?,
            ff2: if s.conv_kernel > 0 {
                Some(FeedForward::load(
                    w,
                    &format!("{key}.feed_forward2"),
                    d,
                    s.dim,
                    s.ff_multiple,
                    activation,
                )?)
            } else {
                None
            },
            attention: Attention::load(w, &format!("{key}.self_attn"), d, s)?,
            conv: if s.conv_kernel > 0 {
                Some(Convolution::load(
                    w,
                    &format!("{key}.conv_module"),
                    d,
                    s,
                    activation,
                )?)
            } else {
                None
            },
            norm: RmsNorm::load(w, &format!("{key}.norm"), d, s.dim)?,
        })
    }
    fn forward(
        &self,
        residual: Tensor<3>,
        rope: &(Tensor<4>, Tensor<4>),
        mask: &Tensor<2, Bool>,
        counts: &Tensor<3>,
    ) -> Tensor<3> {
        let mut x = residual.clone() + self.ff1.forward(residual.clone());
        x = x.clone() + self.attention.forward(residual.clone(), x, rope, mask);
        if let Some(conv) = &self.conv {
            x = x.clone() + conv.forward(x, mask, counts);
        }
        if let Some(ff) = &self.ff2 {
            x = x.clone() + ff.forward(x);
        }
        self.norm.forward(residual.clone() + (x - residual) * 0.25)
    }
}

fn scalars(w: &mut Weights, key: &str, shape: &[usize]) -> Result<Vec<f32>> {
    let (actual, values) = w.values(key)?;
    ensure!(
        actual == shape && values.iter().all(|v| v.is_finite()),
        "invalid learned scalar {key}: {actual:?}"
    );
    Ok(values)
}

struct Stack {
    proj: Tensor<2>,
    layers: Vec<Layer>,
    residual_scales: Vec<f32>,
    input_scale: f32,
    factor: usize,
    query_dim: usize,
}
impl Stack {
    fn load(
        w: &mut Weights,
        key: &str,
        d: &Device,
        s: &StackConfig,
        embed_dim: usize,
        activation: Activation,
    ) -> Result<Self> {
        let input = embed_dim * s.factor;
        let proj = w.checked_tensor(&format!("{key}.proj.weight"), [s.dim, input], d)?
            * ((input as f64).powf(-0.5) / 0.3);
        let mut layers = Vec::with_capacity(s.layers);
        for i in 0..s.layers {
            layers.push(Layer::load(
                w,
                &format!("{key}.layers.{i}"),
                d,
                s,
                activation,
            )?);
        }
        Ok(Self {
            proj,
            layers,
            residual_scales: scalars(w, &format!("{key}.residual_scales"), &[s.layers + 1])?,
            input_scale: scalars(w, &format!("{key}.input_scale"), &[1])?[0],
            factor: s.factor,
            query_dim: s.query_dim,
        })
    }
    fn forward(
        &self,
        residual: Tensor<3>,
        lengths: &[usize],
        rope: &(Tensor<4>, Tensor<4>),
        condition: Option<&LanguageRequest<'_>>,
        layer_offset: usize,
    ) -> Tensor<3> {
        let [b, t, width] = residual.dims();
        let device = residual.device();
        let [dim, _] = self.proj.dims();
        let mask = invalid_mask(lengths, t, &device);
        let counts = Tensor::from_data(
            TensorData::new(
                lengths.iter().map(|&n| n.max(1) as f32).collect::<Vec<_>>(),
                [b, 1, 1],
            ),
            &device,
        );
        let projected = residual
            .clone()
            .reshape([b * t, width])
            .matmul(self.proj.clone().transpose())
            .reshape([b, t, dim]);
        let mut bypass = projected.clone() * self.residual_scales[0];
        let mut x = projected * self.input_scale;
        for (i, layer) in self.layers.iter().enumerate() {
            x = layer.forward(x, rope, &mask, &counts);
            if let Some(condition) = condition {
                x = condition.apply_layer(layer_offset + i, x);
            }
            bypass = bypass + x.clone() * self.residual_scales[i + 1];
        }
        residual
            + bypass
                .reshape([b * t, dim])
                .matmul(self.proj.clone())
                .reshape([b, t, width])
    }
}

/// Native offline full-context Zapformer. Context-limited and language-conditioned execution are separate gates.
pub struct ZapformerEncoder {
    pub frontend: ZapformerFrontend,
    config: ZapformerConfig,
    stacks: Vec<Stack>,
    ropes: BTreeMap<usize, RopeTable>,
    out_scale: f32,
    out_eps: f32,
}
impl ZapformerEncoder {
    pub fn load(w: &mut Weights, key: &str, d: &Device, config: ZapformerConfig) -> Result<Self> {
        config.validate()?;
        ensure!(
            config.supports_full_context(),
            "checkpoint has no full-context mode"
        );
        let frontend = ZapformerFrontend::load(w, &format!("{key}.encoder_embed"), d, &config)?;
        let mut ropes = BTreeMap::new();
        let mut stacks = Vec::with_capacity(config.stacks.len());
        for (i, s) in config.stacks.iter().enumerate() {
            if let std::collections::btree_map::Entry::Vacant(entry) = ropes.entry(s.query_dim) {
                entry.insert(RopeTable::load(
                    w,
                    &format!("{key}.encoder.rope_bank.tables"),
                    d,
                    s.query_dim,
                    config.max_seq_len,
                )?);
            }
            stacks.push(Stack::load(
                w,
                &format!("{key}.encoder.encoders.{i}"),
                d,
                s,
                config.embed_dim,
                config.activation,
            )?);
        }
        let out_scale = scalars(w, &format!("{key}.encoder.out_norm.scale"), &[])?[0];
        let out_eps = scalars(w, &format!("{key}.encoder.out_norm.eps"), &[])?[0];
        ensure!(
            out_eps != 0.0,
            "output normalization epsilon cannot be zero"
        );
        Ok(Self {
            frontend,
            config,
            stacks,
            ropes,
            out_scale,
            out_eps,
        })
    }

    /// Input [batch,time,mel], output [batch,time,embed]. No decoder or mutable stream state.
    pub fn forward(
        &self,
        features: Tensor<3>,
        lengths: &[usize],
    ) -> Result<(Tensor<3>, Vec<usize>)> {
        self.forward_conditioned(features, lengths, None)
    }

    pub fn forward_conditioned(
        &self,
        features: Tensor<3>,
        lengths: &[usize],
        condition: Option<&LanguageRequest<'_>>,
    ) -> Result<(Tensor<3>, Vec<usize>)> {
        let (x, lengths) = self.frontend.forward(features, lengths)?;
        Ok((
            self.forward_from_frontend_conditioned(x, &lengths, condition)?,
            lengths,
        ))
    }

    pub fn forward_from_frontend(&self, x: Tensor<3>, lengths: &[usize]) -> Result<Tensor<3>> {
        self.forward_from_frontend_conditioned(x, lengths, None)
    }

    fn forward_from_frontend_conditioned(
        &self,
        mut x: Tensor<3>,
        lengths: &[usize],
        condition: Option<&LanguageRequest<'_>>,
    ) -> Result<Tensor<3>> {
        let [b, original, embed] = x.dims();
        ensure!(
            b > 0
                && original > 0
                && embed == self.config.embed_dim
                && lengths.len() == b
                && lengths.iter().all(|&n| n <= original),
            "invalid Zapformer encoded geometry"
        );
        ensure!(
            condition.is_none_or(|c| c.batch_size() == b),
            "language embedding batch mismatch"
        );
        let max_ds = self
            .stacks
            .iter()
            .map(|s| s.factor)
            .max()
            .context("no stacks")?;
        let padded = original
            .div_ceil(max_ds)
            .checked_mul(max_ds)
            .context("padded length overflow")?;
        ensure!(
            self.stacks
                .iter()
                .all(|s| padded / s.factor <= self.config.max_seq_len),
            "sequence exceeds RoPE table"
        );
        let device = x.device();
        if padded > original {
            x = x.pad([(0, padded - original), (0, 0)], PadMode::Constant(0.0));
        }
        let mask = invalid_mask(lengths, padded, &device).unsqueeze_dim::<3>(2);
        let mut layer_offset = 0;
        for stack in &self.stacks {
            if stack.factor > 1 {
                x = x.mask_fill(mask.clone(), 0.0);
            }
            let grouped_lengths = lengths
                .iter()
                .map(|n| n.div_ceil(stack.factor))
                .collect::<Vec<_>>();
            let grouped = padded / stack.factor;
            let rope = self.ropes[&stack.query_dim].positions(grouped);
            x = stack
                .forward(
                    x.reshape([b, grouped, embed * stack.factor]),
                    &grouped_lengths,
                    &rope,
                    condition,
                    layer_offset,
                )
                .reshape([b, padded, embed]);
            layer_offset += stack.layers.len();
        }
        let x = x.slice([0..b, 0..original, 0..embed]);
        let variance = (x.clone() * x.clone()).mean_dim(2) + self.out_eps * self.out_eps;
        Ok(x * (variance.sqrt().recip() * self.out_scale))
    }
}

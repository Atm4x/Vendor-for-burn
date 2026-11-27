use super::{
    config::{Activation, ZapformerConfig},
    layers::{Conv2, invalid_mask},
};
use crate::{model::Linear, weights::Weights};
use anyhow::{Result, ensure};
use burn::tensor::{Device, Tensor, ops::PadMode};

struct ConvNeXt {
    depthwise: Conv2,
    expand: Conv2,
    reduce: Conv2,
    causal: bool,
    activation: Activation,
}
impl ConvNeXt {
    fn load(
        w: &mut Weights,
        key: &str,
        d: &Device,
        dim: usize,
        causal: bool,
        activation: Activation,
    ) -> Result<Self> {
        let kernel = if causal { 3 } else { 7 };
        Ok(Self {
            depthwise: Conv2::load(
                w,
                &format!("{key}.depthwise_conv"),
                d,
                dim,
                dim,
                kernel,
                1,
                if causal { [0; 2] } else { [3; 2] },
                dim,
            )?,
            expand: Conv2::load(
                w,
                &format!("{key}.pointwise_conv1"),
                d,
                dim,
                dim * 3,
                1,
                1,
                [0; 2],
                1,
            )?,
            reduce: Conv2::load(
                w,
                &format!("{key}.pointwise_conv2"),
                d,
                dim * 3,
                dim,
                1,
                1,
                [0; 2],
                1,
            )?,
            causal,
            activation,
        })
    }
    fn forward(&self, x: Tensor<4>) -> Tensor<4> {
        let y = if self.causal {
            x.clone().pad((1, 1, 2, 0), PadMode::Constant(0.0))
        } else {
            x.clone()
        };
        x + self.reduce.forward(
            self.activation
                .forward(self.expand.forward(self.depthwise.forward(y)), true),
        )
    }
}

/// Shared offline frontend parameters. Stream caches will be owned separately.
pub struct ZapformerFrontend {
    convs: [Conv2; 3],
    convnext: ConvNeXt,
    out: Linear,
    causal: bool,
    activation: Activation,
    in_channels: usize,
}
impl ZapformerFrontend {
    pub fn load(w: &mut Weights, key: &str, d: &Device, config: &ZapformerConfig) -> Result<Self> {
        config.validate()?;
        let [c1, c2, c3] = config.frontend_channels;
        let causal = config.causal_frontend;
        let activation = config.activation;
        Ok(Self {
            convs: [
                Conv2::load(
                    w,
                    &format!("{key}.conv1"),
                    d,
                    1,
                    c1,
                    3,
                    1,
                    if causal { [0; 2] } else { [1; 2] },
                    1,
                )?,
                Conv2::load(w, &format!("{key}.conv2"), d, c1, c2, 3, 2, [1, 0], 1)?,
                Conv2::load(w, &format!("{key}.conv3"), d, c2, c3, 3, 2, [1, 0], 1)?,
            ],
            convnext: ConvNeXt::load(w, &format!("{key}.convnext"), d, c3, causal, activation)?,
            out: Linear::load_checked(
                w,
                &format!("{key}.out"),
                d,
                config.projection_width()?,
                config.embed_dim,
            )?,
            causal,
            activation,
            in_channels: config.in_channels,
        })
    }

    /// One padded packet. Caches retain only the preceding convolution context, not the recording.
    pub(super) fn forward_stream(&self, features: Tensor<3>, mut length: usize, cache: &mut Vec<Tensor<4>>) -> Result<(Tensor<3>, usize)> {
        let [batch, time, freq] = features.dims();
        ensure!(self.causal && batch == 1 && time > 0 && time % 4 == 0 && freq == self.in_channels && length <= time, "invalid streaming frontend geometry");
        ensure!(cache.is_empty() || cache.len() == 4, "invalid streaming frontend cache");
        let device = features.device();
        let mut mask = invalid_mask(&[length], time, &device);
        let mut x = features.mask_fill(mask.clone().unsqueeze_dim::<3>(2), 0.0).unsqueeze_dim::<4>(1);
        let mut next = Vec::with_capacity(4);
        let join = |x: Tensor<4>, index: usize, valid: usize, context: usize, next: &mut Vec<Tensor<4>>| {
            let [_, channels, _, freq] = x.dims();
            let past = cache.get(index).cloned().unwrap_or_else(|| Tensor::zeros([1, channels, context, freq], &device));
            let joined = Tensor::cat(vec![past, x], 2);
            next.push(joined.clone().slice([0..1, 0..channels, valid..valid+context, 0..freq]));
            joined
        };
        for (i, conv) in self.convs.iter().enumerate() {
            let joined = join(x, i, length, if i == 0 {2} else {1}, &mut next);
            x = self.activation.forward(conv.forward_unpadded(joined, usize::from(i == 0)), false);
            if i > 0 { length = length.div_ceil(2); }
            mask = invalid_mask(&[length], x.dims()[2], &device);
            x = x.mask_fill(mask.clone().unsqueeze_dims::<4>(&[1,3]), 0.0);
        }
        let joined = join(x.clone(), 3, length, 2, &mut next);
        let conv = &self.convnext;
        let y = conv.depthwise.forward_unpadded(joined, 1);
        let y = conv.activation.forward(conv.expand.forward(y), true);
        x = x + conv.reduce.forward(y);
        let [b,c,t,f] = x.dims();
        let x = self.out.forward(x.permute([0,2,1,3]).reshape([b,t,c*f]));
        *cache = next;
        Ok((x.mask_fill(mask.unsqueeze_dim::<3>(2), 0.0), length))
    }

    /// Input and output are [batch, time, channels]; lengths are independent valid widths.
    pub fn forward(&self, x: Tensor<3>, lengths: &[usize]) -> Result<(Tensor<3>, Vec<usize>)> {
        let [batch, time, freq] = x.dims();
        ensure!(
            batch > 0
                && time > 0
                && freq == self.in_channels
                && lengths.len() == batch
                && lengths.iter().all(|&n| n <= time),
            "invalid Zapformer frontend input"
        );
        let device = x.device();
        let mut lengths = lengths.to_vec();
        let mut mask = invalid_mask(&lengths, time, &device);
        let mut x = x
            .mask_fill(mask.clone().unsqueeze_dim::<3>(2), 0.0)
            .unsqueeze_dim::<4>(1);
        for (i, conv) in self.convs.iter().enumerate() {
            if i == 0 && self.causal {
                x = x.pad((1, 1, 2, 0), PadMode::Constant(0.0));
            }
            x = self.activation.forward(conv.forward(x), false);
            if i > 0 {
                for len in &mut lengths {
                    *len = len.div_ceil(2);
                }
            }
            mask = invalid_mask(&lengths, x.dims()[2], &device);
            x = x.mask_fill(mask.clone().unsqueeze_dims::<4>(&[1, 3]), 0.0);
        }
        let x = self.convnext.forward(x);
        let [b, c, t, f] = x.dims();
        let x = self
            .out
            .forward(x.permute([0, 2, 1, 3]).reshape([b, t, c * f]));
        Ok((x.mask_fill(mask.unsqueeze_dim::<3>(2), 0.0), lengths))
    }
}

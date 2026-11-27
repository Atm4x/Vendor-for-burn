use super::config::Activation;
use crate::{model::Linear, weights::Weights};
use anyhow::Result;
use burn::tensor::{Bool, Device, Tensor, TensorData, activation, module, ops::ConvOptions};

pub(super) fn invalid_mask(lengths: &[usize], time: usize, device: &Device) -> Tensor<2, Bool> {
    let values: Vec<bool> = lengths
        .iter()
        .flat_map(|&len| (0..time).map(move |i| i >= len))
        .collect();
    Tensor::from_data(TensorData::new(values, [lengths.len(), time]), device)
}

impl Activation {
    pub(super) fn forward<const D: usize>(self, x: Tensor<D>, left: bool) -> Tensor<D> {
        match (self, left) {
            (Self::Silu, _) => activation::silu(x),
            (Self::Swish, false) => {
                activation::silu(x.clone() * 0.683202862739563 - 0.31602001190185547)
                    - x * 0.012898006476461887
                    + 0.133248575988922
            }
            (Self::Swish, true) => {
                activation::silu(x.clone() * 0.639521598815918 - 2.087040662765503)
                    - x * 0.021007252857089043
                    + 0.2303316444158554
            }
        }
    }
}

pub(super) struct Conv2 {
    weight: Tensor<4>,
    bias: Tensor<1>,
    options: ConvOptions<2>,
}

impl Conv2 {
    pub(super) fn load(
        w: &mut Weights,
        key: &str,
        d: &Device,
        input: usize,
        output: usize,
        kernel: usize,
        stride: usize,
        padding: [usize; 2],
        groups: usize,
    ) -> Result<Self> {
        Ok(Self {
            weight: w.checked_tensor(
                &format!("{key}.weight"),
                [output, input / groups, kernel, kernel],
                d,
            )?,
            bias: w.checked_tensor(&format!("{key}.bias"), [output], d)?,
            options: ConvOptions::new([stride; 2], padding, [1; 2], groups),
        })
    }
    pub(super) fn forward(&self, x: Tensor<4>) -> Tensor<4> {
        module::conv2d(
            x,
            self.weight.clone(),
            Some(self.bias.clone()),
            self.options.clone(),
        )
    }
}

pub(super) struct RmsNorm {
    weight: Tensor<1>,
}
impl RmsNorm {
    pub(super) fn load(w: &mut Weights, key: &str, d: &Device, dim: usize) -> Result<Self> {
        Ok(Self {
            weight: w.checked_tensor(&format!("{key}.weight"), [dim], d)?,
        })
    }
    pub(super) fn forward(&self, x: Tensor<3>) -> Tensor<3> {
        let variance = (x.clone() * x.clone()).mean_dim(2);
        let width = self.weight.dims()[0];
        x * (variance + 1e-6).sqrt().recip() * self.weight.clone().reshape([1, 1, width])
    }
}

pub(super) struct FeedForward {
    input: Linear,
    output: Linear,
    activation: Activation,
}
impl FeedForward {
    pub(super) fn load(
        w: &mut Weights,
        key: &str,
        d: &Device,
        dim: usize,
        multiple: usize,
        activation: Activation,
    ) -> Result<Self> {
        Ok(Self {
            input: Linear::load_checked(w, &format!("{key}.in_proj"), d, dim, dim * multiple)?,
            output: Linear::load_checked(w, &format!("{key}.out_proj"), d, dim * multiple, dim)?,
            activation,
        })
    }
    pub(super) fn forward(&self, x: Tensor<3>) -> Tensor<3> {
        self.output
            .forward(self.activation.forward(self.input.forward(x), false))
    }
}

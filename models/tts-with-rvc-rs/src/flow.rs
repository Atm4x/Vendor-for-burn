use burn::tensor::{Device, Tensor, activation, module, ops::ConvOptions};
use serde::Deserialize;

use crate::{checkpoint::RvcCheckpoint, conv1d_layout::{default_conv1d_layout, load_conv1d_weight}};

struct Conv1d {
    weight: Tensor<3>,
    bias: Tensor<1>,
    dilation: usize,
}

impl Conv1d {
    fn load(checkpoint: &RvcCheckpoint, prefix: &str, dilation: usize, device: &Device) -> Result<Self, String> {
        let normalized = checkpoint.contains(&format!("{prefix}.weight_v"));
        let weight = load_conv1d_weight(checkpoint, prefix, normalized, default_conv1d_layout(), device)?;
        let bias = checkpoint.tensor::<1>(&format!("{prefix}.bias"), device)?;
        if weight.dims()[0] != bias.dims()[0] || dilation == 0 {
            return Err(format!("{prefix}: invalid convolution dimensions"));
        }
        Ok(Self { weight, bias, dilation })
    }

    fn forward(&self, x: Tensor<3>) -> Tensor<3> {
        let kernel = self.weight.dims()[2];
        let pad = (kernel - 1) * self.dilation / 2;
        let options = ConvOptions::new_with_padding([1], [(pad, pad)], [self.dilation], 1);
        module::conv1d(x, self.weight.clone(), Some(self.bias.clone()), options)
    }
}

struct WaveNetLayer {
    filter: Conv1d,
    residual_skip: Conv1d,
}

struct WaveNet {
    hidden_channels: usize,
    layers: Vec<WaveNetLayer>,
    condition: Option<Conv1d>,
}

impl WaveNet {
    fn load(checkpoint: &RvcCheckpoint, prefix: &str, device: &Device) -> Result<Self, String> {
        let mut layers = Vec::new();
        while checkpoint.contains(&format!("{prefix}.in_layers.{}.bias", layers.len())) {
            let index = layers.len();
            let dilation = 1;
            let filter = Conv1d::load(checkpoint, &format!("{prefix}.in_layers.{index}"), dilation, device)?;
            let residual_skip = Conv1d::load(checkpoint, &format!("{prefix}.res_skip_layers.{index}"), 1, device)?;
            layers.push(WaveNetLayer { filter, residual_skip });
        }
        if layers.is_empty() {
            return Err(format!("{prefix}: no WaveNet layers"));
        }
        let hidden_channels = layers[0].filter.weight.dims()[1];
        for (index, layer) in layers.iter().enumerate() {
            if layer.filter.weight.dims()[0] != 2 * hidden_channels
                || layer.filter.weight.dims()[1] != hidden_channels
                || layer.residual_skip.weight.dims()[1] != hidden_channels
                || layer.residual_skip.weight.dims()[0] != (if index + 1 < layers.len() { 2 } else { 1 }) * hidden_channels
            {
                return Err(format!("{prefix}: layer {index} has incompatible dimensions"));
            }
        }
        let condition = checkpoint.contains(&format!("{prefix}.cond_layer.bias"))
            .then(|| Conv1d::load(checkpoint, &format!("{prefix}.cond_layer"), 1, device))
            .transpose()?;
        if let Some(conv) = &condition {
            if conv.weight.dims()[0] != layers.len() * 2 * hidden_channels {
                return Err(format!("{prefix}: unexpected condition channel count"));
            }
        }
        Ok(Self { hidden_channels, layers, condition })
    }

    fn forward(&self, mut x: Tensor<3>, mask: Tensor<3>, g: Option<&Tensor<3>>) -> Tensor<3> {
        let [batch, hidden, time] = x.dims();
        assert_eq!(hidden, self.hidden_channels);
        assert_eq!(mask.dims(), [batch, 1, time]);
        let conditioning = match (&self.condition, g) {
            (Some(conv), Some(g)) => Some(conv.forward(g.clone())),
            _ => None,
        };
        let mut output = Tensor::<3>::zeros([batch, hidden, time], &x.device());
        for (index, layer) in self.layers.iter().enumerate() {
            let filter_output = layer.filter.forward(x.clone());
            let filter_output = if let Some(g) = &conditioning {
                filter_output + g.clone().slice([
                    0..batch,
                    index * 2 * hidden..(index + 1) * 2 * hidden,
                    0..g.dims()[2],
                ])
            } else {
                filter_output
            };
            let first = filter_output.clone().slice([0..batch, 0..hidden, 0..time]);
            let second = filter_output.slice([0..batch, hidden..2 * hidden, 0..time]);
            let acts = activation::tanh(first) * activation::sigmoid(second);
            let residual_skip = layer.residual_skip.forward(acts);
            if index + 1 < self.layers.len() {
                x = (x + residual_skip.clone().slice([0..batch, 0..hidden, 0..time])) * mask.clone();
                output = output + residual_skip.slice([0..batch, hidden..2 * hidden, 0..time]);
            } else {
                output = output + residual_skip;
            }
        }
        output * mask
    }
}

struct ResidualCouplingLayer {
    half_channels: usize,
    pre: Conv1d,
    net: WaveNet,
    post: Conv1d,
}

impl ResidualCouplingLayer {
    fn load(checkpoint: &RvcCheckpoint, prefix: &str, device: &Device) -> Result<Self, String> {
        let pre = Conv1d::load(checkpoint, &format!("{prefix}.pre"), 1, device)?;
        let net = WaveNet::load(checkpoint, &format!("{prefix}.enc"), device)?;
        let post = Conv1d::load(checkpoint, &format!("{prefix}.post"), 1, device)?;
        let half_channels = pre.weight.dims()[1];
        if pre.weight.dims()[0] != net.hidden_channels
            || post.weight.dims() != [half_channels, net.hidden_channels, 1]
        {
            return Err(format!("{prefix}: unsupported non-mean-only RVC flow or invalid dimensions"));
        }
        Ok(Self { half_channels, pre, net, post })
    }

    fn forward(&self, x: Tensor<3>, mask: Tensor<3>, g: Option<&Tensor<3>>, reverse: bool) -> Tensor<3> {
        let [batch, channels, time] = x.dims();
        let half = self.half_channels;
        assert_eq!(channels, half * 2);
        let x0 = x.clone().slice([0..batch, 0..half, 0..time]);
        let x1 = x.slice([0..batch, half..channels, 0..time]);
        let h = self.pre.forward(x0.clone()) * mask.clone();
        let h = self.net.forward(h, mask.clone(), g);
        let m = self.post.forward(h) * mask.clone();
        let x1 = if reverse { (x1 - m) * mask } else { m + x1 * mask };
        Tensor::cat(vec![x0, x1], 1)
    }
}

pub struct ResidualCouplingBlock {
    flows: Vec<ResidualCouplingLayer>,
    channels: usize,
}

impl ResidualCouplingBlock {
    pub fn load(checkpoint: &RvcCheckpoint, device: &Device) -> Result<Self, String> {
        let mut flows = Vec::new();
        while checkpoint.contains(&format!("flow.flows.{}.pre.weight", flows.len() * 2)) {
            let prefix = format!("flow.flows.{}", flows.len() * 2);
            flows.push(ResidualCouplingLayer::load(checkpoint, &prefix, device)?);
        }
        if flows.is_empty() {
            return Err("checkpoint does not contain RVC ResidualCouplingBlock".into());
        }
        let channels = flows[0].half_channels * 2;
        if flows.iter().any(|flow| flow.half_channels * 2 != channels) {
            return Err("flow channel counts differ".into());
        }
        Ok(Self { flows, channels })
    }

    pub fn len(&self) -> usize {
        self.flows.len()
    }

    pub fn forward(&self, mut x: Tensor<3>, mask: Tensor<3>, g: Option<Tensor<3>>, reverse: bool) -> Tensor<3> {
        let [batch, channels, time] = x.dims();
        assert_eq!(channels, self.channels);
        assert_eq!(mask.dims(), [batch, 1, time]);
        if reverse {
            for flow in self.flows.iter().rev() {
                x = flow.forward(x.flip([1]), mask.clone(), g.as_ref(), true);
            }
        } else {
            for flow in &self.flows {
                x = flow.forward(x, mask.clone(), g.as_ref(), false).flip([1]);
            }
        }
        x
    }
}

#[derive(Deserialize)]
pub struct FlowFixture {
    pub batch: usize,
    pub channels: usize,
    pub time: usize,
    pub gin_channels: usize,
    pub x: Vec<f32>,
    pub mask: Vec<f32>,
    pub condition: Vec<f32>,
    pub expected_forward: Vec<f32>,
    pub expected_reverse: Vec<f32>,
}

impl FlowFixture {
    pub fn tensors(&self, device: &Device) -> (Tensor<3>, Tensor<3>, Tensor<3>) {
        let (batch, channels, time) = (self.batch, self.channels, self.time);
        let x = Tensor::from_data(burn::tensor::TensorData::new(self.x.clone(), [batch, channels, time]), device);
        let mask = Tensor::from_data(burn::tensor::TensorData::new(self.mask.clone(), [batch, 1, time]), device);
        let g = Tensor::from_data(burn::tensor::TensorData::new(self.condition.clone(), [batch, self.gin_channels, 1]), device);
        (x, mask, g)
    }
}

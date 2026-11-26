use crate::checkpoint::Checkpoint;
use burn::tensor::{Device, Tensor, TensorData, activation};

pub(super) struct LayerNorm {
    weight: Tensor<1>,
    bias: Tensor<1>,
    eps: f32,
}

impl LayerNorm {
    pub(super) fn load(ckpt: &Checkpoint, prefix: &str, device: &Device) -> Result<Self, String> {
        let weight = ckpt.tensor::<1>(&format!("{prefix}.weight"), device)?;
        let bias = ckpt.tensor::<1>(&format!("{prefix}.bias"), device)?;
        if weight.dims() != bias.dims() { return Err(format!("{prefix}: wrong LayerNorm dims")); }
        Ok(Self { weight, bias, eps: 1e-5 })
    }

    pub(super) fn forward(&self, x: Tensor<3>) -> Tensor<3> {
        let d = self.weight.dims()[0];
        let centered = x.clone() - x.mean_dim(2);
        let variance = centered.clone().powf_scalar(2.0).mean_dim(2);
        centered * (variance + self.eps).sqrt().recip()
            * self.weight.clone().reshape([1, 1, d])
            + self.bias.clone().reshape([1, 1, d])
    }
}

struct Linear {
    weight: Tensor<2>,
    bias: Tensor<1>,
}

impl Linear {
    fn load(ckpt: &Checkpoint, prefix: &str, device: &Device) -> Result<Self, String> {
        let weight = ckpt.tensor::<2>(&format!("{prefix}.weight"), device)?;
        let bias = ckpt.tensor::<1>(&format!("{prefix}.bias"), device)?;
        if weight.dims()[0] != bias.dims()[0] { return Err(format!("{prefix}: invalid Linear bias")); }
        Ok(Self { weight: weight.transpose(), bias })
    }

    fn forward_3d(&self, x: Tensor<3>) -> Tensor<3> {
        let [in_dim, out_dim] = self.weight.dims();
        x.matmul(self.weight.clone().reshape([1, in_dim, out_dim]))
            + self.bias.clone().reshape([1, 1, out_dim])
    }
}

pub struct AxialAttention {
    axis: Axis,
    norm: LayerNorm,
    qkv: Linear,
    out: Linear,
    gamma: Tensor<1>,
    rel_bias: Option<Tensor<4>>,
    rope_cache: Option<(Tensor<4>, Tensor<4>)>,
    heads: usize,
}

#[derive(Clone, Copy)]
pub enum Axis { Pitch, Time }

impl AxialAttention {
    pub fn load(ckpt: &Checkpoint, prefix: &str, device: &Device, axis: Axis, heads: usize) -> Result<Self, String> {
        if heads == 0 { return Err("attention head count must be positive".into()); }
        let norm = LayerNorm::load(ckpt, &format!("{prefix}.ln"), device)?;
        let qkv = Linear::load(ckpt, &format!("{prefix}.qkv"), device)?;
        let out = Linear::load(ckpt, &format!("{prefix}.out"), device)?;
        let gamma = ckpt.tensor::<1>(&format!("{prefix}.gamma"), device)?;
        let c = gamma.dims()[0];
        if c % heads != 0 || qkv.weight.dims() != [c, 3 * c] || out.weight.dims() != [c, c] {
            return Err(format!("{prefix}: incompatible attention projection dimensions"));
        }
        let rel_bias = if matches!(axis, Axis::Pitch) {
            let (shape, bias) = ckpt.values(&format!("{prefix}.rel_bias"))?;
            if shape != [heads, 255] { return Err(format!("{prefix}: wrong pitch relative bias")); }
            let mut dense = Vec::with_capacity(heads * 128 * 128);
            for h in 0..heads {
                for i in 0..128 {
                    for j in 0..128 {
                        dense.push(bias[h * 255 + (j as isize - i as isize + 127) as usize]);
                    }
                }
            }
            Some(Tensor::from_data(TensorData::new(dense, [1, heads, 128, 128]), device))
        } else { None };
        let rope_cache = if matches!(axis, Axis::Time) && c / heads >= 2 {
            Some(Self::rope_tables(512, (c / heads) / 2, device))
        } else { None };
        Ok(Self { axis, norm, qkv, out, gamma, rel_bias, rope_cache, heads })
    }

    fn rope_tables(time: usize, half: usize, device: &Device) -> (Tensor<4>, Tensor<4>) {
        let mut cos = Vec::with_capacity(time * half);
        let mut sin = Vec::with_capacity(time * half);
        for t in 0..time {
            for i in 0..half {
                let freq = 1.0f32 / 10000.0f32.powf(i as f32 / half as f32);
                let angle = t as f32 * freq;
                cos.push(angle.cos());
                sin.push(angle.sin());
            }
        }
        (
            Tensor::from_data(TensorData::new(cos, [1, 1, time, half]), device),
            Tensor::from_data(TensorData::new(sin, [1, 1, time, half]), device),
        )
    }

    fn rope(x: Tensor<4>, cache: Option<&(Tensor<4>, Tensor<4>)>) -> Tensor<4> {
        let [_, _, time, dim] = x.dims();
        let half = dim / 2;
        if half == 0 { return x; }
        let tables = if let Some(tables) = cache.filter(|tables| tables.0.dims()[2] >= time) {
            (tables.0.clone().narrow(2, 0, time), tables.1.clone().narrow(2, 0, time))
        } else {
            Self::rope_tables(time, half, &x.device())
        };
        let (c, s) = tables;
        let left = x.clone().narrow(3, 0, half);
        let right = x.clone().narrow(3, half, half);
        let first = left.clone() * c.clone() - right.clone() * s.clone();
        let second = left * s + right * c;
        let mut pieces = vec![first, second];
        if dim > 2 * half { pieces.push(x.narrow(3, 2 * half, dim - 2 * half)); }
        Tensor::cat(pieces, 3)
    }

    pub fn forward(&self, x: Tensor<4>) -> Tensor<4> {
        let [batch, channels, time, pitch] = x.dims();
        let (length, n) = match self.axis {
            Axis::Time => (time, batch * pitch),
            Axis::Pitch => (pitch, batch * time),
        };
        let seq = match self.axis {
            Axis::Time => x.clone().permute([0, 3, 2, 1]).reshape([n, length, channels]),
            Axis::Pitch => x.clone().permute([0, 2, 3, 1]).reshape([n, length, channels]),
        };
        let normalized = self.norm.forward(seq);
        let dimension = channels / self.heads;
        let qkv = self.qkv.forward_3d(normalized).reshape([n, length, 3, self.heads, dimension]);
        let project = |index| {
            qkv.clone().narrow(2, index, 1)
                .reshape([n, length, self.heads, dimension]).permute([0, 2, 1, 3])
        };
        let mut q = project(0);
        let mut k = project(1);
        let v = project(2);
        if matches!(self.axis, Axis::Time) {
            q = Self::rope(q, self.rope_cache.as_ref());
            k = Self::rope(k, self.rope_cache.as_ref());
        }
        let mut logits = q.matmul(k.transpose()) * (dimension as f32).powf(-0.5);
        if let Some(rel_bias) = &self.rel_bias {
            assert!(length <= 128, "original pitch relative-bias table is limited to 128 lanes");
            logits = logits + rel_bias.clone().narrow(2, 0, length).narrow(3, 0, length);
        }
        let scores = activation::softmax(logits, 3);
        let attention = scores.matmul(v).permute([0, 2, 1, 3]).reshape([n, length, channels]);
        let projected = self.out.forward_3d(attention)
            * self.gamma.clone().reshape([1, 1, channels]);
        let projected = match self.axis {
            Axis::Time => projected.reshape([batch, pitch, time, channels]).permute([0, 3, 2, 1]),
            Axis::Pitch => projected.reshape([batch, time, pitch, channels]).permute([0, 3, 1, 2]),
        };
        x + projected
    }
}

struct GruDirection {
    input_w: Tensor<2>,
    hidden_w: Tensor<2>,
    input_b: Tensor<1>,
    hidden_b: Tensor<1>,
}

impl GruDirection {
    fn load(ckpt: &Checkpoint, prefix: &str, suffix: &str, device: &Device) -> Result<Self, String> {
        let w_i = ckpt.tensor::<2>(&format!("{prefix}.gru.weight_ih_l0{suffix}"), device)?;
        let w_h = ckpt.tensor::<2>(&format!("{prefix}.gru.weight_hh_l0{suffix}"), device)?;
        let b_i = ckpt.tensor::<1>(&format!("{prefix}.gru.bias_ih_l0{suffix}"), device)?;
        let b_h = ckpt.tensor::<1>(&format!("{prefix}.gru.bias_hh_l0{suffix}"), device)?;
        if w_h.dims()[0] != w_h.dims()[1] * 3
            || w_i.dims()[0] != w_h.dims()[0]
            || b_i.dims()[0] != w_h.dims()[0]
            || b_h.dims()[0] != w_h.dims()[0]
        { return Err(format!("{prefix}: invalid GRU gate dimensions")); }
        Ok(Self { input_w: w_i.transpose(), hidden_w: w_h.transpose(), input_b: b_i, hidden_b: b_h })
    }

    fn forward(&self, x: Tensor<3>, reverse: bool) -> Tensor<3> {
        let [batch, time, input_dim] = x.dims();
        let hidden = self.hidden_w.dims()[0];
        let device = x.device();
        let input_gates = x.matmul(self.input_w.clone().reshape([1, input_dim, 3 * hidden]))
            + self.input_b.clone().reshape([1, 1, 3 * hidden]);
        #[cfg(feature = "persistent-gru")]
        if let Some(output) = super::persistent_gru::forward(
            input_gates.clone(), self.hidden_w.clone(), self.hidden_b.clone(), reverse,
        ) {
            return output;
        }
        let mut state = Tensor::<2>::zeros([batch, hidden], &device);
        let mut outputs: Vec<Option<Tensor<3>>> = vec![None; time];
        for offset in 0..time {
            let t = if reverse { time - 1 - offset } else { offset };
            let xg = input_gates.clone().narrow(1, t, 1).reshape([batch, 3 * hidden]);
            let hg = state.clone().matmul(self.hidden_w.clone()) + self.hidden_b.clone().reshape([1, 3 * hidden]);
            let ir = xg.clone().narrow(1, 0, hidden);
            let iz = xg.clone().narrow(1, hidden, hidden);
            let inn = xg.narrow(1, 2 * hidden, hidden);
            let hr = hg.clone().narrow(1, 0, hidden);
            let hz = hg.clone().narrow(1, hidden, hidden);
            let hn = hg.narrow(1, 2 * hidden, hidden);
            let reset = activation::sigmoid(ir + hr);
            let update = activation::sigmoid(iz + hz);
            let candidate = (inn + reset * hn).tanh();
            state = (Tensor::<2>::ones([batch, hidden], &device) - update.clone()) * candidate + update * state;
            outputs[t] = Some(state.clone().reshape([batch, 1, hidden]));
        }
        Tensor::cat(outputs.into_iter().map(Option::unwrap).collect(), 1)
    }
}

pub struct LaneGru {
    norm: LayerNorm,
    forwards: GruDirection,
    backwards: GruDirection,
    output: Linear,
}

impl LaneGru {
    pub fn load(ckpt: &Checkpoint, prefix: &str, device: &Device) -> Result<Self, String> {
        Ok(Self {
            norm: LayerNorm::load(ckpt, &format!("{prefix}.ln"), device)?,
            forwards: GruDirection::load(ckpt, prefix, "", device)?,
            backwards: GruDirection::load(ckpt, prefix, "_reverse", device)?,
            output: Linear::load(ckpt, &format!("{prefix}.out"), device)?,
        })
    }

    pub fn forward(&self, x: Tensor<4>) -> Tensor<4> {
        let [batch, channels, time, pitch] = x.dims();
        let input = x.clone().permute([0, 3, 2, 1]).reshape([batch * pitch, time, channels]);
        let input = self.norm.forward(input);
        let forward = self.forwards.forward(input.clone(), false);
        let backward = self.backwards.forward(input, true);
        let features = Tensor::cat(vec![forward, backward], 2);
        let output = self.output.forward_3d(features)
            .reshape([batch, pitch, time, channels]).permute([0, 3, 2, 1]);
        x + output
    }
}

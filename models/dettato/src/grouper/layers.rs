use crate::checkpoint::Checkpoint;
use burn::tensor::{Device, Tensor, TensorData, activation};

#[derive(Clone)]
pub(super) struct Dense {
    weight: Tensor<2>,
    bias: Tensor<1>,
}

impl Dense {
    pub fn load(ckpt: &Checkpoint, name: &str, device: &Device) -> Result<Self, String> {
        let weight = ckpt.tensor::<2>(&format!("{name}.weight"), device)?.transpose();
        let bias = ckpt.tensor::<1>(&format!("{name}.bias"), device)?;
        if weight.dims()[1] != bias.dims()[0] { return Err(format!("{name}: linear shape mismatch")); }
        Ok(Self { weight, bias })
    }

    pub fn forward(&self, x: Tensor<3>) -> Tensor<3> {
        let [i, o] = self.weight.dims();
        x.matmul(self.weight.clone().reshape([1, i, o])) + self.bias.clone().reshape([1, 1, o])
    }
}

#[derive(Clone)]
pub(super) struct Norm {
    weight: Tensor<1>,
    bias: Tensor<1>,
    eps: f32,
}

impl Norm {
    pub fn load(ckpt: &Checkpoint, name: &str, device: &Device, eps: f32) -> Result<Self, String> {
        let weight = ckpt.tensor::<1>(&format!("{name}.weight"), device)?;
        let bias = ckpt.tensor::<1>(&format!("{name}.bias"), device)?;
        if weight.dims() != bias.dims() { return Err(format!("{name}: norm shape mismatch")); }
        Ok(Self { weight, bias, eps })
    }

    pub fn forward(&self, x: Tensor<3>) -> Tensor<3> {
        let channels = x.dims()[2];
        let centered = x.clone() - x.mean_dim(2);
        let variance = centered.clone().powf_scalar(2.0).mean_dim(2);
        centered * (variance + self.eps).sqrt().recip()
            * self.weight.clone().reshape([1, 1, channels])
            + self.bias.clone().reshape([1, 1, channels])
    }
}

pub(super) struct TransformerBlock {
    norm1: Norm,
    norm2: Norm,
    qkv: Dense,
    output: Dense,
    ff1: Dense,
    ff2: Dense,
    heads: usize,
}

impl TransformerBlock {
    pub fn load(ckpt: &Checkpoint, key: &str, device: &Device, heads: usize) -> Result<Self, String> {
        Ok(Self {
            norm1: Norm::load(ckpt, &format!("{key}.ln1"), device, 1e-5)?,
            norm2: Norm::load(ckpt, &format!("{key}.ln2"), device, 1e-5)?,
            qkv: Dense::load(ckpt, &format!("{key}.qkv"), device)?,
            output: Dense::load(ckpt, &format!("{key}.o"), device)?,
            ff1: Dense::load(ckpt, &format!("{key}.ff.0"), device)?,
            ff2: Dense::load(ckpt, &format!("{key}.ff.2"), device)?,
            heads,
        })
    }

    fn rope(x: Tensor<4>, cos: &Tensor<4>, sin: &Tensor<4>) -> Tensor<4> {
        let dim = x.dims()[3];
        let half = dim / 2;
        let x1 = x.clone().narrow(3, 0, half);
        let x2 = x.clone().narrow(3, half, half);
        let a = x1.clone() * cos.clone() - x2.clone() * sin.clone();
        let b = x1 * sin.clone() + x2 * cos.clone();
        if dim > half * 2 { Tensor::cat(vec![a, b, x.narrow(3, 2 * half, dim - 2 * half)], 3) }
        else { Tensor::cat(vec![a, b], 3) }
    }

    pub fn forward(&self, x: Tensor<3>, positions: &[f32], valid: &[bool], device: &Device) -> Result<Tensor<3>, String> {
        let [batch, n, d] = x.dims();
        if batch != 1 || positions.len() != n || valid.len() != n || d % self.heads != 0 {
            return Err("invalid Grouper transformer input".into());
        }
        let hd = d / self.heads;
        let h = hd / 2;
        let mut cos = Vec::with_capacity(n * h);
        let mut sin = Vec::with_capacity(n * h);
        for &p in positions {
            for i in 0..h {
                let a = p * 10000.0f32.powf(-(i as f32 / h as f32));
                cos.push(a.cos()); sin.push(a.sin());
            }
        }
        let co = Tensor::<4>::from_data(TensorData::new(cos, [1, 1, n, h]), device);
        let si = Tensor::<4>::from_data(TensorData::new(sin, [1, 1, n, h]), device);
        let projection = self.qkv.forward(self.norm1.forward(x.clone()))
            .reshape([1, n, 3, self.heads, hd]);
        let unpack = |part: usize| projection.clone().narrow(2, part, 1)
            .reshape([1, n, self.heads, hd]).permute([0, 2, 1, 3]);
        let q = Self::rope(unpack(0), &co, &si);
        let k = Self::rope(unpack(1), &co, &si);
        let v = unpack(2);
        let mask: Vec<f32> = valid.iter().map(|&v| if v { 0.0 } else { -1e9 }).collect();
        let mask = Tensor::<4>::from_data(TensorData::new(mask, [1, 1, 1, n]), device);
        let attn = activation::softmax(q.matmul(k.transpose()) * (hd as f32).powf(-0.5) + mask, 3);
        let context = attn.matmul(v).permute([0, 2, 1, 3]).reshape([1, n, d]);
        let residual = x + self.output.forward(context);
        let ff = self.ff1.forward(self.norm2.forward(residual.clone()));
        Ok(residual + self.ff2.forward(activation::gelu(ff)))
    }
}

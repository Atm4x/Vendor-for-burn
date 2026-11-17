use crate::checkpoint::Checkpoint;
use burn::tensor::ops::ConvOptions;
use burn::tensor::{Device, Tensor, TensorData, activation, module};
use std::{collections::HashMap, sync::Mutex};

struct Linear {
    weight: Tensor<3>,
    bias: Option<Tensor<3>>,
}

impl Linear {
    fn load(ckpt: &Checkpoint, key: &str, device: &Device, bias: bool) -> Result<Self, String> {
        let weights = ckpt.tensor::<2>(&format!("{key}.weight"), device)?;
        let [out_dim, in_dim] = weights.dims();
        let bias = if bias {
            Some(
                ckpt.tensor::<1>(&format!("{key}.bias"), device)?
                    .reshape([1, 1, out_dim]),
            )
        } else {
            None
        };
        Ok(Self {
            weight: weights.transpose().reshape([1, in_dim, out_dim]),
            bias,
        })
    }

    fn forward(&self, x: Tensor<3>) -> Tensor<3> {
        let out = x.matmul(self.weight.clone());
        match &self.bias {
            Some(bias) => out + bias.clone(),
            None => out,
        }
    }
}

struct Norm {
    gamma: Tensor<3>,
    size: f32,
}

impl Norm {
    fn load(ckpt: &Checkpoint, prefix: &str, device: &Device) -> Result<Self, String> {
        let gamma = ckpt.tensor::<1>(&format!("{prefix}.gamma"), device)?;
        let size = gamma.dims()[0];
        Ok(Self {
            gamma: gamma.reshape([1, 1, size]),
            size: (size as f32).sqrt(),
        })
    }

    fn forward(&self, x: Tensor<3>) -> Tensor<3> {
        let squared = x.clone().powf_scalar(2.0).sum_dim(2);
        // PyTorch F.normalize: x / max(L2(x), 1e-12)
        let norm = squared.sqrt().clamp_min(1e-12).recip();
        x * norm * self.size * self.gamma.clone()
    }
}

struct BatchNorm {
    scale: Tensor<1>,
    bias: Tensor<1>,
}

impl BatchNorm {
    fn load(ckpt: &Checkpoint, prefix: &str, device: &Device) -> Result<Self, String> {
        let (_, weights) = ckpt.values(&format!("{prefix}.weight"))?;
        let (_, biases) = ckpt.values(&format!("{prefix}.bias"))?;
        let (_, means) = ckpt.values(&format!("{prefix}.running_mean"))?;
        let (_, vars) = ckpt.values(&format!("{prefix}.running_var"))?;
        if weights.len() != biases.len()
            || weights.len() != means.len()
            || weights.len() != vars.len()
        {
            return Err(format!("invalid BN: {prefix}"));
        }
        let mut scale = Vec::new();
        let mut shift = Vec::new();
        for ((&w, &b), (&m, &v)) in weights
            .iter()
            .zip(biases.iter())
            .zip(means.iter().zip(vars.iter()))
        {
            let s = w / (v + 1e-5).sqrt();
            scale.push(s);
            shift.push(b - m * s);
        }
        let len = scale.len();
        Ok(Self {
            scale: Tensor::from_data(TensorData::new(scale, [len]), device),
            bias: Tensor::from_data(TensorData::new(shift, [len]), device),
        })
    }

    fn forward_1d(&self, x: Tensor<3>) -> Tensor<3> {
        let len = self.scale.dims()[0];
        x * self.scale.clone().reshape([1, len, 1]) + self.bias.clone().reshape([1, len, 1])
    }

    fn forward_2d(&self, x: Tensor<4>) -> Tensor<4> {
        let len = self.scale.dims()[0];
        x * self.scale.clone().reshape([1, len, 1, 1]) + self.bias.clone().reshape([1, len, 1, 1])
    }
}

struct Rotary {
    freqs: Vec<f32>,
    device: Device,
    cache: Mutex<HashMap<usize, (Tensor<5>, Tensor<5>)>>,
}

impl Rotary {
    fn load(ckpt: &Checkpoint, device: &Device) -> Result<Self, String> {
        let (_, freqs) = ckpt.values("frontend.blocks.0.partial.attnF.rotary_embed.freqs")?;
        if freqs.len() != 16 {
            return Err(format!(
                "expected 16 RoPE frequencies, found {}",
                freqs.len()
            ));
        }
        Ok(Self {
            freqs,
            device: device.clone(),
            cache: Mutex::new(HashMap::new()),
        })
    }

    fn angle(&self, seq_len: usize) -> (Tensor<5>, Tensor<5>) {
        if let Some(rotations) = self.cache.lock().unwrap().get(&seq_len) {
            return rotations.clone();
        }
        let mut cosines = Vec::with_capacity(seq_len * self.freqs.len());
        let mut sines = Vec::with_capacity(seq_len * self.freqs.len());
        for pos in 0..seq_len {
            for &frequency in &self.freqs {
                let angle = pos as f32 * frequency;
                cosines.push(angle.cos());
                sines.push(angle.sin());
            }
        }
        let shape = [1, 1, seq_len, self.freqs.len(), 1];
        let pair = (
            Tensor::from_data(TensorData::new(cosines, shape), &self.device),
            Tensor::from_data(TensorData::new(sines, shape), &self.device),
        );
        self.cache.lock().unwrap().insert(seq_len, pair.clone());
        pair
    }

    fn rotate(&self, q: Tensor<4>, cos: &Tensor<5>, sin: &Tensor<5>) -> Tensor<4> {
        let [batch, heads, time, dim] = q.dims();
        let paired = q.reshape([batch, heads, time, dim / 2, 2]);
        let even = paired.clone().narrow(4, 0, 1);
        let odd = paired.narrow(4, 1, 1);
        Tensor::cat(
            vec![
                even.clone() * cos.clone() - odd.clone() * sin.clone(),
                odd * cos.clone() + even * sin.clone(),
            ],
            4,
        )
        .reshape([batch, heads, time, dim])
    }
}

struct Attention {
    norm: Norm,
    to_qkv: Linear,
    to_gates: Linear,
    to_out: Linear,
    heads: usize,
    head_dim: usize,
}

impl Attention {
    fn load(ckpt: &Checkpoint, prefix: &str, device: &Device) -> Result<Self, String> {
        let norm = Norm::load(ckpt, &format!("{prefix}.norm"), device)?;
        let to_qkv = Linear::load(ckpt, &format!("{prefix}.to_qkv"), device, false)?;
        let to_gates = Linear::load(ckpt, &format!("{prefix}.to_gates"), device, true)?;
        let to_out = Linear::load(ckpt, &format!("{prefix}.to_out.0"), device, false)?;
        let heads = to_gates.weight.dims()[2];
        let head_dim = to_qkv.weight.dims()[2] / (3 * heads);
        if head_dim != 32 {
            return Err(format!("expected head_dim 32 at {prefix}, got {head_dim}"));
        }
        Ok(Self {
            norm,
            to_qkv,
            to_gates,
            to_out,
            heads,
            head_dim,
        })
    }

    fn forward(&self, x: Tensor<3>, rotary: &Rotary) -> Tensor<3> {
        let [batch, seq_len, _] = x.dims();
        let norm = self.norm.forward(x);
        let qkv = self.to_qkv.forward(norm.clone()).reshape([
            batch,
            seq_len,
            3,
            self.heads,
            self.head_dim,
        ]);
        let projection = |index: usize| {
            qkv.clone()
                .narrow(2, index, 1)
                .reshape([batch, seq_len, self.heads, self.head_dim])
                .swap_dims(1, 2)
        };
        let q = projection(0);
        let k = projection(1);
        let v = projection(2);
        let (cos, sin) = rotary.angle(seq_len);
        let q = rotary.rotate(q, &cos, &sin);
        let k = rotary.rotate(k, &cos, &sin);
        let attention = activation::softmax(
            q.matmul(k.transpose()) * (self.head_dim as f32).powf(-0.5),
            3,
        );
        let context = attention.matmul(v);
        let gates = activation::sigmoid(self.to_gates.forward(norm))
            .reshape([batch, seq_len, self.heads, 1])
            .swap_dims(1, 2);
        let gated =
            (context * gates)
                .swap_dims(1, 2)
                .reshape([batch, seq_len, self.heads * self.head_dim]);
        self.to_out.forward(gated)
    }
}

struct FeedForward {
    norm: Norm,
    in_proj: Linear,
    out_proj: Linear,
}

impl FeedForward {
    fn load(ckpt: &Checkpoint, prefix: &str, device: &Device) -> Result<Self, String> {
        Ok(Self {
            norm: Norm::load(ckpt, &format!("{prefix}.net.0"), device)?,
            in_proj: Linear::load(ckpt, &format!("{prefix}.net.1"), device, true)?,
            out_proj: Linear::load(ckpt, &format!("{prefix}.net.4"), device, true)?,
        })
    }

    fn forward(&self, x: Tensor<3>) -> Tensor<3> {
        self.out_proj
            .forward(activation::gelu(self.in_proj.forward(self.norm.forward(x))))
    }
}

struct PartialFT {
    attn_freq: Attention,
    ff_freq: FeedForward,
    attn_time: Attention,
    ff_time: FeedForward,
}

impl PartialFT {
    fn load(ckpt: &Checkpoint, prefix: &str, device: &Device) -> Result<Self, String> {
        Ok(Self {
            attn_freq: Attention::load(ckpt, &format!("{prefix}.attnF"), device)?,
            ff_freq: FeedForward::load(ckpt, &format!("{prefix}.ffF"), device)?,
            attn_time: Attention::load(ckpt, &format!("{prefix}.attnT"), device)?,
            ff_time: FeedForward::load(ckpt, &format!("{prefix}.ffT"), device)?,
        })
    }

    fn forward(&self, x: Tensor<4>, rotary: &Rotary) -> Tensor<4> {
        let [batch, channels, freqs, time] = x.dims();
        let mut freq = x
            .permute([0, 3, 2, 1])
            .reshape([batch * time, freqs, channels]);
        freq = freq.clone() + self.attn_freq.forward(freq.clone(), rotary);
        freq = freq.clone() + self.ff_freq.forward(freq.clone());
        let mut time_seq = freq
            .reshape([batch, time, freqs, channels])
            .permute([0, 2, 1, 3])
            .reshape([batch * freqs, time, channels]);
        time_seq = time_seq.clone() + self.attn_time.forward(time_seq.clone(), rotary);
        time_seq = time_seq.clone() + self.ff_time.forward(time_seq.clone());
        time_seq
            .reshape([batch, freqs, time, channels])
            .permute([0, 3, 1, 2])
    }
}

struct FrontendBlock {
    partial: PartialFT,
    conv_weight: Tensor<4>,
    norm: BatchNorm,
}

impl FrontendBlock {
    fn load(ckpt: &Checkpoint, index: usize, device: &Device) -> Result<Self, String> {
        let prefix = format!("frontend.blocks.{index}");
        Ok(Self {
            partial: PartialFT::load(ckpt, &format!("{prefix}.partial"), device)?,
            conv_weight: ckpt.tensor::<4>(&format!("{prefix}.conv2d.weight"), device)?,
            norm: BatchNorm::load(ckpt, &format!("{prefix}.norm"), device)?,
        })
    }

    fn forward(&self, x: Tensor<4>, rotary: &Rotary) -> Tensor<4> {
        let x = self.partial.forward(x, rotary);
        let x = module::conv2d(
            x,
            self.conv_weight.clone(),
            None,
            ConvOptions::new([2, 1], [0, 1], [1, 1], 1),
        );
        activation::gelu(self.norm.forward_2d(x))
    }
}

struct TransformerBlock {
    attention: Attention,
    ff: FeedForward,
}

impl TransformerBlock {
    fn load(ckpt: &Checkpoint, index: usize, device: &Device) -> Result<Self, String> {
        Ok(Self {
            attention: Attention::load(
                ckpt,
                &format!("transformer_blocks.layers.{index}.0"),
                device,
            )?,
            ff: FeedForward::load(
                ckpt,
                &format!("transformer_blocks.layers.{index}.1"),
                device,
            )?,
        })
    }
    fn forward(&self, mut x: Tensor<3>, rotary: &Rotary) -> Tensor<3> {
        x = x.clone() + self.attention.forward(x.clone(), rotary);
        x.clone() + self.ff.forward(x)
    }
}

pub struct BeatThisModel {
    device: Device,
    rotary: Rotary,
    stem_bn_input: BatchNorm,
    stem_weight: Tensor<4>,
    stem_bn_output: BatchNorm,
    frontend: Vec<FrontendBlock>,
    frontend_linear: Linear,
    blocks: Vec<TransformerBlock>,
    final_norm: Norm,
    head: Linear,
    pub transformer_dim: usize,
    pub parameter_tensors: usize,
}

impl BeatThisModel {
    pub fn from_checkpoint(ckpt: &Checkpoint, device: &Device) -> Result<Self, String> {
        let transformer_dim = ckpt.dimension("transformer_blocks.norm.gamma")?;
        let rotary = Rotary::load(ckpt, device)?;
        let frontend: Result<Vec<_>, _> = (0..3)
            .map(|i| FrontendBlock::load(ckpt, i, device))
            .collect();
        let mut blocks = Vec::new();
        for i in 0..32 {
            if !ckpt.contains(&format!("transformer_blocks.layers.{i}.0.to_qkv.weight")) {
                break;
            }
            blocks.push(TransformerBlock::load(ckpt, i, device)?);
        }
        if blocks.len() != 6 {
            return Err(format!(
                "expected 6 transformer blocks, found {}",
                blocks.len()
            ));
        }
        Ok(Self {
            device: device.clone(),
            rotary,
            stem_bn_input: BatchNorm::load(ckpt, "frontend.stem.bn1d", device)?,
            stem_weight: ckpt.tensor::<4>("frontend.stem.conv2d.weight", device)?,
            stem_bn_output: BatchNorm::load(ckpt, "frontend.stem.bn2d", device)?,
            frontend: frontend?,
            frontend_linear: Linear::load(ckpt, "frontend.linear", device, true)?,
            blocks,
            final_norm: Norm::load(ckpt, "transformer_blocks.norm", device)?,
            head: Linear::load(ckpt, "task_heads.beat_downbeat_lin", device, true)?,
            transformer_dim,
            parameter_tensors: ckpt.count(),
        })
    }

    pub fn infer(&self, frames: &[[f32; 128]]) -> Result<(Vec<f32>, Vec<f32>), String> {
        if frames.is_empty() {
            return Err("cannot infer on zero frames".into());
        }
        let time = frames.len();
        let flattened = frames.iter().flat_map(|row| row.iter().copied()).collect();
        let x = Tensor::<3>::from_data(TensorData::new(flattened, [1, time, 128]), &self.device);
        let stem = self
            .stem_bn_input
            .forward_1d(x.swap_dims(1, 2))
            .reshape([1, 1, 128, time]);
        let stem = module::conv2d(
            stem,
            self.stem_weight.clone(),
            None,
            ConvOptions::new([4, 1], [0, 1], [1, 1], 1),
        );
        let mut x = activation::gelu(self.stem_bn_output.forward_2d(stem));
        for block in &self.frontend {
            x = block.forward(x, &self.rotary);
        }
        let [batch, channels, freqs, time] = x.dims();
        let mut x = self
            .frontend_linear
            .forward(
                x.permute([0, 3, 1, 2])
                    .reshape([batch, time, channels * freqs]),
            );
        for block in &self.blocks {
            x = block.forward(x, &self.rotary);
        }
        let x = self.head.forward(self.final_norm.forward(x));
        let values: Vec<f32> = x
            .into_data()
            .try_into_vec::<f32>()
            .map_err(|e| format!("reading model output: {e}"))?;
        if values.len() != time * 2 {
            return Err("unexpected model output size".into());
        }
        let mut beat = Vec::with_capacity(time);
        let mut downbeat = Vec::with_capacity(time);
        for pair in values.chunks_exact(2) {
            beat.push(pair[0] + pair[1]);
            downbeat.push(pair[1]);
        }
        Ok((beat, downbeat))
    }
}

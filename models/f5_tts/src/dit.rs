use burn::{
    module::Param,
    nn::{Linear, LinearConfig},
    tensor::{
        Bool, Device, Tensor, TensorData,
        activation::{sigmoid, softmax},
    },
};

use crate::{F5Error, Result, checkpoint::Checkpoint};

const NORM_EPSILON: f32 = 1e-6;

pub(super) fn load_linear(
    checkpoint: &Checkpoint,
    key: &str,
    input: usize,
    output: usize,
    device: &Device,
) -> Result<Linear> {
    let weights = checkpoint.read_f32(&format!("{key}.weight"), &[output, input])?;
    let bias = checkpoint.read_f32(&format!("{key}.bias"), &[output])?;
    let weights =
        Tensor::<2>::from_data(TensorData::new(weights, [output, input]), device).transpose();
    let bias = Tensor::<1>::from_data(TensorData::new(bias, [output]), device);
    Ok(Linear {
        weight: Param::from_tensor(weights),
        bias: Some(Param::from_tensor(bias)),
    })
}

fn silu<const D: usize>(x: Tensor<D>) -> Tensor<D> {
    x.clone() * sigmoid(x)
}

fn gelu_tanh<const D: usize>(x: Tensor<D>) -> Tensor<D> {
    let inner = (x.clone() + x.clone().powf_scalar(3.0) * 0.044_715)
        * (2.0_f32 / core::f32::consts::PI).sqrt();
    x * (inner.tanh() + 1.0) * 0.5
}

fn layer_norm(x: Tensor<3>) -> Tensor<3> {
    let mean = x.clone().mean_dim(2);
    let centered = x - mean;
    let variance = centered.clone().powf_scalar(2.0).mean_dim(2);
    centered / (variance + NORM_EPSILON).sqrt()
}

pub struct TimeEmbedding {
    input: Linear,
    output: Linear,
    dimension: usize,
}

impl TimeEmbedding {
    pub fn new(dimension: usize, device: &Device) -> Result<Self> {
        if dimension == 0 {
            return Err(F5Error::InvalidConfiguration(
                "Time embedding dimension must be positive",
            ));
        }
        Ok(Self {
            input: LinearConfig::new(256, dimension).init(device),
            output: LinearConfig::new(dimension, dimension).init(device),
            dimension,
        })
    }

    pub fn from_checkpoint(
        checkpoint: &Checkpoint,
        dimension: usize,
        device: &Device,
    ) -> Result<Self> {
        Ok(Self {
            input: load_linear(
                checkpoint,
                "transformer.time_embed.time_mlp.0",
                256,
                dimension,
                device,
            )?,
            output: load_linear(
                checkpoint,
                "transformer.time_embed.time_mlp.2",
                dimension,
                dimension,
                device,
            )?,
            dimension,
        })
    }

    pub fn forward(&self, time: &[f32], device: &Device) -> Result<Tensor<2>> {
        if time.is_empty() || time.iter().any(|t| !t.is_finite()) {
            return Err(F5Error::InvalidConfiguration(
                "Time values must be finite and nonempty",
            ));
        }
        let mut frequencies = Vec::with_capacity(time.len() * 256);
        let exponent = -(10_000.0_f32.ln()) / 127.0;
        for &t in time {
            for index in 0..128 {
                frequencies.push((1000.0 * t * (index as f32 * exponent).exp()).sin());
            }
            for index in 0..128 {
                frequencies.push((1000.0 * t * (index as f32 * exponent).exp()).cos());
            }
        }
        let data = TensorData::new(frequencies, [time.len(), 256]);
        let embedded = Tensor::from_data(data, device);
        let result = self.output.forward(silu(self.input.forward(embedded)));
        debug_assert_eq!(result.dims(), [time.len(), self.dimension]);
        Ok(result)
    }
}

const ATTENTION_QUERY_ALIGNMENT: usize = 128;

pub struct RotaryPositions {
    cos: Tensor<4>,
    sin: Tensor<4>,
    sequence: usize,
    valid_sequence: usize,
    head_dim: usize,
}

impl RotaryPositions {
    pub fn new(sequence: usize, head_dim: usize, device: &Device) -> Result<Self> {
        if sequence == 0 || head_dim == 0 || !head_dim.is_multiple_of(2) {
            return Err(F5Error::InvalidConfiguration(
                "RoPE requires nonempty sequence and even head dimension",
            ));
        }
        let mut cosine = Vec::with_capacity(sequence * head_dim);
        let mut sine = Vec::with_capacity(sequence * head_dim);
        let half = head_dim / 2;
        for pos in 0..sequence {
            for i in 0..head_dim {
                let frequency =
                    (-(10_000.0_f32.ln()) * (i % half) as f32 * 2.0 / head_dim as f32).exp();
                let angle = pos as f32 * frequency;
                cosine.push(angle.cos());
                sine.push(angle.sin());
            }
        }
        Ok(Self {
            cos: Tensor::from_data(TensorData::new(cosine, [1, 1, sequence, head_dim]), device),
            sin: Tensor::from_data(TensorData::new(sine, [1, 1, sequence, head_dim]), device),
            sequence,
            valid_sequence: sequence,
            head_dim,
        })
    }

    fn padded(&self, sequence: usize) -> Self {
        let padding = [(0, 0), (0, sequence - self.sequence), (0, 0)];
        Self {
            cos: self.cos.clone().pad(padding, 0.0),
            sin: self.sin.clone().pad(padding, 0.0),
            sequence,
            valid_sequence: self.sequence,
            head_dim: self.head_dim,
        }
    }

    pub fn apply(&self, x: Tensor<4>) -> Result<Tensor<4>> {
        let [batch, heads, sequence, dim] = x.dims();
        if sequence != self.sequence || dim != self.head_dim {
            return Err(F5Error::InvalidShape(
                "Attention input does not match cached RoPE positions",
            ));
        }
        let half = dim / 2;
        let first = x.clone().slice([0..batch, 0..heads, 0..sequence, 0..half]);
        let second = x
            .clone()
            .slice([0..batch, 0..heads, 0..sequence, half..dim]);
        let rotated = Tensor::cat(vec![second.neg(), first], 3);
        Ok(x * self.cos.clone() + rotated * self.sin.clone())
    }
}

enum AttentionProjection {
    Separate {
        query: Linear,
        key: Linear,
        value: Linear,
    },
    Packed(Linear),
}

impl AttentionProjection {
    fn from_linears(query: Linear, key: Linear, value: Linear) -> Self {
        if query.weight.val().dtype() != burn::tensor::DType::F16 {
            return Self::Separate { query, key, value };
        }
        let weights = Tensor::cat(
            vec![query.weight.val(), key.weight.val(), value.weight.val()],
            1,
        );
        let bias = match (query.bias, key.bias, value.bias) {
            (Some(q), Some(k), Some(v)) => Some(Param::from_tensor(Tensor::cat(
                vec![q.val(), k.val(), v.val()],
                0,
            ))),
            (None, None, None) => None,
            _ => unreachable!("F5 attention projections share a bias configuration"),
        };
        Self::Packed(Linear {
            weight: Param::from_tensor(weights),
            bias,
        })
    }

    fn forward(&self, input: Tensor<3>, dim: usize) -> (Tensor<3>, Tensor<3>, Tensor<3>) {
        match self {
            Self::Separate { query, key, value } => (
                query.forward(input.clone()),
                key.forward(input.clone()),
                value.forward(input),
            ),
            Self::Packed(linear) => {
                let [batch, frames, _] = input.dims();
                let output = linear.forward(input);
                let slice = |part: usize| {
                    output
                        .clone()
                        .slice([0..batch, 0..frames, part * dim..(part + 1) * dim])
                };
                (slice(0), slice(1), slice(2))
            }
        }
    }
}

pub struct DiTAttention {
    qkv: AttentionProjection,
    out: Linear,
    heads: usize,
    head_dim: usize,
}

impl DiTAttention {
    pub fn new(dim: usize, heads: usize, device: &Device) -> Result<Self> {
        if heads == 0 || dim == 0 || !dim.is_multiple_of(heads) || !(dim / heads).is_multiple_of(2)
        {
            return Err(F5Error::InvalidConfiguration(
                "Attention dimension must divide into even-sized heads",
            ));
        }
        Ok(Self {
            qkv: AttentionProjection::from_linears(
                LinearConfig::new(dim, dim).init(device),
                LinearConfig::new(dim, dim).init(device),
                LinearConfig::new(dim, dim).init(device),
            ),
            out: LinearConfig::new(dim, dim).init(device),
            heads,
            head_dim: dim / heads,
        })
    }

    pub fn from_checkpoint(
        checkpoint: &Checkpoint,
        prefix: &str,
        dim: usize,
        heads: usize,
        device: &Device,
    ) -> Result<Self> {
        if heads == 0 || dim == 0 || !dim.is_multiple_of(heads) || !(dim / heads).is_multiple_of(2)
        {
            return Err(F5Error::InvalidConfiguration(
                "Invalid DiT attention dimensions",
            ));
        }
        Ok(Self {
            qkv: AttentionProjection::from_linears(
                load_linear(checkpoint, &format!("{prefix}.to_q"), dim, dim, device)?,
                load_linear(checkpoint, &format!("{prefix}.to_k"), dim, dim, device)?,
                load_linear(checkpoint, &format!("{prefix}.to_v"), dim, dim, device)?,
            ),
            out: load_linear(checkpoint, &format!("{prefix}.to_out.0"), dim, dim, device)?,
            heads,
            head_dim: dim / heads,
        })
    }

    pub fn forward(&self, x: Tensor<3>, rope: &RotaryPositions) -> Result<Tensor<3>> {
        self.forward_inner(x, rope, None)
    }

    pub fn forward_masked(
        &self,
        x: Tensor<3>,
        rope: &RotaryPositions,
        mask: Tensor<3, Bool>,
    ) -> Result<Tensor<3>> {
        self.forward_inner(x, rope, Some(mask))
    }

    fn forward_inner(
        &self,
        x: Tensor<3>,
        rope: &RotaryPositions,
        mask: Option<Tensor<3, Bool>>,
    ) -> Result<Tensor<3>> {
        let [batch, sequence, dim] = x.dims();
        if dim != self.heads * self.head_dim
            || mask
                .as_ref()
                .is_some_and(|m| m.dims() != [batch, sequence, 1])
        {
            return Err(F5Error::InvalidShape(
                "Attention input dimensions do not match model and mask",
            ));
        }
        let reshape = |tensor: Tensor<3>| {
            tensor
                .reshape([batch, sequence, self.heads, self.head_dim])
                .swap_dims(1, 2)
        };
        let (q, k, v) = self.qkv.forward(x, dim);
        let q = rope.apply(reshape(q))?;
        let k = rope.apply(reshape(k))?;
        let v = reshape(v);
        let output = if mask.is_none()
            && q.dtype() == burn::tensor::DType::F16
            && sequence >= 128
            && self.head_dim.is_multiple_of(16)
        {
            // CubeCL's accelerated attention requires complete query tiles. Extra
            // queries are independent, so K/V and the real softmax are unchanged.
            let padded_sequence = sequence.next_multiple_of(ATTENTION_QUERY_ALIGNMENT);
            let q = if padded_sequence == sequence {
                q
            } else {
                q.pad(
                    [(0, 0), (0, 0), (0, padded_sequence - sequence), (0, 0)],
                    burn::tensor::ops::PadMode::Constant(0.0f32.into()),
                )
            };
            let trim_keys = |tensor: Tensor<4>| {
                if rope.valid_sequence == sequence {
                    tensor
                } else {
                    tensor.slice([
                        0..batch,
                        0..self.heads,
                        0..rope.valid_sequence,
                        0..self.head_dim,
                    ])
                }
            };
            burn::tensor::module::attention(
                q,
                trim_keys(k),
                trim_keys(v),
                None,
                None,
                Default::default(),
            )
            .slice([0..batch, 0..self.heads, 0..sequence, 0..self.head_dim])
            .swap_dims(1, 2)
            .reshape([batch, sequence, dim])
        } else {
            let mut scores = q.matmul(k.swap_dims(2, 3)) * (self.head_dim as f32).sqrt().recip();
            if let Some(valid) = &mask {
                scores = scores.mask_fill(
                    valid.clone().reshape([batch, 1, 1, sequence]).bool_not(),
                    -f32::MAX,
                );
            }
            softmax(scores, 3)
                .matmul(v)
                .swap_dims(1, 2)
                .reshape([batch, sequence, dim])
        };
        let output = self.out.forward(output);
        Ok(if let Some(valid) = mask {
            output.mask_fill(valid.bool_not(), 0.0)
        } else {
            output
        })
    }
}

pub struct DiTBlock {
    modulation: Linear,
    attention: DiTAttention,
    ff_in: Linear,
    ff_out: Linear,
    dim: usize,
}

impl DiTBlock {
    pub fn new(dim: usize, heads: usize, ff_mult: usize, device: &Device) -> Result<Self> {
        if ff_mult == 0 || dim.checked_mul(ff_mult).is_none() || dim.checked_mul(6).is_none() {
            return Err(F5Error::InvalidConfiguration(
                "Invalid DiT feedforward or modulation dimension",
            ));
        }
        Ok(Self {
            modulation: LinearConfig::new(dim, 6 * dim).init(device),
            attention: DiTAttention::new(dim, heads, device)?,
            ff_in: LinearConfig::new(dim, dim * ff_mult).init(device),
            ff_out: LinearConfig::new(dim * ff_mult, dim).init(device),
            dim,
        })
    }

    pub fn from_checkpoint(
        checkpoint: &Checkpoint,
        block: usize,
        dim: usize,
        heads: usize,
        ff_mult: usize,
        device: &Device,
    ) -> Result<Self> {
        if ff_mult == 0 || dim.checked_mul(ff_mult).is_none() || dim.checked_mul(6).is_none() {
            return Err(F5Error::InvalidConfiguration(
                "Invalid DiT block dimensions",
            ));
        }
        let prefix = format!("transformer.transformer_blocks.{block}");
        Ok(Self {
            modulation: load_linear(
                checkpoint,
                &format!("{prefix}.attn_norm.linear"),
                dim,
                6 * dim,
                device,
            )?,
            attention: DiTAttention::from_checkpoint(
                checkpoint,
                &format!("{prefix}.attn"),
                dim,
                heads,
                device,
            )?,
            ff_in: load_linear(
                checkpoint,
                &format!("{prefix}.ff.ff.0.0"),
                dim,
                dim * ff_mult,
                device,
            )?,
            ff_out: load_linear(
                checkpoint,
                &format!("{prefix}.ff.ff.2"),
                dim * ff_mult,
                dim,
                device,
            )?,
            dim,
        })
    }

    pub fn forward(&self, x: Tensor<3>, t: Tensor<2>, rope: &RotaryPositions) -> Result<Tensor<3>> {
        let [batch, _sequence, dim] = x.dims();
        if dim != self.dim || t.dims() != [batch, dim] {
            return Err(F5Error::InvalidShape(
                "DiT block requires [B, T, D] input and [B, D] time",
            ));
        }
        let params = self.modulation.forward(silu(t));
        let parts = params.chunk(6, 1);
        let shifted = |tensor: Tensor<2>| tensor.reshape([batch, 1, dim]);
        let attn_input =
            layer_norm(x.clone()) * (shifted(parts[1].clone()) + 1.0) + shifted(parts[0].clone());
        let attn_out = self.attention.forward(attn_input, rope)?;
        let x = x + attn_out * shifted(parts[2].clone());
        let ff_input =
            layer_norm(x.clone()) * (shifted(parts[4].clone()) + 1.0) + shifted(parts[3].clone());
        let ff_out = self.ff_out.forward(gelu_tanh(self.ff_in.forward(ff_input)));
        Ok(x + ff_out * shifted(parts[5].clone()))
    }
}

pub struct DiTBackbone {
    blocks: Vec<DiTBlock>,
    final_modulation: Linear,
    projection: Linear,
    dim: usize,
}

impl DiTBackbone {
    pub fn new(
        dim: usize,
        heads: usize,
        depth: usize,
        ff_mult: usize,
        mel_channels: usize,
        device: &Device,
    ) -> Result<Self> {
        if depth == 0 || mel_channels == 0 || dim.checked_mul(2).is_none() {
            return Err(F5Error::InvalidConfiguration(
                "Invalid DiT backbone dimensions",
            ));
        }
        let mut blocks = Vec::with_capacity(depth);
        for _ in 0..depth {
            blocks.push(DiTBlock::new(dim, heads, ff_mult, device)?);
        }
        Ok(Self {
            blocks,
            final_modulation: LinearConfig::new(dim, 2 * dim).init(device),
            projection: LinearConfig::new(dim, mel_channels).init(device),
            dim,
        })
    }

    pub fn from_checkpoint(
        checkpoint: &Checkpoint,
        dim: usize,
        heads: usize,
        depth: usize,
        ff_mult: usize,
        mel_channels: usize,
        device: &Device,
    ) -> Result<Self> {
        if depth == 0 || dim == 0 || mel_channels == 0 || dim.checked_mul(2).is_none() {
            return Err(F5Error::InvalidConfiguration(
                "Invalid DiT backbone dimensions",
            ));
        }
        let mut blocks = Vec::with_capacity(depth);
        for index in 0..depth {
            blocks.push(DiTBlock::from_checkpoint(
                checkpoint, index, dim, heads, ff_mult, device,
            )?);
        }
        Ok(Self {
            blocks,
            final_modulation: load_linear(
                checkpoint,
                "transformer.norm_out.linear",
                dim,
                2 * dim,
                device,
            )?,
            projection: load_linear(
                checkpoint,
                "transformer.proj_out",
                dim,
                mel_channels,
                device,
            )?,
            dim,
        })
    }

    pub fn forward(
        &self,
        x: Tensor<3>,
        time: Tensor<2>,
        rope: &RotaryPositions,
    ) -> Result<Tensor<3>> {
        self.forward_with_observer(x, time, rope, |_| {})
    }

    fn forward_with_observer<F: FnMut(&Tensor<3>)>(
        &self,
        mut x: Tensor<3>,
        time: Tensor<2>,
        rope: &RotaryPositions,
        mut observe: F,
    ) -> Result<Tensor<3>> {
        let [batch, sequence, dim] = x.dims();
        if time.dims() != [batch, self.dim] || dim != self.dim || rope.sequence != sequence {
            return Err(F5Error::InvalidShape(
                "DiT backbone time, RoPE or model dimension mismatch",
            ));
        }
        // The block is tokenwise outside attention. Extra query rows improve GEMM
        // tiling, while the original K/V length keeps the softmax domain intact.
        let aligned_sequence = sequence.next_multiple_of(ATTENTION_QUERY_ALIGNMENT);
        let aligned_rope = (x.dtype() == burn::tensor::DType::F16
            && sequence >= ATTENTION_QUERY_ALIGNMENT
            && rope.head_dim.is_multiple_of(16)
            && aligned_sequence != sequence
            && aligned_sequence - sequence <= sequence / 16)
            .then(|| rope.padded(aligned_sequence));
        let rope = if let Some(aligned_rope) = &aligned_rope {
            x = x.pad([(0, 0), (0, aligned_sequence - sequence), (0, 0)], 0.0);
            aligned_rope
        } else {
            rope
        };
        for block in &self.blocks {
            x = block.forward(x, time.clone(), rope)?;
            observe(&x);
        }
        let modulation = self.final_modulation.forward(silu(time)).chunk(2, 1);
        let scale = modulation[0].clone().reshape([batch, 1, dim]);
        let shift = modulation[1].clone().reshape([batch, 1, dim]);
        let output = self
            .projection
            .forward(layer_norm(x) * (scale + 1.0) + shift);
        let output = if aligned_rope.is_some() {
            let [_, _, channels] = output.dims();
            output.slice([0..batch, 0..sequence, 0..channels])
        } else {
            output
        };
        Ok(output)
    }

    #[cfg(test)]
    pub(crate) fn forward_traced(
        &self,
        x: Tensor<3>,
        time: Tensor<2>,
        rope: &RotaryPositions,
    ) -> Result<(Tensor<3>, Vec<Tensor<3>>)> {
        let mut states = Vec::with_capacity(self.blocks.len());
        let output =
            self.forward_with_observer(x, time, rope, |state| states.push(state.clone()))?;
        Ok((output, states))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn model_specific_shapes_reject_invalid_configurations() {
        let device = Device::default();
        assert!(RotaryPositions::new(16, 7, &device).is_err());
        assert!(DiTBlock::new(15, 4, 2, &device).is_err());
        assert!(TimeEmbedding::new(0, &device).is_err());
    }

    #[cfg(feature = "cubecl-cpu")]
    #[test]
    fn cubecl_cpu_matches_independent_pytorch_dit_block() {
        use std::path::Path;
        let device = Device::cpu();
        let fixture_dir = Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures");
        let checkpoint = Checkpoint::open(fixture_dir.join("dit_block_small.safetensors")).unwrap();
        let spec: serde_json::Value =
            serde_json::from_str(include_str!("../tests/fixtures/dit_block_expected.json"))
                .unwrap();
        let values = |key: &str| {
            spec[key]
                .as_array()
                .unwrap()
                .iter()
                .map(|value| value.as_f64().unwrap() as f32)
                .collect::<Vec<_>>()
        };
        let input = Tensor::from_data(TensorData::new(values("input"), [1, 3, 8]), &device);
        let time = Tensor::from_data(
            TensorData::new(values("timestep_embedding"), [1, 8]),
            &device,
        );
        let model = DiTBlock::from_checkpoint(&checkpoint, 0, 8, 2, 2, &device).unwrap();
        let rope = RotaryPositions::new(3, 4, &device).unwrap();
        let output = model
            .forward(input, time, &rope)
            .unwrap()
            .into_data()
            .try_to_vec::<f32>()
            .unwrap();
        let expected = values("output");
        let error = output
            .iter()
            .zip(expected.iter())
            .map(|(actual, reference)| (actual - reference).abs())
            .fold(0.0_f32, f32::max);
        assert!(
            error <= 2e-4,
            "DiT block must match independent PyTorch, max_abs={error}"
        );
    }

    #[cfg(feature = "cubecl-cpu")]
    #[test]
    fn cubecl_cpu_block_preserves_dimensions_and_is_finite() {
        let device = Device::cpu();
        let input = Tensor::<3>::ones([1, 3, 16], &device);
        let t = Tensor::<2>::ones([1, 16], &device);
        let rope = RotaryPositions::new(3, 4, &device).unwrap();
        let block = DiTBlock::new(16, 4, 2, &device).unwrap();
        let output = block.forward(input, t, &rope).unwrap();
        assert_eq!(output.dims(), [1, 3, 16]);
        let output = output.into_data();
        assert!(
            output
                .try_to_vec::<f32>()
                .unwrap()
                .iter()
                .all(|v| v.is_finite())
        );
    }
}

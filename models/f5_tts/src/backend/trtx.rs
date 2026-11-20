//! F5 DiT-specific TensorRT-RTX backbone: weight schema and immutable tensor contracts.
//! The input-embedding frontend and Vocos decoder are separate graphs, not CubeCL fallbacks.

use std::{collections::BTreeMap, path::Path};

use crate::{F5Error, Result, checkpoint::Checkpoint};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct BackboneShape {
    pub batch: usize,
    pub frames: usize,
    pub dim: usize,
    pub heads: usize,
    pub depth: usize,
    pub ff_mult: usize,
    pub mel_channels: usize,
}

impl BackboneShape {
    pub const V1: Self = Self {
        batch: 2,
        frames: 256,
        dim: 1024,
        heads: 16,
        depth: 22,
        ff_mult: 2,
        mel_channels: 100,
    };

    pub fn validate(self) -> Result<()> {
        if self.batch == 0
            || self.batch > 16
            || self.frames == 0
            || self.frames > 65_536
            || self.dim == 0
            || self.heads == 0
            || self.dim % self.heads != 0
            || (self.dim / self.heads) % 2 != 0
            || self.depth == 0
            || self.depth > 22
            || self.ff_mult == 0
            || self.mel_channels == 0
            || self.mel_channels > 100
            || self.dim.checked_mul(self.ff_mult).is_none()
            || self.dim.checked_mul(6).is_none()
            || self
                .batch
                .checked_mul(self.frames)
                .and_then(|x| x.checked_mul(self.dim))
                .is_none()
        {
            return Err(F5Error::InvalidConfiguration(
                "Invalid F5 TensorRT backbone shape",
            ));
        }
        Ok(())
    }

    pub fn buffer_sizes(self) -> Result<BTreeMap<&'static str, usize>> {
        self.validate()?;
        let mut sizes = BTreeMap::new();
        for (name, width) in [("hidden", self.dim), ("output", self.mel_channels)] {
            sizes.insert(
                name,
                self.batch
                    .checked_mul(self.frames)
                    .and_then(|n| n.checked_mul(width))
                    .and_then(|n| n.checked_mul(size_of::<f32>()))
                    .ok_or(F5Error::InvalidShape("TensorRT buffer size overflow"))?,
            );
        }
        sizes.insert("time_embedding", self.batch * self.dim * size_of::<f32>());
        Ok(sizes)
    }
}

#[derive(Debug, Clone)]
pub struct TrtLinear {
    pub input: usize,
    pub output: usize,
    pub weights: Vec<f32>,
    pub bias: Vec<f32>,
}

impl TrtLinear {
    pub fn from_checkpoint(
        checkpoint: &Checkpoint,
        key: &str,
        input: usize,
        output: usize,
    ) -> Result<Self> {
        Ok(Self {
            input,
            output,
            weights: checkpoint.read_f32(&format!("{key}.weight"), &[output, input])?,
            bias: checkpoint.read_f32(&format!("{key}.bias"), &[output])?,
        })
    }
}

#[derive(Debug, Clone)]
pub struct TrtBlockWeights {
    pub modulation: TrtLinear,
    pub query: TrtLinear,
    pub key: TrtLinear,
    pub value: TrtLinear,
    pub attention_out: TrtLinear,
    pub ff_in: TrtLinear,
    pub ff_out: TrtLinear,
}

#[derive(Debug, Clone)]
pub struct TrtBackboneWeights {
    pub shape: BackboneShape,
    pub blocks: Vec<TrtBlockWeights>,
    pub final_modulation: TrtLinear,
    pub projection: TrtLinear,
}

impl TrtBackboneWeights {
    pub fn from_path(path: impl AsRef<Path>, shape: BackboneShape) -> Result<Self> {
        Self::from_checkpoint(&Checkpoint::open(path)?, shape)
    }

    pub fn from_checkpoint(checkpoint: &Checkpoint, shape: BackboneShape) -> Result<Self> {
        shape.validate()?;
        let mut blocks = Vec::with_capacity(shape.depth);
        for n in 0..shape.depth {
            let p = format!("transformer.transformer_blocks.{n}");
            blocks.push(TrtBlockWeights {
                modulation: TrtLinear::from_checkpoint(
                    checkpoint,
                    &format!("{p}.attn_norm.linear"),
                    shape.dim,
                    6 * shape.dim,
                )?,
                query: TrtLinear::from_checkpoint(
                    checkpoint,
                    &format!("{p}.attn.to_q"),
                    shape.dim,
                    shape.dim,
                )?,
                key: TrtLinear::from_checkpoint(
                    checkpoint,
                    &format!("{p}.attn.to_k"),
                    shape.dim,
                    shape.dim,
                )?,
                value: TrtLinear::from_checkpoint(
                    checkpoint,
                    &format!("{p}.attn.to_v"),
                    shape.dim,
                    shape.dim,
                )?,
                attention_out: TrtLinear::from_checkpoint(
                    checkpoint,
                    &format!("{p}.attn.to_out.0"),
                    shape.dim,
                    shape.dim,
                )?,
                ff_in: TrtLinear::from_checkpoint(
                    checkpoint,
                    &format!("{p}.ff.ff.0.0"),
                    shape.dim,
                    shape.dim * shape.ff_mult,
                )?,
                ff_out: TrtLinear::from_checkpoint(
                    checkpoint,
                    &format!("{p}.ff.ff.2"),
                    shape.dim * shape.ff_mult,
                    shape.dim,
                )?,
            });
        }
        Ok(Self {
            shape,
            blocks,
            final_modulation: TrtLinear::from_checkpoint(
                checkpoint,
                "transformer.norm_out.linear",
                shape.dim,
                2 * shape.dim,
            )?,
            projection: TrtLinear::from_checkpoint(
                checkpoint,
                "transformer.proj_out",
                shape.dim,
                shape.mel_channels,
            )?,
        })
    }
}

#[derive(Debug, Clone)]
pub struct TrtTimeWeights {
    pub input: TrtLinear,
    pub output: TrtLinear,
    pub batch: usize,
    pub dim: usize,
}

impl TrtTimeWeights {
    pub fn from_checkpoint(checkpoint: &Checkpoint, batch: usize, dim: usize) -> Result<Self> {
        if batch == 0 || batch > 16 || dim == 0 || dim > 1024 {
            return Err(F5Error::InvalidConfiguration(
                "Invalid F5 TensorRT timestep shape",
            ));
        }
        Ok(Self {
            input: TrtLinear::from_checkpoint(
                checkpoint,
                "transformer.time_embed.time_mlp.0",
                256,
                dim,
            )?,
            output: TrtLinear::from_checkpoint(
                checkpoint,
                "transformer.time_embed.time_mlp.2",
                dim,
                dim,
            )?,
            batch,
            dim,
        })
    }
}

/// Sinusoidal input to F5's 256-dimensional timestep MLP, identical to the CubeCL formula.
pub fn timestep_frequencies(times: &[f32]) -> Result<Vec<f32>> {
    if times.is_empty() || times.len() > 16 || times.iter().any(|x| !x.is_finite()) {
        return Err(F5Error::InvalidConfiguration(
            "F5 timestep batch must be finite and nonempty",
        ));
    }
    let exponent = -(10_000.0_f32.ln()) / 127.0;
    let mut result = Vec::with_capacity(times.len() * 256);
    for &time in times {
        for index in 0..128 {
            result.push((1000.0 * time * (index as f32 * exponent).exp()).sin());
        }
        for index in 0..128 {
            result.push((1000.0 * time * (index as f32 * exponent).exp()).cos());
        }
    }
    Ok(result)
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct DeviceBuffer {
    pub address: usize,
    pub bytes: usize,
}

impl DeviceBuffer {
    pub fn validate(self, minimum_bytes: usize) -> Result<()> {
        if self.address == 0 || self.bytes < minimum_bytes {
            return Err(F5Error::InvalidShape(
                "TensorRT device buffer is null or undersized",
            ));
        }
        Ok(())
    }
}

#[derive(Debug, Clone, Copy)]
pub struct BackboneBindings {
    pub hidden: DeviceBuffer,
    pub time_embedding: DeviceBuffer,
    pub output: DeviceBuffer,
}

impl BackboneBindings {
    pub fn validate(&self, shape: BackboneShape) -> Result<()> {
        let sizes = shape.buffer_sizes()?;
        for (name, region) in [
            ("hidden", self.hidden),
            ("time_embedding", self.time_embedding),
            ("output", self.output),
        ] {
            region.validate(sizes[name])?;
        }
        let overlap = |left: DeviceBuffer, right: DeviceBuffer| -> Result<bool> {
            let end_left = left
                .address
                .checked_add(left.bytes)
                .ok_or(F5Error::InvalidShape("TensorRT buffer address overflow"))?;
            let end_right = right
                .address
                .checked_add(right.bytes)
                .ok_or(F5Error::InvalidShape("TensorRT buffer address overflow"))?;
            Ok(left.address < end_right && right.address < end_left)
        };
        if overlap(self.hidden, self.output)?
            || overlap(self.hidden, self.time_embedding)?
            || overlap(self.time_embedding, self.output)?
        {
            return Err(F5Error::InvalidShape(
                "TensorRT backbone buffers must not overlap",
            ));
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn exact_v1_shapes_and_buffer_bytes() {
        let sizes = BackboneShape::V1.buffer_sizes().unwrap();
        assert_eq!(sizes["hidden"], 2 * 256 * 1024 * 4);
        assert_eq!(sizes["time_embedding"], 2 * 1024 * 4);
        assert_eq!(sizes["output"], 2 * 256 * 100 * 4);
    }

    #[test]
    fn timestep_mlp_weights_match_reduced_pytorch_fixture() {
        let fixture = concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/tests/fixtures/dit_full_small.safetensors"
        );
        let checkpoint = Checkpoint::open(fixture).unwrap();
        let time = TrtTimeWeights::from_checkpoint(&checkpoint, 2, 16).unwrap();
        assert_eq!(time.input.weights.len(), 16 * 256);
        assert_eq!(time.output.weights.len(), 16 * 16);
        let values = timestep_frequencies(&[0.0, 1.0]).unwrap();
        assert_eq!(values.len(), 512);
        assert!(values[..128].iter().all(|&x| x == 0.0));
        assert!(values[128..256].iter().all(|&x| x == 1.0));
        assert!(timestep_frequencies(&[f32::NAN]).is_err());
    }

    #[test]
    fn reject_invalid_head_dimensions() {
        let mut s = BackboneShape::V1;
        s.dim = 1019;
        assert!(s.validate().is_err());
        s.dim = 1024;
        s.heads = 32;
        assert!(s.validate().is_ok());
        s.depth = 23;
        assert!(s.validate().is_err());
    }

    #[test]
    fn exact_checkpoint_schema_for_all_22_blocks() {
        let fixture = concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/tests/fixtures/dit_full_22_blocks.safetensors"
        );
        let config = BackboneShape {
            batch: 2,
            frames: 7,
            dim: 16,
            heads: 4,
            depth: 22,
            ff_mult: 2,
            mel_channels: 2,
        };
        let weights = TrtBackboneWeights::from_path(fixture, config).unwrap();
        assert_eq!(weights.blocks.len(), 22);
        assert_eq!(weights.blocks[21].modulation.weights.len(), 16 * 96);
        assert_eq!(weights.blocks[0].ff_in.weights.len(), 16 * 32);
        assert_eq!(weights.projection.weights.len(), 32);
    }

    #[test]
    fn invalid_buffers_and_overlapping_io_are_rejected() {
        let shape = BackboneShape {
            batch: 1,
            frames: 3,
            dim: 8,
            heads: 2,
            depth: 1,
            ff_mult: 2,
            mel_channels: 2,
        };
        let good = BackboneBindings {
            hidden: DeviceBuffer {
                address: 0x1000,
                bytes: 96,
            },
            time_embedding: DeviceBuffer {
                address: 0x2000,
                bytes: 32,
            },
            output: DeviceBuffer {
                address: 0x3000,
                bytes: 24,
            },
        };
        good.validate(shape).unwrap();
        let mut bad = good;
        bad.output.bytes -= 4;
        assert!(bad.validate(shape).is_err());
        bad = good;
        bad.output.address = bad.hidden.address + 12;
        assert!(bad.validate(shape).is_err());
        bad = good;
        bad.output.address = usize::MAX - 4;
        assert!(bad.validate(shape).is_err());
    }
}

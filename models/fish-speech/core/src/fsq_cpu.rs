use crate::error::{FishError, Result};
use crate::firefly_cpu::AudioTensor;

/// Finite Scalar Quantization index representation in vector-quantize-pytorch 1.14.24.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct FsqCodebook {
    pub levels: Vec<usize>,
    pub bases: Vec<usize>,
    pub codebook_size: usize,
}

impl FsqCodebook {
    pub fn new(levels: &[usize]) -> Result<Self> {
        if levels.is_empty() || levels.iter().any(|&n| n < 2) {
            return Err(FishError::InvalidConfig("FSQ levels must be >=2"));
        }
        let mut bases = Vec::with_capacity(levels.len());
        let mut size = 1usize;
        for &level in levels {
            bases.push(size);
            size = size
                .checked_mul(level)
                .ok_or(FishError::InvalidShape("FSQ codebook size overflow"))?;
        }
        if size > i32::MAX as usize {
            return Err(FishError::InvalidConfig("FSQ index size exceeds i32"));
        }
        Ok(Self {
            levels: levels.to_vec(),
            bases,
            codebook_size: size,
        })
    }

    pub fn decode_index(&self, index: i32) -> Result<Vec<f32>> {
        let idx =
            usize::try_from(index).map_err(|_| FishError::InvalidShape("negative FSQ index"))?;
        if idx >= self.codebook_size {
            return Err(FishError::InvalidShape("FSQ index exceeds codebook size"));
        }
        let mut decoded = Vec::with_capacity(self.levels.len());
        for (&level, &basis) in self.levels.iter().zip(&self.bases) {
            let digit = (idx / basis) % level;
            let half_width = level / 2;
            decoded.push((digit as f32 - half_width as f32) / half_width as f32);
        }
        Ok(decoded)
    }

    pub fn encode_level_indices(&self, digits: &[usize]) -> Result<i32> {
        if digits.len() != self.levels.len()
            || digits
                .iter()
                .zip(&self.levels)
                .any(|(&digit, &level)| digit >= level)
        {
            return Err(FishError::InvalidShape("invalid FSQ level indices"));
        }
        let index = digits
            .iter()
            .zip(&self.bases)
            .try_fold(0usize, |sum, (&digit, &basis)| {
                sum.checked_add(digit.checked_mul(basis)?)
            })
            .ok_or(FishError::InvalidShape("FSQ index overflow"))?;
        i32::try_from(index).map_err(|_| FishError::InvalidShape("FSQ index exceeds i32"))
    }
}

#[derive(Debug, Clone)]
pub struct FsqGroupProjection {
    /// PyTorch nn.Linear project_out: [group_dim, len(levels)] in row-major order.
    pub weights: Vec<f32>,
    pub bias: Vec<f32>,
}

#[derive(Debug, Clone)]
pub struct GroupedResidualFsqDecoder {
    pub codebook: FsqCodebook,
    pub num_quantizers: usize,
    pub projections: Vec<FsqGroupProjection>,
}

impl GroupedResidualFsqDecoder {
    pub fn validate(&self) -> Result<()> {
        let dims = self.codebook.levels.len();
        if self.num_quantizers == 0
            || self.projections.is_empty()
            || self.projections.iter().any(|p| {
                p.bias.is_empty() || p.bias.len().checked_mul(dims) != Some(p.weights.len())
            })
        {
            return Err(FishError::InvalidShape(
                "invalid residual FSQ projection shape",
            ));
        }
        let first_width = self.projections[0].bias.len();
        if self.projections.iter().any(|p| p.bias.len() != first_width) {
            return Err(FishError::InvalidShape(
                "FSQ groups must have equal feature width",
            ));
        }
        Ok(())
    }

    /// Indices are [batch, groups * quantizers, frames] as in Fish fsq.py::decode.
    /// Output is an [batch, dim, frames] tensor before FishTransConvNet upsampling.
    pub fn decode(&self, indices: &[i32], batch: usize, frames: usize) -> Result<AudioTensor> {
        self.validate()?;
        let groups = self.projections.len();
        let group_width = self.projections[0].bias.len();
        let input_channels = groups
            .checked_mul(self.num_quantizers)
            .ok_or(FishError::InvalidShape("FSQ codebook channel overflow"))?;
        let input_len = batch
            .checked_mul(input_channels)
            .and_then(|x| x.checked_mul(frames))
            .ok_or(FishError::InvalidShape("FSQ code input shape overflow"))?;
        if batch == 0 || frames == 0 || indices.len() != input_len {
            return Err(FishError::InvalidShape("FSQ indices BCT shape mismatch"));
        }
        let output_channels = group_width
            .checked_mul(groups)
            .ok_or(FishError::InvalidShape("FSQ output shape overflow"))?;
        let output_len = batch
            .checked_mul(output_channels)
            .and_then(|x| x.checked_mul(frames))
            .ok_or(FishError::InvalidShape("FSQ output tensor overflow"))?;
        let mut output = vec![0.0f32; output_len];
        let mut residual = vec![0.0f32; self.codebook.levels.len()];
        for b in 0..batch {
            for (g, projection) in self.projections.iter().enumerate() {
                for t in 0..frames {
                    residual.fill(0.0);
                    for q in 0..self.num_quantizers {
                        let index = indices
                            [(b * input_channels + g * self.num_quantizers + q) * frames + t];
                        if index == -1 {
                            continue;
                        }
                        let decoded = self.codebook.decode_index(index)?;
                        for (d, &val) in decoded.iter().enumerate() {
                            let scale = ((self.codebook.levels[d] - 1) as f32).powi(-(q as i32));
                            residual[d] += val * scale;
                        }
                    }
                    for out_c in 0..group_width {
                        let mut value = projection.bias[out_c];
                        for (d, &code) in residual.iter().enumerate() {
                            value += projection.weights[out_c * residual.len() + d] * code;
                        }
                        output[(b * output_channels + g * group_width + out_c) * frames + t] =
                            value;
                    }
                }
            }
        }
        AudioTensor::new(batch, output_channels, frames, output)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn fish_fsq_exact_mixed_radix_basis() {
        let c = FsqCodebook::new(&[8, 5, 5, 5]).unwrap();
        assert_eq!(c.bases, vec![1, 8, 40, 200]);
        assert_eq!(c.codebook_size, 1000);
        assert_eq!(c.decode_index(0).unwrap(), vec![-1.0, -1.0, -1.0, -1.0]);
        assert_eq!(c.decode_index(999).unwrap(), vec![0.75, 1.0, 1.0, 1.0]);
        assert_eq!(c.decode_index(40).unwrap(), vec![-1.0, -1.0, -0.5, -1.0]);
    }
    #[test]
    fn fish_fsq_all_1000_codes_roundtrip() {
        let c = FsqCodebook::new(&[8, 5, 5, 5]).unwrap();
        for index in 0..1000usize {
            let mut x = index;
            let digits = c
                .levels
                .iter()
                .map(|&level| {
                    let digit = x % level;
                    x /= level;
                    digit
                })
                .collect::<Vec<_>>();
            assert_eq!(c.encode_level_indices(&digits).unwrap(), index as i32);
        }
    }
    #[test]
    fn fsq_rejects_index_outside_codebook() {
        let c = FsqCodebook::new(&[8, 5, 5, 5]).unwrap();
        assert!(c.decode_index(1000).is_err());
        assert!(c.decode_index(-1).is_err());
        assert!(c.encode_level_indices(&[8, 0, 0, 0]).is_err());
    }
    #[test]
    fn grouped_projection_bct_layout_and_dropped_codebook() {
        let c = FsqCodebook::new(&[2, 3]).unwrap();
        let model = GroupedResidualFsqDecoder {
            codebook: c,
            num_quantizers: 2,
            projections: vec![
                FsqGroupProjection {
                    weights: vec![1.0, 0.0, 0.0, 1.0],
                    bias: vec![0.5, 1.0],
                },
                FsqGroupProjection {
                    weights: vec![2.0, 0.0, 0.0, 3.0],
                    bias: vec![0.0, 0.0],
                },
            ],
        };
        let out = model.decode(&[0, 1, 2, -1, 3, 4, 5, 0], 1, 2).unwrap();
        assert_eq!((out.batch, out.channels, out.frames), (1, 4, 2));
        // q=0 on group0 is 0 -> [-1,-1], q=1 is 2 -> [-1,+0.0], scale = [1, 1/2].
        assert_eq!(out.data.len(), 8);
        assert!(out.data.iter().all(|x| x.is_finite()));
        assert!(model.decode(&[10000, 1, 2, -1, 3, 4, 5, 0], 1, 2).is_err());
    }
}

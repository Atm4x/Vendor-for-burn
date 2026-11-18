use crate::error::{FishError, Result};
use crate::firefly_ir::{FireflyGraph, FireflyOp, FireflyStage};
use crate::fsq_cpu::{FsqCodebook, FsqGroupProjection, GroupedResidualFsqDecoder};
use crate::graph_ir::ScalarType;

pub const FSQ_LEVELS: [usize; 4] = [8, 5, 5, 5];
pub const FSQ_CODEBOOK_SIZE: usize = 1000;

/// A rank-2 [groups * codes_per_group, features_per_group] lookup table.
/// Each row pre-combines original FSQ mixed-radix decode and learned project_out.
/// This is specific to Fish Speech 1.5's single residual quantizer per group.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct FsqDecodeLookupPlan {
    pub name: String,
    pub node_id: usize,
    pub input_node: usize,
    pub input_nchw: [usize; 4],
    pub output_nchw: [usize; 4],
    pub groups: usize,
    pub codebook_size: usize,
    pub features_per_group: usize,
    pub checkpoint_keys: Vec<String>,
}

impl FsqDecodeLookupPlan {
    pub fn from_node(graph: &FireflyGraph, index: usize) -> Result<Option<Self>> {
        graph.validate()?;
        let node = graph
            .nodes
            .get(index)
            .ok_or(FishError::InvalidShape("FSQ node outside graph"))?;
        let (groups, quantizers) = match node.operation {
            FireflyOp::FsqDecode { groups, quantizers } => (groups, quantizers),
            _ => return Ok(None),
        };
        if graph.stage != FireflyStage::WaveformDecode
            || groups != 8
            || quantizers != 1
            || node.inputs.len() != 1
            || node.checkpoint_keys.len() != groups * 2
        {
            return Err(FishError::InvalidShape(
                "unsupported grouped FSQ lookup contract",
            ));
        }
        let before = graph.nodes[node.inputs[0]].output;
        let after = node.output;
        if before.dtype != ScalarType::I32
            || before.channels != groups
            || after.dtype != ScalarType::F32
            || before.batch != after.batch
            || before.frames != after.frames
            || after.channels % groups != 0
        {
            return Err(FishError::InvalidShape("invalid FSQ input/output shape"));
        }
        let feature_width = after.channels / groups;
        for group in 0..groups {
            for (offset, expected) in [vec![feature_width, 4], vec![feature_width]]
                .iter()
                .enumerate()
            {
                let param = graph
                    .weights
                    .get(&node.checkpoint_keys[group * 2 + offset])
                    .ok_or(FishError::InvalidShape("missing FSQ project_out parameter"))?;
                if param.shape != *expected || param.dtype != ScalarType::F32 {
                    return Err(FishError::InvalidShape(
                        "FSQ project_out weight shape mismatch",
                    ));
                }
            }
        }
        Ok(Some(Self {
            name: node.name.clone(),
            node_id: index,
            input_node: node.inputs[0],
            input_nchw: [before.batch, before.channels, 1, before.frames],
            output_nchw: [after.batch, after.channels, 1, after.frames],
            groups,
            codebook_size: FSQ_CODEBOOK_SIZE,
            features_per_group: feature_width,
            checkpoint_keys: node.checkpoint_keys.clone(),
        }))
    }

    pub fn table_f32_bytes(&self, projections: Vec<FsqGroupProjection>) -> Result<Vec<u8>> {
        let model = GroupedResidualFsqDecoder {
            codebook: FsqCodebook::new(&FSQ_LEVELS)?,
            num_quantizers: 1,
            projections,
        };
        model.validate()?;
        if model.projections.len() != self.groups
            || model
                .projections
                .iter()
                .any(|p| p.bias.len() != self.features_per_group)
        {
            return Err(FishError::InvalidShape(
                "wrong FSQ lookup projection groups",
            ));
        }
        let entries = self
            .groups
            .checked_mul(self.codebook_size + 1)
            .and_then(|n| n.checked_mul(self.features_per_group))
            .ok_or(FishError::InvalidShape("FSQ LUT size overflow"))?;
        let mut bytes = Vec::with_capacity(
            entries
                .checked_mul(4)
                .ok_or(FishError::InvalidShape("FSQ LUT byte overflow"))?,
        );
        for projection in &model.projections {
            for bias in &projection.bias {
                bytes.extend_from_slice(&bias.to_le_bytes());
            }
            for code in 0..self.codebook_size {
                let scalar = model.codebook.decode_index(code as i32)?;
                for channel in 0..self.features_per_group {
                    let mut value = projection.bias[channel];
                    for dim in 0..4 {
                        value += projection.weights[channel * 4 + dim] * scalar[dim];
                    }
                    bytes.extend_from_slice(&value.to_le_bytes());
                }
            }
        }
        Ok(bytes)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn lookup_matches_grouped_residual_fsq_decode() {
        let graph = FireflyGraph::waveform_decode(1, 7, 6).unwrap();
        let id = graph
            .nodes
            .iter()
            .position(|n| n.name == "quantizer.fsq_decode")
            .unwrap();
        let plan = FsqDecodeLookupPlan::from_node(&graph, id).unwrap().unwrap();
        let projections = (0..8)
            .map(|g| FsqGroupProjection {
                weights: (0..256)
                    .map(|i| ((i * 7 + g * 3) % 101) as f32 / 89.0 - 0.45)
                    .collect(),
                bias: (0..64)
                    .map(|i| ((i * 3 + g) % 31) as f32 / 51.0 - 0.2)
                    .collect(),
            })
            .collect::<Vec<_>>();
        let table = plan.table_f32_bytes(projections.clone()).unwrap();
        assert_eq!(table.len(), 8 * 1001 * 64 * 4);
        let model = GroupedResidualFsqDecoder {
            codebook: FsqCodebook::new(&FSQ_LEVELS).unwrap(),
            num_quantizers: 1,
            projections,
        };
        let input = (0..(8 * 7))
            .map(|i| ((i * 53 + 137) % 1000) as i32)
            .collect::<Vec<_>>();
        let reference = model.decode(&input, 1, 7).unwrap();
        for group in 0..8 {
            for frame in 0..7 {
                let code = input[group * 7 + frame] as usize;
                for channel in 0..64 {
                    let offset = ((group * 1001 + code + 1) * 64 + channel) * 4;
                    let value = f32::from_le_bytes(table[offset..offset + 4].try_into().unwrap());
                    let expected = reference.data[(group * 64 + channel) * 7 + frame];
                    assert_eq!(
                        value, expected,
                        "group={group}, frame={frame}, channel={channel}"
                    );
                }
            }
        }
    }
    #[test]
    fn all_1000_indices_in_all_eight_groups_match_cpu_reference() {
        let graph = FireflyGraph::waveform_decode(1, 1, 1).unwrap();
        let id = graph
            .nodes
            .iter()
            .position(|n| n.name == "quantizer.fsq_decode")
            .unwrap();
        let plan = FsqDecodeLookupPlan::from_node(&graph, id).unwrap().unwrap();
        let projections = (0..8)
            .map(|g| FsqGroupProjection {
                weights: (0..256)
                    .map(|i| ((i * 19 + g * 11) % 107) as f32 / 103.0 - 0.6)
                    .collect(),
                bias: (0..64)
                    .map(|i| ((i * 13 + g) % 47) as f32 / 101.0 - 0.2)
                    .collect(),
            })
            .collect::<Vec<_>>();
        let table = plan.table_f32_bytes(projections.clone()).unwrap();
        let reference = GroupedResidualFsqDecoder {
            codebook: FsqCodebook::new(&FSQ_LEVELS).unwrap(),
            num_quantizers: 1,
            projections,
        };
        let indices = (0..8).flat_map(|_| 0..1000i32).collect::<Vec<_>>();
        let output = reference.decode(&indices, 1, 1000).unwrap();
        for group in 0..8 {
            for index in 0..1000 {
                for channel in 0..64 {
                    let offset = ((group * 1001 + index + 1) * 64 + channel) * 4;
                    let lookup = f32::from_le_bytes(table[offset..offset + 4].try_into().unwrap());
                    assert_eq!(lookup, output.data[(group * 64 + channel) * 1000 + index]);
                }
            }
        }
    }
    #[test]
    fn minus_one_drop_token_uses_project_out_bias() {
        let graph = FireflyGraph::waveform_decode(1, 1, 1).unwrap();
        let id = graph
            .nodes
            .iter()
            .position(|n| n.name == "quantizer.fsq_decode")
            .unwrap();
        let plan = FsqDecodeLookupPlan::from_node(&graph, id).unwrap().unwrap();
        let projections = (0..8)
            .map(|g| FsqGroupProjection {
                weights: vec![0.25; 256],
                bias: vec![g as f32 / 10.0; 64],
            })
            .collect::<Vec<_>>();
        let table = plan.table_f32_bytes(projections.clone()).unwrap();
        let model = GroupedResidualFsqDecoder {
            codebook: FsqCodebook::new(&FSQ_LEVELS).unwrap(),
            num_quantizers: 1,
            projections,
        };
        let output = model.decode(&vec![-1; 8], 1, 1).unwrap();
        for group in 0..8 {
            for channel in 0..64 {
                let offset = (group * 1001 * 64 + channel) * 4;
                let lookup = f32::from_le_bytes(table[offset..offset + 4].try_into().unwrap());
                assert_eq!(lookup, output.data[group * 64 + channel]);
            }
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct FsqEncodePlan {
    pub name: String,
    pub node_id: usize,
    pub input_node: usize,
    pub input_nchw: [usize; 4],
    pub output_nchw: [usize; 4],
    pub groups: usize,
    pub group_width: usize,
    pub checkpoint_keys: Vec<String>,
}

impl FsqEncodePlan {
    pub fn from_node(graph: &FireflyGraph, index: usize) -> Result<Option<Self>> {
        graph.validate()?;
        let node = graph
            .nodes
            .get(index)
            .ok_or(FishError::InvalidShape("FSQ encode node outside graph"))?;
        let (groups, quantizers) = match node.operation {
            FireflyOp::FsqEncode { groups, quantizers } => (groups, quantizers),
            _ => return Ok(None),
        };
        if graph.stage != FireflyStage::ReferenceEncode
            || groups != 8
            || quantizers != 1
            || node.inputs.len() != 1
            || node.checkpoint_keys.len() != groups * 2
        {
            return Err(FishError::InvalidShape("unsupported FSQ encoder contract"));
        }
        let input = graph.nodes[node.inputs[0]].output;
        let output = node.output;
        if input.dtype != ScalarType::F32
            || input.channels % groups != 0
            || output.dtype != ScalarType::I32
            || output.channels != groups
            || input.batch != output.batch
            || input.frames != output.frames
        {
            return Err(FishError::InvalidShape("FSQ encode input/output mismatch"));
        }
        let width = input.channels / groups;
        for group in 0..groups {
            for (offset, expected) in [vec![4, width], vec![4]].iter().enumerate() {
                let name = &node.checkpoint_keys[group * 2 + offset];
                let meta = graph.weights.get(name).ok_or(FishError::InvalidShape(
                    "FSQ encoder checkpoint parameter not declared",
                ))?;
                if meta.shape != *expected || meta.dtype != ScalarType::F32 {
                    return Err(FishError::InvalidShape(
                        "FSQ project_in dimensions mismatch",
                    ));
                }
            }
        }
        Ok(Some(Self {
            name: node.name.clone(),
            node_id: index,
            input_node: node.inputs[0],
            input_nchw: [input.batch, input.channels, 1, input.frames],
            output_nchw: [output.batch, output.channels, 1, output.frames],
            groups,
            group_width: width,
            checkpoint_keys: node.checkpoint_keys.clone(),
        }))
    }

    pub fn grouped_conv_weights(
        &self,
        projections: &[FsqGroupProjection],
    ) -> Result<(Vec<u8>, Vec<u8>)> {
        if projections.len() != self.groups
            || projections.iter().any(|p| {
                p.weights.len() != 4 * self.group_width
                    || p.bias.len() != 4
                    || p.weights.iter().chain(&p.bias).any(|x| !x.is_finite())
            })
        {
            return Err(FishError::InvalidShape(
                "invalid FSQ encoder projection weights",
            ));
        }
        let mut kernel = Vec::with_capacity(self.groups * 4 * self.group_width * 4);
        let mut bias = Vec::with_capacity(self.groups * 4 * 4);
        for projection in projections {
            for value in &projection.weights {
                kernel.extend_from_slice(&value.to_le_bytes());
            }
            for value in &projection.bias {
                bias.extend_from_slice(&value.to_le_bytes());
            }
        }
        Ok((kernel, bias))
    }

    pub fn scalar_constants(&self) -> (Vec<f32>, Vec<f32>, Vec<f32>, Vec<f32>, Vec<f32>) {
        let mut shift = Vec::with_capacity(self.groups * 4);
        let mut half = Vec::with_capacity(self.groups * 4);
        let mut offset = Vec::with_capacity(self.groups * 4);
        let mut digit_half = Vec::with_capacity(self.groups * 4);
        let mut bases = Vec::with_capacity(self.groups * 4);
        for _ in 0..self.groups {
            let mut basis = 1usize;
            for &level in &FSQ_LEVELS {
                let h = (level - 1) as f32 * 1.001 / 2.0;
                let o = if level % 2 == 0 { 0.5 } else { 0.0 };
                shift.push((o / h).atanh());
                half.push(h);
                offset.push(o);
                digit_half.push((level / 2) as f32);
                bases.push(basis as f32);
                basis *= level;
            }
        }
        (shift, half, offset, digit_half, bases)
    }
}

#[cfg(test)]
mod encode_tests {
    use super::*;
    use crate::firefly_cpu::AudioTensor;
    use crate::fsq_cpu::GroupedResidualFsqEncoder;

    #[test]
    fn encoder_weights_pack_identically_to_grouped_conv1x1() {
        let graph = FireflyGraph::reference_encode(2, 12_345).unwrap();
        let id = graph
            .nodes
            .iter()
            .position(|n| matches!(n.operation, FireflyOp::FsqEncode { .. }))
            .unwrap();
        let plan = FsqEncodePlan::from_node(&graph, id).unwrap().unwrap();
        assert_eq!(plan.input_nchw, [2, 512, 1, 6]);
        assert_eq!(plan.output_nchw, [2, 8, 1, 6]);
        let projections: Vec<_> = (0..8)
            .map(|g| FsqGroupProjection {
                weights: (0..256)
                    .map(|i| (i as f32 - 120.0 + g as f32) / 440.0)
                    .collect(),
                bias: (0..4).map(|i| (i as f32 - g as f32) / 33.0).collect(),
            })
            .collect();
        let (weight, bias) = plan.grouped_conv_weights(&projections).unwrap();
        assert_eq!(weight.len(), 8 * 4 * 64 * 4);
        assert_eq!(bias.len(), 8 * 4 * 4);
        let data: Vec<f32> = (0..(2 * 512 * 6))
            .map(|i| ((i * 13) % 137) as f32 / 70.0 - 1.0)
            .collect();
        let features = AudioTensor::new(2, 512, 6, data).unwrap();
        let reference = GroupedResidualFsqEncoder {
            codebook: FsqCodebook::new(&FSQ_LEVELS).unwrap(),
            num_quantizers: 1,
            projections,
        }
        .encode(&features)
        .unwrap();
        let (shift, half, offset, digit_half, bases) = plan.scalar_constants();
        let mut actual = Vec::new();
        for b in 0..2 {
            for g in 0..8 {
                for t in 0..6 {
                    let mut code = 0i32;
                    for d in 0..4 {
                        let channel = g * 4 + d;
                        let mut y = f32::from_le_bytes(
                            bias[channel * 4..channel * 4 + 4].try_into().unwrap(),
                        );
                        for c in 0..64 {
                            let x = features.data[(b * 512 + g * 64 + c) * 6 + t];
                            let p = (channel * 64 + c) * 4;
                            y += f32::from_le_bytes(weight[p..p + 4].try_into().unwrap()) * x;
                        }
                        for _ in 0..2 {
                            y = (y + shift[channel]).tanh() * half[channel] - offset[channel];
                        }
                        code +=
                            ((y.round_ties_even() + digit_half[channel]) * bases[channel]) as i32;
                    }
                    actual.push((b, g, t, code));
                }
            }
        }
        for (b, g, t, code) in actual {
            assert_eq!(code, reference[(b * 8 + g) * 6 + t]);
        }
    }

    #[test]
    fn reject_invalid_grouped_projection() {
        let graph = FireflyGraph::reference_encode(1, 4096).unwrap();
        let idx = graph
            .nodes
            .iter()
            .position(|n| matches!(n.operation, FireflyOp::FsqEncode { .. }))
            .unwrap();
        let plan = FsqEncodePlan::from_node(&graph, idx).unwrap().unwrap();
        let mut projections = vec![
            FsqGroupProjection {
                weights: vec![0.0; 256],
                bias: vec![0.0; 4]
            };
            8
        ];
        assert!(plan.grouped_conv_weights(&projections).is_ok());
        projections[3].weights[0] = f32::NAN;
        assert!(plan.grouped_conv_weights(&projections).is_err());
    }
}

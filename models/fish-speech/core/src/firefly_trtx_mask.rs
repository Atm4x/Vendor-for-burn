use crate::audio::{firefly_decode_lengths, firefly_reference_lengths, HOP_LENGTH};
use crate::error::{FishError, Result};
use crate::firefly_ir::{FireflyGraph, FireflyOp, FireflyStage};
use crate::graph_ir::ScalarType;

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct FireflyMaskPlan {
    pub name: String,
    pub node_id: usize,
    pub input_node: usize,
    pub nchw: [usize; 4],
    pub valid_frames: usize,
}

impl FireflyMaskPlan {
    pub fn from_node(graph: &FireflyGraph, index: usize) -> Result<Option<Self>> {
        graph.validate()?;
        let node = graph
            .nodes
            .get(index)
            .ok_or(FishError::InvalidShape("mask node out of bounds"))?;
        if node.operation != FireflyOp::Mask {
            return Ok(None);
        }
        if node.inputs.len() != 1
            || !node.checkpoint_keys.is_empty()
            || node.output.dtype != ScalarType::F32
        {
            return Err(FishError::InvalidShape(
                "unexpected Firefly mask arity or dtype",
            ));
        }
        let valid_frames = match (graph.stage, node.name.as_str()) {
            (FireflyStage::ReferenceEncode, "reference.mel_mask" | "reference.feature_mask") => {
                let samples = graph.nodes[graph.input].output.frames;
                firefly_reference_lengths(samples)?.mel_frames
            }
            (FireflyStage::WaveformDecode, "waveform.quantized_mel_mask") => {
                let code = graph.nodes[graph.input].output.frames;
                let valid_code = graph
                    .valid_frames
                    .checked_div(4 * HOP_LENGTH)
                    .ok_or(FishError::InvalidShape("decode mask length overflow"))?;
                if graph.valid_frames % (4 * HOP_LENGTH) != 0 {
                    return Err(FishError::InvalidShape("unaligned Firefly output mask"));
                }
                firefly_decode_lengths(code, valid_code)?.valid_mel_frames
            }
            (FireflyStage::WaveformDecode, "waveform.audio_mask") => graph.valid_frames,
            _ => return Err(FishError::InvalidShape("unknown Firefly mask boundary")),
        };
        let out = node.output;
        if valid_frames > out.frames || graph.nodes[node.inputs[0]].output != out {
            return Err(FishError::InvalidShape(
                "Firefly mask length/input mismatch",
            ));
        }
        Ok(Some(Self {
            name: node.name.clone(),
            node_id: index,
            input_node: node.inputs[0],
            nchw: [out.batch, out.channels, 1, out.frames],
            valid_frames,
        }))
    }

    pub fn mask_f32_bytes(&self) -> Result<Vec<u8>> {
        let total_bytes = self.nchw[3]
            .checked_mul(4)
            .ok_or(FishError::InvalidShape("mask allocation overflow"))?;
        let mut bytes = Vec::with_capacity(total_bytes);
        for index in 0..self.nchw[3] {
            bytes.extend_from_slice(
                &if index < self.valid_frames {
                    1f32
                } else {
                    0f32
                }
                .to_le_bytes(),
            );
        }
        Ok(bytes)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn reference_masks_match_audio_hop_length() {
        let graph = FireflyGraph::reference_encode(2, 12345).unwrap();
        for name in ["reference.mel_mask", "reference.feature_mask"] {
            let id = graph.nodes.iter().position(|v| v.name == name).unwrap();
            let mask = FireflyMaskPlan::from_node(&graph, id).unwrap().unwrap();
            assert_eq!(mask.valid_frames, 12345 / 512);
            assert_eq!(mask.nchw[0], 2);
            assert_eq!(mask.nchw[3], 12345 / 512);
        }
    }
    #[test]
    fn waveform_mask_keeps_only_valid_audio() {
        let graph = FireflyGraph::waveform_decode(1, 7, 6).unwrap();
        for (name, frames, valid) in [
            ("waveform.quantized_mel_mask", 28, 24),
            ("waveform.audio_mask", 7 * 2048, 6 * 2048),
        ] {
            let id = graph.nodes.iter().position(|v| v.name == name).unwrap();
            let mask = FireflyMaskPlan::from_node(&graph, id).unwrap().unwrap();
            assert_eq!(mask.nchw[3], frames);
            assert_eq!(mask.valid_frames, valid);
            let data = mask.mask_f32_bytes().unwrap();
            assert_eq!(data.len(), frames * 4);
            assert_eq!(
                f32::from_le_bytes(data[(valid - 1) * 4..valid * 4].try_into().unwrap()),
                1.
            );
            assert_eq!(
                f32::from_le_bytes(data[valid * 4..valid * 4 + 4].try_into().unwrap()),
                0.
            );
        }
    }
}

use std::collections::BTreeMap;

use crate::trtx_firefly_fsq::{lower_fsq_decode_from_checkpoint, lower_fsq_encode_from_checkpoint};
use crate::trtx_firefly_mask::lower_firefly_mask;
use crate::trtx_firefly_pointwise::lower_firefly_pointwise_from_checkpoint;
use crate::trtx_lowering::{LoweringError, lower_firefly_conv2d_from_checkpoint};
use fish_speech_core::firefly_ir::FireflyGraph;
use fish_speech_core::firefly_trtx_conv::FireflyConv2dPlan;
use fish_speech_core::firefly_trtx_fsq::{FsqDecodeLookupPlan, FsqEncodePlan};
use fish_speech_core::firefly_trtx_mask::FireflyMaskPlan;
use fish_speech_core::firefly_trtx_pointwise::PointwisePlan;
use fish_speech_core::firefly_trtx_segment::{SegmentOp, plan_segment};
use fish_speech_core::graph_ir::ScalarType;
use safetensors::tensor::Dtype;
use trtx::{DataType, NetworkDefinition, Tensor};

fn invalid(msg: impl Into<String>) -> LoweringError {
    LoweringError::InvalidBlueprint(msg.into())
}

/// Composes actual TensorRT layers into a connected Firefly subgraph.
/// External tensors are explicit boundaries (e.g. verified log-mel or FSQ output).
/// All parameter bindings are validated before any TRT network mutation.
/// Does not implicitly implement masks, FSQ, STFT or a full TTS pipeline.
pub fn lower_firefly_segment<'n>(
    network: &mut NetworkDefinition<'n>,
    graph: &FireflyGraph,
    output_id: usize,
    externals: Vec<(usize, Tensor<'n>)>,
    checkpoint: &crate::weights::LoadedWeights,
) -> Result<Tensor<'n>, LoweringError> {
    let external_ids = externals.iter().map(|(id, _)| *id).collect::<Vec<_>>();
    let segment = plan_segment(graph, output_id, &external_ids)
        .map_err(|e| invalid(format!("Firefly segment: {e:?}")))?;
    let mut values = BTreeMap::new();
    for (id, tensor) in externals {
        let shape = graph.nodes[id].output;
        let required = vec![
            shape.batch as i64,
            shape.channels as i64,
            1,
            shape.frames as i64,
        ];
        let dtype = match shape.dtype {
            ScalarType::F32 => DataType::kFLOAT,
            ScalarType::I32 => DataType::kINT32,
            ScalarType::BF16 => DataType::kBF16,
        };
        if tensor
            .dimensions(network)
            .map_err(|e| LoweringError::TensorRt(e.to_string()))?
            != required
            || tensor.data_type(network) != dtype
        {
            return Err(invalid(format!(
                "{}: external tensor shape or dtype mismatch",
                graph.nodes[id].name
            )));
        }
        values.insert(id, tensor);
    }
    for step in &segment.steps {
        if step.op == SegmentOp::External {
            continue;
        }
        for key in &graph.nodes[step.index].checkpoint_keys {
            let wanted = &graph.weights[key];
            let tensor = checkpoint
                .tensor(key)
                .map_err(|e| invalid(format!("{key}: {e}")))?;
            if tensor.dtype() != Dtype::F32 || tensor.shape() != wanted.shape {
                return Err(invalid(format!(
                    "{key}: expected strict F32 weight shape {:?}",
                    wanted.shape
                )));
            }
        }
    }
    for step in &segment.steps {
        if step.op == SegmentOp::External {
            continue;
        }
        let args = graph.nodes[step.index]
            .inputs
            .iter()
            .map(|id| {
                values
                    .get(id)
                    .ok_or_else(|| invalid(format!("{}: missing node {id}", step.name)))
            })
            .collect::<Result<Vec<_>, _>>()?;
        let value = match step.op {
            SegmentOp::Conv => {
                let plan = FireflyConv2dPlan::from_node(graph, step.index)
                    .map_err(|e| invalid(format!("{}: {e:?}", step.name)))?
                    .ok_or_else(|| invalid("missing Conv2D plan"))?;
                lower_firefly_conv2d_from_checkpoint(network, args[0], &plan, checkpoint)?
            }
            SegmentOp::Pointwise => {
                let plan = PointwisePlan::from_node(graph, step.index)
                    .map_err(|e| invalid(format!("{}: {e:?}", step.name)))?
                    .ok_or_else(|| invalid("missing pointwise plan"))?;
                lower_firefly_pointwise_from_checkpoint(network, &args, &plan, checkpoint)?
            }
            SegmentOp::Mask => {
                let plan = FireflyMaskPlan::from_node(graph, step.index)
                    .map_err(|e| invalid(format!("{}: {e:?}", step.name)))?
                    .ok_or_else(|| invalid("missing Firefly mask plan"))?;
                lower_firefly_mask(network, args[0], &plan)?
            }
            SegmentOp::FsqEncode => {
                let plan = FsqEncodePlan::from_node(graph, step.index)
                    .map_err(|e| invalid(format!("{}: {e:?}", step.name)))?
                    .ok_or_else(|| invalid("missing FSQ encode plan"))?;
                lower_fsq_encode_from_checkpoint(network, args[0], &plan, checkpoint)?
            }
            SegmentOp::FsqDecode => {
                let plan = FsqDecodeLookupPlan::from_node(graph, step.index)
                    .map_err(|e| invalid(format!("{}: {e:?}", step.name)))?
                    .ok_or_else(|| invalid("missing FSQ decode lookup plan"))?;
                lower_fsq_decode_from_checkpoint(network, args[0], &plan, checkpoint)?
            }
            SegmentOp::External => unreachable!("external nodes are already present"),
        };
        values.insert(step.index, value);
    }
    values
        .remove(&output_id)
        .ok_or_else(|| invalid("Firefly segment output not bound"))
}

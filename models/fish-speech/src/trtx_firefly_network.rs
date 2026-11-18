use crate::trtx_firefly_segment::lower_firefly_segment;
use crate::trtx_lowering::LoweringError;
use fish_speech_core::firefly_ir::{FireflyGraph, FireflyOp, FireflyStage};
use fish_speech_core::firefly_trtx_segment::{SegmentOp, plan_segment};
use fish_speech_core::graph_ir::ScalarType;
use trtx::{DataType, NetworkDefinition};

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct FireflyNetworkInterface {
    pub external_node_id: usize,
    pub external_name: String,
    pub external_shape: [usize; 4],
    pub external_type: ScalarType,
    pub output_shape: [usize; 4],
    pub output_type: ScalarType,
    pub internal_nodes: usize,
}

pub fn preflight_firefly_network(
    graph: &FireflyGraph,
) -> Result<FireflyNetworkInterface, LoweringError> {
    let external_node_id = match graph.stage {
        FireflyStage::ReferenceEncode => graph
            .nodes
            .iter()
            .position(|n| n.operation == FireflyOp::LogMel),
        FireflyStage::WaveformDecode => Some(graph.input),
    }
    .ok_or_else(|| LoweringError::InvalidBlueprint("missing Firefly network boundary".into()))?;
    let segment = plan_segment(graph, graph.output, &[external_node_id])
        .map_err(|e| LoweringError::InvalidBlueprint(format!("Firefly network: {e:?}")))?;
    if segment.steps.first().map(|s| s.op) != Some(SegmentOp::External)
        || segment.steps.last().map(|s| s.index) != Some(graph.output)
    {
        return Err(LoweringError::InvalidBlueprint(
            "unclosed Firefly dependency graph".into(),
        ));
    }
    let from = &graph.nodes[external_node_id];
    let to = &graph.nodes[graph.output];
    Ok(FireflyNetworkInterface {
        external_node_id,
        external_name: from.name.clone(),
        external_shape: [
            from.output.batch,
            from.output.channels,
            1,
            from.output.frames,
        ],
        external_type: from.output.dtype,
        output_shape: [to.output.batch, to.output.channels, 1, to.output.frames],
        output_type: to.output.dtype,
        internal_nodes: segment.steps.len() - 1,
    })
}

pub fn define_firefly_network(
    network: &mut NetworkDefinition<'_>,
    graph: &FireflyGraph,
    checkpoint: &crate::weights::LoadedWeights,
) -> Result<FireflyNetworkInterface, LoweringError> {
    let spec = preflight_firefly_network(graph)?;
    let input_type = match spec.external_type {
        ScalarType::F32 => DataType::kFLOAT,
        ScalarType::I32 => DataType::kINT32,
        ScalarType::BF16 => DataType::kBF16,
    };
    let dims = spec
        .external_shape
        .map(|dim| {
            i64::try_from(dim).map_err(|_| {
                LoweringError::InvalidBlueprint("Firefly input dimension exceeds i64".into())
            })
        })
        .into_iter()
        .collect::<Result<Vec<_>, _>>()?;
    let input = network
        .add_input(&spec.external_name, input_type, &dims)
        .map_err(|e| LoweringError::TensorRt(e.to_string()))?;
    let output = lower_firefly_segment(
        network,
        graph,
        graph.output,
        vec![(spec.external_node_id, input)],
        checkpoint,
    )?;
    network.mark_output(&output);
    Ok(spec)
}

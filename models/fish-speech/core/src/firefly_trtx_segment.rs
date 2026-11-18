use std::collections::BTreeSet;

use crate::error::{FishError, Result};
use crate::firefly_ir::{FireflyGraph, FireflyOp};
use crate::firefly_trtx_conv::FireflyConv2dPlan;
use crate::firefly_trtx_fsq::{FsqDecodeLookupPlan, FsqEncodePlan};
use crate::firefly_trtx_mask::FireflyMaskPlan;
use crate::firefly_trtx_pointwise::PointwisePlan;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SegmentOp {
    External,
    Conv,
    Pointwise,
    Mask,
    FsqDecode,
    FsqEncode,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SegmentStep {
    pub index: usize,
    pub name: String,
    pub op: SegmentOp,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct FireflySegment {
    pub output: usize,
    pub steps: Vec<SegmentStep>,
}

/// Resolve the dependency closure for a subgraph before modifying a TRT network.
/// Unsupported stages (LogMel, FSQ, Mask, physical graph I/O) are never silently
/// replaced with identity operations. They must be explicitly supplied as external
/// graph boundaries by the caller with the correct dtype and length contracts.
pub fn plan_segment(
    graph: &FireflyGraph,
    output: usize,
    externals: &[usize],
) -> Result<FireflySegment> {
    graph.validate()?;
    if output >= graph.nodes.len() || externals.iter().any(|&v| v >= graph.nodes.len()) {
        return Err(FishError::InvalidShape(
            "segment boundary outside Firefly DAG",
        ));
    }
    let external = externals.iter().copied().collect::<BTreeSet<_>>();
    if external.len() != externals.len() {
        return Err(FishError::InvalidShape(
            "duplicate Firefly external boundary",
        ));
    }
    let mut required = BTreeSet::new();
    let mut stack = vec![output];
    while let Some(i) = stack.pop() {
        if !required.insert(i) || external.contains(&i) {
            continue;
        }
        stack.extend(graph.nodes[i].inputs.iter().copied());
    }
    let mut steps = Vec::with_capacity(required.len());
    for &index in &required {
        let node = &graph.nodes[index];
        let op = if external.contains(&index) {
            SegmentOp::External
        } else if FireflyConv2dPlan::from_node(graph, index)?.is_some() {
            SegmentOp::Conv
        } else if PointwisePlan::from_node(graph, index)?.is_some() {
            SegmentOp::Pointwise
        } else if FireflyMaskPlan::from_node(graph, index)?.is_some() {
            SegmentOp::Mask
        } else if FsqDecodeLookupPlan::from_node(graph, index)?.is_some() {
            SegmentOp::FsqDecode
        } else if FsqEncodePlan::from_node(graph, index)?.is_some() {
            SegmentOp::FsqEncode
        } else {
            return Err(FishError::InvalidShape(match node.operation {
                FireflyOp::Mask => "Firefly Mask needs explicit sequence-length semantics",
                FireflyOp::LogMel => "Firefly LogMel requires a separate spectrogram pipeline",
                FireflyOp::FsqEncode { .. } | FireflyOp::FsqDecode { .. } => {
                    "Firefly FSQ requires a codebook transform"
                }
                _ => "Firefly model boundary requires an input tensor",
            }));
        };
        steps.push(SegmentStep {
            index,
            name: node.name.clone(),
            op,
        });
    }
    Ok(FireflySegment { output, steps })
}

pub fn structural_coverage(graph: &FireflyGraph) -> Result<(usize, usize, Vec<(usize, String)>)> {
    graph.validate()?;
    let mut eligible = 0;
    let mut blocked = Vec::new();
    for (index, node) in graph.nodes.iter().enumerate() {
        if FireflyConv2dPlan::from_node(graph, index)?.is_some()
            || PointwisePlan::from_node(graph, index)?.is_some()
            || FireflyMaskPlan::from_node(graph, index)?.is_some()
            || FsqDecodeLookupPlan::from_node(graph, index)?.is_some()
            || FsqEncodePlan::from_node(graph, index)?.is_some()
        {
            eligible += 1;
        } else {
            blocked.push((index, format!("{}: {:?}", node.name, node.operation)));
        }
    }
    Ok((eligible, graph.nodes.len(), blocked))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn convnext_block_lowering_preserves_all_dependencies() {
        let graph = FireflyGraph::reference_encode(1, 8192).unwrap();
        let start = graph
            .nodes
            .iter()
            .position(|n| n.name == "backbone.stages.0.0.dwconv")
            .unwrap();
        let incoming = graph.nodes[start].inputs[0];
        let end = graph
            .nodes
            .iter()
            .position(|n| n.name == "backbone.stages.0.0.residual")
            .unwrap();
        let segment = plan_segment(&graph, end, &[incoming]).unwrap();
        assert_eq!(segment.steps.first().unwrap().index, incoming);
        assert_eq!(segment.steps.last().unwrap().index, end);
        assert_eq!(segment.steps.len(), 8);
        assert_eq!(
            segment
                .steps
                .iter()
                .filter(|s| s.op == SegmentOp::Conv)
                .count(),
            3
        );
        assert_eq!(
            segment
                .steps
                .iter()
                .filter(|s| s.op == SegmentOp::Pointwise)
                .count(),
            4
        );
    }

    #[test]
    fn mask_and_fsq_cannot_be_silently_ignored() {
        let graph = FireflyGraph::waveform_decode(1, 4, 3).unwrap();
        assert!(plan_segment(&graph, graph.output, &[]).is_err());
        let (covered, total, blocked) = structural_coverage(&graph).unwrap();
        assert_eq!(total, graph.nodes.len());
        assert_eq!(covered + blocked.len(), total);
        assert!(!blocked.iter().any(|(_, s)| s.contains("FsqDecode")));
        assert!(!blocked.iter().any(|(_, s)| s.contains("Mask")));
    }
    #[test]
    fn entire_waveform_decoder_closes_from_explicit_code_input() {
        let graph = FireflyGraph::waveform_decode(1, 7, 6).unwrap();
        let segment = plan_segment(&graph, graph.output, &[graph.input]).unwrap();
        assert_eq!(segment.steps.len(), graph.nodes.len());
        assert_eq!(segment.steps[0].op, SegmentOp::External);
        assert_eq!(segment.steps.last().unwrap().index, graph.output);
        assert_eq!(
            segment
                .steps
                .iter()
                .filter(|s| s.op == SegmentOp::FsqDecode)
                .count(),
            1
        );
        assert_eq!(
            segment
                .steps
                .iter()
                .filter(|s| s.op == SegmentOp::Mask)
                .count(),
            2
        );
    }
    #[test]
    fn entire_reference_encoder_closes_from_external_logmel() {
        let graph = FireflyGraph::reference_encode(2, 12_345).unwrap();
        let mel = graph
            .nodes
            .iter()
            .position(|node| node.operation == FireflyOp::LogMel)
            .unwrap();
        let segment = plan_segment(&graph, graph.output, &[mel]).unwrap();
        assert_eq!(segment.steps.first().unwrap().op, SegmentOp::External);
        assert_eq!(segment.steps.last().unwrap().op, SegmentOp::FsqEncode);
        assert_eq!(segment.steps.len(), graph.nodes.len() - 1);
        let (supported, total, blocked) = structural_coverage(&graph).unwrap();
        assert_eq!((supported, total), (154, 156));
        assert!(blocked.iter().any(|(_, s)| s.contains("AudioInput")));
        assert!(blocked.iter().any(|(_, s)| s.contains("LogMel")));
    }
    #[test]
    fn reference_encoder_closes_between_external_logmel_and_fsq_input() {
        let graph = FireflyGraph::reference_encode(1, 12345).unwrap();
        let start = graph
            .nodes
            .iter()
            .position(|v| v.operation == FireflyOp::LogMel)
            .unwrap();
        let fsq = graph
            .nodes
            .iter()
            .position(|v| matches!(v.operation, FireflyOp::FsqEncode { .. }))
            .unwrap();
        let end = graph.nodes[fsq].inputs[0];
        let segment = plan_segment(&graph, end, &[start]).unwrap();
        assert_eq!(segment.steps.first().unwrap().index, start);
        assert_eq!(segment.steps.last().unwrap().index, end);
        assert_eq!(
            segment
                .steps
                .iter()
                .filter(|s| s.op == SegmentOp::Mask)
                .count(),
            2
        );
    }
}

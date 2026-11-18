use crate::error::{FishError, Result};
use crate::firefly_ir::{FireflyGraph, FireflyOp};
use crate::graph_ir::ScalarType;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PointwiseKind {
    SiLu,
    GeluErf,
    Tanh,
    Add,
    ParallelMean,
    LayerScale,
    ChannelLayerNorm,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PointwisePlan {
    pub name: String,
    pub node_id: usize,
    pub kind: PointwiseKind,
    pub inputs: Vec<usize>,
    pub nchw: [usize; 4],
    pub checkpoint_keys: Vec<String>,
}

impl PointwisePlan {
    pub fn from_node(graph: &FireflyGraph, id: usize) -> Result<Option<Self>> {
        graph.validate()?;
        let node = graph
            .nodes
            .get(id)
            .ok_or(FishError::InvalidShape("invalid Firefly pointwise node"))?;
        let kind = match node.operation {
            FireflyOp::SiLu => PointwiseKind::SiLu,
            FireflyOp::Gelu => PointwiseKind::GeluErf,
            FireflyOp::Tanh => PointwiseKind::Tanh,
            FireflyOp::Add => PointwiseKind::Add,
            FireflyOp::ParallelMean => PointwiseKind::ParallelMean,
            FireflyOp::LayerScale => PointwiseKind::LayerScale,
            FireflyOp::LayerNorm => PointwiseKind::ChannelLayerNorm,
            _ => return Ok(None),
        };
        let inputs = match kind {
            PointwiseKind::Add => 2,
            PointwiseKind::ParallelMean => 3,
            _ => 1,
        };
        if node.inputs.len() != inputs || node.output.dtype != ScalarType::F32 {
            return Err(FishError::InvalidShape(
                "pointwise operand arity/dtype mismatch",
            ));
        }
        if node
            .inputs
            .iter()
            .any(|&input| graph.nodes[input].output != node.output)
        {
            return Err(FishError::InvalidShape(
                "pointwise operands must have identical shape",
            ));
        }
        let weights = match kind {
            PointwiseKind::ChannelLayerNorm => 2,
            PointwiseKind::LayerScale => 1,
            _ => 0,
        };
        if node.checkpoint_keys.len() != weights {
            return Err(FishError::InvalidShape(
                "incorrect channel-affine binding count",
            ));
        }
        for key in &node.checkpoint_keys {
            let param = graph
                .weights
                .get(key)
                .ok_or(FishError::InvalidShape("missing affine tensor"))?;
            if param.shape != [node.output.channels] || param.dtype != ScalarType::F32 {
                return Err(FishError::InvalidShape(
                    "incorrect affine channel count/dtype",
                ));
            }
        }
        Ok(Some(Self {
            name: node.name.clone(),
            node_id: id,
            kind,
            inputs: node.inputs.clone(),
            nchw: [
                node.output.batch,
                node.output.channels,
                1,
                node.output.frames,
            ],
            checkpoint_keys: node.checkpoint_keys.clone(),
        }))
    }

    pub fn validate_affine_bytes(&self, slices: &[&[u8]]) -> Result<()> {
        let expected = self.nchw[1]
            .checked_mul(4)
            .ok_or(FishError::InvalidShape("affine byte length overflow"))?;
        if slices.len() != self.checkpoint_keys.len() || slices.iter().any(|s| s.len() != expected)
        {
            return Err(FishError::InvalidShape("affine weight bytes mismatch"));
        }
        Ok(())
    }
}

pub fn plan_pointwise(graph: &FireflyGraph) -> Result<Vec<PointwisePlan>> {
    graph.validate()?;
    graph
        .nodes
        .iter()
        .enumerate()
        .filter_map(|(id, _)| match PointwisePlan::from_node(graph, id) {
            Ok(Some(plan)) => Some(Ok(plan)),
            Ok(None) => None,
            Err(e) => Some(Err(e)),
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn convnext_norm_and_channel_scale_shapes() {
        let graph = FireflyGraph::reference_encode(2, 12_345).unwrap();
        let plans = plan_pointwise(&graph).unwrap();
        let norm = plans
            .iter()
            .find(|p| p.name == "backbone.stages.0.0.norm")
            .unwrap();
        assert_eq!(norm.kind, PointwiseKind::ChannelLayerNorm);
        assert_eq!(norm.nchw[0..3], [2, 128, 1]);
        assert_eq!(norm.checkpoint_keys.len(), 2);
        assert!(norm
            .validate_affine_bytes(&[&vec![0u8; 512], &vec![0u8; 512]])
            .is_ok());
        assert!(norm
            .validate_affine_bytes(&[&vec![0u8; 12], &vec![0u8; 512]])
            .is_err());
        let scale = plans
            .iter()
            .find(|p| p.name == "backbone.stages.0.0.layer_scale")
            .unwrap();
        assert_eq!(scale.kind, PointwiseKind::LayerScale);
        assert_eq!(scale.checkpoint_keys, ["backbone.stages.0.0.gamma"]);
        assert!(plans.iter().any(|p| p.kind == PointwiseKind::GeluErf));
        assert!(plans.iter().any(|p| p.kind == PointwiseKind::Add));
    }
    #[test]
    fn hifigan_parallel_mean_has_three_inputs() {
        let graph = FireflyGraph::waveform_decode(1, 7, 6).unwrap();
        let plans = plan_pointwise(&graph).unwrap();
        let mean = plans
            .iter()
            .find(|p| p.name == "head.resblocks.0.mean")
            .unwrap();
        assert_eq!(mean.kind, PointwiseKind::ParallelMean);
        assert_eq!(mean.inputs.len(), 3);
        assert!(mean
            .inputs
            .iter()
            .all(|&i| graph.nodes[i].output == graph.nodes[mean.node_id].output));
        assert!(plans.iter().any(|p| p.kind == PointwiseKind::SiLu));
        assert!(plans.iter().any(|p| p.kind == PointwiseKind::Tanh));
    }
}

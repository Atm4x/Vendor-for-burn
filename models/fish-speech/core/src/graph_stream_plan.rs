//! Backend-neutral Graph IR execution phases. No TensorRT blueprint is required.

use crate::{
    error::{FishError, Result},
    graph_ir::{Graph, Op, Origin, ValueId},
};

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum GraphExecutionPhase {
    Compute(Vec<usize>),
    KvWrites(Vec<usize>),
}

impl GraphExecutionPhase {
    pub fn node_indices(&self) -> &[usize] {
        match self {
            Self::Compute(indices) | Self::KvWrites(indices) => indices,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct GraphKvWrite {
    pub source_cache: ValueId,
    pub updated_cache: ValueId,
    pub position: usize,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct GraphExecutionPlan {
    phases: Vec<GraphExecutionPhase>,
    kv_writes: Vec<GraphKvWrite>,
}

impl GraphExecutionPlan {
    pub fn from_graph(graph: &Graph) -> Result<Self> {
        graph.validate()?;
        let mut phases = Vec::new();
        let mut compute = Vec::new();
        let mut writes = Vec::new();
        let mut kv_writes = Vec::new();
        for (index, node) in graph.nodes.iter().enumerate() {
            if let Op::KvUpdate { position } = node.op {
                if !compute.is_empty() {
                    phases.push(GraphExecutionPhase::Compute(std::mem::take(&mut compute)));
                }
                let source = node.inputs[0];
                if !matches!(graph.values[source.0].origin, Origin::Input)
                    || graph.values[source.0].ty != graph.values[node.result.0].ty
                {
                    return Err(FishError::InvalidShape(
                        "Graph IR KV update must preserve source cache geometry",
                    ));
                }
                kv_writes.push(GraphKvWrite {
                    source_cache: source,
                    updated_cache: node.result,
                    position,
                });
                writes.push(index);
            } else {
                if !writes.is_empty() {
                    phases.push(GraphExecutionPhase::KvWrites(std::mem::take(&mut writes)));
                }
                compute.push(index);
            }
        }
        if !writes.is_empty() {
            phases.push(GraphExecutionPhase::KvWrites(writes));
        }
        if !compute.is_empty() {
            phases.push(GraphExecutionPhase::Compute(compute));
        }
        if phases.is_empty() {
            return Err(FishError::InvalidShape(
                "Graph IR execution plan has no nodes",
            ));
        }
        Ok(Self { phases, kv_writes })
    }

    pub fn phases(&self) -> &[GraphExecutionPhase] {
        &self.phases
    }

    pub fn kv_writes(&self) -> &[GraphKvWrite] {
        &self.kv_writes
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{config::FishModelConfig, transformer_ir::TransformerIrFactory};

    #[test]
    fn transformer_profiles_have_ordered_backend_neutral_phases() {
        let config = FishModelConfig::synthetic_dual_ar_fixture();
        let f = TransformerIrFactory::new(&config).unwrap();
        for graph in [
            f.slow_prefill(3).unwrap(),
            f.slow_decode(4).unwrap(),
            f.fast_decode(0).unwrap(),
        ] {
            let plan = GraphExecutionPlan::from_graph(&graph).unwrap();
            let indices = plan
                .phases()
                .iter()
                .flat_map(|phase| phase.node_indices().iter().copied())
                .collect::<Vec<_>>();
            assert_eq!(indices, (0..graph.nodes.len()).collect::<Vec<_>>());
            assert_eq!(
                plan.kv_writes().len(),
                graph
                    .nodes
                    .iter()
                    .filter(|n| matches!(n.op, Op::KvUpdate { .. }))
                    .count()
            );
            assert!(plan
                .phases()
                .iter()
                .any(|p| matches!(p, GraphExecutionPhase::KvWrites(_))));
        }
    }

    #[test]
    fn kv_writes_retain_identity_and_source_position() {
        let config = FishModelConfig::synthetic_dual_ar_fixture();
        let graph = TransformerIrFactory::new(&config)
            .unwrap()
            .slow_decode(4)
            .unwrap();
        let plan = GraphExecutionPlan::from_graph(&graph).unwrap();
        for write in plan.kv_writes() {
            assert_eq!(write.position, 3);
            assert!(graph.values[write.source_cache.0].name.ends_with("_cache"));
            assert!(graph.outputs.contains(&write.updated_cache));
            assert_eq!(
                graph.values[write.source_cache.0].ty,
                graph.values[write.updated_cache.0].ty
            );
        }
    }
}

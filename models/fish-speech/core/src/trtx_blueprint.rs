use std::collections::{BTreeMap, BTreeSet};

fn f32_to_bf16(x: f32) -> u16 {
    let bits = x.to_bits();
    if x.is_nan() {
        return ((bits >> 16) as u16) | 0x0040;
    }
    ((bits.wrapping_add(0x7fff + ((bits >> 16) & 1))) >> 16) as u16
}

use crate::ar_attention::{causal_gqa_cpu, GqaShape};
use crate::ar_rotary::rotate_bhtd;
use crate::cpu_ir::{NamedTensors, TensorData};
use crate::error::{FishError, Result};
use crate::graph_ir::{Graph, Op, Origin, QkvPart, TensorType, ValueId};

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum Binding {
    Value(ValueId),
    Temp(usize),
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum TrtPrimitive {
    Input,
    ConstantWeight,
    MatrixMultiplyTransposeRhs,
    AddBroadcastBias,
    Sigmoid,
    ElementwiseProduct,
    ElementwiseSum,
    Identity,
    CastToBf16,
    RmsNorm {
        epsilon_bits: u32,
    },
    SliceLastSequenceToken,
    EmbeddingGather,
    FishEmbedding {
        num_codebooks: usize,
        codebook_size: usize,
        share_codebook_embeddings: bool,
    },
    SplitPackedQkv {
        part: QkvPart,
        q_heads: usize,
        kv_heads: usize,
        head_dim: usize,
    },
    RotaryInterleaved {
        position: usize,
        base_bits: u32,
    },
    KvActivePrefix {
        active_len: usize,
    },
    GqaAttention {
        position: usize,
        q_heads: usize,
        kv_heads: usize,
    },
    Deferred {
        original: Op,
        reason: &'static str,
    },
}

#[derive(Debug, Clone)]
pub struct TrtStep {
    pub name: String,
    pub primitive: TrtPrimitive,
    pub arguments: Vec<Binding>,
    pub result: Binding,
    pub output_type: TensorType,
}

#[derive(Debug, Clone)]
pub struct TrtBlueprint {
    pub steps: Vec<TrtStep>,
    pub outputs: Vec<Binding>,
    pub deferred: BTreeMap<&'static str, usize>,
    pub weight_count: usize,
}

impl TrtBlueprint {
    pub fn from_graph(graph: &Graph) -> Result<Self> {
        graph.validate()?;
        let mut steps = Vec::new();
        let mut temporary = 0;
        let mut deferred = BTreeMap::new();
        let mut weight_count = 0;
        for (idx, value) in graph.values.iter().enumerate() {
            let primitive = match value.origin {
                Origin::Input => TrtPrimitive::Input,
                Origin::Weight { .. } => {
                    weight_count += 1;
                    TrtPrimitive::ConstantWeight
                }
                Origin::Node(_) => continue,
            };
            steps.push(TrtStep {
                name: value.name.clone(),
                primitive,
                arguments: Vec::new(),
                result: Binding::Value(ValueId(idx)),
                output_type: value.ty.clone(),
            });
        }
        for node in &graph.nodes {
            let output_type = graph.tensor(node.result)?.clone();
            let args = node
                .inputs
                .iter()
                .copied()
                .map(Binding::Value)
                .collect::<Vec<_>>();
            let result = Binding::Value(node.result);
            let mut push = |name: String,
                            primitive: TrtPrimitive,
                            arguments: Vec<Binding>,
                            result: Binding,
                            ty: TensorType| {
                steps.push(TrtStep {
                    name,
                    primitive,
                    arguments,
                    result,
                    output_type: ty,
                });
            };
            match node.op {
                Op::Linear | Op::LinearBias => {
                    let linear_output = if matches!(node.op, Op::LinearBias) {
                        let id = Binding::Temp(temporary);
                        temporary += 1;
                        id
                    } else {
                        result
                    };
                    push(
                        format!("{}.gemm", node.name),
                        TrtPrimitive::MatrixMultiplyTransposeRhs,
                        args[..2].to_vec(),
                        linear_output,
                        output_type.clone(),
                    );
                    if matches!(node.op, Op::LinearBias) {
                        push(
                            node.name.clone(),
                            TrtPrimitive::AddBroadcastBias,
                            vec![linear_output, args[2]],
                            result,
                            output_type,
                        );
                    }
                }
                Op::SiLu => {
                    let sigmoid = Binding::Temp(temporary);
                    temporary += 1;
                    push(
                        format!("{}.sigmoid", node.name),
                        TrtPrimitive::Sigmoid,
                        args.clone(),
                        sigmoid,
                        output_type.clone(),
                    );
                    push(
                        node.name.clone(),
                        TrtPrimitive::ElementwiseProduct,
                        vec![args[0], sigmoid],
                        result,
                        output_type,
                    );
                }
                Op::FishEmbedding {
                    num_codebooks,
                    codebook_size,
                    share_codebook_embeddings,
                } => {
                    push(
                        node.name.clone(),
                        TrtPrimitive::FishEmbedding {
                            num_codebooks,
                            codebook_size,
                            share_codebook_embeddings,
                        },
                        args,
                        result,
                        output_type,
                    );
                }
                Op::TokenEmbedding => {
                    push(
                        node.name.clone(),
                        TrtPrimitive::EmbeddingGather,
                        args,
                        result,
                        output_type,
                    );
                }
                Op::SplitQkv {
                    part,
                    q_heads,
                    kv_heads,
                    head_dim,
                } => {
                    push(
                        node.name.clone(),
                        TrtPrimitive::SplitPackedQkv {
                            part,
                            q_heads,
                            kv_heads,
                            head_dim,
                        },
                        args,
                        result,
                        output_type,
                    );
                }
                Op::RoPe {
                    position,
                    base_bits,
                } => {
                    push(
                        node.name.clone(),
                        TrtPrimitive::RotaryInterleaved {
                            position,
                            base_bits,
                        },
                        args,
                        result,
                        output_type,
                    );
                }
                Op::GqaAttention {
                    position,
                    q_heads,
                    kv_heads,
                } => {
                    push(
                        node.name.clone(),
                        TrtPrimitive::GqaAttention {
                            position,
                            q_heads,
                            kv_heads,
                        },
                        args,
                        result,
                        output_type,
                    );
                }
                Op::KvPrefix { active_len } => {
                    push(
                        node.name.clone(),
                        TrtPrimitive::KvActivePrefix { active_len },
                        args,
                        result,
                        output_type,
                    );
                }
                Op::RmsNorm { epsilon_bits } => {
                    push(
                        node.name.clone(),
                        TrtPrimitive::RmsNorm { epsilon_bits },
                        args,
                        result,
                        output_type,
                    );
                }
                Op::Mul | Op::Add | Op::Identity | Op::LastToken => {
                    let primitive = match node.op {
                        Op::Mul => TrtPrimitive::ElementwiseProduct,
                        Op::Add => TrtPrimitive::ElementwiseSum,
                        Op::Identity => TrtPrimitive::Identity,
                        Op::LastToken => TrtPrimitive::SliceLastSequenceToken,
                        _ => unreachable!(),
                    };
                    push(node.name.clone(), primitive, args, result, output_type);
                }
                other => {
                    let reason = match other {
                        Op::KvUpdate { .. } => "external mutable KV buffers and update",
                        _ => unreachable!(),
                    };
                    *deferred.entry(reason).or_insert(0) += 1;
                    push(
                        node.name.clone(),
                        TrtPrimitive::Deferred {
                            original: other,
                            reason,
                        },
                        args,
                        result,
                        output_type,
                    );
                }
            }
        }
        let plan = Self {
            steps,
            outputs: graph.outputs.iter().copied().map(Binding::Value).collect(),
            deferred,
            weight_count,
        };
        plan.validate()?;
        Ok(plan)
    }

    pub fn validate(&self) -> Result<()> {
        let mut produced = BTreeSet::new();
        for step in &self.steps {
            if step.name.is_empty() {
                return Err(FishError::InvalidShape("empty TRT blueprint step name"));
            }
            if step.output_type.shape.is_empty() || step.output_type.shape.contains(&0) {
                return Err(FishError::InvalidShape(
                    "invalid TRT blueprint output shape",
                ));
            }
            if step.arguments.iter().any(|arg| !produced.contains(arg)) {
                return Err(FishError::InvalidShape("TRT blueprint has dangling input"));
            }
            if !produced.insert(step.result) {
                return Err(FishError::InvalidShape(
                    "TRT blueprint has multiple producers",
                ));
            }
        }
        if self.outputs.is_empty() || self.outputs.iter().any(|o| !produced.contains(o)) {
            return Err(FishError::InvalidShape("TRT blueprint missing output"));
        }
        Ok(())
    }

    pub fn execute_cpu_reference(&self, bindings: &NamedTensors) -> Result<NamedTensors> {
        self.require_no_deferred_ops()?;
        self.validate()?;
        let mut intermediates: BTreeMap<Binding, (TensorType, TensorData)> = BTreeMap::new();
        for step in &self.steps {
            let data = match step.primitive {
                TrtPrimitive::Input | TrtPrimitive::ConstantWeight => bindings
                    .get(&step.name)
                    .ok_or(FishError::InvalidShape("missing blueprint CPU input"))?
                    .clone(),
                _ => {
                    let args = step
                        .arguments
                        .iter()
                        .map(|id| {
                            intermediates
                                .get(id)
                                .ok_or(FishError::InvalidShape("blueprint CPU missing parent"))
                        })
                        .collect::<Result<Vec<_>>>()?;
                    let read = |index: usize| -> Result<&[f32]> {
                        match &args[index].1 {
                            TensorData::F32(data) => Ok(data),
                            _ => Err(FishError::InvalidShape("blueprint CPU expects FP32 data")),
                        }
                    };
                    let values = match step.primitive {
                        TrtPrimitive::FishEmbedding {
                            num_codebooks,
                            codebook_size,
                            share_codebook_embeddings,
                        } => {
                            let TensorData::I32(tokens) = &args[0].1 else {
                                return Err(FishError::InvalidShape(
                                    "FishEmbedding tokens must be I32",
                                ));
                            };
                            let TensorData::I32(lut) = &args[3].1 else {
                                return Err(FishError::InvalidShape(
                                    "FishEmbedding gate LUT must be I32",
                                ));
                            };
                            crate::ar_embedding::fish_embedding_cpu(
                                tokens,
                                args[0].0.shape[0],
                                args[0].0.shape[2],
                                num_codebooks,
                                codebook_size,
                                read(1)?,
                                read(2)?,
                                args[1].0.shape[1],
                                lut,
                                share_codebook_embeddings,
                            )?
                        }
                        TrtPrimitive::EmbeddingGather => {
                            let TensorData::I32(ids) = &args[0].1 else {
                                return Err(FishError::InvalidShape(
                                    "embedding indices must be I32",
                                ));
                            };
                            let weights = read(1)?;
                            let vocab = args[1].0.shape[0];
                            let width = args[1].0.shape[1];
                            let mut gathered = Vec::with_capacity(ids.len() * width);
                            for &id in ids {
                                let index = usize::try_from(id).map_err(|_| {
                                    FishError::InvalidShape("negative embedding index")
                                })?;
                                if index >= vocab {
                                    return Err(FishError::InvalidShape(
                                        "embedding index outside vocab",
                                    ));
                                }
                                gathered.extend_from_slice(
                                    &weights[index * width..(index + 1) * width],
                                );
                            }
                            gathered
                        }
                        TrtPrimitive::SplitPackedQkv {
                            part,
                            q_heads,
                            kv_heads,
                            head_dim,
                        } => {
                            let packed = read(0)?;
                            let (batch, frames, total) =
                                (args[0].0.shape[0], args[0].0.shape[1], args[0].0.shape[2]);
                            let (heads, offset) = match part {
                                QkvPart::Query => (q_heads, 0),
                                QkvPart::Key => (kv_heads, q_heads * head_dim),
                                QkvPart::Value => (kv_heads, (q_heads + kv_heads) * head_dim),
                            };
                            let mut out = vec![0f32; batch * heads * frames * head_dim];
                            for batch_id in 0..batch {
                                for head in 0..heads {
                                    for frame in 0..frames {
                                        let source = (batch_id * frames + frame) * total
                                            + offset
                                            + head * head_dim;
                                        let destination =
                                            ((batch_id * heads + head) * frames + frame) * head_dim;
                                        out[destination..destination + head_dim]
                                            .copy_from_slice(&packed[source..source + head_dim]);
                                    }
                                }
                            }
                            out
                        }
                        TrtPrimitive::GqaAttention {
                            position,
                            q_heads,
                            kv_heads,
                        } => {
                            let q_shape = &args[0].0.shape;
                            let kv_shape = &args[1].0.shape;
                            causal_gqa_cpu(
                                read(0)?,
                                read(1)?,
                                read(2)?,
                                GqaShape {
                                    batch: q_shape[0],
                                    query_heads: q_heads,
                                    kv_heads,
                                    query_len: q_shape[2],
                                    kv_len: kv_shape[2],
                                    head_dim: q_shape[3],
                                    first_position: position,
                                },
                            )?
                        }
                        TrtPrimitive::KvActivePrefix { active_len } => {
                            let input = read(0)?;
                            let shape = &args[0].0.shape;
                            let (batch, heads, capacity, dim) =
                                (shape[0], shape[1], shape[2], shape[3]);
                            let mut output = vec![0f32; batch * heads * active_len * dim];
                            for row in 0..batch * heads {
                                let source = row * capacity * dim;
                                let target = row * active_len * dim;
                                output[target..target + active_len * dim]
                                    .copy_from_slice(&input[source..source + active_len * dim]);
                            }
                            output
                        }
                        TrtPrimitive::RotaryInterleaved {
                            position,
                            base_bits,
                        } => {
                            let shape = &args[0].0.shape;
                            rotate_bhtd(
                                read(0)?,
                                shape[0],
                                shape[1],
                                position,
                                shape[2],
                                shape[3],
                                f32::from_bits(base_bits),
                            )?
                        }
                        TrtPrimitive::MatrixMultiplyTransposeRhs => {
                            let a = read(0)?;
                            let w = read(1)?;
                            let input_dim = args[1].0.shape[1];
                            let output_dim = args[1].0.shape[0];
                            let mut data = Vec::with_capacity((a.len() / input_dim) * output_dim);
                            for row in a.chunks_exact(input_dim) {
                                for weight in w.chunks_exact(input_dim) {
                                    data.push(
                                        row.iter().zip(weight).map(|(x, y)| x * y).sum::<f32>(),
                                    );
                                }
                            }
                            data
                        }
                        TrtPrimitive::AddBroadcastBias => {
                            let a = read(0)?;
                            let bias = read(1)?;
                            if bias.len()
                                != *step
                                    .output_type
                                    .shape
                                    .last()
                                    .ok_or(FishError::InvalidShape("missing output dimension"))?
                            {
                                return Err(FishError::InvalidShape("bias broadcast mismatch"));
                            }
                            a.iter()
                                .enumerate()
                                .map(|(i, x)| x + bias[i % bias.len()])
                                .collect()
                        }
                        TrtPrimitive::Sigmoid => {
                            read(0)?.iter().map(|x| 1.0 / (1.0 + (-x).exp())).collect()
                        }
                        TrtPrimitive::ElementwiseProduct => {
                            read(0)?.iter().zip(read(1)?).map(|(a, b)| a * b).collect()
                        }
                        TrtPrimitive::ElementwiseSum => {
                            read(0)?.iter().zip(read(1)?).map(|(a, b)| a + b).collect()
                        }
                        TrtPrimitive::RmsNorm { epsilon_bits } => {
                            let x = read(0)?;
                            let gamma = read(1)?;
                            let width = gamma.len();
                            let mut out = Vec::with_capacity(x.len());
                            for row in x.chunks_exact(width) {
                                let mean = row.iter().map(|v| v * v).sum::<f32>() / width as f32;
                                let inv = 1.0 / (mean + f32::from_bits(epsilon_bits)).sqrt();
                                out.extend(row.iter().zip(gamma).map(|(v, w)| v * inv * w));
                            }
                            out
                        }
                        TrtPrimitive::CastToBf16 => read(0)?
                            .iter()
                            .map(|&x| f32::from_bits((f32_to_bf16(x) as u32) << 16))
                            .collect(),
                        TrtPrimitive::Identity => read(0)?.to_vec(),
                        TrtPrimitive::SliceLastSequenceToken => {
                            let a = read(0)?;
                            let ty = &args[0].0;
                            let (batch, frames, dim) = (ty.shape[0], ty.shape[1], ty.shape[2]);
                            let mut data = Vec::with_capacity(batch * dim);
                            for b in 0..batch {
                                let start = (b * frames + frames - 1) * dim;
                                data.extend_from_slice(&a[start..start + dim]);
                            }
                            data
                        }
                        _ => return Err(FishError::Unsupported("unexpected unlowered operation")),
                    };
                    TensorData::F32(values)
                }
            };
            let expected: usize = step.output_type.shape.iter().product();
            let valid = match (&data, step.output_type.dtype) {
                (
                    TensorData::F32(v),
                    crate::graph_ir::ScalarType::F32 | crate::graph_ir::ScalarType::BF16,
                ) => v.len() == expected,
                (TensorData::I32(v), crate::graph_ir::ScalarType::I32) => v.len() == expected,
                _ => false,
            };
            if !valid {
                return Err(FishError::InvalidShape(
                    "blueprint CPU result shape/dtype mismatch",
                ));
            }
            intermediates.insert(step.result, (step.output_type.clone(), data));
        }
        let mut outputs = NamedTensors::new();
        for id in &self.outputs {
            let step = self
                .steps
                .iter()
                .find(|step| step.result == *id)
                .ok_or(FishError::InvalidShape("output producer missing"))?;
            outputs.insert(step.name.clone(), intermediates[id].1.clone());
        }
        Ok(outputs)
    }

    pub fn require_no_deferred_ops(&self) -> Result<()> {
        if self.deferred.is_empty() {
            Ok(())
        } else {
            Err(FishError::Unsupported(
                "TRT blueprint contains non-lowered graph operations",
            ))
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::FishModelConfig;
    use crate::graph::Stage;
    use crate::graph_ir::{Layout, ScalarType};
    use crate::transformer_ir::TransformerIrFactory;

    #[test]
    fn direct_primitives_are_wired_and_decomposed() {
        let mut graph = Graph::new(Stage::SlowDecode, None, None);
        let x = graph
            .input(
                "x",
                TensorType::new([1, 2, 3], ScalarType::F32, Layout::SequenceBtd),
            )
            .unwrap();
        let w = graph
            .weight("w", TensorType::new([4, 3], ScalarType::F32, Layout::Plain))
            .unwrap();
        let bias = graph
            .weight("bias", TensorType::new([4], ScalarType::F32, Layout::Plain))
            .unwrap();
        let y = graph
            .emit("project", Op::LinearBias, &[x, w, bias])
            .unwrap();
        let z = graph.emit("silu", Op::SiLu, &[y]).unwrap();
        graph.output(z).unwrap();
        let plan = TrtBlueprint::from_graph(&graph).unwrap();
        plan.require_no_deferred_ops().unwrap();
        assert_eq!(plan.steps.len(), 7);
        assert_eq!(plan.weight_count, 2);
        assert_eq!(
            plan.steps[3].primitive,
            TrtPrimitive::MatrixMultiplyTransposeRhs
        );
        assert_eq!(plan.steps[4].primitive, TrtPrimitive::AddBroadcastBias);
        assert_eq!(plan.steps[5].primitive, TrtPrimitive::Sigmoid);
        assert_eq!(plan.steps[6].primitive, TrtPrimitive::ElementwiseProduct);
    }

    #[test]
    fn rms_norm_matches_pytorch_and_is_not_deferred() {
        let mut graph = Graph::new(Stage::SlowDecode, None, None);
        let input = graph
            .input(
                "x",
                TensorType::new([1, 2, 4], ScalarType::F32, Layout::SequenceBtd),
            )
            .unwrap();
        let weight = graph
            .weight(
                "gamma",
                TensorType::new([4], ScalarType::F32, Layout::Plain),
            )
            .unwrap();
        let y = graph
            .emit(
                "norm",
                Op::RmsNorm {
                    epsilon_bits: 1e-5f32.to_bits(),
                },
                &[input, weight],
            )
            .unwrap();
        graph.output(y).unwrap();
        let plan = TrtBlueprint::from_graph(&graph).unwrap();
        plan.require_no_deferred_ops().unwrap();
        assert!(matches!(
            plan.steps[2].primitive,
            TrtPrimitive::RmsNorm { .. }
        ));
        let mut bindings = NamedTensors::new();
        bindings.insert(
            "x".into(),
            TensorData::F32(vec![1.25, -0.75, 2.5, -4.0, 0.002, 0.0, -0.003, 0.005]),
        );
        bindings.insert("gamma".into(), TensorData::F32(vec![0.25, -1.5, 0.8, 2.25]));
        let output = plan.execute_cpu_reference(&bindings).unwrap();
        let TensorData::F32(got) = &output["norm"] else {
            panic!("expected FP32 RMSNorm")
        };
        let golden = include_str!("../fixtures/generate_rmsnorm_golden.txt")
            .trim()
            .split(',')
            .map(|v| v.parse::<f32>().unwrap())
            .collect::<Vec<_>>();
        assert_eq!(got.len(), golden.len());
        for (actual, expected) in got.iter().zip(&golden) {
            assert!(
                (actual - expected).abs() <= 3e-6,
                "RMSNorm mismatch: {actual} != {expected}"
            );
        }
        assert!(TrtBlueprint::from_graph(&graph)
            .unwrap()
            .deferred
            .is_empty());
    }

    #[test]
    fn full_ar_graph_reports_exact_deferred_blockers() {
        let config = FishModelConfig::synthetic_dual_ar_fixture();
        let factory = TransformerIrFactory::new(&config).unwrap();
        for graph in [
            factory.slow_prefill(4).unwrap(),
            factory.slow_decode(6).unwrap(),
            factory.fast_decode(0).unwrap(),
            factory.fast_decode(2).unwrap(),
        ] {
            let plan = TrtBlueprint::from_graph(&graph).unwrap();
            plan.validate().unwrap();
            assert!(!plan.deferred.is_empty());
            assert!(!plan
                .deferred
                .contains_key("RMSNorm reduction and epsilon semantics"));
            assert!(plan.require_no_deferred_ops().is_err());
            assert_eq!(plan.outputs.len(), graph.outputs.len());
            assert_eq!(plan.weight_count, graph.weight_requirements().len());
        }
    }

    #[test]
    fn internal_temporary_must_be_produced_before_use() {
        let config = FishModelConfig::synthetic_dual_ar_fixture();
        let g = TransformerIrFactory::new(&config)
            .unwrap()
            .slow_decode(2)
            .unwrap();
        let mut plan = TrtBlueprint::from_graph(&g).unwrap();
        let idx = plan
            .steps
            .iter()
            .position(|step| matches!(step.primitive, TrtPrimitive::MatrixMultiplyTransposeRhs))
            .unwrap();
        plan.steps[idx].arguments[0] = Binding::Temp(usize::MAX);
        assert!(plan.validate().is_err());
    }

    #[test]
    fn primitive_decomposition_matches_original_cpu_ir() {
        use crate::cpu_ir::CpuIrInterpreter;
        let mut g = Graph::new(Stage::SlowDecode, None, None);
        let x = g
            .input(
                "x",
                TensorType::new([1, 2, 3], ScalarType::F32, Layout::SequenceBtd),
            )
            .unwrap();
        let w = g
            .weight("w", TensorType::new([4, 3], ScalarType::F32, Layout::Plain))
            .unwrap();
        let b = g
            .weight("b", TensorType::new([4], ScalarType::F32, Layout::Plain))
            .unwrap();
        let y = g.emit("linear", Op::LinearBias, &[x, w, b]).unwrap();
        let y = g.emit("silu", Op::SiLu, &[y]).unwrap();
        let z = g.emit("last", Op::LastToken, &[y]).unwrap();
        g.output(z).unwrap();
        let mut bind = NamedTensors::new();
        bind.insert("x".into(), TensorData::F32(vec![1., -2., 3., 4., -5., 6.]));
        bind.insert(
            "w".into(),
            TensorData::F32(vec![1., 2., 3., -1., 0., 1., 0.5, 0.25, -1., 0., 2., 1.]),
        );
        bind.insert("b".into(), TensorData::F32(vec![1., -2., 0., 4.]));
        let expect = CpuIrInterpreter::run(&g, &bind).unwrap();
        let got = TrtBlueprint::from_graph(&g)
            .unwrap()
            .execute_cpu_reference(&bind)
            .unwrap();
        assert_eq!(got, expect);
    }

    #[test]
    fn blueprint_cpu_reference_rejects_bad_bindings_and_deferred_ops() {
        let mut g = Graph::new(Stage::SlowDecode, None, None);
        let x = g
            .input(
                "x",
                TensorType::new([1, 1, 2], ScalarType::F32, Layout::SequenceBtd),
            )
            .unwrap();
        let y = g.emit("identity", Op::Identity, &[x]).unwrap();
        g.output(y).unwrap();
        let plan = TrtBlueprint::from_graph(&g).unwrap();
        assert!(plan.execute_cpu_reference(&NamedTensors::new()).is_err());
        let mut b = NamedTensors::new();
        b.insert("x".into(), TensorData::F32(vec![1., 2.]));
        assert_eq!(
            plan.execute_cpu_reference(&b).unwrap()["identity"],
            TensorData::F32(vec![1., 2.])
        );
        b.insert("x".into(), TensorData::F32(vec![3.]));
        assert!(plan.execute_cpu_reference(&b).is_err());
        let c = FishModelConfig::synthetic_dual_ar_fixture();
        let graph = TransformerIrFactory::new(&c)
            .unwrap()
            .slow_decode(1)
            .unwrap();
        assert!(TrtBlueprint::from_graph(&graph)
            .unwrap()
            .execute_cpu_reference(&NamedTensors::new())
            .is_err());
    }

    #[test]
    fn strict_gate_never_calls_deferred_plan_gpu_ready() {
        let config = FishModelConfig::synthetic_dual_ar_fixture();
        let factory = TransformerIrFactory::new(&config).unwrap();
        let first = TrtBlueprint::from_graph(&factory.fast_decode(0).unwrap()).unwrap();
        let later = TrtBlueprint::from_graph(&factory.fast_decode(1).unwrap()).unwrap();
        assert!(first
            .steps
            .iter()
            .any(|step| matches!(step.primitive, TrtPrimitive::GqaAttention { .. })));
        assert!(first.deferred.keys().any(|reason| reason.contains("KV")));
        assert!(!later
            .deferred
            .keys()
            .any(|reason| reason.contains("embedding gather")));
        assert!(later
            .steps
            .iter()
            .any(|step| step.primitive == TrtPrimitive::EmbeddingGather));
        assert!(first.require_no_deferred_ops().is_err());
        assert!(later.require_no_deferred_ops().is_err());
    }
}

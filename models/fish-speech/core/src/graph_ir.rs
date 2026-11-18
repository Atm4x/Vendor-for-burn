use std::collections::{BTreeMap, HashSet};

use crate::error::{FishError, Result};
use crate::graph::Stage;

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct ValueId(pub usize);

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ScalarType {
    I32,
    F32,
    BF16,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Layout {
    TokensBct,
    SequenceBtd,
    HeadsBhtd,
    CacheBhtd,
    Plain,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TensorType {
    pub shape: Vec<usize>,
    pub dtype: ScalarType,
    pub layout: Layout,
}

impl TensorType {
    pub fn new(shape: impl Into<Vec<usize>>, dtype: ScalarType, layout: Layout) -> Self {
        Self {
            shape: shape.into(),
            dtype,
            layout,
        }
    }

    fn valid(&self) -> bool {
        !self.shape.is_empty()
            && self.shape.iter().all(|&x| x > 0)
            && self
                .shape
                .iter()
                .try_fold(1usize, |p, &x| p.checked_mul(x))
                .is_some()
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Origin {
    Input,
    Weight { checkpoint_key: String },
    Node(usize),
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Value {
    pub name: String,
    pub ty: TensorType,
    pub origin: Origin,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum QkvPart {
    Query,
    Key,
    Value,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Op {
    FishEmbedding {
        num_codebooks: usize,
        codebook_size: usize,
        share_codebook_embeddings: bool,
    },
    RmsNorm {
        epsilon_bits: u32,
    },
    Linear,
    LinearBias,
    TokenEmbedding,
    Identity,
    SplitQkv {
        part: QkvPart,
        q_heads: usize,
        kv_heads: usize,
        head_dim: usize,
    },
    RoPe {
        position: usize,
        base_bits: u32,
    },
    KvUpdate {
        position: usize,
    },
    KvPrefix {
        active_len: usize,
    },
    GqaAttention {
        position: usize,
        q_heads: usize,
        kv_heads: usize,
    },
    SiLu,
    Mul,
    Add,
    LastToken,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Node {
    pub name: String,
    pub op: Op,
    pub inputs: Vec<ValueId>,
    pub result: ValueId,
}

#[derive(Debug, Clone)]
pub struct Graph {
    pub stage: Stage,
    pub values: Vec<Value>,
    pub nodes: Vec<Node>,
    pub outputs: Vec<ValueId>,
    pub active_kv_len: Option<usize>,
    pub kv_capacity: Option<usize>,
}

fn expect(condition: bool, message: &'static str) -> Result<()> {
    if condition {
        Ok(())
    } else {
        Err(FishError::InvalidShape(message))
    }
}

fn check_type(ty: &TensorType, rank: usize, dtype: ScalarType, layout: Layout) -> Result<()> {
    expect(
        ty.shape.len() == rank && ty.dtype == dtype && ty.layout == layout,
        "operand rank, dtype or layout mismatch",
    )
}

impl Graph {
    pub fn new(stage: Stage, active_kv_len: Option<usize>, kv_capacity: Option<usize>) -> Self {
        Self {
            stage,
            values: Vec::new(),
            nodes: Vec::new(),
            outputs: Vec::new(),
            active_kv_len,
            kv_capacity,
        }
    }

    pub fn input(&mut self, name: impl Into<String>, ty: TensorType) -> Result<ValueId> {
        self.declare(name.into(), ty, Origin::Input)
    }

    pub fn weight(&mut self, key: impl Into<String>, ty: TensorType) -> Result<ValueId> {
        let key = key.into();
        if let Some((id, value)) = self.values.iter().enumerate().find(|(_, v)| v.name == key) {
            expect(
                value.ty == ty
                    && value.origin
                        == (Origin::Weight {
                            checkpoint_key: key,
                        }),
                "conflicting checkpoint weight specification",
            )?;
            return Ok(ValueId(id));
        }
        self.declare(
            key.clone(),
            ty,
            Origin::Weight {
                checkpoint_key: key,
            },
        )
    }

    fn declare(&mut self, name: String, ty: TensorType, origin: Origin) -> Result<ValueId> {
        expect(
            ty.valid(),
            "invalid tensor extent or element count overflow",
        )?;
        expect(
            !name.is_empty() && !self.values.iter().any(|v| v.name == name),
            "duplicate or empty value name",
        )?;
        let id = ValueId(self.values.len());
        self.values.push(Value { name, ty, origin });
        Ok(id)
    }

    pub fn emit(&mut self, name: impl Into<String>, op: Op, inputs: &[ValueId]) -> Result<ValueId> {
        let inferred = op.infer(inputs, &self.values)?;
        let node_idx = self.nodes.len();
        let name = name.into();
        let result = self.declare(name.clone(), inferred, Origin::Node(node_idx))?;
        self.nodes.push(Node {
            name,
            op,
            inputs: inputs.to_vec(),
            result,
        });
        Ok(result)
    }

    pub fn output(&mut self, value: ValueId) -> Result<()> {
        expect(self.values.get(value.0).is_some(), "unbound output value")?;
        expect(!self.outputs.contains(&value), "duplicate graph output")?;
        self.outputs.push(value);
        Ok(())
    }

    pub fn tensor(&self, id: ValueId) -> Result<&TensorType> {
        self.values
            .get(id.0)
            .map(|v| &v.ty)
            .ok_or(FishError::InvalidShape("unknown value ID"))
    }

    pub fn weight_requirements(&self) -> BTreeMap<String, TensorType> {
        self.values
            .iter()
            .filter_map(|v| match &v.origin {
                Origin::Weight { checkpoint_key } => Some((checkpoint_key.clone(), v.ty.clone())),
                _ => None,
            })
            .collect()
    }

    pub fn validate(&self) -> Result<()> {
        expect(
            !self.nodes.is_empty() && !self.outputs.is_empty(),
            "graph has no nodes or outputs",
        )?;
        match (self.active_kv_len, self.kv_capacity) {
            (Some(active), Some(capacity)) => expect(
                active >= 1 && active <= capacity,
                "invalid active-prefix bounds",
            )?,
            (None, None) => {}
            _ => return Err(FishError::InvalidShape("KV length/capacity mismatch")),
        }
        let mut names = HashSet::new();
        let mut produced = vec![false; self.values.len()];
        for (i, value) in self.values.iter().enumerate() {
            expect(
                value.ty.valid() && names.insert(&value.name),
                "invalid or duplicate value",
            )?;
            match &value.origin {
                Origin::Input => produced[i] = true,
                Origin::Weight { checkpoint_key } => {
                    expect(
                        checkpoint_key == &value.name,
                        "checkpoint key/name divergence",
                    )?;
                    produced[i] = true;
                }
                Origin::Node(idx) => expect(*idx < self.nodes.len(), "invalid node producer")?,
            }
        }
        for (idx, node) in self.nodes.iter().enumerate() {
            for input in &node.inputs {
                expect(
                    produced.get(input.0) == Some(&true),
                    "dangling or forward-referenced input",
                )?;
            }
            let inferred = node.op.infer(&node.inputs, &self.values)?;
            let value = self
                .values
                .get(node.result.0)
                .ok_or(FishError::InvalidShape("node output missing"))?;
            expect(
                value.ty == inferred
                    && value.name == node.name
                    && value.origin == Origin::Node(idx),
                "node output shape or producer mismatch",
            )?;
            expect(!produced[node.result.0], "value has multiple producers")?;
            produced[node.result.0] = true;
        }
        for &out in &self.outputs {
            expect(
                produced.get(out.0) == Some(&true),
                "unproduced graph output",
            )?;
        }
        Ok(())
    }
}

impl Op {
    fn infer(&self, ids: &[ValueId], values: &[Value]) -> Result<TensorType> {
        let args = ids
            .iter()
            .map(|id| {
                values
                    .get(id.0)
                    .map(|v| &v.ty)
                    .ok_or(FishError::InvalidShape("unknown input value ID"))
            })
            .collect::<Result<Vec<_>>>()?;
        let expected_arity = match self {
            Op::FishEmbedding { .. } => 4,
            Op::RmsNorm { .. }
            | Op::Linear
            | Op::TokenEmbedding
            | Op::KvUpdate { .. }
            | Op::Mul
            | Op::Add => 2,
            Op::GqaAttention { .. } | Op::LinearBias => 3,
            _ => 1,
        };
        expect(args.len() == expected_arity, "operator arity mismatch")?;
        let a = args[0];
        match *self {
            Op::FishEmbedding {
                num_codebooks,
                codebook_size,
                share_codebook_embeddings: _,
            } => {
                check_type(a, 3, ScalarType::I32, Layout::TokensBct)?;
                check_type(args[1], 2, ScalarType::F32, Layout::Plain)?;
                check_type(args[2], 2, ScalarType::F32, Layout::Plain)?;
                check_type(args[3], 1, ScalarType::I32, Layout::Plain)?;
                let dim = args[1].shape[1];
                expect(
                    a.shape[1] == num_codebooks + 1
                        && args[2].shape == [num_codebooks * codebook_size, dim]
                        && args[3].shape == [args[1].shape[0]],
                    "Fish token / embedding / semantic mask mismatch",
                )?;
                Ok(TensorType::new(
                    vec![a.shape[0], a.shape[2], dim],
                    ScalarType::F32,
                    Layout::SequenceBtd,
                ))
            }
            Op::RmsNorm { epsilon_bits } => {
                expect(
                    f32::from_bits(epsilon_bits).is_finite() && f32::from_bits(epsilon_bits) > 0.0,
                    "invalid RMSNorm epsilon",
                )?;
                check_type(a, 3, ScalarType::F32, Layout::SequenceBtd)?;
                check_type(args[1], 1, ScalarType::F32, Layout::Plain)?;
                expect(
                    args[1].shape == [a.shape[2]],
                    "RMSNorm gamma shape mismatch",
                )?;
                Ok(a.clone())
            }
            Op::TokenEmbedding => {
                check_type(a, 2, ScalarType::I32, Layout::Plain)?;
                check_type(args[1], 2, ScalarType::F32, Layout::Plain)?;
                Ok(TensorType::new(
                    vec![a.shape[0], a.shape[1], args[1].shape[1]],
                    ScalarType::F32,
                    Layout::SequenceBtd,
                ))
            }
            Op::Identity => Ok(a.clone()),
            Op::Linear | Op::LinearBias => {
                check_type(a, 3, ScalarType::F32, Layout::SequenceBtd)?;
                check_type(args[1], 2, ScalarType::F32, Layout::Plain)?;
                expect(
                    a.shape[2] == args[1].shape[1],
                    "linear input/weight dimension mismatch",
                )?;
                if matches!(self, Op::LinearBias) {
                    check_type(args[2], 1, ScalarType::F32, Layout::Plain)?;
                    expect(
                        args[2].shape == [args[1].shape[0]],
                        "linear bias dimension mismatch",
                    )?;
                }
                Ok(TensorType::new(
                    vec![a.shape[0], a.shape[1], args[1].shape[0]],
                    ScalarType::F32,
                    Layout::SequenceBtd,
                ))
            }
            Op::SplitQkv {
                part,
                q_heads,
                kv_heads,
                head_dim,
            } => {
                check_type(a, 3, ScalarType::F32, Layout::SequenceBtd)?;
                expect(
                    q_heads > 0
                        && kv_heads > 0
                        && q_heads % kv_heads == 0
                        && head_dim > 0
                        && a.shape[2] == (q_heads + 2 * kv_heads) * head_dim,
                    "invalid packed QKV and GQA dimensions",
                )?;
                let h = if part == QkvPart::Query {
                    q_heads
                } else {
                    kv_heads
                };
                Ok(TensorType::new(
                    vec![a.shape[0], h, a.shape[1], head_dim],
                    ScalarType::F32,
                    Layout::HeadsBhtd,
                ))
            }
            Op::RoPe {
                position,
                base_bits,
            } => {
                expect(
                    f32::from_bits(base_bits).is_finite() && f32::from_bits(base_bits) > 0.0,
                    "invalid RoPE base",
                )?;
                check_type(a, 4, ScalarType::F32, Layout::HeadsBhtd)?;
                expect(
                    a.shape[3] % 2 == 0 && position.checked_add(a.shape[2]).is_some(),
                    "RoPE requires even head dimension and valid position",
                )?;
                Ok(a.clone())
            }
            Op::KvUpdate { position } => {
                check_type(a, 4, ScalarType::BF16, Layout::CacheBhtd)?;
                check_type(args[1], 4, ScalarType::F32, Layout::HeadsBhtd)?;
                expect(
                    a.shape[0] == args[1].shape[0]
                        && a.shape[1] == args[1].shape[1]
                        && a.shape[3] == args[1].shape[3]
                        && position
                            .checked_add(args[1].shape[2])
                            .is_some_and(|n| n <= a.shape[2]),
                    "KV update exceeds capacity or has mismatched heads",
                )?;
                Ok(a.clone())
            }
            Op::KvPrefix { active_len } => {
                check_type(a, 4, ScalarType::BF16, Layout::CacheBhtd)?;
                expect(
                    active_len >= 1 && active_len <= a.shape[2],
                    "KV prefix exceeds physical cache",
                )?;
                Ok(TensorType::new(
                    vec![a.shape[0], a.shape[1], active_len, a.shape[3]],
                    ScalarType::BF16,
                    Layout::HeadsBhtd,
                ))
            }
            Op::GqaAttention {
                position,
                q_heads,
                kv_heads,
            } => {
                check_type(a, 4, ScalarType::F32, Layout::HeadsBhtd)?;
                check_type(args[1], 4, ScalarType::BF16, Layout::HeadsBhtd)?;
                check_type(args[2], 4, ScalarType::BF16, Layout::HeadsBhtd)?;
                expect(
                    q_heads > 0
                        && kv_heads > 0
                        && q_heads % kv_heads == 0
                        && a.shape[1] == q_heads
                        && args[1].shape[1] == kv_heads
                        && args[1] == args[2]
                        && args[1].shape[0] == a.shape[0]
                        && args[1].shape[3] == a.shape[3],
                    "GQA heads, cache or head dimension mismatch",
                )?;
                expect(
                    position.checked_add(a.shape[2]) == Some(args[1].shape[2]),
                    "attention must see exactly the active logical prefix",
                )?;
                Ok(TensorType::new(
                    vec![a.shape[0], a.shape[2], q_heads * a.shape[3]],
                    ScalarType::F32,
                    Layout::SequenceBtd,
                ))
            }
            Op::SiLu | Op::LastToken => {
                check_type(a, 3, ScalarType::F32, Layout::SequenceBtd)?;
                let mut ty = a.clone();
                if matches!(*self, Op::LastToken) {
                    ty.shape[1] = 1;
                }
                Ok(ty)
            }
            Op::Mul | Op::Add => {
                check_type(a, 3, ScalarType::F32, Layout::SequenceBtd)?;
                expect(a == args[1], "elementwise operands must match exactly")?;
                Ok(a.clone())
            }
        }
    }

    pub fn lowering_status(&self) -> LoweringStatus {
        match self {
            Op::FishEmbedding { .. } => LoweringStatus::CandidateTrtPrimitive,
            Op::KvUpdate { .. } => LoweringStatus::RequiresGpuValidation,
            Op::GqaAttention { .. } => LoweringStatus::CandidateTrtPrimitive,
            Op::TokenEmbedding | Op::SplitQkv { .. } | Op::RoPe { .. } | Op::KvPrefix { .. } => {
                LoweringStatus::CandidateTrtPrimitive
            }
            Op::Identity
            | Op::RmsNorm { .. }
            | Op::Linear
            | Op::LinearBias
            | Op::SiLu
            | Op::Mul
            | Op::Add
            | Op::LastToken => LoweringStatus::CandidateTrtPrimitive,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum LoweringStatus {
    CandidateTrtPrimitive,
    NeedsDecomposition,
    RequiresGpuValidation,
}

#[derive(Debug, Clone)]
pub struct LoweringAudit {
    pub counts: BTreeMap<LoweringStatus, usize>,
    pub weight_count: usize,
    pub input_count: usize,
    pub output_count: usize,
}

impl Graph {
    pub fn audit(&self) -> Result<LoweringAudit> {
        self.validate()?;
        let mut counts = BTreeMap::new();
        for n in &self.nodes {
            *counts.entry(n.op.lowering_status()).or_insert(0) += 1;
        }
        Ok(LoweringAudit {
            counts,
            weight_count: self.weight_requirements().len(),
            input_count: self
                .values
                .iter()
                .filter(|v| v.origin == Origin::Input)
                .count(),
            output_count: self.outputs.len(),
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    fn t(shape: &[usize], dtype: ScalarType, layout: Layout) -> TensorType {
        TensorType::new(shape.to_vec(), dtype, layout)
    }
    #[test]
    fn forward_reference_and_wrong_shape_rejected() {
        let mut g = Graph::new(Stage::SlowDecode, Some(1), Some(4));
        let a = g
            .input("a", t(&[1, 1, 8], ScalarType::F32, Layout::SequenceBtd))
            .unwrap();
        let b = g
            .input("b", t(&[1, 1, 8], ScalarType::F32, Layout::SequenceBtd))
            .unwrap();
        let c = g.emit("c", Op::Add, &[a, b]).unwrap();
        g.output(c).unwrap();
        assert!(g.validate().is_ok());
        g.nodes[0].inputs[0] = c;
        assert!(g.validate().is_err());
    }
    #[test]
    fn type_conflict_detected() {
        let mut g = Graph::new(Stage::SlowDecode, None, None);
        let a = g
            .input("x", t(&[1, 1, 8], ScalarType::F32, Layout::SequenceBtd))
            .unwrap();
        let b = g
            .input("bad", t(&[1, 1, 8], ScalarType::BF16, Layout::SequenceBtd))
            .unwrap();
        assert!(g.emit("out", Op::Add, &[a, b]).is_err());
        assert!(g
            .input("x", t(&[1, 1, 8], ScalarType::F32, Layout::SequenceBtd))
            .is_err());
    }
    #[test]
    fn conflicting_weight_specifications_fail() {
        let mut g = Graph::new(Stage::SlowDecode, None, None);
        g.weight("w", t(&[8, 8], ScalarType::F32, Layout::Plain))
            .unwrap();
        assert!(g
            .weight("w", t(&[7, 8], ScalarType::F32, Layout::Plain))
            .is_err());
    }
    #[test]
    fn gqa_must_consume_only_right_aligned_prefix() {
        let mut g = Graph::new(Stage::SlowDecode, Some(8), Some(16));
        let q = g
            .input("q", t(&[1, 8, 1, 8], ScalarType::F32, Layout::HeadsBhtd))
            .unwrap();
        let k = g
            .input("k", t(&[1, 2, 8, 8], ScalarType::BF16, Layout::HeadsBhtd))
            .unwrap();
        let v = g
            .input("v", t(&[1, 2, 8, 8], ScalarType::BF16, Layout::HeadsBhtd))
            .unwrap();
        assert!(g
            .emit(
                "correct",
                Op::GqaAttention {
                    position: 7,
                    q_heads: 8,
                    kv_heads: 2
                },
                &[q, k, v]
            )
            .is_ok());
        assert!(g
            .emit(
                "wrong",
                Op::GqaAttention {
                    position: 0,
                    q_heads: 8,
                    kv_heads: 2
                },
                &[q, k, v]
            )
            .is_err());
    }
}

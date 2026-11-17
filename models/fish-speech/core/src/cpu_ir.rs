use std::collections::BTreeMap;

use crate::error::{FishError, Result};
use crate::graph_ir::{Graph, Op, Origin, QkvPart, ScalarType, TensorType};

#[derive(Debug, Clone, PartialEq)]
pub enum TensorData {
    F32(Vec<f32>),
    I32(Vec<i32>),
}

impl TensorData {
    fn floats(&self) -> Result<&[f32]> {
        match self {
            Self::F32(v) => Ok(v),
            _ => Err(FishError::InvalidShape("expected float tensor")),
        }
    }
    fn ints(&self) -> Result<&[i32]> {
        match self {
            Self::I32(v) => Ok(v),
            _ => Err(FishError::InvalidShape("expected integer tensor")),
        }
    }
    fn valid_for(&self, ty: &TensorType) -> bool {
        let len: usize = ty.shape.iter().product();
        match (ty.dtype, self) {
            (ScalarType::I32, TensorData::I32(v)) => v.len() == len,
            (ScalarType::F32 | ScalarType::BF16, TensorData::F32(v)) => v.len() == len,
            _ => false,
        }
    }
}

pub type NamedTensors = BTreeMap<String, TensorData>;

pub struct CpuIrInterpreter;

impl CpuIrInterpreter {
    pub fn run(graph: &Graph, bindings: &NamedTensors) -> Result<NamedTensors> {
        graph.validate()?;
        let mut buffers: Vec<Option<TensorData>> = vec![None; graph.values.len()];
        for (i, value) in graph.values.iter().enumerate() {
            if matches!(value.origin, Origin::Input | Origin::Weight { .. }) {
                let data = bindings
                    .get(&value.name)
                    .ok_or(FishError::InvalidShape("missing CPU tensor binding"))?;
                if !data.valid_for(&value.ty) {
                    return Err(FishError::InvalidShape(
                        "CPU binding dtype or shape mismatch",
                    ));
                }
                buffers[i] = Some(data.clone());
            }
        }
        for node in &graph.nodes {
            let operands = node
                .inputs
                .iter()
                .map(|id| {
                    buffers[id.0]
                        .as_ref()
                        .ok_or(FishError::InvalidShape("CPU missing dependency"))
                })
                .collect::<Result<Vec<_>>>()?;
            let input_types = node
                .inputs
                .iter()
                .map(|id| &graph.values[id.0].ty)
                .collect::<Vec<_>>();
            let data = eval(node.op, &operands, &input_types)?;
            if !data.valid_for(&graph.values[node.result.0].ty) {
                return Err(FishError::InvalidShape(
                    "CPU node produced wrong tensor type",
                ));
            }
            buffers[node.result.0] = Some(data);
        }
        let mut outputs = BTreeMap::new();
        for output in &graph.outputs {
            let v = &graph.values[output.0];
            outputs.insert(
                v.name.clone(),
                buffers[output.0]
                    .as_ref()
                    .ok_or(FishError::InvalidShape("CPU graph output missing"))?
                    .clone(),
            );
        }
        Ok(outputs)
    }
}

fn eval(op: Op, args: &[&TensorData], ty: &[&TensorType]) -> Result<TensorData> {
    let a = args[0];
    match op {
        Op::FishEmbedding {
            num_codebooks,
            codebook_size,
            share_codebook_embeddings,
        } => {
            let tokens = a.ints()?;
            let word_weights = args[1].floats()?;
            let codebook_weights = args[2].floats()?;
            let semantic_ids = args[3].ints()?;
            let (batch, frames, dim) = (ty[0].shape[0], ty[0].shape[2], ty[1].shape[1]);
            let mut out = vec![0f32; batch * frames * dim];
            for b in 0..batch {
                for frame in 0..frames {
                    let index = tokens[(b * (num_codebooks + 1)) * frames + frame];
                    let word = usize::try_from(index)
                        .map_err(|_| FishError::InvalidShape("negative semantic token"))?;
                    if word >= ty[1].shape[0] {
                        return Err(FishError::InvalidShape(
                            "semantic token outside embedding vocabulary",
                        ));
                    }
                    let condition = semantic_ids.contains(&index);
                    let o = (b * frames + frame) * dim;
                    for j in 0..dim {
                        out[o + j] = word_weights[word * dim + j];
                    }
                    if condition {
                        for c in 0..num_codebooks {
                            let index = tokens[(b * (num_codebooks + 1) + 1 + c) * frames + frame];
                            let code = usize::try_from(index)
                                .map_err(|_| FishError::InvalidShape("negative acoustic code"))?;
                            if code >= codebook_size {
                                return Err(FishError::InvalidShape(
                                    "acoustic code outside codebook",
                                ));
                            }
                            let row = code
                                + if share_codebook_embeddings {
                                    c * codebook_size
                                } else {
                                    0
                                };
                            for j in 0..dim {
                                out[o + j] += codebook_weights[row * dim + j];
                            }
                        }
                    }
                }
            }
            Ok(TensorData::F32(out))
        }
        Op::TokenEmbedding => {
            let indices = a.ints()?;
            let weights = args[1].floats()?;
            let (vocab, dim) = (ty[1].shape[0], ty[1].shape[1]);
            let mut out = Vec::with_capacity(indices.len() * dim);
            for &idx in indices {
                let idx = usize::try_from(idx)
                    .map_err(|_| FishError::InvalidShape("negative token embedding ID"))?;
                if idx >= vocab {
                    return Err(FishError::InvalidShape("token embedding ID out of range"));
                }
                out.extend_from_slice(&weights[idx * dim..(idx + 1) * dim]);
            }
            Ok(TensorData::F32(out))
        }
        Op::Identity => Ok(a.clone()),
        Op::RmsNorm { epsilon_bits } => {
            let x = a.floats()?;
            let w = args[1].floats()?;
            let d = w.len();
            let mut out = vec![0f32; x.len()];
            for (src, dst) in x.chunks_exact(d).zip(out.chunks_exact_mut(d)) {
                let mean = src.iter().map(|v| v * v).sum::<f32>() / d as f32;
                let scale = 1.0 / (mean + f32::from_bits(epsilon_bits)).sqrt();
                for j in 0..d {
                    dst[j] = src[j] * scale * w[j];
                }
            }
            Ok(TensorData::F32(out))
        }
        Op::Linear | Op::LinearBias => {
            let x = a.floats()?;
            let w = args[1].floats()?;
            let bias = if matches!(op, Op::LinearBias) {
                Some(args[2].floats()?)
            } else {
                None
            };
            let (input, output) = (ty[1].shape[1], ty[1].shape[0]);
            let mut out = vec![0f32; (x.len() / input) * output];
            for (row, dst) in x.chunks_exact(input).zip(out.chunks_exact_mut(output)) {
                for o in 0..output {
                    let mut sum = bias.map_or(0.0, |b| b[o]);
                    for i in 0..input {
                        sum += row[i] * w[o * input + i];
                    }
                    dst[o] = sum;
                }
            }
            Ok(TensorData::F32(out))
        }
        Op::SplitQkv {
            part,
            q_heads,
            kv_heads,
            head_dim,
        } => {
            let x = a.floats()?;
            let (batch, frames, full) = (ty[0].shape[0], ty[0].shape[1], ty[0].shape[2]);
            let (head_count, off) = match part {
                QkvPart::Query => (q_heads, 0),
                QkvPart::Key => (kv_heads, q_heads * head_dim),
                QkvPart::Value => (kv_heads, (q_heads + kv_heads) * head_dim),
            };
            let mut out = vec![0f32; batch * head_count * frames * head_dim];
            for b in 0..batch {
                for h in 0..head_count {
                    for t in 0..frames {
                        for d in 0..head_dim {
                            out[((b * head_count + h) * frames + t) * head_dim + d] =
                                x[(b * frames + t) * full + off + h * head_dim + d];
                        }
                    }
                }
            }
            Ok(TensorData::F32(out))
        }
        Op::RoPe {
            position,
            base_bits,
        } => {
            let x = a.floats()?;
            let (batch, heads, frames, dim) = (
                ty[0].shape[0],
                ty[0].shape[1],
                ty[0].shape[2],
                ty[0].shape[3],
            );
            let mut out = x.to_vec();
            let base = f32::from_bits(base_bits);
            for b in 0..batch {
                for h in 0..heads {
                    for t in 0..frames {
                        for pair in 0..dim / 2 {
                            let freq = 1.0 / base.powf((pair * 2) as f32 / dim as f32);
                            let angle = (position + t) as f32 * freq;
                            let (sin, cos) = angle.sin_cos();
                            let ix = ((b * heads + h) * frames + t) * dim + 2 * pair;
                            out[ix] = x[ix] * cos - x[ix + 1] * sin;
                            out[ix + 1] = x[ix] * sin + x[ix + 1] * cos;
                        }
                    }
                }
            }
            Ok(TensorData::F32(out))
        }
        Op::KvUpdate { position } => {
            let mut cache = a.floats()?.to_vec();
            let update = args[1].floats()?;
            let (batch, heads, capacity, dim) = (
                ty[0].shape[0],
                ty[0].shape[1],
                ty[0].shape[2],
                ty[0].shape[3],
            );
            let frames = ty[1].shape[2];
            for b in 0..batch {
                for h in 0..heads {
                    for frame in 0..frames {
                        for d in 0..dim {
                            cache[((b * heads + h) * capacity + position + frame) * dim + d] =
                                update[((b * heads + h) * frames + frame) * dim + d];
                        }
                    }
                }
            }
            Ok(TensorData::F32(cache))
        }
        Op::KvPrefix { active_len } => {
            let x = a.floats()?;
            let (batch, heads, capacity, dim) = (
                ty[0].shape[0],
                ty[0].shape[1],
                ty[0].shape[2],
                ty[0].shape[3],
            );
            let mut out = vec![0f32; batch * heads * active_len * dim];
            for b in 0..batch {
                for h in 0..heads {
                    let src = (b * heads + h) * capacity * dim;
                    let dst = (b * heads + h) * active_len * dim;
                    out[dst..dst + active_len * dim]
                        .copy_from_slice(&x[src..src + active_len * dim]);
                }
            }
            Ok(TensorData::F32(out))
        }
        Op::GqaAttention {
            position,
            q_heads,
            kv_heads,
        } => {
            let q = a.floats()?;
            let k = args[1].floats()?;
            let v = args[2].floats()?;
            let (batch, frames, hd, kv_len) = (
                ty[0].shape[0],
                ty[0].shape[2],
                ty[0].shape[3],
                ty[1].shape[2],
            );
            let mut out = vec![0f32; batch * frames * q_heads * hd];
            for b in 0..batch {
                for head in 0..q_heads {
                    for frame in 0..frames {
                        let kv_head = head / (q_heads / kv_heads);
                        let visible = position + frame + 1;
                        let mut scores = vec![0f32; visible];
                        for j in 0..visible {
                            let mut sum = 0f32;
                            for d in 0..hd {
                                sum += q[((b * q_heads + head) * frames + frame) * hd + d]
                                    * k[((b * kv_heads + kv_head) * kv_len + j) * hd + d];
                            }
                            scores[j] = sum / (hd as f32).sqrt();
                        }
                        let max = scores.iter().copied().fold(f32::NEG_INFINITY, f32::max);
                        let mut denominator = 0f32;
                        for score in &mut scores {
                            *score = (*score - max).exp();
                            denominator += *score;
                        }
                        for d in 0..hd {
                            let mut sum = 0f32;
                            for j in 0..visible {
                                sum += scores[j] / denominator
                                    * v[((b * kv_heads + kv_head) * kv_len + j) * hd + d];
                            }
                            out[(b * frames + frame) * (q_heads * hd) + head * hd + d] = sum;
                        }
                    }
                }
            }
            Ok(TensorData::F32(out))
        }
        Op::SiLu => Ok(TensorData::F32(
            a.floats()?.iter().map(|x| x / (1.0 + (-x).exp())).collect(),
        )),
        Op::Mul | Op::Add => {
            let aa = a.floats()?;
            let bb = args[1].floats()?;
            let iter = aa.iter().zip(bb);
            Ok(TensorData::F32(if matches!(op, Op::Mul) {
                iter.map(|(x, y)| x * y).collect()
            } else {
                iter.map(|(x, y)| x + y).collect()
            }))
        }
        Op::LastToken => {
            let x = a.floats()?;
            let (batch, frames, dim) = (ty[0].shape[0], ty[0].shape[1], ty[0].shape[2]);
            let mut out = vec![0f32; batch * dim];
            for b in 0..batch {
                out[b * dim..(b + 1) * dim].copy_from_slice(
                    &x[(b * frames + frames - 1) * dim..(b * frames + frames) * dim],
                );
            }
            Ok(TensorData::F32(out))
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::FishModelConfig;
    use crate::graph_ir::Layout;
    use crate::transformer_ir::TransformerIrFactory;

    fn fixture_bindings(graph: &Graph, poison_tail: bool) -> NamedTensors {
        let mut bind = NamedTensors::new();
        for value in &graph.values {
            let count: usize = value.ty.shape.iter().product();
            let data = match &value.origin {
                Origin::Input => {
                    if value.name == "tokens" {
                        let mut values = vec![0i32; count];
                        let seq = value.ty.shape[2];
                        for item in values.iter_mut().take(seq) {
                            *item = 12;
                        }
                        TensorData::I32(values)
                    } else if value.name == "semantic_token_ids" {
                        TensorData::I32((0..1024).collect())
                    } else if value.ty.dtype == ScalarType::I32 {
                        TensorData::I32(vec![0; count])
                    } else if value.ty.layout == Layout::CacheBhtd {
                        let mut data = vec![0f32; count];
                        if poison_tail && graph.active_kv_len.is_some() {
                            let active = graph.active_kv_len.unwrap();
                            let cap = value.ty.shape[2];
                            let head_dim = value.ty.shape[3];
                            for head in 0..value.ty.shape[1] {
                                for pos in active..cap {
                                    for d in 0..head_dim {
                                        data[(head * cap + pos) * head_dim + d] = f32::NAN;
                                    }
                                }
                            }
                        }
                        TensorData::F32(data)
                    } else {
                        TensorData::F32(vec![0.01; count])
                    }
                }
                Origin::Weight { .. } => {
                    let data = if value.name.ends_with("norm.weight") || value.name == "norm.weight"
                    {
                        vec![1f32; count]
                    } else {
                        (0..count)
                            .map(|i| ((i % 19) as f32 - 9.0) / 1000.0)
                            .collect()
                    };
                    TensorData::F32(data)
                }
                Origin::Node(_) => continue,
            };
            bind.insert(value.name.clone(), data);
        }
        bind
    }

    #[test]
    fn full_synthetic_slow_prefill_executes() {
        let c = FishModelConfig::synthetic_dual_ar_fixture();
        let g = TransformerIrFactory::new(&c)
            .unwrap()
            .slow_prefill(3)
            .unwrap();
        let out = CpuIrInterpreter::run(&g, &fixture_bindings(&g, false)).unwrap();
        assert_eq!(out["logits"].floats().unwrap().len(), c.vocab_size);
        assert_eq!(
            out["layers.0.k_updated"].floats().unwrap().len(),
            2 * c.max_seq_len * 8
        );
        assert!(out["logits"]
            .floats()
            .unwrap()
            .iter()
            .all(|v| v.is_finite()));
    }

    #[test]
    fn decoder_poisoned_tail_cannot_change_logits() {
        let c = FishModelConfig::synthetic_dual_ar_fixture();
        let g = TransformerIrFactory::new(&c)
            .unwrap()
            .slow_decode(4)
            .unwrap();
        let clean = CpuIrInterpreter::run(&g, &fixture_bindings(&g, false)).unwrap();
        let dirty = CpuIrInterpreter::run(&g, &fixture_bindings(&g, true)).unwrap();
        assert_eq!(clean["logits"], dirty["logits"]);
        assert!(dirty["logits"]
            .floats()
            .unwrap()
            .iter()
            .all(|x| x.is_finite()));
    }

    #[test]
    fn full_synthetic_fast_codebook_decode_executes() {
        let c = FishModelConfig::synthetic_dual_ar_fixture();
        let g = TransformerIrFactory::new(&c)
            .unwrap()
            .fast_decode(3)
            .unwrap();
        let out = CpuIrInterpreter::run(&g, &fixture_bindings(&g, false)).unwrap();
        assert_eq!(out["fast_logits"].floats().unwrap().len(), c.codebook_size);
        assert!(out["fast_logits"]
            .floats()
            .unwrap()
            .iter()
            .all(|v| v.is_finite()));
    }

    #[test]
    fn refuses_missing_checkpoint_tensors() {
        let c = FishModelConfig::synthetic_dual_ar_fixture();
        let g = TransformerIrFactory::new(&c)
            .unwrap()
            .fast_decode(0)
            .unwrap();
        let mut bind = fixture_bindings(&g, false);
        bind.remove("fast_layers.0.attention.wqkv.weight");
        assert!(CpuIrInterpreter::run(&g, &bind).is_err());
    }

    #[test]
    fn fast_step_zero_consumes_exact_slow_context_output() {
        let c = FishModelConfig::synthetic_dual_ar_fixture();
        let f = TransformerIrFactory::new(&c).unwrap();
        let slow = f.slow_prefill(1).unwrap();
        let slow_output = CpuIrInterpreter::run(&slow, &fixture_bindings(&slow, false)).unwrap();
        let step_zero = f.fast_decode(0).unwrap();
        let mut input = fixture_bindings(&step_zero, false);
        input.insert("fast_context".into(), slow_output["fast_context"].clone());
        let outputs = CpuIrInterpreter::run(&step_zero, &input).unwrap();
        assert!(outputs["fast_logits"]
            .floats()
            .unwrap()
            .iter()
            .all(|v| v.is_finite()));
    }

    #[test]
    fn fast_token_embedding_uses_last_generated_code() {
        use crate::graph::Stage;
        let mut graph = Graph::new(Stage::FastDecode, None, None);
        let token = graph
            .input(
                "token",
                TensorType::new(vec![1, 1], ScalarType::I32, Layout::Plain),
            )
            .unwrap();
        let w = graph
            .weight(
                "weight",
                TensorType::new(vec![3, 2], ScalarType::F32, Layout::Plain),
            )
            .unwrap();
        let out = graph
            .emit("embedded", Op::TokenEmbedding, &[token, w])
            .unwrap();
        graph.output(out).unwrap();
        let mut bindings = NamedTensors::new();
        bindings.insert("token".into(), TensorData::I32(vec![2]));
        bindings.insert(
            "weight".into(),
            TensorData::F32(vec![1., 2., 3., 4., 5., 6.]),
        );
        let outputs = CpuIrInterpreter::run(&graph, &bindings).unwrap();
        assert_eq!(outputs["embedded"], TensorData::F32(vec![5., 6.]));
        bindings.insert("token".into(), TensorData::I32(vec![-1]));
        assert!(CpuIrInterpreter::run(&graph, &bindings).is_err());
        bindings.insert("token".into(), TensorData::I32(vec![3]));
        assert!(CpuIrInterpreter::run(&graph, &bindings).is_err());
    }

    #[test]
    fn projection_dimension_change_is_executable() {
        let mut c = FishModelConfig::synthetic_dual_ar_fixture();
        let f = c.fast.as_mut().unwrap();
        f.dim = 32;
        f.n_heads = 4;
        f.n_kv_heads = 2;
        f.intermediate_size = 96;
        let factory = TransformerIrFactory::new(&c).unwrap();
        let slow = factory.slow_prefill(1).unwrap();
        let slow_outputs = CpuIrInterpreter::run(&slow, &fixture_bindings(&slow, false)).unwrap();
        let fast = factory.fast_decode(0).unwrap();
        let mut bindings = fixture_bindings(&fast, false);
        bindings.insert("fast_context".into(), slow_outputs["fast_context"].clone());
        let out = CpuIrInterpreter::run(&fast, &bindings).unwrap();
        assert_eq!(out["fast_logits"].floats().unwrap().len(), c.codebook_size);
    }

    #[test]
    fn explicit_embedding_gating_and_offset_parity() {
        use crate::graph::Stage;
        for share in [false, true] {
            let mut g = Graph::new(Stage::SlowPrefill, None, None);
            let tok = g
                .input(
                    "tok",
                    TensorType::new(vec![1, 3, 2], ScalarType::I32, Layout::TokensBct),
                )
                .unwrap();
            let w = g
                .weight(
                    "word",
                    TensorType::new(vec![10, 1], ScalarType::F32, Layout::Plain),
                )
                .unwrap();
            let cb = g
                .weight(
                    "cb",
                    TensorType::new(vec![4, 1], ScalarType::F32, Layout::Plain),
                )
                .unwrap();
            let sem = g
                .input(
                    "sem",
                    TensorType::new(vec![1024], ScalarType::I32, Layout::Plain),
                )
                .unwrap();
            let out = g
                .emit(
                    "out",
                    Op::FishEmbedding {
                        num_codebooks: 2,
                        codebook_size: 2,
                        share_codebook_embeddings: share,
                    },
                    &[tok, w, cb, sem],
                )
                .unwrap();
            g.output(out).unwrap();
            let mut bind = NamedTensors::new();
            bind.insert("tok".into(), TensorData::I32(vec![5, 9, 1, 1, 1, 0]));
            bind.insert(
                "word".into(),
                TensorData::F32((0..10).map(|i| i as f32).collect()),
            );
            bind.insert("cb".into(), TensorData::F32(vec![0., 10., 20., 30.]));
            let mut semantic = vec![-1; 1024];
            semantic[0] = 5;
            bind.insert("sem".into(), TensorData::I32(semantic));
            let result = CpuIrInterpreter::run(&g, &bind).unwrap();
            assert_eq!(
                result["out"],
                TensorData::F32(vec![if share { 45. } else { 25. }, 9.])
            );
        }
    }

    #[test]
    fn explicit_rope_rotation_uses_absolute_position() {
        use crate::graph::Stage;
        let mut g = Graph::new(Stage::SlowDecode, None, None);
        let q = g
            .input(
                "q",
                TensorType::new(vec![1, 1, 1, 2], ScalarType::F32, Layout::HeadsBhtd),
            )
            .unwrap();
        let y = g
            .emit(
                "y",
                Op::RoPe {
                    position: 1,
                    base_bits: 10000f32.to_bits(),
                },
                &[q],
            )
            .unwrap();
        g.output(y).unwrap();
        let mut bind = NamedTensors::new();
        bind.insert("q".into(), TensorData::F32(vec![1., 0.]));
        let result = CpuIrInterpreter::run(&g, &bind).unwrap();
        let data = result["y"].floats().unwrap();
        assert!((data[0] - 1f32.cos()).abs() < 1e-6);
        assert!((data[1] - 1f32.sin()).abs() < 1e-6);
    }

    #[test]
    fn prefill_attention_causality_is_right_aligned() {
        use crate::graph::Stage;
        let mut g = Graph::new(Stage::SlowPrefill, Some(3), Some(16));
        let q = g
            .input(
                "q",
                TensorType::new(vec![1, 2, 3, 1], ScalarType::F32, Layout::HeadsBhtd),
            )
            .unwrap();
        let k = g
            .input(
                "k",
                TensorType::new(vec![1, 1, 3, 1], ScalarType::BF16, Layout::HeadsBhtd),
            )
            .unwrap();
        let v = g
            .input(
                "v",
                TensorType::new(vec![1, 1, 3, 1], ScalarType::BF16, Layout::HeadsBhtd),
            )
            .unwrap();
        let out = g
            .emit(
                "out",
                Op::GqaAttention {
                    position: 0,
                    q_heads: 2,
                    kv_heads: 1,
                },
                &[q, k, v],
            )
            .unwrap();
        g.output(out).unwrap();
        let mut bind = NamedTensors::new();
        bind.insert("q".into(), TensorData::F32(vec![0.; 6]));
        bind.insert("k".into(), TensorData::F32(vec![1., 1., 1.]));
        bind.insert("v".into(), TensorData::F32(vec![1., 3., 100.]));
        let result = CpuIrInterpreter::run(&g, &bind).unwrap();
        let data = result["out"].floats().unwrap();
        assert_eq!(data.len(), 6);
        for h in 0..2 {
            assert!((data[h] - 1.).abs() < 1e-6);
            assert!((data[2 + h] - 2.).abs() < 1e-6);
            assert!((data[4 + h] - (104. / 3.)).abs() < 1e-4);
        }
    }

    #[test]
    fn cpu_ir_rejects_binding_with_wrong_type() {
        let c = FishModelConfig::synthetic_dual_ar_fixture();
        let g = TransformerIrFactory::new(&c)
            .unwrap()
            .slow_decode(1)
            .unwrap();
        let mut bind = fixture_bindings(&g, false);
        bind.insert("tokens".into(), TensorData::F32(vec![0.; 9]));
        assert!(CpuIrInterpreter::run(&g, &bind).is_err());
    }

    #[test]
    fn cached_next_token_matches_full_prefill_on_synthetic_weights() {
        let c = FishModelConfig::synthetic_dual_ar_fixture();
        let f = TransformerIrFactory::new(&c).unwrap();
        let short = f.slow_prefill(1).unwrap();
        let full = f.slow_prefill(2).unwrap();
        let cached = f.slow_decode(2).unwrap();
        let first = CpuIrInterpreter::run(&short, &fixture_bindings(&short, false)).unwrap();
        let full_result = CpuIrInterpreter::run(&full, &fixture_bindings(&full, false)).unwrap();
        let mut bindings = fixture_bindings(&cached, false);
        for layer in 0..c.slow.layers {
            for prefix in ["k", "v"] {
                let source = format!("layers.{layer}.{prefix}_updated");
                let target = format!("layers.{layer}.{prefix}_cache");
                bindings.insert(target, first[&source].clone());
            }
        }
        let next = CpuIrInterpreter::run(&cached, &bindings).unwrap();
        let got = next["logits"].floats().unwrap();
        let expect = full_result["logits"].floats().unwrap();
        assert_eq!(got.len(), expect.len());
        for (a, b) in got.iter().zip(expect) {
            assert!(
                (a - b).abs() <= 1e-5,
                "full prefill and cached decode diverged: {a} vs {b}"
            );
        }
    }
}

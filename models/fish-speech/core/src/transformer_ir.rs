use crate::config::{FishModelConfig, ModelKind, TransformerDims};
use crate::error::{FishError, Result};
use crate::graph::Stage;
use crate::graph_ir::{Graph, Layout, Op, QkvPart, ScalarType, TensorType, ValueId};

fn t(shape: &[usize], dtype: ScalarType, layout: Layout) -> TensorType {
    TensorType::new(shape.to_vec(), dtype, layout)
}

struct GraphBuilder<'a> {
    graph: Graph,
    dims: &'a TransformerDims,
    prefix: &'static str,
    context_len: usize,
    capacity: usize,
    rope_base: f32,
    norm_eps: f32,
    first_position: usize,
}

impl GraphBuilder<'_> {
    fn weight(&mut self, key: impl Into<String>, shape: &[usize]) -> Result<ValueId> {
        self.graph
            .weight(key, t(shape, ScalarType::F32, Layout::Plain))
    }

    fn linear(
        &mut self,
        name: &str,
        input: ValueId,
        key: &str,
        out_dim: usize,
        bias: bool,
    ) -> Result<ValueId> {
        let input_dim = self.graph.tensor(input)?.shape[2];
        let w = self.weight(format!("{key}.weight"), &[out_dim, input_dim])?;
        if bias {
            let b = self.weight(format!("{key}.bias"), &[out_dim])?;
            self.graph.emit(name, Op::LinearBias, &[input, w, b])
        } else {
            self.graph.emit(name, Op::Linear, &[input, w])
        }
    }

    fn norm(&mut self, name: &str, x: ValueId, key: &str) -> Result<ValueId> {
        let dim = self.graph.tensor(x)?.shape[2];
        let weight = self.weight(key, &[dim])?;
        self.graph.emit(
            name,
            Op::RmsNorm {
                epsilon_bits: self.norm_eps.to_bits(),
            },
            &[x, weight],
        )
    }

    fn build_layers(&mut self, mut x: ValueId) -> Result<ValueId> {
        let head_dim = self.dims.head_dim();
        for layer in 0..self.dims.layers {
            let base = if self.prefix == "layers" {
                format!("layers.{layer}")
            } else {
                format!("fast_layers.{layer}")
            };
            let att_norm = self.norm(
                &format!("{base}.att_norm"),
                x,
                &format!("{base}.attention_norm.weight"),
            )?;
            let packed = self.linear(
                &format!("{base}.qkv"),
                att_norm,
                &format!("{base}.attention.wqkv"),
                self.dims.qkv_width(),
                self.dims.qkv_bias,
            )?;
            let split = |part| Op::SplitQkv {
                part,
                q_heads: self.dims.n_heads,
                kv_heads: self.dims.n_kv_heads,
                head_dim,
            };
            let q = self
                .graph
                .emit(format!("{base}.q"), split(QkvPart::Query), &[packed])?;
            let k = self
                .graph
                .emit(format!("{base}.k"), split(QkvPart::Key), &[packed])?;
            let v = self
                .graph
                .emit(format!("{base}.v"), split(QkvPart::Value), &[packed])?;
            let q = self.graph.emit(
                format!("{base}.q_rope"),
                Op::RoPe {
                    position: self.first_position,
                    base_bits: self.rope_base.to_bits(),
                },
                &[q],
            )?;
            let k = self.graph.emit(
                format!("{base}.k_rope"),
                Op::RoPe {
                    position: self.first_position,
                    base_bits: self.rope_base.to_bits(),
                },
                &[k],
            )?;
            let cache_ty = t(
                &[1, self.dims.n_kv_heads, self.capacity, head_dim],
                ScalarType::BF16,
                Layout::CacheBhtd,
            );
            let k_buf = self
                .graph
                .input(format!("{base}.k_cache"), cache_ty.clone())?;
            let v_buf = self.graph.input(format!("{base}.v_cache"), cache_ty)?;
            let k_full = self.graph.emit(
                format!("{base}.k_updated"),
                Op::KvUpdate {
                    position: self.first_position,
                },
                &[k_buf, k],
            )?;
            let v_full = self.graph.emit(
                format!("{base}.v_updated"),
                Op::KvUpdate {
                    position: self.first_position,
                },
                &[v_buf, v],
            )?;
            self.graph.output(k_full)?;
            self.graph.output(v_full)?;
            let k_active = self.graph.emit(
                format!("{base}.k_active"),
                Op::KvPrefix {
                    active_len: self.context_len,
                },
                &[k_full],
            )?;
            let v_active = self.graph.emit(
                format!("{base}.v_active"),
                Op::KvPrefix {
                    active_len: self.context_len,
                },
                &[v_full],
            )?;
            let attention = self.graph.emit(
                format!("{base}.attention"),
                Op::GqaAttention {
                    position: self.first_position,
                    q_heads: self.dims.n_heads,
                    kv_heads: self.dims.n_kv_heads,
                },
                &[q, k_active, v_active],
            )?;
            let projected = self.linear(
                &format!("{base}.wo"),
                attention,
                &format!("{base}.attention.wo"),
                self.dims.dim,
                false,
            )?;
            let residual = self.graph.emit(
                format!("{base}.attention_residual"),
                Op::Add,
                &[x, projected],
            )?;
            let ffn_norm = self.norm(
                &format!("{base}.ffn_norm"),
                residual,
                &format!("{base}.ffn_norm.weight"),
            )?;
            let a = self.linear(
                &format!("{base}.w1"),
                ffn_norm,
                &format!("{base}.feed_forward.w1"),
                self.dims.intermediate_size,
                false,
            )?;
            let b = self.linear(
                &format!("{base}.w3"),
                ffn_norm,
                &format!("{base}.feed_forward.w3"),
                self.dims.intermediate_size,
                false,
            )?;
            let activated = self.graph.emit(format!("{base}.silu"), Op::SiLu, &[a])?;
            let gated = self
                .graph
                .emit(format!("{base}.gated"), Op::Mul, &[activated, b])?;
            let down = self.linear(
                &format!("{base}.w2"),
                gated,
                &format!("{base}.feed_forward.w2"),
                self.dims.dim,
                false,
            )?;
            x = self
                .graph
                .emit(format!("{base}.ffn_residual"), Op::Add, &[residual, down])?;
        }
        Ok(x)
    }
}

pub struct TransformerIrFactory<'a> {
    config: &'a FishModelConfig,
}

impl<'a> TransformerIrFactory<'a> {
    pub fn new(config: &'a FishModelConfig) -> Result<Self> {
        config.validate()?;
        Ok(Self { config })
    }

    pub fn slow_prefill(&self, prompt_len: usize) -> Result<Graph> {
        self.slow(Stage::SlowPrefill, prompt_len, prompt_len)
    }

    pub fn slow_decode(&self, active_len: usize) -> Result<Graph> {
        self.slow(Stage::SlowDecode, active_len, 1)
    }

    fn slow(&self, stage: Stage, active_len: usize, query_len: usize) -> Result<Graph> {
        let c = self.config;
        if active_len == 0 || query_len == 0 || active_len > c.max_seq_len || active_len < query_len
        {
            return Err(FishError::InvalidShape(
                "invalid Slow AR logical sequence length",
            ));
        }
        let mut b = GraphBuilder {
            graph: Graph::new(stage, Some(active_len), Some(c.max_seq_len)),
            dims: &c.slow,
            prefix: "layers",
            context_len: active_len,
            capacity: c.max_seq_len,
            rope_base: c.rope_base,
            norm_eps: c.norm_eps,
            first_position: active_len - query_len,
        };
        let tokens = b.graph.input(
            "tokens",
            t(
                &[1, c.num_codebooks + 1, query_len],
                ScalarType::I32,
                Layout::TokensBct,
            ),
        )?;
        let words = b.weight("embeddings.weight", &[c.vocab_size, c.slow.dim])?;
        let cb = b.weight(
            "codebook_embeddings.weight",
            &[c.num_codebooks * c.codebook_size, c.slow.dim],
        )?;
        let semantic_ids = b.graph.input(
            "semantic_membership_lut",
            t(&[c.vocab_size], ScalarType::I32, Layout::Plain),
        )?;
        let embedded = b.graph.emit(
            "semantic_and_codebook_embeddings",
            Op::FishEmbedding {
                num_codebooks: c.num_codebooks,
                codebook_size: c.codebook_size,
                share_codebook_embeddings: c.share_codebook_embeddings,
            },
            &[tokens, words, cb, semantic_ids],
        )?;
        let x = b.build_layers(embedded)?;
        let last = b.graph.emit("hidden_last", Op::LastToken, &[x])?;
        b.graph.output(last)?;
        if c.kind == ModelKind::DualAr {
            let fast_dim = c.fast.as_ref().expect("validated Dual AR").dim;
            let projected = if fast_dim == c.slow.dim {
                b.graph.emit("fast_context", Op::Identity, &[last])?
            } else {
                b.linear("fast_context", last, "fast_project_in", fast_dim, true)?
            };
            b.graph.output(projected)?;
        }
        let normed = b.norm("final_norm", last, "norm.weight")?;
        let logits = if c.tie_word_embeddings {
            b.graph.emit("logits", Op::Linear, &[normed, words])?
        } else {
            b.linear("logits", normed, "output", c.vocab_size, false)?
        };
        b.graph.output(logits)?;
        if c.kind == ModelKind::Naive {
            let normalized = b.norm("codebook_norm", last, "codebook_norm.weight")?;
            let codebook_logits = b.linear(
                "codebook_logits",
                normalized,
                "codebook_output",
                c.num_codebooks * c.codebook_size,
                false,
            )?;
            b.graph.output(codebook_logits)?;
        }
        b.graph.validate()?;
        Ok(b.graph)
    }

    pub fn fast_decode(&self, step: usize) -> Result<Graph> {
        let c = self.config;
        if c.kind != ModelKind::DualAr {
            return Err(FishError::Unsupported("Fast AR absent in Naive model"));
        }
        if step >= c.num_codebooks {
            return Err(FishError::InvalidPosition {
                position: step,
                capacity: c.num_codebooks,
            });
        }
        let dims = c
            .fast
            .as_ref()
            .ok_or(FishError::InvalidConfig("missing Fast AR dimensions"))?;
        let mut b = GraphBuilder {
            graph: Graph::new(Stage::FastDecode, Some(step + 1), Some(c.num_codebooks)),
            dims,
            prefix: "fast_layers",
            context_len: step + 1,
            capacity: c.num_codebooks,
            rope_base: c.rope_base,
            norm_eps: c.norm_eps,
            first_position: step,
        };
        let input = if step == 0 {
            b.graph.input(
                "fast_context",
                t(&[1, 1, dims.dim], ScalarType::F32, Layout::SequenceBtd),
            )?
        } else {
            let token = b.graph.input(
                "previous_acoustic_code",
                t(&[1, 1], ScalarType::I32, Layout::Plain),
            )?;
            let embeddings = b.weight("fast_embeddings.weight", &[c.codebook_size, dims.dim])?;
            b.graph.emit(
                "fast_token_embedding",
                Op::TokenEmbedding,
                &[token, embeddings],
            )?
        };
        let x = b.build_layers(input)?;
        let normed = b.norm("fast_final_norm", x, "fast_norm.weight")?;
        let logits = b.linear("fast_logits", normed, "fast_output", c.codebook_size, false)?;
        b.graph.output(logits)?;
        b.graph.validate()?;
        Ok(b.graph)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::graph_ir::{LoweringStatus, Origin};

    fn factory_fixture() -> FishModelConfig {
        FishModelConfig::synthetic_dual_ar_fixture()
    }

    #[test]
    fn slow_prefill_wires_real_weight_paths_and_outputs() {
        let c = factory_fixture();
        let g = TransformerIrFactory::new(&c)
            .unwrap()
            .slow_prefill(17)
            .unwrap();
        assert!(g.validate().is_ok());
        let weights = g.weight_requirements();
        assert_eq!(weights["layers.0.attention.wqkv.weight"].shape, [96, 64]);
        assert_eq!(weights["layers.1.feed_forward.w2.weight"].shape, [64, 128]);
        assert_eq!(weights["embeddings.weight"].shape, [1024, 64]);
        assert!(!weights.contains_key("output.weight"));
        assert_eq!(
            g.tensor(*g.outputs.last().unwrap()).unwrap().shape,
            [1, 1, 1024]
        );
        assert_eq!(g.outputs.len(), 3 + c.slow.layers * 2);
        assert_eq!(g.active_kv_len, Some(17));
        assert!(g.nodes.iter().any(|node| node.op
            == Op::GqaAttention {
                position: 0,
                q_heads: 8,
                kv_heads: 2
            }));
    }

    #[test]
    fn slow_cached_decode_does_not_attend_to_capacity() {
        let c = factory_fixture();
        let g = TransformerIrFactory::new(&c)
            .unwrap()
            .slow_decode(37)
            .unwrap();
        assert!(g
            .nodes
            .iter()
            .any(|n| n.op == Op::KvPrefix { active_len: 37 }));
        assert!(g.nodes.iter().any(|n| n.op
            == Op::GqaAttention {
                position: 36,
                q_heads: 8,
                kv_heads: 2
            }));
        assert!(!g
            .nodes
            .iter()
            .any(|n| n.op == Op::KvPrefix { active_len: 256 }));
        assert!(g.validate().is_ok());
    }

    #[test]
    fn fast_cache_is_independent_and_prefix_scoped() {
        let c = factory_fixture();
        let b = TransformerIrFactory::new(&c).unwrap();
        for step in 0..c.num_codebooks {
            let g = b.fast_decode(step).unwrap();
            assert_eq!(g.active_kv_len, Some(step + 1));
            assert_eq!(g.kv_capacity, Some(c.num_codebooks));
            assert!(g.values.iter().any(|v| v.name == "fast_layers.0.k_cache"));
            assert!(!g.values.iter().any(|v| v.name == "layers.0.k_cache"));
        }
        assert!(b.fast_decode(c.num_codebooks).is_err());
    }

    #[test]
    fn untied_embeddings_and_qkv_bias_require_explicit_weights() {
        let mut c = factory_fixture();
        c.tie_word_embeddings = false;
        c.slow.qkv_bias = true;
        let g = TransformerIrFactory::new(&c)
            .unwrap()
            .slow_decode(1)
            .unwrap();
        let weights = g.weight_requirements();
        assert!(weights.contains_key("output.weight"));
        assert!(weights.contains_key("layers.0.attention.wqkv.bias"));
        assert!(g.nodes.iter().any(|n| n.op == Op::LinearBias));
    }

    #[test]
    fn bad_mutation_of_real_graph_caught() {
        let c = factory_fixture();
        let mut g = TransformerIrFactory::new(&c)
            .unwrap()
            .slow_decode(16)
            .unwrap();
        let idx = g
            .nodes
            .iter()
            .position(|n| matches!(n.op, Op::GqaAttention { .. }))
            .unwrap();
        g.nodes[idx].op = Op::GqaAttention {
            position: 0,
            q_heads: 8,
            kv_heads: 2,
        };
        assert!(g.validate().is_err());
    }

    #[test]
    fn validate_no_unbound_checkpoint_weights() {
        let c = factory_fixture();
        let g = TransformerIrFactory::new(&c)
            .unwrap()
            .slow_decode(1)
            .unwrap();
        let a = g.audit().unwrap();
        assert!(a.weight_count > 10);
        assert!(a.counts[&LoweringStatus::RequiresGpuValidation] >= c.slow.layers * 2);
        assert!(
            g.values
                .iter()
                .filter(|v| matches!(v.origin, Origin::Weight { .. }))
                .count()
                == a.weight_count
        );
    }

    #[test]
    fn first_fast_step_reads_slow_context_not_acoustic_embedding() {
        let c = factory_fixture();
        let factory = TransformerIrFactory::new(&c).unwrap();
        let slow = factory.slow_decode(1).unwrap();
        assert!(slow.values.iter().any(|v| v.name == "fast_context"));
        let first = factory.fast_decode(0).unwrap();
        assert!(first.values.iter().any(|v| v.name == "fast_context"));
        assert!(!first
            .values
            .iter()
            .any(|v| v.name == "fast_embeddings.weight"));
        assert!(first.nodes.iter().all(|n| n.op != Op::TokenEmbedding));
    }

    #[test]
    fn remaining_fast_steps_embed_previously_generated_code() {
        let c = factory_fixture();
        let factory = TransformerIrFactory::new(&c).unwrap();
        for step in 1..c.num_codebooks {
            let graph = factory.fast_decode(step).unwrap();
            assert!(graph
                .values
                .iter()
                .any(|v| v.name == "previous_acoustic_code"));
            assert_eq!(
                graph.weight_requirements()["fast_embeddings.weight"].shape,
                [c.codebook_size, c.fast.as_ref().unwrap().dim]
            );
            assert!(graph.nodes.iter().any(|n| n.op == Op::TokenEmbedding));
            assert!(!graph.values.iter().any(|v| v.name == "fast_context"));
        }
    }

    #[test]
    fn mismatched_fast_dim_projects_slow_hidden_with_bias() {
        let mut c = factory_fixture();
        let fast = c.fast.as_mut().unwrap();
        fast.dim = 32;
        fast.n_heads = 4;
        fast.n_kv_heads = 2;
        fast.intermediate_size = 96;
        let factory = TransformerIrFactory::new(&c).unwrap();
        let slow = factory.slow_decode(2).unwrap();
        assert_eq!(
            slow.weight_requirements()["fast_project_in.weight"].shape,
            [32, 64]
        );
        assert_eq!(
            slow.weight_requirements()["fast_project_in.bias"].shape,
            [32]
        );
        assert!(slow
            .nodes
            .iter()
            .any(|n| n.name == "fast_context" && n.op == Op::LinearBias));
        let first = factory.fast_decode(0).unwrap();
        assert_eq!(
            first
                .values
                .iter()
                .find(|v| v.name == "fast_context")
                .unwrap()
                .ty
                .shape,
            [1, 1, 32]
        );
        assert!(first.validate().is_ok());
    }

    #[test]
    fn bad_sequence_bounds_rejected() {
        let c = factory_fixture();
        let b = TransformerIrFactory::new(&c).unwrap();
        assert!(b.slow_prefill(0).is_err());
        assert!(b.slow_decode(c.max_seq_len + 1).is_err());
    }
}

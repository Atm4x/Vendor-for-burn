use super::*;
use fish_speech_core::{
    cpu_ir::{NamedTensors, TensorData},
    error::FishError,
    graph_ir::{Graph, Op, QkvPart, ScalarType},
};

#[cfg(test)]
mod preflight_tests {
    use super::*;
    use fish_speech_core::{
        graph::Stage,
        graph_ir::{Layout, TensorType},
    };

    #[test]
    fn supported_graph_compiles_without_runtime_or_tensorrt() {
        let mut g = Graph::new(Stage::SlowPrefill, None, None);
        let a = g
            .input(
                "a",
                TensorType::new([1, 2, 3], ScalarType::F32, Layout::SequenceBtd),
            )
            .unwrap();
        let b = g
            .input(
                "b",
                TensorType::new([1, 2, 3], ScalarType::F32, Layout::SequenceBtd),
            )
            .unwrap();
        let out = g.emit("sum", Op::Add, &[a, b]).unwrap();
        g.output(out).unwrap();
        let compiled = CubeClGraph::compile(&g).unwrap();
        let mut inputs = NamedTensors::new();
        inputs.insert("a".into(), TensorData::F32(vec![1.0; 6]));
        inputs.insert("b".into(), TensorData::F32(vec![2.0; 6]));
        assert!(compiled.validate_inputs(&inputs).is_ok());
        inputs.remove("b");
        assert!(compiled.validate_inputs(&inputs).is_err());
        inputs.insert("b".into(), TensorData::I32(vec![2; 6]));
        assert!(compiled.validate_inputs(&inputs).is_err());
        inputs.insert("b".into(), TensorData::F32(vec![2.0; 6]));
        inputs.insert("unknown".into(), TensorData::F32(vec![1.0]));
        assert!(compiled.validate_inputs(&inputs).is_err());
    }

    #[test]
    fn token_embedding_is_supported_and_rejects_invalid_ids() {
        let mut g = Graph::new(Stage::SlowPrefill, None, None);
        let tokens = g
            .input(
                "tokens",
                TensorType::new([1, 2], ScalarType::I32, Layout::Plain),
            )
            .unwrap();
        let weight = g
            .weight(
                "weight",
                TensorType::new([12, 4], ScalarType::F32, Layout::Plain),
            )
            .unwrap();
        let out = g
            .emit("embedding", Op::TokenEmbedding, &[tokens, weight])
            .unwrap();
        g.output(out).unwrap();
        let plan = CubeClGraph::compile(&g).unwrap();
        let mut inputs = NamedTensors::new();
        inputs.insert("tokens".into(), TensorData::I32(vec![0, 11]));
        inputs.insert("weight".into(), TensorData::F32(vec![0.0; 48]));
        plan.validate_inputs(&inputs).unwrap();
        for invalid in [-1, 12, i32::MAX] {
            inputs.insert("tokens".into(), TensorData::I32(vec![0, invalid]));
            assert!(plan.validate_inputs(&inputs).is_err());
        }
    }

    #[test]
    fn fish_embedding_checks_gated_codebooks_and_semantic_lut() {
        let mut g = Graph::new(Stage::SlowPrefill, None, None);
        let tok = g
            .input(
                "tok",
                TensorType::new([1, 3, 2], ScalarType::I32, Layout::TokensBct),
            )
            .unwrap();
        let words = g
            .weight(
                "words",
                TensorType::new([10, 2], ScalarType::F32, Layout::Plain),
            )
            .unwrap();
        let codes = g
            .weight(
                "codes",
                TensorType::new([4, 2], ScalarType::F32, Layout::Plain),
            )
            .unwrap();
        let lut = g
            .input("lut", TensorType::new([10], ScalarType::I32, Layout::Plain))
            .unwrap();
        let y = g
            .emit(
                "y",
                Op::FishEmbedding {
                    num_codebooks: 2,
                    codebook_size: 2,
                    share_codebook_embeddings: true,
                },
                &[tok, words, codes, lut],
            )
            .unwrap();
        g.output(y).unwrap();
        let plan = CubeClGraph::compile(&g).unwrap();
        let mut inputs = NamedTensors::new();
        inputs.insert("tok".into(), TensorData::I32(vec![5, 9, 1, -15, 1, 99]));
        inputs.insert("words".into(), TensorData::F32(vec![1.0; 20]));
        inputs.insert("codes".into(), TensorData::F32(vec![1.0; 8]));
        let mut semantic = vec![0; 10];
        semantic[5] = 1;
        inputs.insert("lut".into(), TensorData::I32(semantic.clone()));
        plan.validate_inputs(&inputs).unwrap();
        semantic[5] = 2;
        inputs.insert("lut".into(), TensorData::I32(semantic.clone()));
        assert!(plan.validate_inputs(&inputs).is_err());
        semantic[5] = 1;
        inputs.insert("lut".into(), TensorData::I32(semantic));
        inputs.insert("tok".into(), TensorData::I32(vec![5, 9, 2, -15, 1, 99]));
        assert!(plan.validate_inputs(&inputs).is_err());
        inputs.insert("tok".into(), TensorData::I32(vec![10, 9, 1, -15, 1, 99]));
        assert!(plan.validate_inputs(&inputs).is_err());
    }

    #[test]
    fn split_qkv_and_rope_compile_as_typed_device_operations() {
        let mut g = Graph::new(Stage::SlowDecode, None, None);
        let packed = g
            .input(
                "packed",
                TensorType::new([2, 3, 16], ScalarType::F32, Layout::SequenceBtd),
            )
            .unwrap();
        let q = g
            .emit(
                "q",
                Op::SplitQkv {
                    part: QkvPart::Query,
                    q_heads: 4,
                    kv_heads: 2,
                    head_dim: 2,
                },
                &[packed],
            )
            .unwrap();
        let k = g
            .emit(
                "k",
                Op::SplitQkv {
                    part: QkvPart::Key,
                    q_heads: 4,
                    kv_heads: 2,
                    head_dim: 2,
                },
                &[packed],
            )
            .unwrap();
        let v = g
            .emit(
                "v",
                Op::SplitQkv {
                    part: QkvPart::Value,
                    q_heads: 4,
                    kv_heads: 2,
                    head_dim: 2,
                },
                &[packed],
            )
            .unwrap();
        let rotated = g
            .emit(
                "rotated",
                Op::RoPe {
                    position: 3,
                    base_bits: 10_000f32.to_bits(),
                },
                &[q],
            )
            .unwrap();
        for id in [rotated, k, v] {
            g.output(id).unwrap();
        }
        let graph = CubeClGraph::compile(&g).unwrap();
        assert_eq!(graph.graph().tensor(q).unwrap().shape, [2, 4, 3, 2]);
        assert_eq!(graph.graph().tensor(k).unwrap().shape, [2, 2, 3, 2]);
    }

    #[test]
    fn original_dual_ar_and_naive_transformer_graphs_compile_without_tensorrt() {
        use fish_speech_core::{
            config::{FishModelConfig, ModelKind},
            transformer_ir::TransformerIrFactory,
        };
        let c = FishModelConfig::synthetic_dual_ar_fixture();
        let original = TransformerIrFactory::new(&c).unwrap();
        for graph in [
            original.slow_prefill(3).unwrap(),
            original.slow_decode(4).unwrap(),
            original.fast_decode(0).unwrap(),
            original.fast_decode(1).unwrap(),
        ] {
            CubeClGraph::compile(&graph).unwrap();
        }
        let mut naive = c;
        naive.kind = ModelKind::Naive;
        naive.fast = None;
        let factory = TransformerIrFactory::new(&naive).unwrap();
        CubeClGraph::compile(&factory.slow_prefill(2).unwrap()).unwrap();
        CubeClGraph::compile(&factory.slow_decode(3).unwrap()).unwrap();
    }

    #[test]
    fn generated_device_token_ids_fail_compilation_until_device_validation_exists() {
        let mut g = Graph::new(Stage::SlowPrefill, None, None);
        let tokens = g
            .input(
                "tokens",
                TensorType::new([1, 2], ScalarType::I32, Layout::Plain),
            )
            .unwrap();
        let tokens2 = g.emit("tokens2", Op::Identity, &[tokens]).unwrap();
        let weight = g
            .weight(
                "weight",
                TensorType::new([12, 4], ScalarType::F32, Layout::Plain),
            )
            .unwrap();
        let y = g.emit("y", Op::TokenEmbedding, &[tokens2, weight]).unwrap();
        g.output(y).unwrap();
        assert!(matches!(
            CubeClGraph::compile(&g),
            Err(FishError::Unsupported(_))
        ));
    }
}

#[cfg(all(test, feature = "cubecl-cpu"))]
mod tests {
    use super::*;
    use cubecl::Device;
    use fish_speech_core::{
        cpu_ir::CpuIrInterpreter,
        graph::Stage,
        graph_ir::{Layout, TensorType},
    };

    fn ty(shape: &[usize]) -> TensorType {
        TensorType::new(shape.to_vec(), ScalarType::F32, Layout::SequenceBtd)
    }

    fn run_parity(graph: Graph, mut inputs: NamedTensors) {
        for value in &graph.values {
            if value.ty.dtype == ScalarType::BF16 {
                if let Some(TensorData::F32(values)) = inputs.get_mut(&value.name) {
                    for x in values {
                        *x = half::bf16::from_f32(*x).to_f32();
                    }
                }
            }
        }
        let expected = CpuIrInterpreter::run(&graph, &inputs).unwrap();
        let executor = CubeClGraphExecutor::new(Device::Cpu(Default::default()).client());
        let actual = executor
            .run(&CubeClGraph::compile(&graph).unwrap(), &inputs)
            .unwrap();
        assert_eq!(actual.len(), expected.len());
        for (name, target) in expected {
            let got = &actual[&name];
            match (got, target) {
                (TensorData::F32(a), TensorData::F32(b)) => {
                    assert_eq!(a.len(), b.len());
                    for (a, b) in a.iter().zip(b.iter()) {
                        assert!(
                            (a - b).abs() <= 1e-4 * b.abs().max(1.0),
                            "{name}: {a} != {b}"
                        );
                    }
                }
                (a, b) => assert_eq!(*a, b),
            }
        }
    }

    #[test]
    fn cubecl_f32_linear_bias_silu_add_rms_last_token() {
        let mut g = Graph::new(Stage::SlowPrefill, None, None);
        let x = g.input("x", ty(&[1, 3, 2])).unwrap();
        let w = g
            .weight("w", TensorType::new([4, 2], ScalarType::F32, Layout::Plain))
            .unwrap();
        let bias = g
            .weight("bias", TensorType::new([4], ScalarType::F32, Layout::Plain))
            .unwrap();
        let gamma = g
            .weight(
                "gamma",
                TensorType::new([4], ScalarType::F32, Layout::Plain),
            )
            .unwrap();
        let h = g.emit("h", Op::LinearBias, &[x, w, bias]).unwrap();
        let z = g.emit("z", Op::SiLu, &[h]).unwrap();
        let r = g.emit("r", Op::Add, &[h, z]).unwrap();
        let m = g.emit("m", Op::Mul, &[r, z]).unwrap();
        let norm = g
            .emit(
                "norm",
                Op::RmsNorm {
                    epsilon_bits: 1e-5f32.to_bits(),
                },
                &[m, gamma],
            )
            .unwrap();
        let last = g.emit("last", Op::LastToken, &[norm]).unwrap();
        g.output(norm).unwrap();
        g.output(last).unwrap();
        let mut inputs = NamedTensors::new();
        inputs.insert(
            "x".into(),
            TensorData::F32(vec![-1.0, 0.25, 0.0, 1.0, 2.0, -1.0]),
        );
        inputs.insert(
            "w".into(),
            TensorData::F32(vec![1.0, 2.0, 0.5, -1.0, -1.0, 0.25, 2.0, 0.1]),
        );
        inputs.insert("bias".into(), TensorData::F32(vec![0.5, -0.5, 1.0, 0.25]));
        inputs.insert("gamma".into(), TensorData::F32(vec![1.0, 0.75, 2.0, -0.5]));
        run_parity(g, inputs);
    }

    #[test]
    fn cubecl_token_embedding_matches_cpu_reference() {
        let mut g = Graph::new(Stage::SlowPrefill, None, None);
        let tok = g
            .input(
                "tok",
                TensorType::new([2, 3], ScalarType::I32, Layout::Plain),
            )
            .unwrap();
        let weight = g
            .weight(
                "weight",
                TensorType::new([8, 4], ScalarType::F32, Layout::Plain),
            )
            .unwrap();
        let y = g.emit("y", Op::TokenEmbedding, &[tok, weight]).unwrap();
        g.output(y).unwrap();
        let mut bindings = NamedTensors::new();
        bindings.insert("tok".into(), TensorData::I32(vec![0, 7, 2, 5, 3, 1]));
        bindings.insert(
            "weight".into(),
            TensorData::F32((0..32).map(|n| n as f32 * 0.1).collect()),
        );
        run_parity(g, bindings);
    }

    #[test]
    fn cubecl_fish_embedding_matches_cpu_gated_shared_and_nonshared() {
        for shared in [false, true] {
            let mut g = Graph::new(Stage::SlowPrefill, None, None);
            let tok = g
                .input(
                    "tok",
                    TensorType::new([2, 3, 2], ScalarType::I32, Layout::TokensBct),
                )
                .unwrap();
            let word = g
                .weight(
                    "word",
                    TensorType::new([10, 2], ScalarType::F32, Layout::Plain),
                )
                .unwrap();
            let cb = g
                .weight(
                    "cb",
                    TensorType::new([4, 2], ScalarType::F32, Layout::Plain),
                )
                .unwrap();
            let lut = g
                .input("lut", TensorType::new([10], ScalarType::I32, Layout::Plain))
                .unwrap();
            let y = g
                .emit(
                    "y",
                    Op::FishEmbedding {
                        num_codebooks: 2,
                        codebook_size: 2,
                        share_codebook_embeddings: shared,
                    },
                    &[tok, word, cb, lut],
                )
                .unwrap();
            g.output(y).unwrap();
            let mut bindings = NamedTensors::new();
            bindings.insert(
                "tok".into(),
                TensorData::I32(vec![5, 9, 1, 1, 1, 0, 9, 5, -99, 1, 77, 0]),
            );
            bindings.insert(
                "word".into(),
                TensorData::F32((0..20).map(|n| n as f32 * 0.25).collect()),
            );
            bindings.insert(
                "cb".into(),
                TensorData::F32((0..8).map(|n| n as f32 * 0.33).collect()),
            );
            let mut gates = vec![0; 10];
            gates[5] = 1;
            bindings.insert("lut".into(), TensorData::I32(gates));
            run_parity(g, bindings);
        }
    }

    #[test]
    fn cubecl_packed_qkv_and_rope_match_cpu_at_nonzero_position() {
        let mut g = Graph::new(Stage::SlowDecode, None, None);
        let packed = g
            .input(
                "packed",
                TensorType::new([2, 3, 16], ScalarType::F32, Layout::SequenceBtd),
            )
            .unwrap();
        for (part, prefix) in [
            (QkvPart::Query, "q"),
            (QkvPart::Key, "k"),
            (QkvPart::Value, "v"),
        ] {
            let split = g
                .emit(
                    prefix,
                    Op::SplitQkv {
                        part,
                        q_heads: 4,
                        kv_heads: 2,
                        head_dim: 2,
                    },
                    &[packed],
                )
                .unwrap();
            let rotated = g
                .emit(
                    format!("{prefix}_rope"),
                    Op::RoPe {
                        position: 7,
                        base_bits: 10_000f32.to_bits(),
                    },
                    &[split],
                )
                .unwrap();
            g.output(rotated).unwrap();
        }
        let mut bindings = NamedTensors::new();
        bindings.insert(
            "packed".into(),
            TensorData::F32((0..96).map(|n| (n as f32 * 0.04).sin()).collect()),
        );
        run_parity(g, bindings);
    }

    #[test]
    fn cubecl_bf16_kv_update_prefix_and_causal_gqa_match_cpu() {
        let mut g = Graph::new(Stage::SlowDecode, Some(4), Some(5));
        let cache = g
            .input(
                "cache",
                TensorType::new([1, 1, 5, 2], ScalarType::BF16, Layout::CacheBhtd),
            )
            .unwrap();
        let fresh = g
            .input(
                "fresh",
                TensorType::new([1, 1, 2, 2], ScalarType::F32, Layout::HeadsBhtd),
            )
            .unwrap();
        let q = g
            .input(
                "q",
                TensorType::new([1, 2, 1, 2], ScalarType::F32, Layout::HeadsBhtd),
            )
            .unwrap();
        let updated = g
            .emit("updated", Op::KvUpdate { position: 2 }, &[cache, fresh])
            .unwrap();
        let prefix = g
            .emit("prefix", Op::KvPrefix { active_len: 4 }, &[updated])
            .unwrap();
        let attention = g
            .emit(
                "attention",
                Op::GqaAttention {
                    position: 3,
                    q_heads: 2,
                    kv_heads: 1,
                },
                &[q, prefix, prefix],
            )
            .unwrap();
        for id in [updated, prefix, attention] {
            g.output(id).unwrap();
        }
        let mut inputs = NamedTensors::new();
        inputs.insert(
            "cache".into(),
            TensorData::F32(vec![0.0, 1.0, 0.5, -0.5, 0.0, 0.0, 0.0, 0.0, 99.0, 99.0]),
        );
        inputs.insert("fresh".into(), TensorData::F32(vec![0.2, 1.2, -0.1, 0.7]));
        inputs.insert("q".into(), TensorData::F32(vec![0.1, 0.2, 0.5, -0.25]));
        run_parity(g, inputs);
    }

    #[test]
    fn cubecl_plain_linear_identity_and_last_token_across_batches() {
        let mut g = Graph::new(Stage::SlowPrefill, None, None);
        let x = g.input("x", ty(&[2, 3, 4])).unwrap();
        let weights = g
            .weight(
                "weights",
                TensorType::new([5, 4], ScalarType::F32, Layout::Plain),
            )
            .unwrap();
        let projected = g.emit("projected", Op::Linear, &[x, weights]).unwrap();
        let copied = g.emit("copied", Op::Identity, &[projected]).unwrap();
        let last = g.emit("last", Op::LastToken, &[copied]).unwrap();
        for id in [projected, copied, last] {
            g.output(id).unwrap();
        }
        let mut inputs = NamedTensors::new();
        inputs.insert(
            "x".into(),
            TensorData::F32((0..24).map(|i| ((i as f32 - 11.0) * 0.41).sin()).collect()),
        );
        inputs.insert(
            "weights".into(),
            TensorData::F32((0..20).map(|i| ((i as f32 + 3.0) * 0.29).cos()).collect()),
        );
        run_parity(g, inputs);
    }

    #[test]
    fn cubecl_bf16_gqa_multi_batch_multi_head_and_causal_mask() {
        let mut g = Graph::new(Stage::SlowDecode, Some(5), Some(5));
        let query = g
            .input(
                "query",
                TensorType::new([2, 4, 3, 4], ScalarType::F32, Layout::HeadsBhtd),
            )
            .unwrap();
        let key = g
            .input(
                "key",
                TensorType::new([2, 2, 5, 4], ScalarType::BF16, Layout::HeadsBhtd),
            )
            .unwrap();
        let value = g
            .input(
                "value",
                TensorType::new([2, 2, 5, 4], ScalarType::BF16, Layout::HeadsBhtd),
            )
            .unwrap();
        let attention = g
            .emit(
                "attention",
                Op::GqaAttention {
                    position: 2,
                    q_heads: 4,
                    kv_heads: 2,
                },
                &[query, key, value],
            )
            .unwrap();
        g.output(attention).unwrap();

        let mut inputs = NamedTensors::new();
        inputs.insert(
            "query".into(),
            TensorData::F32((0..96).map(|i| ((i as f32 + 1.0) * 0.19).cos()).collect()),
        );
        inputs.insert(
            "key".into(),
            TensorData::F32((0..80).map(|i| ((i as f32 + 2.0) * 0.11).sin()).collect()),
        );
        inputs.insert(
            "value".into(),
            TensorData::F32((0..80).map(|i| ((i as f32 - 3.0) * 0.17).cos()).collect()),
        );
        run_parity(g.clone(), inputs.clone());

        let executor = CubeClGraphExecutor::new(Device::Cpu(Default::default()).client());
        let compiled = CubeClGraph::compile(&g).unwrap();
        let before = executor.run(&compiled, &inputs).unwrap();
        if let TensorData::F32(values) = inputs.get_mut("value").unwrap() {
            for batch in 0..2 {
                for head in 0..2 {
                    for frame in 3..5 {
                        for feature in 0..4 {
                            let index = ((batch * 2 + head) * 5 + frame) * 4 + feature;
                            values[index] = 100.0 + index as f32;
                        }
                    }
                }
            }
        }
        run_parity(g.clone(), inputs.clone());
        let after = executor.run(&compiled, &inputs).unwrap();
        let TensorData::F32(before) = &before["attention"] else {
            panic!("attention must be F32");
        };
        let TensorData::F32(after) = &after["attention"] else {
            panic!("attention must be F32");
        };
        for batch in 0..2 {
            let first_query = batch * 3 * 16;
            for (left, right) in before[first_query..first_query + 16]
                .iter()
                .zip(&after[first_query..first_query + 16])
            {
                assert_eq!(left, right, "future acoustic frames leaked into causal GQA");
            }
        }
    }

    #[test]
    fn cubecl_gqa_prefill_preserves_token_major_heads_through_last_token() {
        let mut graph = Graph::new(Stage::SlowPrefill, None, None);
        let query = graph
            .input(
                "query",
                TensorType::new([2, 4, 3, 2], ScalarType::F32, Layout::HeadsBhtd),
            )
            .unwrap();
        let key = graph
            .input(
                "key",
                TensorType::new([2, 2, 3, 2], ScalarType::BF16, Layout::HeadsBhtd),
            )
            .unwrap();
        let value = graph
            .input(
                "value",
                TensorType::new([2, 2, 3, 2], ScalarType::BF16, Layout::HeadsBhtd),
            )
            .unwrap();
        let identity = graph
            .weight(
                "identity",
                TensorType::new([8, 8], ScalarType::F32, Layout::Plain),
            )
            .unwrap();
        let attention = graph
            .emit(
                "attention",
                Op::GqaAttention {
                    position: 0,
                    q_heads: 4,
                    kv_heads: 2,
                },
                &[query, key, value],
            )
            .unwrap();
        let projected = graph
            .emit("projected", Op::Linear, &[attention, identity])
            .unwrap();
        let last = graph.emit("last", Op::LastToken, &[projected]).unwrap();
        graph.output(attention).unwrap();
        graph.output(last).unwrap();

        let mut inputs = NamedTensors::new();
        inputs.insert("query".into(), TensorData::F32(vec![0.0; 2 * 4 * 3 * 2]));
        inputs.insert("key".into(), TensorData::F32(vec![0.0; 2 * 2 * 3 * 2]));
        let values = (0..2)
            .flat_map(|batch| {
                (0..2).flat_map(move |kv_head| {
                    (0..3).flat_map(move |frame| {
                        (0..2).map(move |feature| {
                            (batch * 100 + kv_head * 40 + frame * 10 + feature + 1) as f32
                        })
                    })
                })
            })
            .collect();
        inputs.insert("value".into(), TensorData::F32(values));
        inputs.insert(
            "identity".into(),
            TensorData::F32(
                (0..8)
                    .flat_map(|row| (0..8).map(move |col| if row == col { 1.0 } else { 0.0 }))
                    .collect(),
            ),
        );
        run_parity(graph, inputs);
    }

    #[test]
    fn cubecl_gqa_single_token_decode_preserves_head_order() {
        let mut graph = Graph::new(Stage::SlowDecode, Some(5), Some(5));
        let query = graph
            .input(
                "query",
                TensorType::new([2, 4, 1, 2], ScalarType::F32, Layout::HeadsBhtd),
            )
            .unwrap();
        let key = graph
            .input(
                "key",
                TensorType::new([2, 2, 5, 2], ScalarType::BF16, Layout::HeadsBhtd),
            )
            .unwrap();
        let value = graph
            .input(
                "value",
                TensorType::new([2, 2, 5, 2], ScalarType::BF16, Layout::HeadsBhtd),
            )
            .unwrap();
        let attention = graph
            .emit(
                "attention",
                Op::GqaAttention {
                    position: 4,
                    q_heads: 4,
                    kv_heads: 2,
                },
                &[query, key, value],
            )
            .unwrap();
        graph.output(attention).unwrap();
        let mut inputs = NamedTensors::new();
        inputs.insert("query".into(), TensorData::F32(vec![0.0; 2 * 4 * 2]));
        inputs.insert("key".into(), TensorData::F32(vec![0.0; 2 * 2 * 5 * 2]));
        inputs.insert(
            "value".into(),
            TensorData::F32((0..2 * 2 * 5 * 2).map(|i| (i as f32 + 1.0) * 0.5).collect()),
        );
        run_parity(graph, inputs);
    }

    #[test]
    fn cubecl_compile_rejects_unsupported_before_device_use() {
        let mut g = Graph::new(Stage::SlowPrefill, None, None);
        let ids = g
            .input(
                "tokens",
                TensorType::new([1, 2], ScalarType::I32, Layout::Plain),
            )
            .unwrap();
        let weights = g
            .weight(
                "embed",
                TensorType::new([8, 3], ScalarType::F32, Layout::Plain),
            )
            .unwrap();
        let out = g
            .emit("embed_out", Op::TokenEmbedding, &[ids, weights])
            .unwrap();
        g.output(out).unwrap();
        let compiled = CubeClGraph::compile(&g).unwrap();
        let mut bindings = NamedTensors::new();
        bindings.insert("tokens".into(), TensorData::I32(vec![1, 8]));
        bindings.insert("embed".into(), TensorData::F32(vec![0.0; 24]));
        assert!(compiled.validate_inputs(&bindings).is_err());
    }
}

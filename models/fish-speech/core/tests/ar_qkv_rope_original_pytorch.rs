use std::collections::BTreeMap;

use fish_speech_core::cpu_ir::{NamedTensors, TensorData};
use fish_speech_core::graph::Stage;
use fish_speech_core::graph_ir::{Graph, Layout, Op, QkvPart, ScalarType, TensorType};
use fish_speech_core::trtx_blueprint::TrtBlueprint;

fn goldens() -> BTreeMap<String, Vec<f32>> {
    include_str!("../fixtures/ar_qkv_rope_pytorch.txt")
        .lines()
        .map(|line| {
            let (key, values) = line.split_once(':').expect("named fixture");
            let data = values
                .split(',')
                .map(|v| v.parse::<f32>().unwrap())
                .collect();
            (key.to_string(), data)
        })
        .collect()
}

#[test]
fn native_blueprint_qkv_and_bf16_cached_rope_match_original_pytorch() {
    let golden = goldens();
    let mut graph = Graph::new(Stage::SlowDecode, None, None);
    let packed = graph
        .input(
            "packed",
            TensorType::new([2, 3, 48], ScalarType::F32, Layout::SequenceBtd),
        )
        .unwrap();
    for part in [QkvPart::Query, QkvPart::Key, QkvPart::Value] {
        let name = match part {
            QkvPart::Query => "q",
            QkvPart::Key => "k",
            QkvPart::Value => "v",
        };
        let heads = if part == QkvPart::Query { 4 } else { 2 };
        let output = graph
            .emit(
                name,
                Op::SplitQkv {
                    part,
                    q_heads: 4,
                    kv_heads: 2,
                    head_dim: 6,
                },
                &[packed],
            )
            .unwrap();
        assert_eq!(graph.tensor(output).unwrap().shape, [2, heads, 3, 6]);
        graph.output(output).unwrap();
        if part != QkvPart::Value {
            let rotated = graph
                .emit(
                    format!("{name}_rot"),
                    Op::RoPe {
                        position: 37,
                        base_bits: 10000f32.to_bits(),
                    },
                    &[output],
                )
                .unwrap();
            graph.output(rotated).unwrap();
        }
    }
    let plan = TrtBlueprint::from_graph(&graph).unwrap();
    plan.require_no_deferred_ops().unwrap();
    let mut inputs = NamedTensors::new();
    inputs.insert("packed".into(), TensorData::F32(golden["packed"].clone()));
    let result = plan.execute_cpu_reference(&inputs).unwrap();
    for (name, expected) in golden.iter().filter(|(k, _)| k.as_str() != "packed") {
        let TensorData::F32(actual) = &result[name] else {
            panic!("missing output {name}")
        };
        assert_eq!(actual.len(), expected.len(), "{name} shape");
        let max_abs = actual
            .iter()
            .zip(expected)
            .map(|(a, b)| (a - b).abs())
            .fold(0f32, f32::max);
        assert!(max_abs <= 1.5e-4, "{name} max difference {max_abs}");
    }
}

#[test]
fn native_token_embedding_gather_checks_indices_and_layout() {
    let mut graph = Graph::new(Stage::FastDecode, None, None);
    let tokens = graph
        .input(
            "tokens",
            TensorType::new([1, 3], ScalarType::I32, Layout::Plain),
        )
        .unwrap();
    let weights = graph
        .weight(
            "weights",
            TensorType::new([4, 2], ScalarType::F32, Layout::Plain),
        )
        .unwrap();
    let output = graph
        .emit("embedding", Op::TokenEmbedding, &[tokens, weights])
        .unwrap();
    graph.output(output).unwrap();
    let plan = TrtBlueprint::from_graph(&graph).unwrap();
    plan.require_no_deferred_ops().unwrap();
    let mut binding = NamedTensors::new();
    binding.insert("tokens".into(), TensorData::I32(vec![1, 3, 0]));
    binding.insert(
        "weights".into(),
        TensorData::F32(vec![0., 1., 2., 3., 4., 5., 6., 7.]),
    );
    let output = plan.execute_cpu_reference(&binding).unwrap();
    assert_eq!(
        output["embedding"],
        TensorData::F32(vec![2., 3., 6., 7., 0., 1.])
    );
    binding.insert("tokens".into(), TensorData::I32(vec![0, 4, 2]));
    assert!(plan.execute_cpu_reference(&binding).is_err());
    binding.insert("tokens".into(), TensorData::I32(vec![0, -1, 2]));
    assert!(plan.execute_cpu_reference(&binding).is_err());
}

#[test]
fn blueprint_active_prefix_never_reads_poisoned_cache_tail() {
    let mut graph = Graph::new(Stage::SlowDecode, Some(3), Some(8));
    let cache = graph
        .input(
            "cache",
            TensorType::new([2, 2, 8, 4], ScalarType::BF16, Layout::CacheBhtd),
        )
        .unwrap();
    let prefix = graph
        .emit("prefix", Op::KvPrefix { active_len: 3 }, &[cache])
        .unwrap();
    graph.output(prefix).unwrap();
    let plan = TrtBlueprint::from_graph(&graph).unwrap();
    plan.require_no_deferred_ops().unwrap();
    let mut input = Vec::new();
    for row in 0..4 {
        for frame in 0..8 {
            for dim in 0..4 {
                input.push(if frame < 3 {
                    (row * 100 + frame * 10 + dim) as f32
                } else {
                    f32::NAN
                });
            }
        }
    }
    let mut values = NamedTensors::new();
    values.insert("cache".into(), TensorData::F32(input));
    let result = plan.execute_cpu_reference(&values).unwrap();
    let TensorData::F32(data) = &result["prefix"] else {
        panic!("expected BF16 test data")
    };
    assert_eq!(data.len(), 2 * 2 * 3 * 4);
    assert!(data.iter().all(|x| x.is_finite()));
    assert_eq!(data[0], 0.);
    assert_eq!(data[4], 10.);
    assert_eq!(data[3 * 4], 100.);
}
